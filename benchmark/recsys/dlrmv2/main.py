# ******************************************************************************
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

from utils import calc_cores, get_args, get_dataset
from model_loader import get_compiled_model
from torch import multiprocessing as mp
import time
import torch
import random
import numpy as np
import os
from torch.profiler import profile, record_function, ProfilerActivity
import sklearn
import zentorch

# The default values for the environment variables are set of a 128 core 2P turin machine.
num_sockets = int(os.getenv("NUM_SOCKETS", 2))  # number of CPU sockets
cpus_per_socket = int(
    os.getenv("CPUS_PER_SOCKET", 128)
)  # number of CPU cores per socket
cpus_for_loadgen = int(
    os.getenv("CPUS_FOR_LOADGEN", 1)
)  # number of CPU cores for loadgen
cpus_per_instance = int(
    os.getenv("CPUS_PER_INSTANCE", 2)
)  # number of CPU cores per instance

args = get_args()

if args.enable_freezing:
    torch._inductor.config.freezing = True

ds = get_dataset(dataset=args.dataset, dataset_path=args.dataset_path)


def dlrm_wrap(model, densex, index, offset):
    with torch.no_grad():
        out = model(densex, index, offset)
    return out


def trace_handler(prof):
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=-1))


def profiled_run(model, densex, index, offset):
    with profile(
        activities=[ProfilerActivity.CPU],
        record_shapes=True,
        on_trace_ready=trace_handler,
    ), record_function("model_inference"):
        dlrm_wrap(model, densex, index, offset)


def sub_process(
    model,
    model_type,
    sample_inputs,
    task_list,
    result_tensor,
    finished_event,
    execute_event,
    is_accuracy_mode,
    num_warmups,
    is_bfloat16,
    enable_profiling,
    affinity,
    init_counter,
    lock,
    i,
):
    zentorch.utils.thread_bind(affinity)
    if model_type == "export_quant32":
        model = model.module(check_guards=False)
        model = torch.compile(model, backend="zentorch")

    densex, index, offset, labels = sample_inputs

    if is_bfloat16:
        densex = densex.to(dtype=torch.bfloat16)

    print(f"warming up in sub_process {i}", flush=True)
    start = time.time()
    with torch.no_grad():
        for _ in range(num_warmups):
            _ = dlrm_wrap(model, densex, index, offset)
    end = time.time()
    print(f"warmed up in sub_process {i} in :{(end - start):.4f} seconds", flush=True)

    if enable_profiling:
        print("Running Profiled run")
        profiled_run(model, densex, index, offset)
    # out, time_taken=dlrm_wrap(model,densex, index, offset)
    # print(f"iteration time for process {i}: {time_taken:.4f} seconds")

    with lock:
        init_counter.value += 1

    if is_accuracy_mode:
        results = []
        execute_event.wait()
        for task in task_list:
            sample_s, sample_e = task
            densex, index, offset, labels = ds.val_data.load_batch(
                range(sample_s, sample_e)
            )
            if is_bfloat16:
                densex = densex.to(dtype=torch.bfloat16)
            out = dlrm_wrap(model, densex, index, offset)
            results.append(torch.cat((out.unsqueeze(1), labels.unsqueeze(1)), dim=1))
        for j in range(len(results)):
            result_tensor[i * len(task_list) + j] = results[j]
        finished_event.set()
    else:

        execute_event.wait()

        for _ in range(len(task_list)):
            out = dlrm_wrap(model, densex, index, offset)
        finished_event.set()


def fill_tasks(task_lists, args):
    print("filling tasks", flush=True)

    sample_count = args.sample_count
    num_of_instances = len(task_lists)
    step = ((sample_count // num_of_instances) // args.batch_size) * args.batch_size
    for i in range(num_of_instances):
        task_lists[i] = [
            (j, j + args.batch_size)
            for j in range(i * step, (i + 1) * step, args.batch_size)
        ]
    print("tasks filled", flush=True)
    return num_of_instances * (step // args.batch_size)


def calculate_accuracy(result_tensor):
    results = result_tensor[:, :, 0].flatten()
    targets = result_tensor[:, :, 1].flatten()

    print("\nTotal ROC AUC = ", sklearn.metrics.roc_auc_score(targets, results))
    print("Verified", flush=True)


if __name__ == "__main__":

    cpus_for_loadgen, proc_inst_start_idx = calc_cores(
        num_sockets,
        cpus_per_socket,
        cpus_per_instance,
        cpus_for_loadgen,
    )
    print(f"Using {cpus_for_loadgen}, cores for loadgen", flush=True)

    zentorch.utils.thread_bind(range(cpus_for_loadgen))

    print("Loading Dataset", flush=True)

    ds = get_dataset(dataset=args.dataset, dataset_path=args.dataset_path)
    sample_inputs = ds.val_data.load_batch(range(0, args.batch_size))

    print("Dataset Loaded", flush=True)

    manager = mp.Manager()
    lock = manager.Lock()
    init_counter = manager.Value("i", 0)
    task_lists = [[] for _ in proc_inst_start_idx]
    finished_event = [manager.Event() for _ in proc_inst_start_idx]
    execute_event = manager.Event()
    mp.set_start_method("spawn", force=True)

    seed = 1234
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    compiled_model = get_compiled_model(args)

    num_of_queries = fill_tasks(task_lists, args)
    result_tensor = torch.empty((num_of_queries, args.batch_size, 2)).share_memory_()

    num_of_instances = len(proc_inst_start_idx)

    consumers = [
        mp.Process(
            target=sub_process,
            args=(
                compiled_model,
                args.model,
                sample_inputs,
                task_lists[i],
                result_tensor,
                finished_event[i],
                execute_event,
                args.accuracy_mode,
                args.num_warmups,
                args.model == "quant16",
                args.enable_profiling,
                list(
                    range(
                        proc_inst_start_idx[i],
                        proc_inst_start_idx[i] + cpus_per_instance,
                    )
                ),
                init_counter,
                lock,
                i,
            ),
        )
        for i in range(num_of_instances)
    ]

    print("Starting sub_process", flush=True)
    for consumer in consumers:
        consumer.start()
    print("Started all sub_process", flush=True)

    while init_counter.value < num_of_instances:
        time.sleep(2)

    print("ALL READY", flush=True)
    start_time = time.time()
    execute_event.set()
    for event in finished_event:
        event.wait()
    end_time = time.time()
    print("ALL DONE", flush=True)
    if args.accuracy_mode:
        calculate_accuracy(result_tensor)
        print(
            "throughput = "
            f"{num_of_queries * args.batch_size / (end_time - start_time)}"
            " Samples per second",
            flush=True,
        )
    else:
        print(
            "throughput = "
            f"{num_of_queries * args.batch_size / (end_time - start_time)}"
            " Samples per second",
            flush=True,
        )
    for consumer in consumers:
        consumer.join()

    manager.shutdown()
