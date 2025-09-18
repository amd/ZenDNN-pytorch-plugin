# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************


# from utils import calc_cores, get_model, get_dataset, get_args

import time
import random
import numpy as np
import os
import psutil
import argparse
import sys
from enum import Enum
from typing import Dict, List, Tuple, Any

import torch
from torch import multiprocessing as mp
from torch.profiler import profile, record_function, ProfilerActivity

from transformers import BertTokenizer, BertModel

import datasets

# # --------------------------------------------
# # DEBUG: Check env variables passed from launch.json
# # --------------------------------------------
# for name, value in os.environ.items():
#     print(f"{name}: {value}")
# sys.exit()


class AffinityMode(Enum):
    pid = 1
    tid = 2

    def __str__(self) -> str:
        return self.name


# --------------------------------------------------------------
# Modifications from Elton
# --------------------------------------------------------------
def is_hyperthreading_enabled() -> bool:
    # Get logical and physical CPU counts
    logical_cpus = psutil.cpu_count(logical=True)
    physical_cpus = psutil.cpu_count(logical=False)
    # print('DEBUG:', 'logical_cpus  =', logical_cpus)
    # print('DEBUG:', 'physical_cpus =', physical_cpus)
    return logical_cpus > physical_cpus


ht_enabled = is_hyperthreading_enabled()
if ht_enabled:
    print("Hyperthreading is enabled", flush=True)
# --------------------------------------------------------------
# end: Modifications from Elton
# --------------------------------------------------------------


num_sockets = int(os.getenv("NUM_SOCKETS", 2))
cpus_per_socket = int(os.getenv("CPUS_PER_SOCKET", 64))
cpus_for_loadgen = int(os.getenv("CPUS_FOR_LOADGEN", 1))
cpus_per_instance = int(os.getenv("CPUS_PER_INSTANCE", 2))
# print('DEBUG:', 'OMP_NUM_THREADS =', os.getenv("OMP_NUM_THREADS"))
# print('DEBUG:', 'cpus_per_instance =', cpus_per_instance)
# AMD:NEW if int(os.getenv("OMP_NUM_THREADS", 0)) > 0:
# AMD:NEW     # cpus_per_instance = int(os.getenv("OMP_NUM_THREADS", 0))                          # ORIGINAL
# AMD:NEW     cpus_per_instance = int(os.getenv("OMP_NUM_THREADS", 0)) * (2 if ht_enabled else 1) # Elton
# AMD:NEW print('DEBUG:', 'cpus_per_instance =', cpus_per_instance)
# sys.exit()

MD_TYPE_LIST: List[str] = [
    "bert",
]

MD_CLASSES: Dict[str, Any] = {
    "bert": (BertModel, BertTokenizer),
}

sst2 = datasets.load_dataset("glue", "sst2")
sst2_sentences = sst2["validation"]["sentence"]

sst2_sentences_tok: List[Any] = []
sst2_sentences_slen: List[Any] = []


def tok_data(tokenizer: Any, args: Any) -> None:
    global sst2_sentences_tok, sst2_sentences_slen  # noqa: F824
    pad = True if args.seq_len > 0 else False
    max_len = args.seq_len
    for d in sst2_sentences:
        if pad:
            enc_d = tokenizer(
                d, padding="max_length", max_length=max_len, return_tensors="pt"
            )
        else:
            enc_d = tokenizer(d, return_tensors="pt")
        sst2_sentences_slen.append(enc_d.input_ids.shape[1])
        sst2_sentences_tok.append(enc_d)
    print(f"Size of converted data {len(sst2_sentences_tok)}", flush=True)


def get_data(start: int, end: int = -1) -> List[Any]:
    if end < 0:
        print(f"Returning data warmup {len(sst2_sentences_tok)}", flush=True)
        return sst2_sentences_tok[:start]
    elif end > len(sst2_sentences_tok):
        print(f"Returning data full {len(sst2_sentences_tok)}", flush=True)
        return sst2_sentences_tok[start:]
    else:
        print("Returning data slice", flush=True)
        return sst2_sentences_tok[start:end]


def get_num_cpus(logical: bool = False) -> int:
    num_cpus = num_sockets * cpus_per_socket
    ps_num_cpus = psutil.cpu_count(logical=logical)
    if num_cpus < ps_num_cpus:
        return num_cpus
    else:
        return ps_num_cpus


@torch.no_grad()
def infer(model: Any, inputs: Any, dtype: Any = torch.float32) -> Any:
    with torch.autocast(device_type="cpu", enabled=dtype == torch.bfloat16):
        return model(**inputs)


def trace_handler(prof):
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=-1))


def run_profiler(model, inputs, inst: int, dtype: Any = torch.float32):
    if inst:
        return
    with profile(
        activities=[ProfilerActivity.CPU],
        # disabling due to known memory issues
        # record_shapes=True,
        on_trace_ready=trace_handler,
    ):
        for d in inputs:
            with record_function("model_inference"):
                infer(model=model, inputs=d, dtype=dtype)


def change_proc_affinity(pid, affinity: List[int]):
    from zentorch.utils import thread_bind

    thread_bind(core_ids=affinity)


def change_proc_affinity_os(pid, affinity: List[int], mode=AffinityMode.pid) -> None:
    if mode == AffinityMode.pid:
        os.sched_setaffinity(pid, affinity)
    else:
        print("Use tid mode", flush=True)
        for tid in map(
            int, os.listdir(os.path.sep.join(["", "proc", str(pid), "task"]))
        ):
            try:
                os.sched_setaffinity(tid, affinity)
            except Exception as e:
                print(e)
                pass


def to_bfloat16(data: Dict[Any, Any]) -> None:
    for _, val in data.items():
        if torch.is_tensor(val):
            val = val.to(dtype=torch.bfloat16)
    return data


def sub_process(
    model,
    sample_inputs: List[Any],
    task: List[Tuple[int, int]],
    # result_tensor,
    finished_event,
    execute_event,
    # is_accuracy_mode,
    num_warmups: int,
    is_bfloat16: bool,
    enable_profiling: bool,
    use_zentorch: bool,
    affinity: List[Any],
    init_counter,
    lock,
    i_inst,
):
    # os based, psutil based and pid/tid based affinity settings may not work
    # with multiprocessing when > 1 threads/sub-process are involved
    # need to use pthread based binding available in zentorch
    # os.sched_setaffinity(os.getpid(), affinity)
    # change_proc_affinity_os(pid=os.getpid(), affinity=affinity, mode=AffinityMode.tid)
    change_proc_affinity(pid=os.getpid(), affinity=affinity)

    wrap_with_zt(model=model, use_zt=use_zentorch)

    print(f"warming up in sub_process {i_inst}", flush=True)
    start = time.time()
    dtype = torch.bfloat16 if is_bfloat16 else torch.float32
    for i in range(num_warmups):
        _ = infer(model=model, inputs=sample_inputs[i], dtype=dtype)
    end = time.time()

    print(
        f"warmed up in sub_process {i_inst} in :{(end - start):.4f} seconds", flush=True
    )

    if enable_profiling:
        print("Running Profiled run")
        run_profiler(model=model, inputs=sample_inputs, inst=i_inst, dtype=dtype)

    data_len = len(task)
    print(f"size of data in sub_process {i_inst} :{data_len}", flush=True)

    with lock:
        init_counter.value += 1
    execute_event.wait()

    for i in range(data_len):
        _ = infer(model=model, inputs=task[i], dtype=dtype)

    finished_event.set()


def get_auto_args(md_type: str, dtype: torch.float32) -> Dict[str, any]:
    try:
        check_md_type(md_type=md_type)
    except Exception as e:
        print(e)

    auto_args = Dict[str, any]
    auto_args = {
        "torchscript": True,
        "return_dict": False,
        "torch_dtype": dtype,
    }

    return auto_args


def wrap_with_zt(model, use_zt: bool = False):
    backend = "inductor"
    if use_zt:
        import zentorch  # noqa: F401

        backend = "zentorch"
        print("Compiling with zentorch backend")
    model.forward = torch.compile(model.forward, backend=backend)
    return model


def check_md_type(md_type: str):
    md_type_list = MD_TYPE_LIST
    md_type_list_str = ",".join(md_type_list)
    md_type = md_type.lower()
    md_key = str()
    for _md_t in md_type_list:
        if _md_t in md_type:
            md_key = _md_t
    if not md_key:
        print("Model type '{}' is not supported yet".format(md_type))
        print("Only these models are supported currently '{}'".format(md_type_list_str))
        raise RuntimeError
    return md_key


def get_model_and_tok(args=None) -> tuple[Any, Any]:
    if args is None:
        return None, None

    md_type = args.model_type
    try:
        md_key = check_md_type(md_type=md_type)
    except Exception as e:
        print(e)
        return None, None

    dtype = args.dtype
    torch_dtype = getattr(torch, dtype)

    model_cls, tokenizer_cls = MD_CLASSES[md_key]
    auto_args = get_auto_args(md_type=md_type, dtype=torch_dtype)
    tokenizer = tokenizer_cls.from_pretrained(md_type, **auto_args)
    model = model_cls.from_pretrained(md_type, **auto_args)
    model = model.cpu().share_memory()

    return model, tokenizer


def fill_tasks(task_list, args=None) -> None:
    print("filling tasks", flush=True)
    sample_count = len(sst2_sentences_tok)
    num_of_instances = len(task_lists)
    step = sample_count // num_of_instances

    for i in range(num_of_instances):
        d_start = i * step
        if d_start + step > sample_count:
            d_end = sample_count
        else:
            d_end = d_start + step
        task_list[i].append((d_start, d_end))

    print("tasks filled", flush=True)


def fill_data_tasks(task_list, task_data_list, args=None) -> None:
    print("filling data tasks", flush=True)
    for i, task in enumerate(task_list):
        start, end = task[0]
        task_data_list[i] = get_data(start=start, end=end)


def proc_inp_noshare(tokenizer, data, num_inf_req: int) -> Tuple[List[Any], List[Any]]:
    data_sz = len(data)
    if num_inf_req > data_sz:
        print(
            "Cannot chunk dataset of size '{}' to '{}' inf requests".format(
                data_sz, num_inf_req
            )
        )
        sys.exit()
    bin_sz = data_sz // num_inf_req
    inpList: List[Any] = []
    slenList: List[Any] = []
    for req in range(num_inf_req):
        d_start = req * bin_sz
        if d_start + bin_sz > data_sz:
            d_end = data_sz
        else:
            d_end = d_start + bin_sz
        slen_d_lst: List[Any] = []
        enc_d_lst: List[Any] = []
        for d in data[d_start:d_end]:
            enc_d = tokenizer(d, return_tensors="pt")
            slen_d_lst.append(enc_d.input_ids.shape[1])
            enc_d_lst.append(enc_d)
        slenList.append(slen_d_lst)
        inpList.append(enc_d_lst)

    return inpList, slenList


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-mt",
        "--model-type",
        type=str,
        required=True,
        help="model type - llama7b/chatglm3",
    )
    parser.add_argument(
        "-ws", "--warmup-steps", type=int, default=3, help="number of warmup steps"
    )
    parser.add_argument(
        "-zt",
        "--use-zentorch",
        action="store_true",
        help="use zentorch as the torch.compile backend",
    )
    parser.add_argument(
        "-prof",
        "--enable-profiler",
        action="store_true",
        help="enable PyTorch profiler",
    )
    parser.add_argument(
        "-dt",
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "bfloat16"],
        help="datatype to use",
    )
    parser.add_argument(
        "-dl",
        "--num-data-loaders",
        type=int,
        default=1,
        choices=[1],
        help="number of data loader processes to spawn",
    )
    parser.add_argument(
        "-nr",
        "--num-requests",
        type=int,
        default=1,
        help="number of concurrent inference requests",
    )
    parser.add_argument(
        "-sl",
        "--seq-len",
        type=int,
        default=-1,
        help="sequence length to which sequences will be padded",
    )
    parser.add_argument(
        "-sys",
        "--system",
        type=str,
        default="baremetal",
        choices=["baremetal", "vm"],
        help="environment where the flow is run: BareMetal or VM",
    )
    parser.add_argument(
        "-hcpu",
        "--num_host_cpu",
        type=int,
        default=0,
        help="number of physical CPU cores reserved for Host if --system='vm'",
    )

    args = parser.parse_args()

    system = args.system  # AMD:NEW
    num_host_cpu = args.num_host_cpu  # AMD:NEW

    if (system == "baremetal") & (num_host_cpu > 0):  # AMD:NEW
        print(
            "Warning: a non-zero value specified in the --num_host_cpu option for --system='baremetal'."
        )  # AMD:NEW
    if (system == "vm") & (num_host_cpu == 0):  # AMD:NEW
        print(
            "Error: the number of host CPU for --system='vm' should be greater than 0.",
            "Use the --num_host_cpu option to specify the number of logical CPUs allocated for Host.",
        )  # AMD:NEW
        sys.exit()

    if int(os.getenv("OMP_NUM_THREADS", 0)) > 0:  # AMD:NEW
        # cpus_per_instance = int(os.getenv("OMP_NUM_THREADS", 0))                          # ORIGINAL                     # AMD:NEW
        # cpus_per_instance = int(os.getenv("OMP_NUM_THREADS", 0)) * (2 if ht_enabled else 1) # Elton                      # AMD:NEW
        cpus_per_instance = int(os.getenv("OMP_NUM_THREADS", 0)) * (
            2 if ht_enabled & (system == "vm") else 1
        )  # Elton     # AMD:NEW
    # print('DEBUG:', 'cpus_per_instance =', cpus_per_instance)                                                            # AMD:NEW

    cpu_count = get_num_cpus()

    cpus_for_loadgen = args.num_data_loaders
    proc_inst_start_idx = [
        x * cpus_per_instance + cpus_for_loadgen for x in range(args.num_requests)
    ]

    # os based, psutil based and pid/tid based affinity settings may not work
    # with multiprocessing when > 1 threads/sub-process are involved
    # need to use pthread based binding available in zentorch
    # os.sched_setaffinity(os.getpid(), range(cpus_for_loadgen))
    # change_proc_affinity_os(pid=os.getpid(), affinity=range(cpus_for_loadgen), mode=AffinityMode.tid)
    change_proc_affinity(pid=os.getpid(), affinity=range(cpus_for_loadgen))

    manager = mp.Manager()
    lock = manager.Lock()
    init_counter = manager.Value("i", 0)
    task_lists = [[] for _ in proc_inst_start_idx]
    task_data_lists = [[] for _ in proc_inst_start_idx]
    finished_event = [manager.Event() for _ in proc_inst_start_idx]
    execute_event = manager.Event()
    mp.set_start_method("spawn", force=True)
    seed = 1234
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    model, tokenizer = get_model_and_tok(args=args)
    if model is None or tokenizer is None:
        print("Cannot initialize either model or tokenizer or both")
        sys.exit()
    tok_data(tokenizer=tokenizer, args=args)

    fill_tasks(task_list=task_lists)
    fill_data_tasks(task_list=task_lists, task_data_list=task_data_lists)
    sample_inputs = get_data(start=args.warmup_steps)
    num_of_instances = len(proc_inst_start_idx)

    consumers = [
        mp.Process(
            target=sub_process,
            args=(
                model,  # compiled_model,
                sample_inputs,
                task_data_lists[i],
                # result_tensor,
                finished_event[i],
                execute_event,
                # args.accuracy_mode,
                args.warmup_steps,  # args.num_warmups,
                args.dtype == "bfloat16",  # args.model == "quant16",
                args.enable_profiler,
                args.use_zentorch,
                list(
                    range(
                        # proc_inst_start_idx[i],                                  # AMD:NEW
                        # proc_inst_start_idx[i] + cpus_per_instance,              # AMD:NEW
                        # (2 if ht_enabled else 1)                       # Elton   # AMD:NEW
                        proc_inst_start_idx[i] + num_host_cpu,  # AMD:NEW
                        proc_inst_start_idx[i]
                        + cpus_per_instance
                        + num_host_cpu,  # AMD:NEW
                        (
                            2 if ht_enabled & (system == "vm") else 1
                        ),  # Elton   # AMD:NEW
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

    duration = end_time - start_time
    sum_seq_len = sum(sst2_sentences_slen)
    print(f"Average sequence length: {sum_seq_len / len(sst2_sentences_tok):.2f}")
    print(f"Average processing time: {duration / len(sst2_sentences_tok) * 1e3:.2f} ms")
    print(f"Duration:                {duration:.2f} seconds")

    for consumer in consumers:
        consumer.join()
