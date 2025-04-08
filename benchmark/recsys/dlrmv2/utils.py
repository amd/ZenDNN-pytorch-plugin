# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************
import argparse
from multihot_criteo import MultihotCriteo


def calc_cores(
    num_sockets,
    cpus_per_socket,
    cpus_per_instance,
    cpus_for_loadgen,
):
    """
    calculate the number of cores for loadgen and
    the starting index of the instances
    """

    assert (
        cpus_for_loadgen < cpus_per_socket
    ), "Multisocket for loadgen is not supported."
    socket0_predict_cores = (
        (cpus_per_socket - cpus_for_loadgen) // cpus_per_instance
    ) * cpus_per_instance
    cpus_for_loadgen = cpus_per_socket - socket0_predict_cores
    proc_inst_start_idx = list(
        range(cpus_for_loadgen, cpus_per_socket, cpus_per_instance)
    )

    for i in range(num_sockets - 1):
        proc_inst_start_idx.extend(
            list(
                range(
                    (i + 1) * cpus_per_socket,
                    (i + 2) * cpus_per_socket - cpus_per_socket % cpus_per_instance,
                    cpus_per_instance,
                )
            )
        )
    return cpus_for_loadgen, proc_inst_start_idx


def get_args():
    """Parse commandline."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True, help="path to the dataset")
    parser.add_argument("--dataset", default="multihot-criteo", help="dataset name")
    parser.add_argument(
        "--sample_count", type=int, default=89000000, help="number of samples to use"
    )
    parser.add_argument("--batch_size", type=int, default=100, help="batch size")
    parser.add_argument(
        "--accuracy_mode",
        action="store_true",
        help="Run in accuracy mode.(default: performance)",
    )
    parser.add_argument(
        "--num_warmups", type=int, default=10, help="number of warmup runs"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path of model to load"
    )
    parser.add_argument(
        "--num_runs", type=int, default=1000, help="number of runs for throughput mode"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="quant32",
        help="type of model to load [quant32,fp32,qdq,quant16]",
    )
    parser.add_argument(
        "--enable_profiling", action="store_true", help="Enable profiling"
    )
    args = parser.parse_args()
    return args


def get_dataset(dataset_path, dataset):
    """Get the corresponding dataset."""

    ds = MultihotCriteo(
        data_path=dataset_path,
        name=dataset,
        memory_map=True,
    )
    return ds
