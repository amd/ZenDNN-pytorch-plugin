#  *****************************************************************************
#  * Modifications Copyright (c) 2025 Advanced Micro Devices, Inc.
#  * All rights reserved.
#  *
#  * Was sourced from
#  * https://github.com/mlcommons/inference_results_v3.1/blob/main/closed/Intel/code/dlrm-v2-99/pytorch-cpu-int8/python/runner.py  # noqa: B950
#  * commit ID: eaf622a
#  ******************************************************************************
"""
mlperf inference benchmarking tool
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import collections
import json
import logging
import os
import sys
from torch import multiprocessing
import threading
import time
import math

import mlperf_loadgen as lg
import numpy as np
from shutil import copyfile

import torch
from items import Item
from consumer import Consumer
from multihot_criteo import get_dataset, SUPPORTED_DATASETS
import sklearn


logging.basicConfig(level=logging.INFO)
log = logging.getLogger("main")

num_sockets = int(os.getenv("NUM_SOCKETS", 2))
cpus_per_socket = int(os.getenv("CPUS_PER_SOCKET", 128))
cpus_per_consumer = int(os.getenv("CPUS_PER_CONSUMER", 128))
cpus_for_loadgen = int(os.getenv("CPUS_FOR_LOADGEN", 1))
cpus_per_instance = int(os.getenv("CPUS_PER_INSTANCE", 2))

NANO_SEC = 1e9

# the datasets we support
# pre-defined command line options so simplify things. They are used as
# defaults and can be overwritten from command line

SUPPORTED_PROFILES = {
    "defaults": {
        "dataset": "multihot-criteo",
        "inputs": "continuous and categorical features",
        "outputs": "probability",
        "backend": "pytorch-native",
        "model": "dlrm",
        "max-batchsize": 2048,
    },
    "dlrm-debug-pytorch": {
        "dataset": "debug",
        "inputs": "continuous and categorical features",
        "outputs": "probability",
        "backend": "pytorch-native",
        "model": "dlrm",
        "max-batchsize": 128,
    },
    "dlrm-multihot-sample-pytorch": {
        "dataset": "multihot-criteo-sample",
        "inputs": "continuous and categorical features",
        "outputs": "probability",
        "backend": "pytorch-native",
        "model": "dlrm",
        "max-batchsize": 2048,
    },
    "dlrm-multihot-pytorch": {
        "dataset": "multihot-criteo",
        "inputs": "continuous and categorical features",
        "outputs": "probability",
        "backend": "pytorch-native",
        "model": "dlrm",
        "max-batchsize": 2048,
    },
}

SCENARIO_MAP = {
    "SingleStream": lg.TestScenario.SingleStream,
    "MultiStream": lg.TestScenario.MultiStream,
    "Server": lg.TestScenario.Server,
    "Offline": lg.TestScenario.Offline,
}

start_time = 0
item_good = 0
item_total = 0
total_instances = 0
item_timing = []
item_results = []
last_timeing = []


def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="name of the mlperf model, ie. dlrm")
    parser.add_argument("--model-path", help="path to the model file")
    parser.add_argument(
        "--dataset", choices=SUPPORTED_DATASETS.keys(), help="dataset"
    )
    parser.add_argument(
        "--dataset-path", required=True, help="path to the dataset"
    )
    parser.add_argument(
        "--profile", choices=SUPPORTED_PROFILES.keys(), help="standard profiles"
    )
    parser.add_argument(
        "--enable-profiling",
        type=bool,
        default=False,
        help="enable pytorch profiling",
    )
    parser.add_argument(
        "--scenario",
        default="Offline",
        help="mlperf benchmark scenario, one of "
        + str(list(SCENARIO_MAP.keys())),
    )
    parser.add_argument(
        "--test-num-workers",
        type=int,
        default=0,
        help="# of workers reading the data",
    )
    parser.add_argument("--max-ind-range", type=int, default=-1)
    parser.add_argument("--data-sub-sample-rate", type=float, default=0.0)
    parser.add_argument(
        "--max-batchsize", type=int, help="max batch size in a single inference"
    )
    parser.add_argument("--output", help="test results")
    parser.add_argument("--inputs", help="model inputs (currently not used)")
    parser.add_argument("--outputs", help="model outputs (currently not used)")
    parser.add_argument("--backend", help="runtime to use")
    parser.add_argument("--use-gpu", action="store_true", default=False)
    parser.add_argument("--use-int8-fp32", action="store_true", default=False)
    parser.add_argument("--use-int8-bf16", action="store_true", default=False)
    parser.add_argument("--use-bf16", action="store_true", default=False)
    parser.add_argument(
        "--int8-model-path",
        default="dlrm_int8.pt",
        type=str,
        metavar="PATH",
        help="path to int8 model, default file name is dlrm_int8.pt",
    )
    parser.add_argument("--threads", default=1, type=int, help="threads")
    parser.add_argument(
        "--cache", type=int, default=0, help="use cache (currently not used)"
    )
    parser.add_argument(
        "--accuracy", action="store_true", help="enable accuracy pass"
    )
    parser.add_argument(
        "--find-peak-performance",
        action="store_true",
        help="enable finding peak performance pass",
    )

    # file to use mlperf rules compliant parameters
    parser.add_argument(
        "--config", default="../mlperf.conf", help="mlperf rules config"
    )
    parser.add_argument(
        "--user-config", default="./user.conf", help="mlperf rules user config"
    )

    # below will override mlperf rules compliant settings - don't use for
    # official submission
    parser.add_argument(
        "--duration", type=int, help="duration in milliseconds (ms)"
    )
    parser.add_argument("--target-qps", type=int, help="target/expected qps")
    parser.add_argument(
        "--max-latency", type=float, help="mlperf max latency in pct tile"
    )
    parser.add_argument(
        "--count-samples", type=int, help="dataset items to use"
    )
    parser.add_argument(
        "--count-queries", type=int, help="number of queries to use"
    )
    parser.add_argument(
        "--samples-per-query-multistream",
        type=int,
        help="query length for multi-stream scenario (in terms of aggregated samples)",
    )
    # --samples-per-query-offline is equivalent to perf_sample_count
    parser.add_argument(
        "--samples-per-query-offline",
        type=int,
        default=2048,
        help="query length for offline scenario (in terms of aggregated samples)",
    )
    parser.add_argument(
        "--samples-to-aggregate-fix",
        type=int,
        help="number of samples to be treated as one",
    )
    parser.add_argument(
        "--samples-to-aggregate-min",
        type=int,
        help="min number of samples to be treated as one in random query size",
    )
    parser.add_argument(
        "--samples-to-aggregate-max",
        type=int,
        help="max number of samples to be treated as one in random query size",
    )
    parser.add_argument(
        "--samples-to-aggregate-quantile-file",
        type=str,
        help=(
            "distribution quantile used to generate number of samples to be"
            "treated as one in random query size",
        ),
    )
    parser.add_argument(
        "--samples-to-aggregate-trace-file",
        type=str,
        default="dlrm_trace_of_aggregated_samples.txt",
    )
    parser.add_argument("--numpy-rand-seed", type=int, default=123)
    args = parser.parse_args()

    # set random seed
    np.random.seed(args.numpy_rand_seed)

    # don't use defaults in argparser. Instead we default to a dict, override
    # that with a profile and take this as default unless command line give
    defaults = SUPPORTED_PROFILES["defaults"]

    if args.profile:
        profile = SUPPORTED_PROFILES[args.profile]
        defaults.update(profile)
    for k, v in defaults.items():
        kc = k.replace("-", "_")
        if getattr(args, kc) is None:
            setattr(args, kc, v)
    if args.inputs:
        args.inputs = args.inputs.split(",")
    if args.outputs:
        args.outputs = args.outputs.split(",")

    if args.scenario not in SCENARIO_MAP:
        parser.error("valid scanarios:" + str(list(SCENARIO_MAP.keys())))
    return args


class OfflineQueueRunner:
    def __init__(self, inQueue, dataset, max_batchsize):
        self.inQueue = inQueue
        self.ds = dataset
        self.max_batchsize = max_batchsize
        self.sample_len_dict = {}
        self.sample_len_num = {}
        self.temp_query_buffer = []
        self.temp_idx_buffer = []
        self.temp_idx_offset_buffer = []

    def enqueue(self, query_id, idx):
        query_len = len(query_id)
        handled_query = 0
        # issued_tasks = 0
        for i in range(0, query_len):
            sample_len = self.ds.items_in_memory_len[idx[i]]
            cur_sample_len = self.sample_len_num[sample_len] + sample_len
            if cur_sample_len < self.max_batchsize:
                self.sample_len_dict[sample_len].append((query_id[i], idx[i]))
                self.sample_len_num[sample_len] = cur_sample_len
            else:
                while self.sample_len_dict[sample_len]:
                    (temp_query_id, temp_idx_id) = self.sample_len_dict[
                        sample_len
                    ].pop()
                    self.temp_query_buffer.append(temp_query_id)
                    self.temp_idx_buffer.append(temp_idx_id)
                    self.temp_idx_offset_buffer.append(sample_len)
                    handled_query = handled_query + 1
                self.temp_query_buffer.append(query_id[i])
                self.temp_idx_buffer.append(idx[i])
                self.temp_idx_offset_buffer.append(sample_len)
                handled_query = handled_query + 1
                self.inQueue.put(
                    Item(
                        self.temp_query_buffer,
                        self.temp_idx_buffer,
                        self.temp_idx_offset_buffer,
                    )
                )
                self.temp_query_buffer = []
                self.temp_idx_buffer = []
                self.temp_idx_offset_buffer = []
                self.sample_len_num[sample_len] = 0

        current_bs = 0
        while handled_query < query_len:
            for sample_len in self.sorted_keys:
                while self.sample_len_dict[sample_len]:
                    current_bs = current_bs + sample_len
                    if current_bs > self.max_batchsize:
                        break
                    (temp_query_id, temp_idx_id) = self.sample_len_dict[
                        sample_len
                    ].pop()
                    self.temp_query_buffer.append(temp_query_id)
                    self.temp_idx_buffer.append(temp_idx_id)
                    self.temp_idx_offset_buffer.append(sample_len)
                    handled_query = handled_query + 1
                if current_bs > self.max_batchsize:
                    break
            self.inQueue.put(
                Item(
                    self.temp_query_buffer,
                    self.temp_idx_buffer,
                    self.temp_idx_offset_buffer,
                )
            )
            self.temp_query_buffer = []
            self.temp_idx_buffer = []
            self.temp_idx_offset_buffer = []
            current_bs = 0

    def load_query_samples(self, sample_list):
        self.ds.count_query_samples_len(sample_list)
        self.sorted_keys = sorted(self.ds.sample_lens_list, reverse=True)
        for sample_len in self.sorted_keys:
            self.sample_len_num[sample_len] = 0
            self.sample_len_dict[sample_len] = []

    def unload_query_samples(self):
        self.ds.unload_query_samples()

    def flush_queries(self):
        self.sample_len_dict = {}
        self.sample_len_num = {}
        # put in end marks
        for _ in range(total_instances):
            self.inQueue.put(None)


class ServerQueueRunner:
    def __init__(self, inQueue, dataset, max_batchsize):
        self.inQueue = inQueue
        self.max_batchsize = max_batchsize
        self.ds = dataset
        self.issues = 0
        self.sample_len = 0
        self.queue_id = 0
        self.num_queues = 4  # len(self.inQueue)
        self.ids = [[] for i in range(self.num_queues)]
        self.idxes = [[] for i in range(self.num_queues)]
        self.idxlens = [[] for i in range(self.num_queues)]
        self.qsize = [0 for i in range(self.num_queues)]
        self.qwaiting = [0 for i in range(self.num_queues)]
        self.qwaiting_flag = False
        self.qorder = collections.deque(range(0, self.num_queues))
        self.qorder.rotate(-1)

    def issue(self, q, query_id, idx, idxlen):
        self.inQueue.put(Item(query_id, idx, idxlen))
        self.ids[q] = []
        self.idxes[q] = []
        self.idxlens[q] = []
        self.qsize[q] = 0
        self.qwaiting[q] = 0

    def find_waiting_q(self, sample_len):
        for q in self.qorder:
            if sample_len <= self.qwaiting[q]:
                return q
        return -1

    # Fill query buffers to create constant batch

    def enqueue(self, query_id, idx):
        increment_q = False
        sample_len = self.ds.items_in_memory_len[idx[0]]
        # print("Enqueue sample of size ", sample_len)
        if sample_len % 100:
            sample_len += 100 - (sample_len % 100)

        self.issues += 1

        q = self.queue_id
        if self.qwaiting_flag:
            old_q = self.find_waiting_q(sample_len)
            if old_q >= 0:
                q = old_q
                # print("Found a prev q ", q, " waiting for sample of size", sample_len)
                # print(self.qwaiting)
                self.qwaiting[q] -= sample_len
                if not sum(x for x in self.qwaiting):
                    self.qwaiting_flag = False
                    # print("Reset qwaiting flag to False")

        self.ids[q].append(query_id[0])
        self.idxes[q].append(idx[0])
        self.idxlens[q].append(sample_len)
        self.qsize[q] += sample_len
        batch_size = self.max_batchsize

        if self.qsize[q] == batch_size:
            self.issue(q, self.ids[q], self.idxes[q], self.idxlens[q])
            if q == self.queue_id:
                increment_q = True
        elif batch_size - self.qsize[q] <= 700:
            to_fill = batch_size - self.qsize[q]
            self.qwaiting[q] = to_fill
            self.qwaiting_flag = True
            if q == self.queue_id:
                increment_q = True

        if increment_q:
            self.qorder.rotate(-1)
            self.queue_id = self.qorder[-1]
            if self.qsize[self.queue_id] > 0:
                # This queue has been waiting for a sample
                self.issue(
                    self.queue_id,
                    self.ids[self.queue_id],
                    self.idxes[self.queue_id],
                    self.idxlens[self.queue_id],
                )

    def load_query_samples(self, sample_list):
        self.ds.load_query_samples(sample_list)
        self.ds.count_query_samples_len(sample_list)

    def unload_query_samples(self):
        self.ds.unload_query_samples()

    def flush_queries(self):
        for q in range(self.num_queues):
            if self.ids[q]:
                self.inQueue.put(
                    Item(self.ids[q], self.idxes[q], self.idxlens[q])
                )
        for _ in range(total_instances):
            self.inQueue.put(None)


def auc_score(results):
    results = np.concatenate(results, axis=0)
    results, targets = list(zip(*results))
    results = np.array(results)
    targets = np.array(targets)
    roc_auc = sklearn.metrics.roc_auc_score(targets, results)
    return roc_auc


def add_results(
    final_results, name, result_dict, result_list, took, show_accuracy=False
):
    percentiles = [50.0, 80.0, 90.0, 95.0, 99.0, 99.9]
    buckets = np.percentile(result_list, percentiles).tolist()
    buckets_str = ",".join(
        ["{}:{:.4f}".format(p, b) for p, b in zip(percentiles, buckets)]
    )

    if result_dict["total"] == 0:
        result_dict["total"] = len(result_list)

    # this is what we record for each run
    result = {
        "took": took,
        "mean": np.mean(result_list),
        "percentiles": {str(k): v for k, v in zip(percentiles, buckets)},
        "qps": len(result_list) / took,
        "count": len(result_list),
        "good_items": result_dict["good"],
        "total_items": result_dict["total"],
    }
    acc_str = ""
    if show_accuracy:
        result["accuracy"] = 100.0 * result_dict["good"] / result_dict["total"]
        acc_str = ", acc={:.3f}%".format(result["accuracy"])
        if "roc_auc" in result_dict:
            result["roc_auc"] = 100.0 * result_dict["roc_auc"]
            acc_str += ", roc_auc={:.3f}%".format(result["roc_auc"])

    # add the result to the result dict
    final_results[name] = result

    # to stdout
    print(
        "{} qps={:.2f}, mean={:.4f}, time={:.3f}{}, queries={}, tiles={}".format(
            name,
            result["qps"],
            result["mean"],
            took,
            acc_str,
            len(result_list),
            buckets_str,
        )
    )


def response_loadgen(outQueue, accuracy, lock):
    global item_good
    global item_total
    global item_timing
    global item_results

    """
    max_outqueue_len = 0
    iter_num = 0
    """
    while True:
        # iter_num += 1
        oitem = outQueue.get()
        if oitem is None:
            break

        response = []
        if accuracy:
            lock.acquire()

        for q_id, arr in zip(oitem.query_ids, oitem.array_ref):
            bi = arr.buffer_info()
            response.append(lg.QuerySampleResponse(q_id, bi[0], bi[1]))
        lg.QuerySamplesComplete(response)
        """
        if iter_num % 20 == 0:
            qsize = outQueue.qsize()
            if qsize > max_outqueue_len:
                max_outqueue_len = qsize
        """

        if accuracy:
            item_good += oitem.good
            item_total += oitem.total
            #            item_timing.append(oitem.timing)
            item_results.append(oitem.presults)
            lock.release()
        item_timing.append(oitem.timing)


def calc_core_bind(
    num_sockets,
    cpus_per_socket,
    cpus_per_consumer,
    cpus_per_instance,
    cpus_for_loadgen=1,
):
    """
    Generate every process instances start_core_idx and re-calculate cpus_for_loadgen
    """
    assert cpus_per_consumer > 0 and cpus_per_instance > 0
    assert (
        cpus_for_loadgen < cpus_per_socket
    ), "Unsupport multisocket for loadgen"
    assert (
        cpus_per_consumer <= cpus_per_socket
    ), "Unsupport cpus_per_consumer > cpus_per_socket"
    assert (
        cpus_per_instance <= cpus_per_consumer
    ), "Unsupport cpus_per_instance > cpus_per_consumer"
    assert (
        cpus_per_socket % cpus_per_consumer == 0
    ), "cpus_per_socket must be divisible by cpus_per_consumer"
    assert (
        cpus_per_consumer % cpus_per_instance == 0
    ), "cpus_per_consumer must be divisible by cpus_per_instance"

    # re-calculate socket0 remained cores for loadgen
    socket0_predict_cores = (
        math.floor((cpus_per_socket - cpus_for_loadgen) / cpus_per_instance)
        * cpus_per_instance
    )
    cpus_for_loadgen = cpus_per_socket - socket0_predict_cores

    n_inst_per_proc = cpus_per_consumer // cpus_per_instance
    idx = np.array(
        list(range(0, num_sockets * cpus_per_socket, cpus_per_instance))
    ).reshape((-1, n_inst_per_proc))

    # pop instances with start_core_id < cpus_for_loadgen
    proc_inst_start_idx = []
    for item in idx:
        tmp = item[item >= cpus_for_loadgen]
        if not len(tmp) > 0:
            continue
        proc_inst_start_idx.append(tmp.tolist())

    total_instances = len([i for ins in proc_inst_start_idx for i in ins])

    return cpus_for_loadgen, proc_inst_start_idx, total_instances


def main():
    global num_sockets
    global cpus_for_loadgen
    global cpus_per_socket
    global cpus_per_consumer
    global cpus_per_instance
    global start_time
    global item_total
    global last_timeing
    global total_instances

    args = get_args()
    log.info(args)
    config = os.path.abspath(args.config)
    user_config = os.path.abspath(args.user_config)

    if not os.path.exists(config):
        log.error("{} not found".format(config))
        sys.exit(1)

    if not os.path.exists(user_config):
        log.error("{} not found".format(user_config))
        sys.exit(1)

    if args.output:
        output_dir = os.path.abspath(args.output)
        os.makedirs(output_dir, exist_ok=True)
        if os.path.exists("./audit.config"):
            copyfile("./audit.config", output_dir + "/audit.config")
        os.chdir(output_dir)

    cpus_for_loadgen, proc_inst_start_idx, total_instances = calc_core_bind(
        num_sockets,
        cpus_per_socket,
        cpus_per_consumer,
        cpus_per_instance,
        cpus_for_loadgen,
    )
    os.sched_setaffinity(os.getpid(), range(cpus_for_loadgen))
    print("cpus for loadgen: {}".format(cpus_for_loadgen))

    lock = multiprocessing.Lock()
    init_counter = multiprocessing.Value("i", 0)
    total_samples = multiprocessing.Value("i", 0)
    dsQueue = multiprocessing.Queue(maxsize=len(proc_inst_start_idx))
    inQueue = multiprocessing.Queue(
        maxsize=len(proc_inst_start_idx) * len(proc_inst_start_idx[-1]) * 2
    )
    if len(proc_inst_start_idx) == 1:
        numOutQ = len(proc_inst_start_idx[0])
        qnum = len(proc_inst_start_idx[0])
    else:
        numOutQ = len(proc_inst_start_idx)
        qnum = 1
    outQueues = [multiprocessing.Queue() for i in range(numOutQ)]
    # inQueue = multiprocessing.JoinableQueue()

    consumers = [
        Consumer(
            inQueue,
            outQueues[i * qnum : (i + 1) * qnum],
            dsQueue,
            lock,
            init_counter,
            i,
            args,
            cpus_per_socket,
            cpus_per_instance,
            proc_inst_start_idx[i],
        )
        for i in range(len(proc_inst_start_idx))
    ]
    for c in consumers:
        c.start()

    # Start response thread
    response_workers = [
        threading.Thread(
            target=response_loadgen, args=(outQueues[i], args.accuracy, lock)
        )
        for i in range(numOutQ)
    ]
    for response_worker in response_workers:
        response_worker.daemon = True
        response_worker.start()

    # Wait until subprocess ready
    while init_counter.value < len(proc_inst_start_idx):
        time.sleep(2)

    torch.set_num_threads(cpus_per_socket * num_sockets)

    dlrm_dataset = get_dataset(args)

    total_samples.value = dlrm_dataset.get_item_count()
    scenario = SCENARIO_MAP[args.scenario]
    runner_map = {
        lg.TestScenario.Server: ServerQueueRunner,
        lg.TestScenario.Offline: OfflineQueueRunner,
    }
    runner = runner_map[scenario](inQueue, dlrm_dataset, args.max_batchsize)

    settings = lg.TestSettings()
    settings.FromConfig(config, args.model, args.scenario)
    settings.FromConfig(user_config, args.model, args.scenario)
    settings.scenario = scenario
    settings.mode = lg.TestMode.PerformanceOnly
    if args.accuracy:
        settings.mode = lg.TestMode.AccuracyOnly
        settings.performance_sample_count_override = total_samples.value

    if args.find_peak_performance:
        settings.mode = lg.TestMode.FindPeakPerformance

    if args.duration:
        settings.min_duration_ms = args.duration
        settings.max_duration_ms = args.duration

    if args.target_qps:
        settings.server_target_qps = float(args.target_qps)
        settings.offline_expected_qps = float(args.target_qps)

    if args.count_queries:
        settings.min_query_count = args.count_queries
        settings.max_query_count = args.count_queries

    if args.samples_per_query_multistream:
        settings.multi_stream_samples_per_query = (
            args.samples_per_query_multistream
        )

    if args.max_latency:
        settings.server_target_latency_ns = int(args.max_latency * NANO_SEC)
        settings.multi_stream_target_latency_ns = int(
            args.max_latency * NANO_SEC
        )

    def load_query_samples(sample_list):
        # Wait until subprocess ready
        global start_time
        # for _ in range(total_instances):
        init_total = len(proc_inst_start_idx) * 2 + len(
            [i for ins in proc_inst_start_idx for i in ins]
        )
        for _ in range(len(proc_inst_start_idx)):
            dsQueue.put(sample_list)
        runner.load_query_samples(sample_list)
        while init_counter.value < init_total:
            time.sleep(2)
        start_time = time.time()

    def unload_query_samples(sample_list):
        runner.unload_query_samples()

    def issue_queries(response_ids, query_sample_indexes):
        runner.enqueue(response_ids, query_sample_indexes)

    def flush_queries():
        runner.flush_queries()

    sut = lg.ConstructFastSUT(issue_queries, flush_queries)
    qsl = lg.ConstructQSL(
        total_samples.value,
        min(total_samples.value, args.samples_per_query_offline),
        load_query_samples,
        unload_query_samples,
    )

    result_dict = {
        "good": 0,
        "total": 0,
        "roc_auc": 0,
        "scenario": str(scenario),
    }
    log.info("starting {}".format(scenario))
    lg.StartTest(sut, qsl, settings)

    for c in consumers:
        c.join()
    for i in range(numOutQ):
        outQueues[i].put(None)

    if not last_timeing:
        last_timeing = item_timing

    if args.accuracy:
        result_dict["good"] = item_good
        result_dict["total"] = item_total
        result_dict["roc_auc"] = auc_score(item_results)

    final_results = {
        "runtime": "pytorch-native-dlrm",
        "version": torch.__version__,
        "time": int(time.time()),
        "cmdline": str(args),
    }

    add_results(
        final_results,
        "{}".format(scenario),
        result_dict,
        last_timeing,
        time.time() - start_time,
        args.accuracy,
    )

    lg.DestroyQSL(qsl)
    lg.DestroyFastSUT(sut)

    # write final results
    if args.output:
        if args.accuracy:
            with open("accuracy.txt", "w") as f:
                json.dump(final_results, f, sort_keys=True, indent=4)
                roc_auc = final_results["{}".format(scenario)]["roc_auc"]
                f.write(f"\nAUC={roc_auc}\n")
        else:
            with open("results.json", "w") as f:
                json.dump(final_results, f, sort_keys=True, indent=4)


if __name__ == "__main__":
    main()
