#  *****************************************************************************
#  * Modifications Copyright (c) 2025 Advanced Micro Devices, Inc.
#  * All rights reserved.
#  *
#  * Was sourced from
#  * https://github.com/mlcommons/inference_results_v3.1/blob/main/closed/Intel/code/dlrm-v2-99/pytorch-cpu-int8/python/consumer.py  # noqa: B950
#  * commit ID: eaf622a
#  ******************************************************************************
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import time
import array

import torch
import numpy as np
import torch.multiprocessing as mp

from items import OItem
from backend_pytorch_native import get_backend
from multihot_criteo import get_dataset

import zentorch


class Consumer(mp.Process):
    def __init__(
        self,
        task_queue,
        result_queue,
        ds_queue,
        lock,
        init_counter,
        proc_num,
        args,
        cpus_per_socket,
        cpus_per_instance,
        inst_start_idx,
    ):
        mp.Process.__init__(self)
        self.args = args
        self.lock = lock
        self.ds_queue = ds_queue
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.rqnum = len(result_queue)
        self.init_counter = init_counter
        self.cpus_per_sockets = cpus_per_socket
        self.cpus_per_instance = cpus_per_instance
        self.inst_start_idx = inst_start_idx

        self.multi_hot = [
            3,
            2,
            1,
            2,
            6,
            1,
            1,
            1,
            1,
            7,
            3,
            8,
            1,
            6,
            9,
            5,
            1,
            1,
            1,
            12,
            100,
            27,
            10,
            3,
            1,
            1,
        ]
        self.workers = []
        print(
            "Process {}: start from core {}, Setup {} instances size".format(
                proc_num, self.inst_start_idx[0], len(self.inst_start_idx)
            )
        )
        print("Process {} instance cores: ", self.inst_start_idx)

    def warmup(self):
        with torch.no_grad():
            for s in range(
                self.args.max_batchsize, self.args.max_batchsize + 800, 100
            ):

                batch_dense_X = torch.randn((s, 13), dtype=torch.float)
                batch_lS_i = []
                batch_lS_o = []
                for _, h in enumerate(self.multi_hot):
                    batch_lS_i.append(torch.ones((s * h), dtype=torch.long))
                    batch_lS_o.append(
                        torch.arange(0, (s + 1) * h, h, dtype=torch.long)
                    )

                self.model(batch_dense_X, batch_lS_i, batch_lS_o)

    def get_samples(self, id_list):
        ls = []
        for i in id_list:
            ls.append(self.items_in_memory[i])
        X_ls = []
        lsi_ls = []
        lbl_ls = []
        for densex, index, _, labels in ls:
            X_ls.append(densex)
            lsi_ls.append(index)
            lbl_ls.append(labels)
        X = torch.cat(X_ls)
        batch = X.shape[0]
        lS_i = []
        lS_o = []
        for i in range(26):
            lsi = [j[i] for j in lsi_ls]
            lsi_all = torch.cat(lsi)
            hot = lsi_all.shape[0] // batch
            lS_i.append(lsi_all)
            lS_o.append(
                torch.arange(0, (batch + 1) * hot, hot, dtype=torch.int)
            )
        T = torch.cat(lbl_ls)
        return (X, lS_i, lS_o, T)

    def handle_tasks(self, i, model, task_queue, result_queue, args, pid):
        torch.set_grad_enabled(False)
        socket_id = self.inst_start_idx[0] // self.cpus_per_sockets
        core_id = self.inst_start_idx[i] - socket_id * self.cpus_per_sockets

        zentorch.utils.thread_bind(
            [
                socket_id * self.cpus_per_sockets + core_id + i
                for i in range(self.cpus_per_instance)
            ]
        )

        instance_name = str(pid) + "-" + str(i)
        if args.enable_profiling:
            filename = "dlrm_mlperf_offline_run_" + instance_name + ".prof"

        self.lock.acquire()
        self.init_counter.value += 1
        self.lock.release()

        total_time = 0

        with torch.autograd.profiler.profile(args.enable_profiling) as prof:
            while True:
                qitem = task_queue.get()
                if qitem is None:
                    break
                get_sample_start = time.time()
                batch_dense_X, batch_lS_i, batch_lS_o, batch_T = (
                    self.get_samples(qitem.content_id)
                )
                idx_offsets = qitem.idx_offsets
                get_sample_timing = time.time() - get_sample_start
                total_time += get_sample_timing
                results = []

                try:
                    with torch.no_grad():
                        results = model(batch_dense_X, batch_lS_i, batch_lS_o)
                    presults = torch.cat(
                        (results.view(-1, 1), batch_T.view(-1, 1)), dim=1
                    )
                    if args.accuracy:
                        total = len(results)
                        good = (
                            (results.round() == batch_T)
                            .nonzero(as_tuple=False)
                            .size(0)
                        )
                        result_timing = time.time() - qitem.start

                except Exception as ex:  # pylint: disable=broad-except
                    print("instance ", instance_name, " failed ", ex)
                    presults = [[]] * len(qitem.query_id)
                finally:
                    response_array_refs = []
                    query_list = qitem.query_id
                    prev_off = 0
                    for idx in range(len(query_list)):
                        cur_off = prev_off + idx_offsets[idx]
                        response_array = array.array(
                            "B",
                            np.array(
                                presults[prev_off:cur_off], np.float32
                            ).tobytes(),
                        )
                        response_array_refs.append(response_array)
                        prev_off = cur_off
                    if args.accuracy:
                        result_queue.put(
                            OItem(
                                np.array(presults, np.float32),
                                query_list,
                                response_array_refs,
                                good,
                                total,
                                result_timing,
                            )
                        )
                    else:
                        result_queue.put(
                            OItem([], query_list, response_array_refs)
                        )
        if args.enable_profiling:
            with open(filename, "w") as prof_f:
                prof_f.write(
                    prof.key_averages().table(sort_by="self_cpu_time_total")
                )

    def run(self):
        os.sched_setaffinity(
            self.pid,
            range(
                self.inst_start_idx[0],
                self.inst_start_idx[-1] + self.cpus_per_instance,
            ),
        )

        backend = get_backend(self.args.backend, self.args.dataset)
        self.model = backend.load(self.args).model
        self.model = torch.compile(self.model, backend="zentorch")
        print("Start warmup.")
        self.warmup()
        print("Warmup done.")

        self.lock.acquire()
        self.init_counter.value += 1
        self.lock.release()

        sample_list = self.ds_queue.get()
        ds = get_dataset(self.args)
        ds.load_query_samples(sample_list)
        self.items_in_memory = ds.items_in_memory
        print(str(self.pid), " : Complete load query samples !!")

        self.lock.acquire()
        self.init_counter.value += 1
        self.lock.release()

        if len(self.inst_start_idx) > 1:
            for i in range(len(self.inst_start_idx)):
                if self.rqnum == 1:
                    worker = mp.Process(
                        target=self.handle_tasks,
                        args=(
                            i,
                            self.model,
                            self.task_queue,
                            self.result_queue[0],
                            self.args,
                            self.pid,
                        ),
                    )
                else:
                    worker = mp.Process(
                        target=self.handle_tasks,
                        args=(
                            i,
                            self.model,
                            self.task_queue,
                            self.result_queue[i],
                            self.args,
                            self.pid,
                        ),
                    )
                self.workers.append(worker)
            for w in self.workers:
                w.start()
            for w in self.workers:
                w.join()
        else:
            self.handle_tasks(
                0,
                self.model,
                self.task_queue,
                self.result_queue[0],
                self.args,
                self.pid,
            )

        ds.unload_query_samples()
