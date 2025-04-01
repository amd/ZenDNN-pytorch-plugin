#  *****************************************************************************
#  * Modifications Copyright (c) 2025 Advanced Micro Devices, Inc.
#  * All rights reserved.
#  *
#  * Was sourced from
#  * https://github.com/mlcommons/inference_results_v3.1/blob/main/closed/Intel/code/dlrm-v2-99/pytorch-cpu-int8/python/multihot_criteo.py  # noqa: B950
#  * commit ID: eaf622a
#  ******************************************************************************
"""
implementation of criteo dataset
"""

import os
import time
import random
import zipfile

import torch
import numpy as np
import sklearn.metrics
from typing import Dict, List

from dataset import Dataset

CAT_FEATURE_COUNT = 26
DAYS = 24
DEFAULT_CAT_NAMES = [
    "cat_0",
    "cat_1",
    "cat_2",
    "cat_3",
    "cat_4",
    "cat_5",
    "cat_6",
    "cat_7",
    "cat_8",
    "cat_9",
    "cat_10",
    "cat_11",
    "cat_12",
    "cat_13",
    "cat_14",
    "cat_15",
    "cat_16",
    "cat_17",
    "cat_18",
    "cat_19",
    "cat_20",
    "cat_21",
    "cat_22",
    "cat_23",
    "cat_24",
    "cat_25",
]


class MultihotCriteo(Dataset):
    def __init__(
        self,
        data_path,
        name,
        num_embeddings_per_feature,
        pre_process,
        count=None,
        samples_to_aggregate_fix=None,
        samples_to_aggregate_min=None,
        samples_to_aggregate_max=None,
        samples_to_aggregate_quantile_file=None,
        samples_to_aggregate_trace_file=None,
        max_ind_range=-1,
        randomize="total",
        memory_map=False,
    ):
        super().__init__()

        self.count = count
        self.random_offsets = []
        self.use_fixed_size = (samples_to_aggregate_quantile_file is None) and (
            samples_to_aggregate_min is None or samples_to_aggregate_max is None
        )
        if self.use_fixed_size:
            # fixed size queries
            self.samples_to_aggregate = (
                1 if samples_to_aggregate_fix is None else samples_to_aggregate_fix
            )
            self.samples_to_aggregate_min = None
            self.samples_to_aggregate_max = None
        else:
            # variable size queries
            self.samples_to_aggregate = 1
            self.samples_to_aggregate_min = samples_to_aggregate_min
            self.samples_to_aggregate_max = samples_to_aggregate_max
            self.samples_to_aggregate_quantile_file = samples_to_aggregate_quantile_file

        if name == "debug":
            stage_files = [
                [os.path.join(data_path, f"day_{DAYS - 1}_dense_debug.npy")],
                [os.path.join(data_path, f"day_{DAYS - 1}_sparse_multi_hot_debug.npz")],
                [os.path.join(data_path, f"day_{DAYS - 1}_labels_debug.npy")],
            ]
        elif name == "multihot-criteo-sample":
            stage_files = [
                [os.path.join(data_path, f"day_{DAYS - 1}_dense_sample.npy")],
                [
                    os.path.join(
                        data_path, f"day_{DAYS - 1}_sparse_multi_hot_sample.npz"
                    )
                ],
                [os.path.join(data_path, f"day_{DAYS - 1}_labels_sample.npy")],
            ]
        elif name == "multihot-criteo":
            stage_files = [
                [os.path.join(data_path, f"day_{DAYS - 1}_dense.npy")],
                [os.path.join(data_path, f"day_{DAYS - 1}_sparse_multi_hot.npz")],
                [os.path.join(data_path, f"day_{DAYS - 1}_labels.npy")],
            ]
        else:
            raise ValueError(
                "only debug|multihot-sample-criteo|multihot-criteo dataset "
                "options are supported"
            )

        self.val_data = MultihotCriteoPipe(
            name,
            "val",
            *stage_files,  # pyre-ignore[6]
            batch_size=self.samples_to_aggregate,
            rank=0,
            world_size=int(os.environ.get("WORLD_SIZE", 1)),
            mmap_mode=memory_map,
        )
        self.test_data = MultihotCriteoPipe(
            name,
            "test",
            *stage_files,
            batch_size=self.samples_to_aggregate,
            rank=0,
            world_size=int(os.environ.get("WORLD_SIZE", 1)),
            mmap_mode=memory_map,
        )
        self.num_individual_samples = len(self.val_data.labels_arrs[0])

        # WARNING: Note that the orignal dataset returns number of samples, while the
        # binary dataset returns the number of batches. Therefore, when using a
        # mini-batch of size samples_to_aggregate as an item we need to adjust the
        # original dataset item_count. On the other hand, data loader always returns
        # number of batches.
        if self.use_fixed_size:
            # the offsets for fixed query size will be generated on-the-fly later on
            print("Using fixed query size: " + str(self.samples_to_aggregate))
            self.num_aggregated_samples = (
                self.num_individual_samples + self.samples_to_aggregate - 1
            ) // self.samples_to_aggregate

        else:
            # the offsets for variable query sizes will be pre-generated here
            if self.samples_to_aggregate_quantile_file is None:
                # generate number of samples in a query from a uniform(min,max)
                # distribution
                print(
                    "Using variable query size: uniform distribution ("
                    + str(self.samples_to_aggregate_min)
                    + ","
                    + str(self.samples_to_aggregate_max)
                    + ")"
                )
                done = False
                qo = 0
                while not done:
                    self.random_offsets.append(int(qo))
                    qs = random.randint(
                        self.samples_to_aggregate_min,
                        self.samples_to_aggregate_max,
                    )
                    qo = min(qo + qs, self.num_individual_samples)
                    if qo >= self.num_individual_samples:
                        done = True
                self.random_offsets.append(int(qo))

                # compute min and max number of samples
                nas_max = (
                    self.num_individual_samples + self.samples_to_aggregate_min - 1
                ) // self.samples_to_aggregate_min
                nas_min = (
                    self.num_individual_samples + self.samples_to_aggregate_max - 1
                ) // self.samples_to_aggregate_max
            else:
                # generate number of samples in a query from a custom distribution,
                # with quantile (inverse of its cdf) given in the file. Note that
                # quantile is related to the concept of percentile in statistics.
                #
                # For instance, assume that we have the following distribution for query length  # noqa: B950
                # length = [100, 200, 300,  400,  500,  600,  700] # x
                # pdf =    [0.1, 0.6, 0.1, 0.05, 0.05, 0.05, 0.05] # p(x)
                # cdf =    [0.1, 0.7, 0.8, 0.85,  0.9, 0.95,  1.0] # f(x) = prefix-sum of p(x)  # noqa: B950
                # The inverse of its cdf with granularity of 0.05 can be written as
                # quantile_p = [.05, .10, .15, .20, .25, .30, .35, .40, .45, .50, .55, .60, .65, .70, .75, .80, .85, .90, .95, 1.0] # p  # noqa: B950
                # quantile_x = [100, 100, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 300, 300, 400, 500, 600, 700] # q(p) = x, such that f(x) >= p   # noqa: B950
                # Notice that once we have quantile, we can apply inverse transform sampling method.  # noqa: B950
                print(
                    "Using variable query size: custom distribution (file "
                    + str(samples_to_aggregate_quantile_file)
                    + ")"
                )
                with open(self.samples_to_aggregate_quantile_file, "r") as f:
                    line = f.readline()
                    quantile = np.fromstring(line, dtype=int, sep=", ")

                len_quantile = len(quantile)
                done = False
                qo = 0
                while not done:
                    self.random_offsets.append(int(qo))
                    pr = np.random.randint(low=0, high=len_quantile)
                    qs = quantile[pr]
                    qo = min(qo + qs, self.num_individual_samples)
                    done = qo >= self.num_individual_samples
                self.random_offsets.append(int(qo))

                # compute min and max number of samples
                nas_max = (self.num_individual_samples + quantile[0] - 1) // quantile[0]
                nas_min = (self.num_individual_samples + quantile[-1] - 1) // quantile[
                    -1
                ]

            # reset num_aggregated_samples
            self.num_aggregated_samples = len(self.random_offsets) - 1

            # check num_aggregated_samples
            if (
                self.num_aggregated_samples < nas_min
                or nas_max < self.num_aggregated_samples
            ):
                raise ValueError("Sannity check failed")

        # limit number of items to count if needed
        if self.count is not None:
            self.num_aggregated_samples = min(self.count, self.num_aggregated_samples)

        # dump the trace of aggregated samples
        if samples_to_aggregate_trace_file is not None:
            with open(samples_to_aggregate_trace_file, "w") as f:
                for i in range(self.num_aggregated_samples):
                    if self.use_fixed_size:
                        s = i * self.samples_to_aggregate
                        e = min(
                            (i + 1) * self.samples_to_aggregate,
                            self.num_individual_samples,
                        )
                    else:
                        s = self.random_offsets[i]
                        e = self.random_offsets[i + 1]
                    f.write(str(s) + ", " + str(e) + ", " + str(e - s) + "\n")

    def get_item_count(self):
        # get number of items in the dataset
        return self.num_aggregated_samples

    """ lg compatibilty routine """

    def unload_query_samples(self, sample_list=None):
        self.items_in_memory = {}
        self.item_sizes = {}

    def count_query_samples_len(self, sample_list):
        self.items_in_memory_len = {}
        self.sample_lens_list = []
        for i in sample_list:
            if self.use_fixed_size:
                s = i * self.samples_to_aggregate
                e = min(
                    (i + 1) * self.samples_to_aggregate,
                    self.num_individual_samples,
                )
            else:
                s = self.random_offsets[i]
                e = self.random_offsets[i + 1]
            sample_len = e - s
            self.items_in_memory_len[i] = sample_len
            if sample_len not in self.sample_lens_list:
                self.sample_lens_list.append(sample_len)

    """ lg compatibilty routine """

    def load_query_samples(self, sample_list):
        print("loading query samples ..")
        self.items_in_memory = {}
        self.item_sizes = {}

        # WARNING: notice that while DataLoader is iterable-style, the Dataset
        # can be iterable- or map-style, and Criteo[Bin]Dataset are the latter
        # This means that we can not index into DataLoader, but can enumerate it,
        # while we can index into the dataset itself.
        for i in sample_list:
            # approach 1: single sample as an item
            """
            self.items_in_memory[l] = self.val_data[l]
            """
            # approach 2: multiple samples as an item
            if self.use_fixed_size:
                s = i * self.samples_to_aggregate
                e = min(
                    (i + 1) * self.samples_to_aggregate,
                    self.num_individual_samples,
                )
            else:
                s = self.random_offsets[i]
                e = self.random_offsets[i + 1]

            ls = list(range(s, e))
            self.items_in_memory[i] = self.val_data.load_batch(ls)
            self.item_sizes[i] = len(ls)

        self.last_loaded = time.time()

    """ lg compatibilty routine """

    def get_samples(self, id_list):
        idx_offsets = [0]
        for item in id_list:
            idx_offsets.append(idx_offsets[-1] + self.item_sizes[item])
        return [self.items_in_memory[item] for item in id_list], idx_offsets

    def get_labels(self, sample):
        if isinstance(sample, list):
            labels = [s.labels for s in sample]
            labels = torch.cat(labels)
            return labels
        else:
            return sample.labels


class MultihotCriteoPipe:
    def __init__(
        self,
        name: str,
        stage: str,
        dense_paths: List[str],
        sparse_paths: List[str],
        labels_paths: List[str],
        batch_size: int,
        rank: int,
        world_size: int,
        mmap_mode: bool = False,
    ) -> None:
        self.stage = stage
        self.dense_paths = dense_paths
        self.sparse_paths = sparse_paths
        self.labels_paths = labels_paths
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.split = self.world_size > 1

        # Load arrays
        m = "r" if mmap_mode else None
        self.dense_arrs: List[np.ndarray] = [
            np.load(f, mmap_mode=m) for f in self.dense_paths
        ]
        self.labels_arrs: List[np.ndarray] = [
            np.load(f, mmap_mode=m) for f in self.labels_paths
        ]
        self.sparse_arrs: List = []
        for sparse_path in self.sparse_paths:
            multi_hot_ids_l = []
            for feat_id_num in range(CAT_FEATURE_COUNT):
                multi_hot_ft_ids = self._load_from_npz(
                    sparse_path, f"{feat_id_num}.npy"
                )
                multi_hot_ids_l.append(multi_hot_ft_ids)
            self.sparse_arrs.append(multi_hot_ids_l)

        len_d0 = len(self.dense_arrs[0])
        second_half_start_index = int(len_d0 // 2 + len_d0 % 2)
        if name == "multihot-criteo":
            if stage == "val":
                self.dense_arrs[0] = self.dense_arrs[0][:second_half_start_index, :]
                self.labels_arrs[0] = self.labels_arrs[0][:second_half_start_index, :]
                self.sparse_arrs[0] = [
                    feats[:second_half_start_index, :] for feats in self.sparse_arrs[0]
                ]
            elif stage == "test":
                self.dense_arrs[0] = self.dense_arrs[0][second_half_start_index:, :]
                self.labels_arrs[0] = self.labels_arrs[0][second_half_start_index:, :]
                self.sparse_arrs[0] = [
                    feats[second_half_start_index:, :] for feats in self.sparse_arrs[0]
                ]

        self.num_rows_per_file: List[int] = list(map(len, self.dense_arrs))
        total_rows = sum(self.num_rows_per_file)
        self.num_full_batches: int = (
            total_rows // batch_size // self.world_size * self.world_size
        )
        self.last_batch_sizes: np.ndarray = np.array(
            [0 for _ in range(self.world_size)]
        )
        remainder = total_rows % (self.world_size * batch_size)
        if remainder < self.world_size:
            self.num_full_batches -= self.world_size
            self.last_batch_sizes += batch_size
        else:
            self.last_batch_sizes += remainder // self.world_size
        self.last_batch_sizes[: remainder % self.world_size] += 1

        self.multi_hot_sizes: List[int] = [
            multi_hot_feat.shape[-1] for multi_hot_feat in self.sparse_arrs[0]
        ]

        self.offsets_pool = {}

        # These values are the same for the KeyedJaggedTensors in all batches, so they
        # are computed once here. This avoids extra work from the KeyedJaggedTensor sync
        # functions.
        self.keys: List[str] = DEFAULT_CAT_NAMES
        self.index_per_key: Dict[str, int] = {
            key: i for (i, key) in enumerate(self.keys)
        }

    def _load_from_npz(self, fname, npy_name):
        # figure out offset of .npy in .npz
        zf = zipfile.ZipFile(fname)
        info = zf.NameToInfo[npy_name]
        assert info.compress_type == 0
        zf.fp.seek(info.header_offset + len(info.FileHeader()) + 20)
        # read .npy header
        zf.open(npy_name, "r")
        version = np.lib.format.read_magic(zf.fp)
        shape, fortran_order, dtype = np.lib.format._read_array_header(zf.fp, version)
        assert (
            dtype == "int32"
        ), f"sparse multi-hot dtype is {dtype} but should be int32"
        offset = zf.fp.tell()
        # create memmap
        return np.memmap(
            zf.filename,
            dtype=dtype,
            shape=shape,
            order="F" if fortran_order else "C",
            mode="r",
            offset=offset,
        )

    def _get_offsets(self, batchsize):
        offsets = []
        for multi_hot_size in self.multi_hot_sizes:
            offsets.append(
                torch.arange(
                    0,
                    (batchsize + 1) * multi_hot_size,
                    multi_hot_size,
                    dtype=torch.long,
                )
            )
        return tuple(offsets)

    def _np_arrays_to_batch(
        self, dense: np.ndarray, sparse: List[np.ndarray], labels: np.ndarray
    ):
        batch_size = len(dense)
        if batch_size not in self.offsets_pool:
            self.offsets_pool[batch_size] = self._get_offsets(batch_size)
        for i in range(len(sparse)):
            sparse[i] = torch.from_numpy(sparse[i]).long().reshape(-1)
        dense = torch.from_numpy(dense.copy())
        index = tuple(sparse)
        offset = self.offsets_pool[batch_size]
        labels = torch.from_numpy(labels.reshape(-1).copy())
        return (dense, index, offset, labels)

    def load_batch(self, sample_list):
        # print("calling load batch from multi-hot-criteo pipe..")
        if self.split:
            # print("split")
            batch = []
            n_samples = len(sample_list)
            limits = [
                i * n_samples // self.world_size for i in range(self.world_size + 1)
            ]
            for i in range(self.world_size):
                dense = self.dense_arrs[0][sample_list[limits[i] : limits[i + 1]], :]
                sparse = [
                    arr[sample_list[limits[i] : limits[i + 1]], :]
                    for arr in self.sparse_arrs[0]
                ]
                labels = self.labels_arrs[0][sample_list[limits[i] : limits[i + 1]], :]
                batch.append(self._np_arrays_to_batch(dense, sparse, labels))
            return batch
        else:
            # print("not split")
            dense = self.dense_arrs[0][sample_list, :]
            sparse = [arr[sample_list, :] for arr in self.sparse_arrs[0]]
            labels = self.labels_arrs[0][sample_list, :]
            return self._np_arrays_to_batch(dense, sparse, labels)


# Pre  processing
def pre_process_criteo_dlrm(x):
    return x


# Post processing
class DlrmPostProcess:
    def __init__(self):
        self.good = 0
        self.total = 0
        self.roc_auc = 0
        self.results = []

    def __call__(self, results, expected=None, result_dict=None):
        processed_results = []
        n = len(results)
        for idx in range(0, n):
            # NOTE: copy from GPU to CPU while post processing, if needed.
            # Alternatively, we could do this on the output of predict function
            # in backend_pytorch_native.py
            result = results[idx].detach().cpu()
            target = expected[idx]

            for r, t in zip(result, target):
                processed_results.append([r, t])
            # debug prints
            # print(result)
            # print(expected[idx])
            # accuracy metric
            self.good += int((result.round() == target).sum().numpy())
            self.total += len(target)
        return processed_results

    def add_results(self, results):
        self.results.append(results)

    def start(self):
        self.good = 0
        self.total = 0
        self.roc_auc = 0
        self.results = []

    def finalize(self, result_dict, ds=False, output_dir=None):
        # AUC metric
        self.results = np.concatenate(self.results, axis=0)
        results, targets = list(zip(*self.results))
        results = np.array(results)
        targets = np.array(targets)
        self.roc_auc = sklearn.metrics.roc_auc_score(targets, results)

        result_dict["good"] = self.good
        result_dict["total"] = self.total
        result_dict["roc_auc"] = self.roc_auc


SUPPORTED_DATASETS = {
    "multihot-criteo": (
        MultihotCriteo,
        pre_process_criteo_dlrm,
        DlrmPostProcess(),
        {"randomize": "total", "memory_map": True},
    ),
}


def get_dataset(args):
    kwargs = {"randomize": "total", "memory_map": True}
    # dataset to use
    # --count-samples can be used to limit the number of samples used for testing
    wanted_dataset, pre_proc, post_proc, kwargs = SUPPORTED_DATASETS[args.dataset]
    ds = wanted_dataset(
        num_embeddings_per_feature=[
            40000000,
            39060,
            17295,
            7424,
            20265,
            3,
            7122,
            1543,
            63,
            40000000,
            3067956,
            405282,
            10,
            2209,
            11938,
            155,
            4,
            976,
            14,
            40000000,
            40000000,
            40000000,
            590152,
            12973,
            108,
            36,
        ],
        data_path=args.dataset_path,
        name=args.dataset,
        pre_process=pre_proc,  # currently an identity function
        count=args.count_samples,
        samples_to_aggregate_fix=args.samples_to_aggregate_fix,
        samples_to_aggregate_min=args.samples_to_aggregate_min,
        samples_to_aggregate_max=args.samples_to_aggregate_max,
        samples_to_aggregate_quantile_file=args.samples_to_aggregate_quantile_file,
        samples_to_aggregate_trace_file=args.samples_to_aggregate_trace_file,
        max_ind_range=args.max_ind_range,
        **kwargs,
    )

    return ds
