#  *****************************************************************************
#  * Modifications Copyright (c) 2025 Advanced Micro Devices, Inc.
#  * All rights reserved.
#  *
#  * Was sourced from
#  * https://github.com/mlcommons/inference_results_v3.1/blob/main/closed/Intel/code/dlrm-v2-99/pytorch-cpu-int8/python/items.py  # noqa: B950
#  * commit ID: eaf622a
#  ******************************************************************************
import time


class Item:
    """An item that we queue for processing by the thread pool."""

    def __init__(self, query_id, content_id, idx_offsets):
        self.query_id = query_id
        self.content_id = content_id
        self.idx_offsets = idx_offsets
        self.start = time.time()


class OItem:
    def __init__(
        self,
        presults,
        query_ids=None,
        array_ref=None,
        good=0,
        total=0,
        timing=0,
    ):
        self.good = good
        self.total = total
        self.timing = timing
        self.presults = presults
        self.query_ids = query_ids
        self.array_ref = array_ref
