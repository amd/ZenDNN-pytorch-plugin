#  *****************************************************************************
#  * Modifications Copyright (c) 2025 Advanced Micro Devices, Inc.
#  * All rights reserved.
#  *
#  * Was sourced from
#  * https://github.com/mlcommons/inference_results_v3.1/blob/main/closed/Intel/code/dlrm-v2-99/pytorch-cpu-int8/python/backend.py  # noqa: B950
#  * commit ID: eaf622a
#  ******************************************************************************
"""
abstract backend class
"""


# TODO: Base class is currenlty not used, can be removed in future.
class Backend:
    def __init__(self):
        self.inputs = []
        self.outputs = []

    def version(self):
        raise NotImplementedError("Backend:version")

    def name(self):
        raise NotImplementedError("Backend:name")

    def load(self, model_path, inputs=None, outputs=None):
        raise NotImplementedError("Backend:load")
