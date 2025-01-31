# ******************************************************************************
# Copyright (c) 2023-2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

from ._build_info import __torchversion__ as buildtime_torchversion
from torch.torch_version import __version__ as runtime_torchversion

# Pytorch lacks symbol-level compatibility, requiring extensions
# to be pinned to the same minor version. To avoid issues, it is
# necessary to error out if the runtime Pytorch version
# differs from the build-time version.

if runtime_torchversion[:3] != buildtime_torchversion[:3]:
    raise ImportError(
        f"Incompatible PyTorch version {runtime_torchversion} detected. "
        f"The installed zentorch binary is only compatible "
        f"with PyTorch versions {buildtime_torchversion[:3]}.x"
    )

from ._optimize import optimize  # noqa
from ._info import __config__, __version__  # noqa
from ._compile_backend import *  # noqa
from ._meta_registrations import *  # noqa
from ._freeze_utils import freezing_enabled  # noqa

# llm optimizations
from . import llm  # noqa

# model reload utility for quantized models
from ._quant_model_reload import load_quantized_model, load_woq_model  # noqa F401
from . import utils  # noqa F401
