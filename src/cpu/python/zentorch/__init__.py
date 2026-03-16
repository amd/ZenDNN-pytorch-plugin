# ******************************************************************************
# Copyright (c) 2023-2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import os
import ctypes

# Load libzentorch.so with RTLD_GLOBAL so that AOTI-compiled modules can
# find the shim functions (aoti_torch_cpu_zentorch_*)
_lib_dir = os.path.dirname(os.path.abspath(__file__))
_lib_path = os.path.join(_lib_dir, "libzentorch.so")
if os.path.exists(_lib_path):
    # The mode parameter must be passed directly to CDLL for RTLD_GLOBAL to work
    ctypes.CDLL(_lib_path, mode=os.RTLD_GLOBAL | os.RTLD_NOW)

from ._build_info import __torchversion__ as buildtime_torchversion  # noqa: E402
from torch.torch_version import __version__ as runtime_torchversion  # noqa: E402

# Pytorch lacks symbol-level compatibility, requiring extensions
# to be pinned to the same minor version. To avoid issues, it is
# necessary to error out if the runtime Pytorch version
# differs from the build-time version.


def _get_minor_version(torch_version):
    """Return the major.minor portion from a PyTorch version string.
    Examples: '2.9.1+cpu' -> '2.9', '2.10.0+cpu' -> '2.10'.
    """
    parts = torch_version.split(".")
    if len(parts) < 2:
        raise ImportError(
            f"Unexpected PyTorch version string {torch_version!r}. "
            "Expected at least major.minor (e.g., 2.9.1 or 2.10.0)."
        )
    return f"{parts[0]}.{parts[1]}"


_runtime_minor = _get_minor_version(runtime_torchversion)
_buildtime_minor = _get_minor_version(buildtime_torchversion)

if _runtime_minor != _buildtime_minor:
    raise ImportError(
        f"Incompatible PyTorch version {runtime_torchversion} detected. "
        f"The installed zentorch binary is only compatible "
        f"with PyTorch versions {_buildtime_minor}.x"
    )

from ._optimize import optimize  # noqa
from ._optimize_for_export import export_optimize_pass  # noqa
from ._info import __config__, __version__  # noqa
from ._compile_backend import *  # noqa
from ._meta_registrations import *  # noqa
from ._lowerings import *  # noqa
from ._freeze_utils import freezing_enabled  # noqa
from . import utils  # noqa F401
from . import llm  # noqa F401
