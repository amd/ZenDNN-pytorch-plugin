# ******************************************************************************
# Copyright (c) 2023-2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

from importlib import metadata as _metadata


def _check_dual_install():
    """Raise ImportError if both zentorch and zentorch-weekly are installed.

    zentorch is published under two PyPI package names: 'zentorch' for stable
    GA releases and 'zentorch-weekly' for weekly development builds. Having
    both installed simultaneously causes version conflicts and undefined
    behaviour, so we detect and reject this early at import time.
    """
    installed_dists = []
    for _name in ("zentorch", "zentorch-weekly"):
        try:
            _metadata.version(_name)
            installed_dists.append(_name)
        except _metadata.PackageNotFoundError:
            pass
    if len(installed_dists) > 1:
        raise ImportError(
            f"Both {' and '.join(installed_dists)} are installed. "
            "Please uninstall one of them, for example:\n"
            "  pip uninstall zentorch\n"
            "  pip uninstall zentorch-weekly"
        )


_check_dual_install()
import os  # noqa: E402
import ctypes  # noqa: E402
import warnings  # noqa: E402
import torch  # noqa: E402

from ._build_info import __torchversion__ as buildtime_torchversion  # noqa: E402
from torch.torch_version import __version__ as runtime_torchversion  # noqa: E402

# Pytorch lacks symbol-level compatibility, so libzentorch.so is pinned to the
# build-time minor version. libzentorch_stable.so carries the stable-ABI subset
# of the ops and loads in its place when the runtime minor version differs.
# Exactly one of the two is loaded per process.


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
# _info.py reads this during package init; keep the assignment before that import.
__stable_abi_only__ = _runtime_minor != _buildtime_minor

_lib_dir = os.path.dirname(os.path.abspath(__file__))
_lib_path = os.path.join(_lib_dir, "libzentorch.so")
_stable_lib_path = os.path.join(_lib_dir, "libzentorch_stable.so")

if not __stable_abi_only__:
    # Load libzentorch.so with RTLD_GLOBAL so that AOTI-compiled modules can
    # find the shim functions (aoti_torch_cpu_zentorch_*).
    # This MUST happen after `torch` has been imported above: libzentorch.so
    # references ATen symbols from libtorch_cpu.so, and RTLD_NOW would fail
    # to resolve them if the torch shared libraries are not yet loaded.
    if os.path.exists(_lib_path):
        # The mode parameter must be passed directly to CDLL for RTLD_GLOBAL to work
        ctypes.CDLL(_lib_path, mode=os.RTLD_GLOBAL | os.RTLD_NOW)
elif os.path.exists(_stable_lib_path):
    warnings.warn(
        f"PyTorch {runtime_torchversion} does not match the zentorch build "
        f"({_buildtime_minor}.x): loading the stable-ABI ops from "
        f"libzentorch_stable.so. Rebuild zentorch to get the full backend.",
        stacklevel=1,
    )
    torch.ops.load_library(_stable_lib_path)
else:
    raise ImportError(
        f"Incompatible PyTorch version {runtime_torchversion} detected. "
        f"The installed zentorch binary is only compatible "
        f"with PyTorch versions {_buildtime_minor}.x"
    )

# Neither of these depends on which native library loaded: the fp16 registry is
# pure Python keyed off an env var, and the version metadata comes from
# _build_info. Keeping them outside the branch keeps the package's Python
# surface identical in both modes.
from ._info import (  # noqa: F401,E402
    __config__,
    __version__,
    __source_tag__,
    __release_type__,
)
from ._fp16_capabilities import update_fp16_registry  # noqa: E402

update_fp16_registry()

if not __stable_abi_only__:
    from ._optimize import optimize  # noqa: F401,E402
    from ._optimize_for_export import export_optimize_pass  # noqa: F401,E402
    from ._compile_backend import *  # noqa: F401,F403,E402
    from ._meta_registrations import *  # noqa: F401,F403,E402
    from ._lowerings import *  # noqa: F401,F403,E402
    from ._freeze_utils import freezing_enabled  # noqa: F401,E402
    from . import utils, llm  # noqa: F401,E402
