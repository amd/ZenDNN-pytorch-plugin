# ******************************************************************************
# Copyright (c) 2023-2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

from . import __stable_abi_only__

if __stable_abi_only__:
    # Do not import _C here. It would also load libzentorch.so, which is the
    # library we already skipped for this torch. try/except ImportError is not
    # enough: a successful load of both .so files is the failure we must avoid.
    # Version metadata still comes from _build_info; only show_config() is lost.
    __config__ = (
        "zentorch built for a different PyTorch minor version; "
        "config unavailable from the portable library"
    )
else:
    # No try/except: in full mode a failure here means a broken install, and
    # reporting that as a version mismatch would send the reader the wrong way.
    import zentorch._C

    __config__ = zentorch._C.show_config()

try:
    from ._build_info import __version__
except ImportError:
    __version__ = "unknown"

try:
    from ._build_info import __source_tag__
except ImportError:
    __source_tag__ = ""

try:
    from ._build_info import __release_type__
except ImportError:
    __release_type__ = "unknown"
