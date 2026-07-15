# ******************************************************************************
# Copyright (c) 2023-2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

from . import __stable_abi_only__

if __stable_abi_only__:
    # Deliberately not importing _C. It carries libzentorch.so in its NEEDED
    # list, and that is the library already ruled out for this torch, while the
    # portable one is loaded by the time this runs - both register the same
    # zentorch::* schemas. Guarding with try/except ImportError would not help:
    # the loader does not necessarily reject the mismatched library, so the case
    # that breaks the exactly-one-library invariant is the dlopen *succeeding*.
    # The version metadata below comes from _build_info and is meaningful in
    # either mode, so only the config string is lost.
    __config__ = "zentorch built for a different PyTorch minor version; \
config unavailable from the portable library"
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
