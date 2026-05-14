# ******************************************************************************
# Copyright (c) 2023-2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

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
