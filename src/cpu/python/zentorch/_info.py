# ******************************************************************************
# Copyright (c) 2023-2024 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import zentorch._C
import sys

__config__ = zentorch._C.show_config()
if sys.version_info >= (3, 8):
    from importlib import metadata
    __version__ = metadata.version('zentorch')
else:
    __version__ = ''
