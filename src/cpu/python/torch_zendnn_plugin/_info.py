# ******************************************************************************
# Copyright (c) 2023 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import torch_zendnn_plugin._C
import sys

__config__ = torch_zendnn_plugin._C.show_config()
if sys.version_info >= (3, 8):
    from importlib import metadata
    __version__ = metadata.version('torch-zendnn-plugin')
else:
    __version__ = ''
