# ******************************************************************************
# Copyright (c) 2024 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

from typing import Union
import torch.nn as nn


def set_op_by_name(
    layer: Union[nn.Module, nn.ModuleList],
    name: str, new_module: nn.Module
) -> None:
    """
    Replaces a submodule in a given neural network layer with a new module
    (e.g. quantized module). The submodule to be replaced is identified by
    the 'name' parameter, which specifies the name of the submodule using
    dot notation. If the name includes dots, it navigates through nested
    submodules to find the specific layer to replace. Otherwise, it
    directly replaces the submodule in the provided layer.

    Parameters:
    - layer: The top-level module containing the submodule.
    - name: name of the submodule, split by dots.
    - new_module: The new module to replace the existing one,
                  for example the quantized module.
    """
    levels = name.split('.')
    if len(levels) > 1:
        mod_ = layer
        for l_idx in range(len(levels) - 1):
            if levels[l_idx].isdigit() and isinstance(mod_, nn.ModuleList):
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        setattr(mod_, levels[-1], new_module)
    else:
        setattr(layer, name, new_module)


def get_op_by_name(
    layer: Union[nn.Module, nn.ModuleList], name: str
) -> Union[nn.Module, nn.ModuleList]:
    levels = name.split('.')
    mod_ = layer
    for l_idx in range(len(levels)):
        if levels[l_idx].isdigit() and isinstance(mod_, nn.ModuleList):
            mod_ = mod_[int(levels[l_idx])]
        else:
            mod_ = getattr(mod_, levels[l_idx])
    return mod_


def get_module_name_str(
    parameter_key: str
) -> str:
    return parameter_key.rsplit('.', 1)[0]
