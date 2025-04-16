# ******************************************************************************
# Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

from typing import Union, Dict, Any, Tuple, Iterable
import torch
import torch.nn as nn


def set_op_by_name(
    layer: Union[nn.Module, nn.ModuleList], name: str, new_module: nn.Module
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
    levels = name.split(".")
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


def get_name_and_info(
    model_info: Dict[str, Any], parent_key: str = ""
) -> Iterable[Tuple[str, Dict[str, Any]]]:
    for key, value in model_info.items():
        new_key = f"{parent_key}.{key}" if parent_key else key
        if isinstance(value, dict):
            if (
                value.get("type", None) is not None
                and value.get("weight", None) is not None
            ):
                yield new_key, value
            else:
                yield from get_name_and_info(value, new_key)
        else:
            continue


def get_torch_type_from_str(str_type):
    if str_type.lower() == "bfloat16":
        return torch.bfloat16
    elif str_type.lower() == "float32":
        return torch.float32
    else:
        raise ValueError("Only float32 or bfloat16 models are supported.")
