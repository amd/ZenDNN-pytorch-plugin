# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import atexit
import json
import os

import torch
import collections
import importlib.util
from typing import List

# import the custom logging module
from ._logging import get_logger
from ._fp16_capabilities import is_fp16_capable
from ._C import is_fp16_supported

# make a logger for this file
logger = get_logger(__name__)

counters = collections.defaultdict(collections.Counter)

at_ops = torch.ops.aten


# When arg_index is none, it will check for node
def get_tensor(fx_graph, node, arg_index=None):
    if arg_index is not None:
        # To identify fake tensors, we check for the 'val' and 'tensor_meta'
        # keys in node.args[arg_index].meta. Till PT <= 2.4.x, presence of
        # metadata implied fake tensors. But from PT 2.5.x,
        # the 'mutation_region_id' default argument is introduced in meta.
        if node.args[arg_index].target == at_ops.clone.default:
            # workaround for CNNs in freezing path
            return node.args[arg_index].args[0].meta["val"]
        if "val" in node.args[arg_index].meta:
            # arg node in fx_graph generated through torch.compile
            # will be fake tensor
            return node.args[arg_index].meta["val"]
        else:
            # while arg node in fx_graph generated through make_fx
            # will not be fake tensor
            return fx_graph._parameters[node.args[arg_index].target]
    else:
        is_fake_tensor = bool(node.meta)
        if is_fake_tensor:
            # arg node in fx_graph generated through torch.compile will be fake tensor
            return node.meta["val"]
        else:
            # while arg node in fx_graph generated through make_fx
            # will not be fake tensor
            return fx_graph._parameters[node.target]


# Compare all the args are same dtype or not
def are_args_same_dtype(fx_graph, node):
    dtype_set = set()
    for i in range(0, len(node.args)):
        dtype_set.add(get_tensor(fx_graph, node, i).dtype)
    return len(dtype_set) == 1


# Checks if the current match has fp16 argument/keyword argument.
# In case it does, it checks if the op and hardware supports fp16.
# By design this function is resposible for rejecting the pass
# for operators which do not support fp16.
# It is not a robust check for the combination of args and kwargs
# which are valid per operators.
def is_valid_fp16(op_name, match):
    for arg in getattr(match, "args", ()):
        # Only consider FX Nodes that have tensor-like meta["val"] with a dtype.
        if not isinstance(arg, torch.fx.Node):
            continue
        meta = getattr(arg, "meta", None)
        if not isinstance(meta, dict):
            continue
        val = meta.get("val", None)
        dtype = getattr(val, "dtype", None)
        if dtype is torch.float16:
            return is_fp16_supported() and is_fp16_capable(op_name)
    for _, value in getattr(match, "kwargs", {None: None}).items():
        if not isinstance(value, torch.fx.Node):
            continue
        meta = getattr(value, "meta", None)
        if not isinstance(meta, dict):
            continue
        val = meta.get("val", None)
        dtype = getattr(val, "dtype", None)
        if dtype is torch.float16:
            return is_fp16_supported() and is_fp16_capable(op_name)
    return True


def numdims_tensor(fx_graph, node, arg_index=None):
    return get_tensor(fx_graph, node, arg_index).ndim


def is_arg_1d_tensor(fx_graph, node, arg_index):
    dims = numdims_tensor(fx_graph, node, arg_index)
    return dims == 1


def is_bias_1d_tensor(fx_graph, node):
    # checks if self/bias tensor is 1-d or not
    # returns true if 1d bias tensor
    return is_arg_1d_tensor(fx_graph, node, 0)


def _dump_counters():
    from datetime import datetime

    counter_data = {k: dict(v) for k, v in counters.items() if v}
    if not counter_data:
        return
    formatted = json.dumps(counter_data, indent=2)

    counters_dir = os.path.join(os.getcwd(), "counters")
    os.makedirs(counters_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(
        counters_dir, f"counters_{timestamp}_pid{os.getpid()}.json"
    )
    with open(output_path, "w") as f:
        f.write(formatted)
    logger.info("Zentorch counters saved to: %s", output_path)


if os.environ.get("ZENTORCH_DUMP_COUNTERS", "0") == "1":
    atexit.register(_dump_counters)


# getattr can result in false negatives if the submodule
# isn't already imported in __init.py__
# To check if a submodule exists without importing it,
# we use importlib.util.find_spec
def is_version_compatible_import(modules: List[str], functions: List[str]) -> bool:
    """
    Checks if the specified modules and functions exist in the current
    version of PyTorch.
    The check is done sequentially for each module and function.

    Args:
        modules (list): A list of module names to check sequentially
        in torch (e.g., [_x1, x2]).
        functions (list): A list of function names to check for within
        the final module (e.g., [a1, a2]).

    Returns:
        bool: True if all modules and functions are available in the current
        PyTorch version, False otherwise.
    """
    current_module = torch  # Start with the base 'torch' module
    full_name = "torch"
    # Sequentially check if each module exists in the hierarchy
    for module_name in modules:
        full_name = f"{full_name}.{module_name}"
        spec = importlib.util.find_spec(full_name)
        if spec is None:
            return False

    # Move to the next level of module
    current_module = importlib.import_module(f"{full_name}")

    # Check if the functions exist in the final module
    for func in functions:
        return all(hasattr(current_module, func) for func in functions)

    # If all checks pass
    return True


def add_version_suffix(major: str, minor: str, patch: str = 0):
    # This function will add a ".dev" substring to the input arguments.
    # This will extend the pytorch version comparisions done using TorchVersion
    # class to include nightly and custom build versions as well.
    # The following tables shows the behaviour of TorchVersion comparisons
    # for release, nightlies and custom binaries, when the substring is used.
    # ".dev" is added to second column i.e A.B.C -> A.B.C.dev

    # This function is intended for only lesser than comparisons.

    #                           X.Y.Z < A.B.C
    # +---------------+----------------+-----------------+
    # | Torch Version |  Torch Version |  Implementation |
    # | used by user  |      to be     |    Behaviour    |
    # |    (X.Y.Z)    |  compared with |                 |
    # |               |    (A.B.C)     |                 |
    # +---------------+----------------+-----------------+
    # |      2.3.1    |      2.4.0     |      True       |
    # +---------------+----------------+-----------------+
    # |      2.4.0    |      2.4.0     |      False      |
    # +---------------+----------------+-----------------+
    # |    2.4.0.dev  |      2.4.0     |      False      |
    # |   (Nightly    |                |                 |
    # |    binaries)  |                |                 |
    # +---------------+----------------+-----------------+
    # |    2.5.0.dev  |      2.4.0     |      False      |
    # |    2.6.0.dev  |                |                 |
    # |   (Nightly    |                |                 |
    # |    binaries)  |                |                 |
    # +---------------+----------------+-----------------+
    # |  2.4.0a0+git  |    2.4.0       |      False      |
    # |    d990dad    |                |                 |
    # +---------------+----------------+-----------------+

    return f"{major}.{minor}.{patch}.dev"


def find_path(fx_graph, start_node, end_node):
    # using iterative dfs to find path from start_node to end_node
    stack = [start_node]
    visited = set()

    while stack:
        current_node = stack.pop()
        if current_node in visited:
            continue
        visited.add(current_node)
        if current_node is end_node:
            return True
        # Add users to the stack
        for user_node in current_node.users:
            if user_node not in visited:
                stack.append(user_node)

    return False


def _is_used_by_zentorch_qlinear(node):
    """Check if any user of this node is a zentorch qlinear* op."""
    zentorch_qlinear_ops = {
        torch.ops.zentorch.zentorch_qlinear.default,
        torch.ops.zentorch.zentorch_qlinear_relu.default,
        torch.ops.zentorch.zentorch_qlinear_sigmoid.default,
        torch.ops.zentorch.zentorch_qlinear_mul_add.default,
    }
    for user in node.users:  # noqa: SIM110
        if user.op == "call_function" and user.target in zentorch_qlinear_ops:
            return True
    return False
