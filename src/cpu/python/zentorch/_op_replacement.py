# ******************************************************************************
# Copyright (c) 2024 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import torch

# import the custom logging module
from ._logging import get_logger

# make a logger for this file
logger = get_logger(__name__)

at_ops = torch.ops.aten
zt_ops = torch.ops.zentorch


# When arg_index is none, it will check for node
def get_tensor(fx_graph, node, arg_index=None):
    if arg_index is not None:
        is_fake_tensor = bool(node.args[arg_index].meta)
        if is_fake_tensor:
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
    if len(dtype_set) == 1:
        return True
    else:
        return False


def numdims_tensor(fx_graph, node, arg_index=None):
    return get_tensor(fx_graph, node, arg_index).ndim


def is_baddbmm_replacable(fx_graph, node):
    return all(numdims_tensor(fx_graph, node, i) == 3 for i in range(0, 3))


def is_arg_1d_tensor(fx_graph, node, arg_index):
    dims = numdims_tensor(fx_graph, node, arg_index)
    if dims == 1:
        return True
    else:
        return False


# checks the two conditions for arg nodes datatypes
# the tensor to check is either directly accessible through
# parameters of fx_graph or is stored as fake in meta dict.
def is_arg_dtype_bfloat16(fx_graph, node, arg_index):
    is_fake_tensor = bool(node.args[arg_index].meta)

    if is_fake_tensor:
        # arg node in fx_graph generated through torch.compile will be fake tensor
        arg_dtype = node.args[arg_index].meta["val"].dtype
    else:
        # while arg node in fx_graph generated through make_fx will not be fake tensor
        arg_dtype = fx_graph._parameters[node.args[arg_index].target].dtype

    if arg_dtype == torch.bfloat16:
        return True
    else:
        return False


def is_embedding_bag_op_replacable(fx_graph, node):
    if is_arg_dtype_bfloat16(fx_graph, node, 0):
        logger.info(
            "embedding_bag op will not be replaced as"
            + " zentorch doesn't support bf16 with it yet!"
        )
        # don't replace embedding bag if autocast is enabled or if the model
        # is mixed precision as zendnn doesn't support it w/ bf16
        return False
    else:
        return True


def is_embedding_op_replacable(fx_graph, node):
    if is_arg_dtype_bfloat16(fx_graph, node, 0):
        logger.info(
            "embedding op will not be replaced as"
            + " zentorch doesn't support bf16 with it yet!"
        )
        # don't replace embedding if autocast is enabled or if the model
        # is mixed precision as zendnn doesn't support it w/ bf16
        return False
    else:
        # Currently zentorch_embedding op only accepts 1-D inputs
        # which is predominantly evident in RecSys models. The
        # embedding op in Langauge models like Bert work with
        # 2-D inputs. In such cases, we do not replace
        # aten embedding with zendnn embedding. The replacement
        # is taken care by getting the input shapes from the graph.
        # returns true if inputs to embedding are 1-D

        if is_arg_1d_tensor(fx_graph, node, 1):
            return True

        logger.info(
            "embedding op will not be replaced as"
            + " zentorch supports only 1-dimensional inputs to the op!"
        )
        return False


def is_bias_1d_tensor(fx_graph, node):
    # checks if self/bias tensor is 1-d or not
    # returns true if 1d bias tensor
    return is_arg_1d_tensor(fx_graph, node, 0)


at_to_zen_op_dict = {
    at_ops._embedding_bag.default: (
        zt_ops.zentorch_embedding_bag.default,
        is_embedding_bag_op_replacable,
    ),
    at_ops.embedding.default: (
        zt_ops.zentorch_embedding.default,
        is_embedding_op_replacable,
    ),
    at_ops.mm.default: (zt_ops.zentorch_mm.default, None),
    at_ops.bmm.default: (zt_ops.zentorch_bmm.default, None),
    at_ops.addmm.default: (zt_ops.zentorch_addmm.default, None),
    at_ops.baddbmm.default: (
        zt_ops.zentorch_baddbmm.default,
        is_baddbmm_replacable,
    ),
}

# currently only zentorch_addmm is the conditional zentorch op
zen_to_zen_op_dict = {
    zt_ops.zentorch_addmm.default: (
        zt_ops.zentorch_addmm_1dbias.default,
        is_bias_1d_tensor,
    ),
}


# generalized function for op-replacement
# this will take a list of dictionaries and iterate over each dict
# for replacement of the ops -> at, ipex [contitional], zen
def replace_with_zentorch_ops(fx_graph: torch.fx.GraphModule, op_dict_lst: list):
    # Loop through the nodes in fx_graph.graph
    # Replacing aten ops with respective zentorch ops
    for node in fx_graph.graph.nodes:
        # Checking for op implementation to be replaced.
        for op_dict in op_dict_lst:
            if node.target in op_dict.keys():
                target_op = node.target
                if op_dict[target_op][1] is not None:
                    if op_dict[target_op][1](fx_graph, node):
                        logger.info(
                            "Now replacing default "
                            + str(target_op)
                            + " with "
                            + str(op_dict[target_op][0])
                            + "!"
                        )
                        node.target = op_dict[target_op][0]
                    else:
                        logger.info(
                            "Not able to replace default "
                            + str(target_op)
                            + " with "
                            + str(op_dict[target_op][0])
                            + " due to non-fulfilment of the condition."
                        )
                else:
                    logger.info(
                        "Now replacing default "
                        + str(target_op)
                        + " with "
                        + str(op_dict[target_op][0])
                        + "!"
                    )
                    node.target = op_dict[target_op][0]

    return fx_graph
