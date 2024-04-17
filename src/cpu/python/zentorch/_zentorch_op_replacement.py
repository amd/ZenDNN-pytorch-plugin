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


def numdims_tensor(fx_graph, node, arg_index):
    is_fake_tensor = bool(node.args[arg_index].meta)

    if is_fake_tensor:
        # arg node in fx_graph generated through torch.compile will be fake tensor
        return node.args[arg_index].meta["val"].ndim
    else:
        # while arg node in fx_graph generated through make_fx will not be fake tensor
        return fx_graph._parameters[node.args[arg_index].target].ndim


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
        logger.warning(
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
        logger.warning(
            "embedding op will not be replaced as"
            + " zentorch doesn't support bf16 with it yet!"
        )
        # don't replace embedding if autocast is enabled or if the model
        # is mixed precision as zendnn doesn't support it w/ bf16
        return False
    else:
        # Currently zendnn_embedding op only accepts 1-D inputs
        # which is predominantly evident in RecSys models. The
        # embedding op in Langauge models like Bert work with
        # 2-D inputs. In such cases, we do not replace
        # aten embedding with zendnn embedding. The replacement
        # is taken care by getting the input shapes from the graph.
        # returns true if inputs to embedding are 1-D

        if is_arg_1d_tensor(fx_graph, node, 1):
            return True

        logger.warning(
            "embedding op will not be replaced as"
            + " zentorch supports only 1-dimensional inputs to the op!"
        )
        return False


def is_bias_1d_tensor(fx_graph, node):
    # checks if self/bias tensor is 1-d or not
    # returns true if 1d bias tensor
    return is_arg_1d_tensor(fx_graph, node, 0)


def replace_with_zentorch_ops(fx_graph):
    op_dict = {
        at_ops._embedding_bag.default: (
            zt_ops.zendnn_embedding_bag.default,
            is_embedding_bag_op_replacable,
        ),
        at_ops.embedding.default: (
            zt_ops.zendnn_embedding.default,
            is_embedding_op_replacable,
        ),
        at_ops.mm.default: (zt_ops.zendnn_mm.default, None),
        at_ops.bmm.default: (zt_ops.zendnn_bmm.default, None),
        at_ops.addmm.default: (zt_ops.zendnn_addmm.default, None),
        at_ops.baddbmm.default: (
            zt_ops.zendnn_baddbmm.default,
            is_baddbmm_replacable,
        ),
    }
    # Loop through the nodes in fx_graph.graph
    # Replacing aten ops with respective zendnn ops
    for node in fx_graph.graph.nodes:
        # Checking for op default implementation to be replaced.
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

            # currently only zendnn_addmm is the conditional zendnn op
            if node.target == zt_ops.zendnn_addmm.default:
                if is_bias_1d_tensor(fx_graph, node):
                    node.target = zt_ops.zendnn_addmm_1dbias.default

    return fx_graph
