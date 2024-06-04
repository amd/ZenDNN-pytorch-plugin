# ******************************************************************************
# Copyright (c) 2023-2024 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import torch
import zentorch._C  # noqa

# import the custom logging module
from ._logging import get_logger
from ._util import save_graph
from ._op_replacement import (
    replace_with_zentorch_ops,
    is_bias_1d_tensor,
    numdims_tensor,
    are_args_same_dtype,
    get_tensor,
)

# TODO: Add support for horizontal_mlp_fusion
from ._custom_op_replacement import (
    emb_ops_horizontal_fusion,
    vertical_mlp_fusion,
    eb_group_mlp_group_fusion,
)

# make a logger for this file
logger = get_logger(__name__)

at_ops = torch.ops.aten
zt_ops = torch.ops.zentorch


def optimize(fx_graph):
    """
    optimize:
    takes in the fx_graph and replaces some of the native ops
    with zentorch implementation of respective ops and fusion
    few ops
    """
    # Dumping of the native graph in svg format
    save_graph(fx_graph, "native_model")

    logger.info("Optimizing the fx_graph with zentorch ops.")

    # Replacing ops to zentorch ops
    optimized_graph = replace_with_zentorch_ops(fx_graph)

    # Op fusions supported by zentorch
    optimized_graph = zentorch_op_fusion(optimized_graph)

    # Fusion of parallel embeddingbags
    optimized_graph = emb_ops_horizontal_fusion(optimized_graph)

    # Vertical fusion of Consecutive MLP layers
    optimized_graph = vertical_mlp_fusion(optimized_graph)

    # Fusion of parallel MLP layers
    # TODO: Add Addtional Support for Horizontal Fusion
    # optimized_graph = horizontal_mlp_fusion(optimized_graph)

    # Horizontal fusion of parallel Group EB op and Group MLP op
    optimized_graph = eb_group_mlp_group_fusion(optimized_graph)

    # Dumping of the optimized graph in svg format
    save_graph(optimized_graph, "zen_optimized_model")

    return optimized_graph


eltwise_targets = {
    zt_ops.zentorch_mm.default: [
        zt_ops.zentorch_mm_relu.default,
        zt_ops.zentorch_mm_gelu_tanh.default,
        zt_ops.zentorch_mm_gelu_erf.default,
    ],
    zt_ops.zentorch_addmm.default: [
        zt_ops.zentorch_addmm_relu.default,
        zt_ops.zentorch_addmm_gelu_tanh.default,
        zt_ops.zentorch_addmm_gelu_erf.default,
    ],
    zt_ops.zentorch_addmm_1dbias.default: [
        zt_ops.zentorch_addmm_1dbias_relu.default,
        zt_ops.zentorch_addmm_1dbias_gelu_tanh.default,
        zt_ops.zentorch_addmm_1dbias_gelu_erf.default,
    ],
}

supported_eltwise_ops = (
    at_ops.relu.default,
    at_ops.relu_.default,
    at_ops.gelu.default,
    at_ops.gelu_.default,
)


# use to fuse relu, gelu(erf/tanh) with mm variants.
def set_fused_target_for_mm(node, post_op):
    if post_op.target == at_ops.relu.default or post_op.target == at_ops.relu_.default:
        node.target = eltwise_targets[node.target][0]
    elif bool(post_op.kwargs) and post_op.kwargs["approximate"] == "tanh":
        node.target = eltwise_targets[node.target][1]
    else:
        node.target = eltwise_targets[node.target][2]


# for now add is not added as post op that's why I created this pattern
add_pattern = (zt_ops.zentorch_bmm.default, zt_ops.zentorch_mm.default)


# list of benign operators
benign_op = [at_ops.clone.default, at_ops.view.default]


def zentorch_op_fusion(fx_graph):
    """
    zentorch_op_fusion:
    takes in the fx_graph and fuses some of the native ops
    with zentorch implementation of respective op fusions
    """
    logger.info("Fusing the zentorch ops in fx graph.")
    # Loop through the nodes in fx_graph.graph
    for node in fx_graph.graph.nodes:
        if len(node.users) > 1:  # Output of node is used by other nodes
            continue
        if node.target == at_ops.clone.default and len(node.kwargs) == 0:
            node.replace_all_uses_with(node.args[0])
            fx_graph.graph.erase_node(node)
    for node in fx_graph.graph.nodes:
        if len(node.users) > 1:  # Output of node is used by other nodes
            continue
        # check the pattern for mm->add or bmm->add
        # TODO Support add fusion when add is farther from matmul node
        # TODO Unit Tests to be added for the farther case
        # TODO Add a negative Unit test for the farther case
        # TODO Support mm add fusion when add is scalar or its dim is 0
        if node.target in add_pattern:
            node_next = list(node.users.keys())[0]
            if node_next.target == at_ops.add.Tensor and node_next == node.next:
                logger.info(
                    "Fusing the "
                    + str(node.target)
                    + "->"
                    + str(node_next.target)
                    + " in fx graph"
                )
                # fuse bmm and add only if dims of bmm and add is same
                should_add_be_fused = (
                    all(
                        numdims_tensor(fx_graph, tensor, dim) == 3
                        for tensor, dim in (
                            (node, 0),
                            (node, 1),
                            (node_next, 0),
                            (node_next, 1),
                        )
                    )
                    if node.target == zt_ops.zentorch_bmm.default
                    else True
                )
                if should_add_be_fused and are_args_same_dtype(fx_graph, node_next):
                    for add_tensor in node_next.args:
                        if add_tensor != node:
                            if (
                                torch.is_tensor(get_tensor(fx_graph, add_tensor))
                                and get_tensor(fx_graph, add_tensor).dim != 0
                            ):
                                # by *node.args we can append all the arguments
                                new_args = (add_tensor, *node.args)
                                node.args = new_args
                                node_next.replace_all_uses_with(node)
                                fx_graph.graph.erase_node(node_next)
                                if node.target == zt_ops.zentorch_mm.default:
                                    if is_bias_1d_tensor(fx_graph, node):
                                        node.target = (
                                            zt_ops.zentorch_addmm_1dbias.default
                                        )
                                    else:
                                        node.target = zt_ops.zentorch_addmm.default
                                else:
                                    node.target = zt_ops.zentorch_baddbmm.default
                else:
                    logger.warning(
                        "baddbmm in zentorch doesnt support "
                        + "non 3 dimentional tensors as of now"
                    )
        # The last node in the graph pattern should be replaced. Eltwise
        # fusion is an exception.
        if node.target in eltwise_targets:
            # create a sub-dict from pattern dict
            if len(node.users) > 1:  # Output of node is used by other nodes
                continue
            op_list = [node]
            # store the user of node in next_node
            next_node = list(node.users.keys())[0]
            # checking for benign op
            while next_node.target in benign_op:
                if len(next_node.users) > 1:  # Output of node is used by other nodes
                    break
                # store benign op in list
                op_list.append(next_node)
                # store user of next_node
                next_node = list(next_node.users.keys())[0]
            if next_node.target in supported_eltwise_ops:
                # call the function for eltwise ops
                set_fused_target_for_mm(node, next_node)
                next_node.replace_all_uses_with(op_list[-1])
                fx_graph.graph.erase_node(next_node)

    logger.info("Recompiling the fx_graph with fusion changes made.")
    fx_graph.graph.set_codegen(torch.fx.graph.CodeGen())
    fx_graph.recompile()
    return fx_graph
