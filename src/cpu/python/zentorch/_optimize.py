# ******************************************************************************
# Copyright (c) 2023-2024 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import torch
import zentorch._C  # noqa

# import the custom logging module
from ._logging import get_logger
from ._util import save_graph
from ._zentorch_op_replacement import (
    replace_with_zentorch_ops,
    is_bias_1d_tensor,
    numdims_tensor,
)
# TODO: Add support for horizontal_mlp_fusion
from ._zentorch_custom_op_replacement import (
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
    with zendnn implementation of respective ops and fusion
    few ops
    """
    # Dumping of the native graph in svg format
    save_graph(fx_graph, "native_model")

    logger.info("Optimizing the fx_graph with zentorch ops.")

    # Replacing ops to zendnn ops
    optimized_graph = replace_with_zentorch_ops(fx_graph)

    # Op fusions supported by ZenDNN
    optimized_graph = zendnn_op_fusion(optimized_graph)

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


# use to fuse relu
def set_relu_mm_fusion_kwargs(node):
    return {**node.kwargs, "fuse": 1}


# use to fuse gelu erf & tanh
def set_gelu_mm_fusion_kwargs(node):
    fuse = 3
    if (
        bool(list(node.users.keys())[0].kwargs)
        and list(node.users.keys())[0].kwargs["approximate"] == "tanh"
    ):
        fuse = 2
    return {**node.kwargs, "fuse": fuse}


# create dict according to fuse
eltwise_patterns = dict.fromkeys(
    (
        zt_ops.zendnn_mm.default,
        zt_ops.zendnn_addmm.default,
        zt_ops.zendnn_addmm_1dbias.default,
    ),
    {
        at_ops.relu.default: set_relu_mm_fusion_kwargs,
        at_ops.relu_.default: set_relu_mm_fusion_kwargs,
        at_ops.gelu.default: set_gelu_mm_fusion_kwargs,
        at_ops.gelu_.default: set_gelu_mm_fusion_kwargs,
    },
)

# for now add is not added as post op that's why I created this pattern
add_pattern = (zt_ops.zendnn_bmm.default, zt_ops.zendnn_mm.default)


# list of benign operators
benign_op = [at_ops.clone.default, at_ops.view.default]


def zendnn_op_fusion(fx_graph):
    """
    zendnn_op_fusion:
    takes in the fx_graph and fuses some of the native ops
    with zendnn implementation of respective op fusions
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
                    if node.target == zt_ops.zendnn_bmm.default
                    else False
                )
                if should_add_be_fused:
                    for add_tensor in node_next.args:
                        if add_tensor != node:
                            # by *node.args we can append all the arguments
                            new_args = (add_tensor, *node.args)
                    node.args = new_args
                    node_next.replace_all_uses_with(node)
                    fx_graph.graph.erase_node(node_next)

                    if node.target == zt_ops.zendnn_mm.default:
                        if is_bias_1d_tensor(fx_graph, node):
                            node.target = zt_ops.zendnn_addmm_1dbias.default
                        else:
                            node.target = zt_ops.zendnn_addmm.default
                    else:
                        node.target = zt_ops.zendnn_baddbmm.default
                else:
                    logger.warning(
                        "baddbdmm in zentorch doesnt support "
                        + "non 3 dimentional tensors as of now"
                    )
        if node.target in eltwise_patterns:
            # create a sub-dict from pattern dict
            if len(node.users) > 1:  # Output of node is used by other nodes
                continue
            op_dict = eltwise_patterns[node.target]
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
            if next_node.target in op_dict:
                # call the function for eltwise ops
                node.kwargs = op_dict[next_node.target](op_list[-1])
                next_node.replace_all_uses_with(op_list[-1])
                fx_graph.graph.erase_node(next_node)

    logger.info("Recompiling the fx_graph with fusion changes made.")
    fx_graph.graph.set_codegen(torch.fx.graph.CodeGen())
    fx_graph.recompile()
    return fx_graph
