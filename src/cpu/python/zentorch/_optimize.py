# ******************************************************************************
# Copyright (c) 2023-2024 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import torch
import zentorch._C  # noqa

# import the custom logging module
from ._logging import get_logger
from ._util import save_graph
from ._zentorch_op_replacement import replace_with_zentorch_ops, is_bias_1d_tensor
from ._zentorch_custom_op_replacement import (
    emb_ops_horizontal_fusion,
    vertical_mlp_fusion,
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

    # Dumping of the optimized graph in svg format
    save_graph(optimized_graph, "zen_optimized_model")

    return optimized_graph


def set_relu_mm_fusion_kwargs(node):
    return {**node.kwargs, "fuse": 1}


def set_gelu_mm_fusion_kwargs(node):
    fuse = 3
    if bool(node.next.kwargs) and node.next.kwargs["approximate"] == "tanh":
        fuse = 2
    return {**node.kwargs, "fuse": fuse}


# create dict according to fuse
op_eltwise_pattern = dict.fromkeys(
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

op_add_pattern = [
    (zt_ops.zendnn_bmm.default, at_ops.add.Tensor),
    (zt_ops.zendnn_mm.default, at_ops.add.Tensor),
]


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
        # check the pattern for mm->add or bmm->add
        if (node.target, node.next.target) in op_add_pattern:
            logger.info(
                "Fusing the "
                + str(node.target)
                + "->"
                + str(node.next.target)
                + " in fx graph"
            )
            for add_tensor in node.next.args:
                if add_tensor != node:
                    # by *node.args we can append all the arguments
                    new_args = (add_tensor, *node.args)
            node.args = new_args
            node.next.replace_all_uses_with(node)
            fx_graph.graph.erase_node(node.next)

            if node.target == zt_ops.zendnn_mm.default:
                if is_bias_1d_tensor(fx_graph, node):
                    node.target = zt_ops.zendnn_addmm_1dbias.default
                else:
                    node.target = zt_ops.zendnn_addmm.default
            else:
                node.target = zt_ops.zendnn_baddbmm.default
        # check the pattern for relu/gelu
        if node.target in op_eltwise_pattern:
            eltwise_op_dict = op_eltwise_pattern[node.target]
            if node.next.target in eltwise_op_dict:
                node.kwargs = eltwise_op_dict[node.next.target](node)
                node.next.replace_all_uses_with(node)
                fx_graph.graph.erase_node(node.next)

    logger.info("Recompiling the fx_graph with fusion changes made.")
    fx_graph.graph.set_codegen(torch.fx.graph.CodeGen())
    fx_graph.recompile()
    return fx_graph