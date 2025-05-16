# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import torch
from ._utils import (
    is_bias_1d_tensor,
    numdims_tensor,
    are_args_same_dtype,
    get_tensor,
)
from ._logging import get_logger

logger = get_logger(__name__)

at_ops = torch.ops.aten
zt_ops = torch.ops.zentorch

# for now add is not added as post op that's why we created this pattern
add_pattern = (zt_ops.zentorch_bmm.default, zt_ops.zentorch_mm.default)


def zentorch_eltwise_binary_fusions(fx_graph):
    logger.info("Fusing the zentorch binary elementwise ops in fx graph.")
    # Loop through the nodes in fx_graph.graph
    for node in fx_graph.graph.nodes:
        if len(node.users) > 1:  # Output of node is used by other nodes
            continue
        # check the pattern for mm->add or bmm->add
        # TODO Support add fusion when add is farther from matmul node
        # TODO Unit Tests to be added for the farther case
        # TODO Add a negative Unit test for the farther case
        # TODO Support mm add fusion when add is scalar or its dim is 0
        if node.target in add_pattern:
            node_next = next(iter(node.users))
            if node_next.target == at_ops.add.Tensor and node_next == node.next:
                logger.info(
                    "Fusing the %s->%s" " in fx graph.", node.target, node_next.target
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
                        if add_tensor != node and (
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
                                    node.target = zt_ops.zentorch_addmm_1dbias.default
                                else:
                                    node.target = zt_ops.zentorch_addmm.default
                            else:
                                node.target = zt_ops.zentorch_baddbmm.default
                else:
                    logger.warning(
                        "baddbmm in zentorch doesnt support "
                        "non 3 dimentional tensors as of now"
                    )
    logger.info("Recompiling the fx_graph with fusion changes made.")

    fx_graph.graph.lint()
    fx_graph.recompile()
    return fx_graph
