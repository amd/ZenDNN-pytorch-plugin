# ******************************************************************************
# Copyright (c) 2023 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import torch
import torch_zendnn_plugin  # noqa

# import the custom logging module
from ._logging import get_logger

# make a logger for this file
logger = get_logger(__name__)


def optimize(fx_graph):
    """
    optimize:
    takes in the fx_graph and replaces some of the native ops
    with zendnn implementation of respective ops
    """
    logger.info("Optimizing the fx_graph with zentorch ops.")
    # Loop through the nodes in fx_graph.graph
    for node in fx_graph.graph.nodes:
        op_dict = {
            "aten::_embedding_bag": [
                torch.ops.zentorch.zendnn_embedding_bag,
                "zendnn_embedding_bag",
            ],
            "aten::mm": [torch.ops.zentorch.zendnn_mm, "zendnn_mm"],
            "aten::bmm": [torch.ops.zentorch.zendnn_bmm, "zendnn_bmm"],
            "aten::addmm": [torch.ops.zentorch.zendnn_addmm, "zendnn_addmm"],
            "aten::baddbmm": [torch.ops.zentorch.zendnn_baddbmm, "zendnn_baddbmm"],
        }

        # Checking for op default implementation to be replaced.
        if (
            isinstance(node.target, torch._ops.OpOverload)
            and node.target.name() in op_dict.keys()
        ):
            op_name = node.target.name()
            logger.info(
                "Now replacing default "
                + op_name
                + " with "
                + op_dict[op_name][1]
                + "!"
            )
            node.target = op_dict[op_name][0]

    logger.info("Recompiling the fx_graph with op replacement changes made.")
    # Recompile the fx_graph with changes made
    fx_graph.graph.set_codegen(torch.fx.graph.CodeGen())
    fx_graph.recompile()

    return op_fusion(fx_graph)


def op_fusion(fx_graph):
    """
    op_fusion:
    takes in the fx_graph and fuses some of the native ops
    with zendnn implementation of respective op fusions
    """
    logger.info("Fusing the zentorch ops in fx_graph.")
    # Loop through the nodes in fx_graph.graph
    for node in fx_graph.graph.nodes:
        if (
            node.target == torch.ops.zentorch.zendnn_mm
            or node.target == torch.ops.zentorch.zendnn_addmm
        ):
            if len(node.users) > 1:  # Output of node is used by other nodes
                continue
            # mm->relu or addmm->relu fusion
            if (
                isinstance(node.next.target, torch._ops.OpOverload)
                and node.next.target.name() == "aten::relu"
            ):
                logger.info("Fusing the mm->relu or addmm->relu in fx_graph.")
                new_kwargs = {**node.kwargs, "fuse_relu": True}
                node.kwargs = new_kwargs
                logger.info(
                    "Replacing the next node[relu] with current " "node from the graph."
                )
                node.next.replace_all_uses_with(node)
                logger.info("Removing the next node[relu] from the graph.")
                fx_graph.graph.erase_node(node.next)
                logger.info("Fused the mm->relu or addmm->relu in fx_graph.")
        if (
            node.target == torch.ops.zentorch.zendnn_mm
            or node.target == torch.ops.zentorch.zendnn_bmm
        ):
            if len(node.users) > 1:  # Output of node is used by other nodes
                continue
            # mm->add or bmm->add fusion
            if (
                isinstance(node.next.target, torch._ops.OpOverload)
                and node.next.target.name() == "aten::add.Tensor"
            ):
                logger.info("Fusing the mm->add or bmm->add in fx_graph.")
                for add_tensor in node.next.args:
                    if add_tensor != node:
                        new_args = (add_tensor, *node.args)
                node.args = new_args
                logger.info(
                    "Replacing the next node[add] with current " "node from the graph."
                )
                node.next.replace_all_uses_with(node)
                logger.info("Removing the next node[add] from the graph.")
                fx_graph.graph.erase_node(node.next)
                if node.target == torch.ops.zentorch.zendnn_mm:
                    logger.info("Fused the mm->add to addmm in fx_graph.")
                    node.target = torch.ops.zentorch.zendnn_addmm
                    # [mm->add]->relu fusion
                    if len(node.users) > 1:  # Output of node is used by other nodes
                        continue
                    elif (
                        isinstance(node.next.target, torch._ops.OpOverload)
                        and node.next.target.name() == "aten::relu"
                    ):
                        logger.info("Fusing the [mm->add]->relu fusion in fx_graph.")
                        new_kwargs = {**node.kwargs, "fuse_relu": True}
                        node.kwargs = new_kwargs
                        logger.info(
                            "Replacing the next node[relu] with current "
                            "node from the graph."
                        )
                        node.next.replace_all_uses_with(node)
                        logger.info("Removing the next node[relu] from the graph.")
                        fx_graph.graph.erase_node(node.next)
                        logger.info("Fused the [mm->add]->relu in fx_graph.")
                else:
                    logger.info("Fused the bmm->add to baddbmm in fx_graph.")
                    node.target = torch.ops.zentorch.zendnn_baddbmm

    logger.info("Recompiling the fx_graph with fusion changes made.")
    fx_graph.graph.set_codegen(torch.fx.graph.CodeGen())
    fx_graph.recompile()

    return fx_graph
