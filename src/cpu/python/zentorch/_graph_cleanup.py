# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import torch
import operator

# import the custom logging module
from ._logging import get_logger

# make a logger for this file
logger = get_logger(__name__)

at_ops = torch.ops.aten


def unused_node_elimination(fx_graph: torch.fx.GraphModule):

    # TODO
    # This function will always be under progress as this will undergo
    # continuous enhancement and changes as generalization of removal of unused
    # nodes and specialization for removal of special unused nodes will keep
    # changing based on the scenarios and the models we encounter.

    """
    unused_node_elimination:
    removes the nodes with no users from the fx_graph
    """
    logger.info("Removing unused nodes from the fx_graph.")

    # Why the following nodes ?
    # operator.getitem
    #   We want to remove the getitem nodes from the graph which are the outputs
    #   of the aten embedding bag if they don't have any users. Based on the
    #   historical evidence, for inference, only the first output of the aten
    #   embedding bag is used and the other getitem nodes are not used. So, we
    #   can safely remove them from the graph.
    # at_ops.clone.default
    #   Clone nodes are used to duplicate a tensor with its properties and
    #   contents to be used by some other node in the graph or model. If its
    #   output is not used by any other node in the graph, then it is unused
    #   and can be removed.
    # at_ops.view.default
    #   View nodes are used to reshape the tensor. If its output is not used by
    #   any other node in the graph, then it is an unused node and can be
    #   removed.
    # at_ops.detach.default
    #   Detach node can impact the number of users. If the output of the detach
    #   node is not used by any other node in the graph, then it is an unused
    #   node and can be removed.

    supported_nodes_for_removal = {
        operator.getitem,
        at_ops.clone.default,
        at_ops.view.default,
        at_ops.detach.default,
    }

    for node in fx_graph.graph.nodes:
        if node.target in supported_nodes_for_removal and len(node.users) == 0:
            fx_graph.graph.erase_node(node)
        # check for replacing redundant clone ops
        elif node.target == at_ops.clone.default and len(node.kwargs) == 0 and len(node.users) == 1:
            node.replace_all_uses_with(node.args[0])
            fx_graph.graph.erase_node(node)

    return fx_graph
