# ******************************************************************************
# Copyright (c) 2023 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import torch
import torch_zendnn_plugin._C


def optimize(fx_graph):
    """
    optimize:
    takes in the fx_graph and replaces the native embedding
    bag with zendnn implementation of embedding bag
    """
    # Loop through the nodes in fx_graph.graph
    for node in fx_graph.graph.nodes:
        # Check for embeddingbag op default implementation
        if (
            type(node.target) == torch._ops.OpOverload
            and node.target.name() == "aten::_embedding_bag"
        ):
            # Replace default embedding_bag with embedding_bag_zendnn
            node.target = torch_zendnn_plugin._C.embedding_bag_zendnn

    # Recompile the fx_graph with changes made
    fx_graph.graph.set_codegen(torch.fx.graph.CodeGen())
    fx_graph.recompile()

    return fx_graph
