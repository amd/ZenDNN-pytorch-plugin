# ******************************************************************************
# Copyright (c) 2023-2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import zentorch._C  # noqa

# import the custom logging module
from ._logging import get_logger
from ._utils import save_graph

# import the graph cleanup module
from ._graph_cleanup import unused_node_elimination

from ._op_replacement import (
    replace_with_zentorch_ops,
    replace_with_composite_zentorch_ops,
)

from ._custom_op_replacement import (
    inplace_cat_fusion,
    emb_ops_horizontal_fusion,
    group_eb_concat_fusion,
    qlinear_reorder_optimizations,
    eb_group_mlp_group_fusion,
    qkv_fusion,
)
from ._eltwise_unary_fusions import zentorch_eltwise_unary_fusions
from ._eltwise_binary_fusions import zentorch_eltwise_binary_fusions
from ._graph_preprocess_matcher import preprocess_graph_pass
from ._fusion_matcher import fusions_graph_pass
from ._qop_replacement import replace_with_zentorch_qops

# make a logger for this file
logger = get_logger(__name__)


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

    # Preprocess the graph to remove the unused nodes
    cleaned_graph = unused_node_elimination(fx_graph)

    # pattern-matcher pass for aten to aten replacement
    # for now we have just composite ops replacement (in older models)
    # for example, some models use decomposed gelu instead of the op directly.
    pattern_matched_model = preprocess_graph_pass(cleaned_graph)

    # Replacing ops with zentorch ops
    optimized_graph = replace_with_zentorch_ops(pattern_matched_model)

    optimized_graph = replace_with_composite_zentorch_ops(optimized_graph)

    # Quantization pattern replacement
    optimized_graph = replace_with_zentorch_qops(optimized_graph)

    # eltwise op fusions supported by zentorch
    optimized_graph = zentorch_eltwise_binary_fusions(optimized_graph)

    # unary fusions happen after binary
    optimized_graph = zentorch_eltwise_unary_fusions(optimized_graph)

    # eltwise fusion replacements
    optimized_graph = fusions_graph_pass(optimized_graph)

    # ZenTorch qlinear reorder optimizations.
    optimized_graph = qlinear_reorder_optimizations(optimized_graph)

    optimized_graph = inplace_cat_fusion(optimized_graph)

    # Fusion of parallel embeddingbags
    optimized_graph = emb_ops_horizontal_fusion(optimized_graph)

    # Fusion of embeddingbag outputs and cat node
    optimized_graph = group_eb_concat_fusion(optimized_graph)

    # Vertical fusion of Consecutive MLP layers
    # TODO: Add support for Vertical Fusion
    # TODO: Fix inconsistent results with unit tests
    # optimized_graph = vertical_mlp_fusion(optimized_graph)

    # Fusion of parallel QKV layers
    optimized_graph = qkv_fusion(optimized_graph)

    # Horizontal fusion of parallel Group EB op and Group MLP op
    optimized_graph = eb_group_mlp_group_fusion(optimized_graph)

    # Dumping of the optimized graph in svg format
    save_graph(optimized_graph, "zen_optimized_model")

    return optimized_graph
