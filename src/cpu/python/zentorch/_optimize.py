# ******************************************************************************
# Copyright (c) 2023-2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import zentorch._C  # noqa
import os
from torch._inductor import config

# import the custom logging module
from ._logging import get_logger

# import the graph cleanup module
from ._graph_cleanup import unused_node_elimination

from ._op_replacement import (
    replace_with_zentorch_ops,
    replace_with_composite_zentorch_ops,
)
from ._op_replacements_new import replace_with_zentorch_ops_new
from ._custom_op_replacement import (
    inplace_cat_fusion,
    emb_ops_horizontal_fusion,
)
from ._prepack_pass import add_zentorch_weight_prepack_ops
from ._eltwise_unary_fusions import zentorch_eltwise_unary_fusions
from ._eltwise_binary_fusions import zentorch_eltwise_binary_fusions
from ._graph_preprocess_matcher import preprocess_graph_pass
from ._fusion_matcher import fusions_graph_pass
from ._unary_fusions import zentorch_unary_post_op_fusions
from ._unary_binary_fusions import zentorch_unary_binary_post_op_fusions
from ._binary_binary_fusions import zentorch_binary_binary_post_op_fusions

# make a logger for this file
logger = get_logger(__name__)


def optimize(fx_graph):
    """
    optimize:
    takes in the fx_graph and replaces some of the native ops
    with zentorch implementation of respective ops and fusion
    few ops
    """

    logger.info("Optimizing the fx_graph with zentorch ops.")

    # Preprocess the graph to remove the unused nodes
    cleaned_graph = unused_node_elimination(fx_graph)

    # pattern-matcher pass for aten to aten replacement
    # for now we have just composite ops replacement (in older models)
    # for example, some models use decomposed gelu instead of the op directly.
    pattern_matched_model = preprocess_graph_pass(cleaned_graph)

    if os.environ.get("ZENTORCH_LINEAR", "0") == "1":
        # for linear replacement (and other new ones to be added, to be moved up or extended)
        pattern_matched_model = replace_with_zentorch_ops_new(pattern_matched_model)

    # Replacing ops with zentorch ops (to be moved down or replaced)
    optimized_graph = replace_with_zentorch_ops(pattern_matched_model)

    optimized_graph = replace_with_composite_zentorch_ops(optimized_graph)

    if (
        config.freezing
        and os.environ.get("ZENTORCH_LINEAR", "0") == "1"
    ):
        # replace zendnn ops with zendnn custom passes
        optimized_graph = add_zentorch_weight_prepack_ops(optimized_graph)

    # do all linear fusions first
    optimized_graph = zentorch_binary_binary_post_op_fusions(optimized_graph)
    optimized_graph = zentorch_unary_binary_post_op_fusions(optimized_graph)
    optimized_graph = zentorch_unary_post_op_fusions(optimized_graph)

    # eltwise op fusions supported by zentorch
    optimized_graph = zentorch_eltwise_binary_fusions(optimized_graph)

    # unary fusions happen after binary
    optimized_graph = zentorch_eltwise_unary_fusions(optimized_graph)

    # eltwise fusion replacements
    optimized_graph = fusions_graph_pass(optimized_graph)

    optimized_graph = inplace_cat_fusion(optimized_graph)

    # Fusion of parallel embeddingbags
    optimized_graph = emb_ops_horizontal_fusion(optimized_graph)

    return optimized_graph
