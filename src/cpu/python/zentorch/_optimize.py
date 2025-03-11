# ******************************************************************************
# Copyright (c) 2023-2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import sys
import torch
import zentorch._C  # noqa

# import the custom logging module
from ._logging import get_logger
from ._utils import save_graph

# import the graph cleanup module
from ._graph_cleanup import unused_node_elimination

from ._op_replacement import (
    replace_with_zentorch_ops,
    replace_with_composite_zentorch_ops,
    at_to_zen_op_dict,
    zen_to_zen_op_dict,
)

from ._custom_op_replacement import (
    inplace_cat_fusion,
    emb_ops_horizontal_fusion,
    group_eb_concat_fusion,
    qlinear_reorder_optimizations,
    eb_group_mlp_group_fusion,
    qkv_fusion,
)
from ._eltwise_fusions import zentorch_eltwise_fusions
from ._graph_preprocess_matcher import preprocess_graph_pass
from ._fusion_matcher import fusions_graph_pass

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

    # Preprocess the graph to remove the unused nodes
    cleaned_graph = unused_node_elimination(fx_graph)

    # pattern-matcher pass for aten to aten replacement
    # for now we have just composite ops replacement (in older models)
    # for example, some models use decomposed gelu instead of the op directly.
    pattern_matched_model = preprocess_graph_pass(cleaned_graph)

    # Replacing ops with zentorch ops
    # first we check if ipex has been imported anywhere in the code,
    # then we append the ipex dict to op replacement
    op_dict_lst = [at_to_zen_op_dict]
    if "intel_extension_for_pytorch" in sys.modules:
        ipex_ops = torch.ops.torch_ipex
        # add ipex rope replacement, no condition as of now
        ipex_to_zen_op_dict = {
            ipex_ops.rotary_position_embedding.default: (
                zt_ops.zentorch_rope.default,
                None,
            ),
            ipex_ops.masked_multihead_self_attention.default: (
                zt_ops.zentorch_masked_multihead_self_attention.default,
                None,
            ),
        }
        # RoPE deepseek is only present in IPEX 2.6 or above
        if hasattr(ipex_ops, "rotary_position_embedding_deepseek"):
            ipex_to_zen_op_dict[ipex_ops.rotary_position_embedding_deepseek.default] = (
                zt_ops.zentorch_rope_deepseek.default,
                None,
            )
        op_dict_lst.append(ipex_to_zen_op_dict)
    op_dict_lst.append(zen_to_zen_op_dict)
    optimized_graph = replace_with_zentorch_ops(pattern_matched_model, op_dict_lst)

    optimized_graph = replace_with_composite_zentorch_ops(optimized_graph)

    # eltwise op fusions supported by zentorch
    optimized_graph = zentorch_eltwise_fusions(optimized_graph)

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
