# ******************************************************************************
# Copyright (c) 2023-2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import zentorch._C  # noqa
import os
import torch
from torch._inductor import config
from torch._inductor.fx_utils import FakeTensorUpdater

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
    qkv_fusion,
    qlinear_reorder_optimizations,
)
from ._prepack_pass import add_zentorch_weight_prepack_ops
from ._eltwise_unary_fusions import zentorch_eltwise_unary_fusions
from ._eltwise_binary_fusions import zentorch_eltwise_binary_fusions
from ._graph_preprocess_matcher import preprocess_graph_pass
from ._fusion_matcher import fusions_graph_pass
from ._qlinear_fusion import qlinear_fusion_pass
from ._qop_replacement import replace_with_zentorch_qops
from ._unary_fusions import zentorch_unary_post_op_fusions
from ._unary_binary_fusions import zentorch_unary_binary_post_op_fusions
from ._binary_binary_fusions import zentorch_binary_binary_post_op_fusions
from ._utils import _is_used_by_zentorch_qlinear

# make a logger for this file
logger = get_logger(__name__)


def constant_fold_full_ops(graph):
    """Replace aten.full nodes with constant scalar args with actual constant tensors."""
    from torch.utils._mode_utils import no_dispatch

    for node in list(graph.nodes):
        if node.op == "call_function" and node.target == torch.ops.aten.full.default:
            size_arg = node.args[0]
            fill_value = node.args[1]
            kwargs = node.kwargs
            # Only fold if arguments are actual constants (not FX nodes)
            if (
                isinstance(size_arg, (list, tuple))
                and all(isinstance(s, int) for s in size_arg)
                and isinstance(fill_value, (int, float))
                and _is_used_by_zentorch_qlinear(node)
            ):
                dtype = kwargs.get("dtype", torch.float32)
                # Create tensor OUTSIDE of FakeTensorMode so it has real data
                with no_dispatch():
                    const_tensor = torch.full(size_arg, fill_value, dtype=dtype)
                # Register as a graph attribute and replace node
                attr_name = f"_folded_constant_{node.name}"
                gm = graph.owning_module
                gm.register_buffer(attr_name, const_tensor)
                with graph.inserting_before(node):
                    new_node = graph.get_attr(attr_name)
                    new_node.meta.update(node.meta)
                    node.replace_all_uses_with(new_node)
                    graph.erase_node(node)
    graph.lint()
    return graph


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

    # for linear replacement (and other new ones to be added, to be moved up or extended)
    pattern_matched_model = replace_with_zentorch_ops_new(pattern_matched_model)

    # Replacing ops with zentorch ops (to be moved down or replaced)
    optimized_graph = replace_with_zentorch_ops(pattern_matched_model)

    fake_tensor_updater = FakeTensorUpdater(optimized_graph)

    optimized_graph = replace_with_composite_zentorch_ops(optimized_graph)

    # Quantization pattern replacement
    optimized_graph = replace_with_zentorch_qops(optimized_graph)

    # Qlinear fusion pass
    optimized_graph = qlinear_fusion_pass(optimized_graph)

    if config.freezing:
        # qkv_fusion pass with zentorch linear ops
        optimized_graph = qkv_fusion(optimized_graph)
        # update fake tensor metadata
        fake_tensor_updater.incremental_update()

    if (
        config.freezing
        and os.environ.get("ZENTORCH_WEIGHT_PREPACK", "1") == "1"
        and os.environ.get("ZENDNNL_MATMUL_ALGO", "1") == "1"
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

    # Reorder optimization for serialized qlinear_* ops.
    optimized_graph = qlinear_reorder_optimizations(optimized_graph)

    optimized_graph = inplace_cat_fusion(optimized_graph)

    # Fusion of parallel embeddingbags
    optimized_graph = emb_ops_horizontal_fusion(optimized_graph)
    # constant folding the aten.full ops will avoid fusing them with zentorch_qlinear ops
    if config.freezing:
        optimized_graph = constant_fold_full_ops(optimized_graph)

    return optimized_graph
