# ***************************************************************************
# Modifications Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
#
# Was sourced from:
# https://github.com/pytorch/pytorch/blob/v2.4.1/torch/_inductor/freezing.py
# ***************************************************************************

import torch
from torch._inductor.freezing import (
    replace_params_with_constants,
    invalidate_eager_modules,
    discard_traced_gm_params,
)
from torch._inductor.compile_fx import fake_tensor_prop
from torch._inductor.pattern_matcher import stable_topological_sort
from torch._inductor.fx_passes.post_grad import view_to_reshape
from torch._functorch.compile_utils import fx_graph_cse
from typing import List, Tuple
from ._logging import get_logger
from ._optimize import optimize
from ._utils import is_version_compatible_import

if is_version_compatible_import(["_inductor", "constant_folding"], ["constant_fold"]):
    from torch._inductor.constant_folding import constant_fold
else:
    from torch._inductor.freezing import constant_fold  # for PT 2.1.x
if is_version_compatible_import(["fx", "_utils"], ["lazy_format_graph_code"]):
    from torch.fx._utils import lazy_format_graph_code
else:
    from torch._dynamo.utils import lazy_format_graph_code  # for PT 2.1.x, 2.2.x, 2.3.x

logger = get_logger(__name__)


# freeze to monkey-patch in PyTorch
# constant propogation is unsupported in forward_compiler_base
# pass, we have to define a custom freeze function and use that
# in forward_compiler_freezing to avoid unnecessary/inapplicable
# optimizations in native (like mkldnn); selectively porting
# constant_fold logic also results in multiple downstream errors.
def freeze(
    dynamo_gm: torch.fx.GraphModule,
    aot_autograd_gm: torch.fx.GraphModule,
    example_inputs: List[torch._subclasses.FakeTensor],
) -> Tuple[torch.fx.GraphModule, List[int]]:
    logger.info("Optimizing the model with zentorch ops.")
    zen_gm = optimize(aot_autograd_gm)
    # we do view to reshape to avoid lowering exception
    view_to_reshape(zen_gm)

    if tracing_context := torch._guards.TracingContext.try_get():
        fw_metadata = tracing_context.fw_metadata
        params_flat = tracing_context.params_flat
        assert fw_metadata is not None and params_flat is not None

        preserved_arg_indices = replace_params_with_constants(
            zen_gm, params_flat, fw_metadata
        )
    else:
        inputs = zen_gm.graph.find_nodes(op="placeholder")
        preserved_arg_indices = list(range(len(inputs)))

    # we eliminate commom subexpressions from the graph (CSE)
    logger.info("Running common subexpression elimination on the fx-graph.")
    cse_graph = fx_graph_cse(zen_gm.graph)
    zen_gm.graph = cse_graph
    zen_gm.recompile()

    aot_example_inputs = [example_inputs[ind] for ind in preserved_arg_indices]
    fake_tensor_prop(zen_gm, aot_example_inputs, force_allow_non_fake_inputs=True)

    logger.info("Constant folding the fx-graph.")
    constant_fold(zen_gm)
    fake_tensor_prop(zen_gm, aot_example_inputs, force_allow_non_fake_inputs=True)
    stable_topological_sort(zen_gm.graph)
    zen_gm.recompile()
    zen_gm.graph.lint()

    # invalidate nn Modules
    if torch._inductor.config.freezing_discard_parameters:
        invalidate_eager_modules()
        discard_traced_gm_params(dynamo_gm)

    logger.debug("%s", lazy_format_graph_code("FROZEN GRAPH", zen_gm))

    return zen_gm, preserved_arg_indices
