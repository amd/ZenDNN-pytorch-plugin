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
from ._logging import get_logger
from ._optimize import optimize
from ._utils import is_version_compatible_import
from torch._inductor.constant_folding import constant_fold
from torch.fx._utils import lazy_format_graph_code
if is_version_compatible_import(["_inductor", "freezing_utils"], ["record_has_frozen_params"]):
    # record_has_frozen_params() is only present in PT 2.7 or above
    from torch._inductor.freezing_utils import record_has_frozen_params
else:
    # TODO: remove this else block when dropping support for PT 2.6
    def record_has_frozen_params(gm):
        # adding a new attribute is supported
        gm._has_frozen_params = True

logger = get_logger(__name__)


# freeze to monkey-patch in PyTorch
# constant propogation is unsupported in forward_compiler_base
# pass, we have to define a custom freeze function and use that
# in forward_compiler_freezing to avoid unnecessary/inapplicable
# optimizations in native (like mkldnn); selectively porting
# constant_fold logic also results in multiple downstream errors.
def _freeze(
    dynamo_gm: torch.fx.GraphModule,
    aot_autograd_gm: torch.fx.GraphModule,
    example_inputs: list[torch._subclasses.FakeTensor],
) -> tuple[torch.fx.GraphModule, list[int]]:
    logger.info("Optimizing the model with zentorch ops.")
    zen_gm = optimize(aot_autograd_gm)
    # we do view to reshape to avoid lowering exception
    view_to_reshape(zen_gm)

    if tracing_context := torch._guards.TracingContext.try_get():
        fw_metadata = tracing_context.fw_metadata
        if hasattr(tracing_context, "params_flat_unwrap_subclasses"):
            # added in PT 2.6.0
            assert tracing_context.params_flat_unwrap_subclasses is not None
            params_flat = tracing_context.params_flat_unwrap_subclasses
        else:
            # for PT 2.5.x and below
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

    record_has_frozen_params(zen_gm)
    return zen_gm, preserved_arg_indices
