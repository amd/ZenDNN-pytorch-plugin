# ******************************************************************************
# Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import torch
import inspect

# pattern matcher relevant imports
from torch._inductor.pattern_matcher import (
    PatternMatcherPass,
    stable_topological_sort,
    init_once_fakemode,
)
from torch._inductor import config
from ._logging import get_logger

logger = get_logger(__name__)

matcher_pass = PatternMatcherPass()


# for fake tensors
@init_once_fakemode
def lazy_init():
    from ._fusion_patterns import _replace_init

    _replace_init()


# applies the registered patterns to fx_graph
def fusions_graph_pass(gm: torch.fx.GraphModule):

    # In PyTorch versions > 2.1, the arguments in `kwargs` are treated as optional.
    # This allows fusion to work whether `kwargs` are provided or not.
    # However, in PyTorch versions <= 2.1, all arguments are considered part of the
    # function signature.
    # As a result, if some arguments are not explicitly defined, it raises an error,
    # complaining that the signature expects x + 2 arguments, but only x are provided in
    # pattern matcher.
    # For this reason we are disabling pattern matcher for 2.1

    sig = inspect.signature(torch._inductor.pattern_matcher.register_replacement)

    if "search_fn_pattern" not in sig.parameters:
        return gm

    lazy_init()
    count = 0
    if config.pattern_matcher:
        count += matcher_pass.apply(gm.graph)
    else:
        logger.info(
            "Inductor config for pattern matching is set to False, "
            "no matcher passes will be run."
        )
    if count:
        stable_topological_sort(gm.graph)
        gm.graph.lint()
        gm.recompile()
    return gm
