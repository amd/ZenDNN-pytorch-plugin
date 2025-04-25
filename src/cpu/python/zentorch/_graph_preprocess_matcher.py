# ******************************************************************************
# Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import torch

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
    from ._graph_preprocess_patterns import _replace_init

    _replace_init()


# applies the registered patterns to fx_graph
def preprocess_graph_pass(gm: torch.fx.GraphModule):
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
