# ******************************************************************************
# Copyright (c) 2023 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import torch # noqa
from torch._dynamo import register_backend
from torch._inductor.compile_fx import compile_fx, compile_fx_inner
from ._optimize import optimize
from typing import Callable, List, Optional
from ._logging import get_logger

logger = get_logger(__name__)


def zentorch_compile_fx_inner(gm: torch.fx.GraphModule,
                              example_inputs: List[torch.Tensor],
                              cudagraphs=None,
                              num_fixed: int = 0,
                              is_backward: bool = False,
                              graph_id: Optional[int] = None,
                              cpp_wrapper: bool = False,
                              aot_mode: bool = False,
                              is_inference: bool = False,
                              boxed_forward_device_index=None,
                              user_visible_outputs=frozenset(),
                              layout_opt: Optional[bool] = None):
    logger.info("Optimizing the model with zentorch ops.")
    # ZenDNN Optimized Implemention starts here###
    zen_gm = optimize(gm)
    # ZenDNN Optimized Implemention ends here###

    logger.info("Model is passed to compile_fx_inner.")
    return compile_fx_inner(zen_gm, example_inputs,
                            cudagraphs=cudagraphs,
                            num_fixed=num_fixed,
                            is_backward=is_backward,
                            graph_id=graph_id
                            )


def zentorch_compile(
    model: torch.fx.GraphModule,
    example_inputs: List[torch.Tensor],
) -> Callable:
    logger.info("Called the zentorch backend.")

    return compile_fx(
        model,
        example_inputs,
        inner_compile=zentorch_compile_fx_inner,
    )


@register_backend
def zentorch(model, inputs):
    return zentorch_compile(model, inputs)
