# ******************************************************************************
# Copyright (c) 2023-2024 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import torch  # noqa
from torch._dynamo import register_backend
from torch._inductor.compile_fx import compile_fx, compile_fx_inner
from ._optimize import optimize
from typing import Callable, List, Optional
from ._logging import get_logger
from torch._functorch.aot_autograd import aot_module_simplified
'''
Pytorch 2.0 has mkldnn_fuse_fx but 2.1 and above Pytorch deprecated \
mkldnn_fuse_fx function. So we are using try catch here. We are \
making use of Pytorch 2.0 mkldnn_fuse_fx which has additional \
conv+add+relu fusion but in Pytorch 2.1 and above this fusion is done \
in Post Grad pass which is why we are using existing mkldnn_fuse_fx \
function instead of generic function mkldnn_fuse_fx for 2.0
'''
try:
    from torch._inductor.mkldnn import mkldnn_fuse_fx
except ImportError:
    from ._mkldnn import mkldnn_fuse_fx
'''
We are making use of existing pytorch functions fuse_conv_bn, remove_identity \
Pytorch 2.0 and (2.1 and above) has integrated these changes at different places
'''
try:
    from torch._inductor.fx_passes.pre_grad import fuse_conv_bn, remove_identity
except ImportError:
    from torch._inductor.overrides import fuse_conv_bn, remove_identity


try:
    from torch._decomp import remove_decompositions
    from torch._inductor.decomposition import decompositions
    REMOVE_DECOMP = True
except ImportError:
    REMOVE_DECOMP = False
disable_inductor_flag = False

logger = get_logger(__name__)


def zentorch_compile_fx_inner(
    gm: torch.fx.GraphModule,
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
    layout_opt: Optional[bool] = None,
):
    logger.info("Optimizing the model with zentorch ops.")
    # ZenDNN Optimized Implemention starts here###
    zen_gm = optimize(gm)
    # ZenDNN Optimized Implemention ends here###
    logger.info("Model is passed to compile_fx_inner.")
    return compile_fx_inner(
        zen_gm,
        example_inputs,
        cudagraphs=cudagraphs,
        num_fixed=num_fixed,
        is_backward=is_backward,
        graph_id=graph_id,
    )


def zentorch_compile(
    gm: torch.fx.GraphModule,
    example_inputs: List[torch.Tensor],
) -> Callable:
    logger.info("Called the zentorch backend.")

    # Pytorch 2.0 has dynamic_shapes to control dynamic shapes from torch.compile
    # but it got deprecated in Pytorch2.1 and above. Pytorch 2.1 introduced
    # automatic_dynamic_shapes which will do the same task as dynamic_shapes

    try:
        dynamic = torch._dynamo.config.automatic_dynamic_shapes
    except AttributeError:
        dynamic = torch._dynamo.config.dynamic_shapes

    if not torch.is_grad_enabled():
        gm = remove_identity(gm)
        gm = fuse_conv_bn(gm)
        if not dynamic:
            gm = mkldnn_fuse_fx(gm, example_inputs)

    return compile_fx(
        gm,
        example_inputs,
        inner_compile=zentorch_compile_fx_inner,
    )


# Compile_fx is heavily intertwined with Inductor namespace.
# This can be seen with the logs generated by TORCH_COMPILE_DEBUG=1.
# Hence, it cannot be used for adding zendnn optimisations.
# Instead of writing a independent compile_fx for this purpose,
# we are using aot_module_simplfied instead.
# Compared to using compilefx from inductor (w/o compile_fx_inner),
# aot_module_simplfified provides better performance as well.


def zentorch_compiler_noinductor(gm, sample_inputs):

    def zentorch_compiler_noinductor_impl(gm, sample_inputs):

        # Applying ZenDNN's optimizations here
        fused_model = optimize(gm)
        return fused_model

    # Invoke AOTAutograd
    return aot_module_simplified(
        gm, sample_inputs, fw_compiler=zentorch_compiler_noinductor_impl
    )


@register_backend
def zentorch(model, inputs):
    if REMOVE_DECOMP:
        remove_decompositions(
            decompositions,
            [torch.ops.aten.gelu_, torch.ops.aten.gelu],
        )
    if disable_inductor_flag:
        logger.info(
            "Inductor Compilation has been disabled."
            + "FX Graph is sent to aot_module_simplified"
        )
        return zentorch_compiler_noinductor(model, inputs)
    else:
        return zentorch_compile(model, inputs)


# This API takes in boolean input to disable or enable inductor backend.
# It is intended for use whenever fx_graphs generated from torch.compile,
# needs to be compared with and without Inductor compilation.
# fx graphs are sent to AOT Autograd using aot_module_simplified.


def disable_inductor(disabled: bool):
    """
    Parameters:
    disabled - True will disable inductor. False will re-enable it.
    """

    logger.warning("TORCH_COMPILE_DEBUG=1 might crash with PT 2.0")
    global disable_inductor_flag
    disable_inductor_flag = disabled
