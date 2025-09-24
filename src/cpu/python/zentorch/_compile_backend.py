# ******************************************************************************
# Copyright (c) 2023-2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import torch  # noqa
import inspect
from torch._dynamo import register_backend
from ._utils import is_version_compatible_import
from torch._inductor.compile_fx import compile_fx, compile_fx_inner
from ._optimize import optimize
from typing import Callable, List, Optional
from ._logging import get_logger
from torch._functorch.aot_autograd import aot_module_simplified
from torch.torch_version import TorchVersion
from ._mkldnn import mkldnn_fuse_fx
from torch._inductor.fx_passes.pre_grad import fuse_conv_bn, remove_identity

if is_version_compatible_import(["_dynamo", "utils"], ["is_parameter_freezing"]):
    from torch._dynamo.utils import is_parameter_freezing
else:

    def is_parameter_freezing():  # for PT 2.4.x or below
        return torch._inductor.config.freezing and not torch.is_grad_enabled()


# Make use of existing decompositions functions if Torch version >= 2.1
# Torch version less than 2.1 doesn't support removal of decompositions

from torch._decomp import remove_decompositions
from torch._inductor.decomposition import decompositions, fast_random_decomps
from torch._inductor.lowering import make_fallback

torch_version = TorchVersion(torch.__version__)
"""
Pytorch 2.0 has mkldnn_fuse_fx but 2.1 and above Pytorch deprecated \
mkldnn_fuse_fx function. In Pytorch 2.1 and above this fusion is done \
in Post Grad pass which is why we are using existing mkldnn_fuse_fx \
function instead of generic function mkldnn_fuse_fx for 2.0.
We are making use of existing pytorch functions fuse_conv_bn, remove_identity \
(2.1 and above) has integrated these changes at different places
"""


# Make use of existing decompositions functions if Torch version >= 2.1
# Torch version less than 2.1 doesn't support removal of decompositions


REMOVE_DECOMP = True

disable_inductor_flag = False

logger = get_logger(__name__)


class ConvConfig:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConvConfig, cls).__new__(cls)
            cls._instance.enable_zentorch_conv_flag = False
        return cls._instance

    def enable_zentorch_conv(self, enabled: bool):
        self.enable_zentorch_conv_flag = enabled


conv_config = ConvConfig()


# this is only invoked for non-freezing path
def zentorch_compile_fx_inner(
    gm: torch.fx.GraphModule,
    example_inputs: List[torch.Tensor],
    cudagraphs=None,
    static_input_idxs: Optional[List[int]] = None,
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
    # zentorch optimization starts here
    zen_gm = optimize(gm)
    # zentorch optimization ends here
    logger.info("Model is passed to compile_fx_inner.")
    # From PT2.4, compile_fx_inner introduced the optional static_input_idxs
    # argument and deprecated the optional num_fixed argument.
    # inspect can not be used directly on compile_fx_inner as the decorator
    # with_fresh_cache_if_config does not preserve the original metadata
    # But for previous versions, inspect furnishes the right signature
    sig = inspect.signature(torch._inductor.compile_fx.compile_fx_inner)
    if "num_fixed" in sig.parameters:
        return compile_fx_inner(
            zen_gm,
            example_inputs,
            cudagraphs=cudagraphs,
            num_fixed=num_fixed,
            is_backward=is_backward,
            graph_id=graph_id,
        )
    else:
        return compile_fx_inner(
            zen_gm,
            example_inputs,
            cudagraphs=cudagraphs,
            static_input_idxs=static_input_idxs,
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

    dynamic = torch._dynamo.config.automatic_dynamic_shapes

    if not torch.is_grad_enabled():
        gm = remove_identity(gm)
        gm = fuse_conv_bn(gm)
        if not conv_config.enable_zentorch_conv_flag and not dynamic:
            gm = mkldnn_fuse_fx(gm, example_inputs)

    # we cannot use options to pass 'freezing' to inductor config here
    # as the dynamo utilizes the freezing flag when it is set by the env
    # variable available in pytorch. As a result of that, whenever the control
    # comes to zentorch compile backend, the dynamo freezing optimizations
    # would not have been performed, hence the first subgraph in case of any
    # graph breaks in the model will always be un-frozen and if there are no
    # graph breaks then the entire model will be unfrozen!
    if is_parameter_freezing():
        # freeze function should already be overwritten by this point
        return compile_fx(
            gm,
            example_inputs,
        )
    return compile_fx(
        gm,
        example_inputs,
        inner_compile=zentorch_compile_fx_inner,
    )


# Compile_fx is heavily intertwined with Inductor namespace.
# This can be seen with the logs generated by TORCH_COMPILE_DEBUG=1.
# Hence, it cannot be used for adding zentorch optimisations.
# Instead of writing a independent compile_fx for this purpose,
# we are using aot_module_simplfied instead.
# Compared to using compilefx from inductor (w/o compile_fx_inner),
# aot_module_simplfified provides better performance as well.


def zentorch_compiler_noinductor(gm, sample_inputs):
    def zentorch_compiler_noinductor_impl(gm, sample_inputs):

        # Applying zentorch optimizations here.
        fused_model = optimize(gm)
        return fused_model

    # Invoke AOTAutograd
    return aot_module_simplified(
        gm, sample_inputs, fw_compiler=zentorch_compiler_noinductor_impl
    )


@register_backend
def zentorch(model, inputs):

    if REMOVE_DECOMP:
        REMOVE_DECOMP_LIST = [
            torch.ops.aten.gelu_,
            torch.ops.aten.gelu,
            torch.ops.aten.silu_,
            torch.ops.aten.silu,
            torch.ops.aten.native_layer_norm,
            torch.ops.aten.empty.memory_format,
        ]
        remove_decompositions(decompositions, REMOVE_DECOMP_LIST)
        # Clear the fast_random_decomps cache to avoid using old decompositions
        # This is necessary as the fast_random_decomps uses LRU cache
        fast_random_decomps.cache_clear()
        # PT will throw an error if CI env variable is set
        # This looks like a bug in PT as this check is unnecessary
        # before registering the fallback
        # This can be avoided by calling the API with the warning mode
        # set to False (torch/_inductor/lowering.py)
        for op in REMOVE_DECOMP_LIST:
            make_fallback(op, warn=False)

    if disable_inductor_flag:
        logger.info(
            "Inductor Compilation has been disabled. "
            "FX Graph is sent to aot_module_simplified"
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


def enable_zentorch_conv(enabled: bool):
    """
    Parameters:
    enabled - True will enable zentorch conv path changes. False will
    go through mkldnn path.
    """
    conv_config.enable_zentorch_conv(enabled)
