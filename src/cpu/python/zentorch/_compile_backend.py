# ******************************************************************************
# Copyright (c) 2023-2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import torch  # noqa
import base64
from torch._inductor import config
from torch._dynamo import register_backend
from torch._inductor.compile_fx import compile_fx
from torch._inductor.custom_graph_pass import CustomGraphPass, get_hash_for_files
from ._optimize import optimize
from typing import Callable, List, Any
from ._logging import get_logger
from ._mkldnn import mkldnn_fuse_fx
from torch._inductor.fx_passes.pre_grad import fuse_conv_bn, remove_identity


class OptimizePass(CustomGraphPass):
    def __call__(self, graph: torch.fx.Graph):
        optimize(graph)

    def __bytes_to_str(self, byte_str: Any) -> Any:
        if not isinstance(byte_str, bytes):
            return byte_str
        byte_str_b = base64.b64encode(byte_str)
        byte_str_s = byte_str_b.decode('utf-8')
        return byte_str_s

    def uuid(self):
        # needed for inductor caching
        uuid_val = get_hash_for_files((__file__,))
        uuid_val_str = self.__bytes_to_str(uuid_val)
        return uuid_val_str

    def __repr__(self):
        try:
            uuid_val = self.uuid()
            uuid_val_str = self.__bytes_to_str(uuid_val)
            return f"OptimizePass(uuid={uuid_val_str!r})"
        except Exception:
            return "OptimizePass(uuid=<error>)"


optimize_pass = OptimizePass()


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

    def __repr__(self):
        return f"ConvConfig(enable_zentorch_conv_flag={self.enable_zentorch_conv_flag})"


conv_config = ConvConfig()


def zentorch_compile(
    gm: torch.fx.GraphModule,
    example_inputs: List[torch.Tensor],
) -> Callable:
    logger.info("Called the zentorch backend.")

    config.joint_custom_post_pass = optimize_pass

    dynamic = torch._dynamo.config.automatic_dynamic_shapes

    if not torch.is_grad_enabled():
        gm = remove_identity(gm)
        gm = fuse_conv_bn(gm)
        if not conv_config.enable_zentorch_conv_flag and not dynamic:
            gm = mkldnn_fuse_fx(gm, example_inputs)

    # All zentorch graph transformations are installed as Inductor's
    # joint_custom_post_pass (set above), so we just hand the FX graph to
    # the upstream Inductor compile_fx pipeline. The freezing path is handled
    # entirely inside compile_fx based on torch._inductor.config.freezing.
    return compile_fx(gm, example_inputs)


@register_backend
def zentorch(model, inputs):
    return zentorch_compile(model, inputs)


def enable_zentorch_conv(enabled: bool):
    """
    Parameters:
    enabled - True will enable zentorch conv path changes. False will
    go through mkldnn path.
    """
    conv_config.enable_zentorch_conv(enabled)
