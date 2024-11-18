# ******************************************************************************
# Modifications Copyright (c) 2024 Advanced Micro Devices, Inc.
# All rights reserved.
#
# Was sourced from
# https://github.com/pytorch/pytorch/blob/release/2.0/torch/_inductor/mkldnn.py
# ******************************************************************************

import copy
import itertools
import operator

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.fx.experimental.optimization import (
    matches_module_pattern,
    replace_node_module,
)
from torch.fx.experimental.symbolic_shapes import guard_int
from torch.fx.passes.shape_prop import ShapeProp
from torch.nn.modules.utils import _pair
from torch._inductor import config

from torch._inductor.fx_utils import matches_module_function_pattern
from torch.utils._pytree import tree_flatten
from typing import Any, Optional, List


class UnaryAttr:
    def __init__(self, op_name: str, scalars_attr=None, algorithm_attr=None):
        self.op_name = op_name
        self.scalars_attr = scalars_attr if scalars_attr else []
        self.algorithm_attr = algorithm_attr if algorithm_attr else ""
        super().__init__()

    def __call__(self, unary_module: nn.Module):
        if type(unary_module) is nn.ReLU6:
            unary_module = nn.Hardtanh(min_val=0, max_val=6)
        assert all(hasattr(unary_module, item) for item in self.scalars_attr)
        scalars = [getattr(unary_module, item) for item in self.scalars_attr]

        algorithm = ""
        if self.algorithm_attr:
            assert hasattr(unary_module, self.algorithm_attr)
            algorithm = getattr(unary_module, self.algorithm_attr)

        return self.op_name, scalars, algorithm


def is_bfloat16_module(m):
    weight_is_bf16 = m.weight.dtype == torch.bfloat16
    bias_is_bf16 = m.bias is None or m.bias.dtype == torch.bfloat16
    return weight_is_bf16 and bias_is_bf16


def is_group_depthwise_conv_transpose(m):
    return (
        type(m) in [nn.ConvTranspose2d] and m.groups > 1 and m.groups == m.in_channels
    )


class ConvUnary2d(nn.Conv2d):
    def __init__(
        self,
        conv: nn.Module,
        unary: Optional[nn.Module],
        input_size: list,
    ):
        super().__init__(
            conv.in_channels,
            conv.out_channels,
            conv.kernel_size,
            conv.stride,
            conv.padding,
            conv.dilation,
            conv.groups,
            conv.bias is not None,
            conv.padding_mode,
            conv.weight.device,
            conv.weight.dtype,
        )
        self._update_module_params(conv, unary, input_size)

    def _update_module_params(self, conv, unary, input_size):
        self.__dict__ = copy.deepcopy(conv.__dict__)
        self.attr = "none"
        self.scalars = []
        self.algorithm = ""
        if unary is not None:
            self.attr, self.scalars, self.algorithm = unary_modules_map[
                unary.__class__
            ](unary)
        # support amp inside mkldnn_conv
        if torch.is_autocast_cpu_enabled():
            self.weight = torch.nn.Parameter(self.weight.bfloat16())
            if self.bias is not None:
                self.bias = torch.nn.Parameter(self.bias.bfloat16())
        self.weight = torch.nn.Parameter(
            torch._C._nn.mkldnn_reorder_conv2d_weight(
                self.weight.to_mkldnn(),
                self.padding,
                self.stride,
                self.dilation,
                self.groups,
                tuple(guard_int(x) for x in input_size),
            ),
            requires_grad=self.weight.requires_grad,
        )

    def _conv_forward(self, input, weight, bias):
        if self.padding_mode != "zeros":
            return torch.ops.mkldnn._convolution_pointwise(
                F.pad(
                    input, self._reversed_padding_repeated_twice, mode=self.padding_mode
                ),
                weight,
                bias,
                _pair(0),
                self.stride,
                self.dilation,
                self.groups,
                self.attr,
                self.scalars,
                self.algorithm,
            )
        return torch.ops.mkldnn._convolution_pointwise(
            input,
            weight,
            bias,
            self.padding,
            self.stride,
            self.dilation,
            self.groups,
            self.attr,
            self.scalars,
            self.algorithm,
        )

    def forward(self, input):
        if torch.is_autocast_cpu_enabled():
            input = input.bfloat16()
        return self._conv_forward(input, self.weight, self.bias)


class ConvBinary2d(nn.Conv2d):
    def __init__(
        self,
        conv: nn.Module,
        binary_op_name: str,
        input_size: list,
    ):
        super().__init__(
            conv.in_channels,
            conv.out_channels,
            conv.kernel_size,
            conv.stride,
            conv.padding,
            conv.dilation,
            conv.groups,
            conv.bias is not None,
            conv.padding_mode,
            conv.weight.device,
            conv.weight.dtype,
        )
        self._update_module_params(conv, binary_op_name, input_size)

    def _update_module_params(self, conv, binary_op_name, input_size):
        self.__dict__ = copy.deepcopy(conv.__dict__)
        self.binary_attr = binary_op_name
        self.binary_alpha = None
        self.unary_attr = None
        self.unary_scalars = []
        self.unary_algorithm = None
        if torch.is_autocast_cpu_enabled():
            self.weight = torch.nn.Parameter(self.weight.bfloat16())
            if self.bias is not None:
                self.bias = torch.nn.Parameter(self.bias.bfloat16())
        self.weight = torch.nn.Parameter(
            torch._C._nn.mkldnn_reorder_conv2d_weight(
                self.weight.to_mkldnn(),
                self.padding,
                self.stride,
                self.dilation,
                self.groups,
                tuple(guard_int(x) for x in input_size),
            ),
            requires_grad=self.weight.requires_grad,
        )

    def _update_unary_params(self, unary):
        self.unary_attr, self.unary_scalars, self.unary_algorithm = unary_modules_map[
            unary.__class__
        ](unary)

    def _conv_forward(self, input, other, weight, bias):
        if self.padding_mode != "zeros":
            return torch.ops.mkldnn._convolution_pointwise(
                F.pad(
                    input, self._reversed_padding_repeated_twice, mode=self.padding_mode
                ),
                other,
                weight,
                bias,
                _pair(0),
                self.stride,
                self.dilation,
                self.groups,
                self.binary_attr,
                self.binary_alpha,
                self.unary_attr,
                self.unary_scalars,
                self.unary_algorithm,
            )
        return torch.ops.mkldnn._convolution_pointwise(
            input,
            other,
            weight,
            bias,
            self.padding,
            self.stride,
            self.dilation,
            self.groups,
            self.binary_attr,
            self.binary_alpha,
            self.unary_attr,
            self.unary_scalars,
            self.unary_algorithm,
        )

    def forward(self, input, other):
        if torch.is_autocast_cpu_enabled():
            input = input.bfloat16()
        return self._conv_forward(input, other, self.weight, self.bias)


class ConvTransposeUnary2d(nn.ConvTranspose2d):
    def __init__(
        self,
        conv_transpose: nn.Module,
        unary: Optional[nn.Module],
        input_size: list,
    ):
        super().__init__(
            conv_transpose.in_channels,
            conv_transpose.out_channels,
            conv_transpose.kernel_size,
            conv_transpose.stride,
            conv_transpose.padding,
            conv_transpose.output_padding,
            conv_transpose.groups,
            conv_transpose.bias is not None,
            conv_transpose.dilation,
            conv_transpose.padding_mode,
            conv_transpose.weight.device,
            conv_transpose.weight.dtype,
        )
        self._update_module_params(conv_transpose, unary, input_size)

    def _update_module_params(self, conv_transpose, unary, input_size):
        self.__dict__ = copy.deepcopy(conv_transpose.__dict__)
        self.attr, self.scalars, self.algorithm = (
            unary_modules_map[unary.__class__](unary) if unary else ("none", [], "")
        )
        if torch.is_autocast_cpu_enabled():
            self.weight = torch.nn.Parameter(self.weight.bfloat16())
            if self.bias is not None:
                self.bias = torch.nn.Parameter(self.bias.bfloat16())

        # Removed support for pytorch version < 2.1
        weight = self.weight
        packed_weight = torch.ops.mkldnn._reorder_convolution_transpose_weight(
            weight,
            self.padding,
            self.output_padding,
            self.stride,
            self.dilation,
            self.groups,
            input_size,
        )
        self.weight = torch.nn.Parameter(
            packed_weight,
            requires_grad=self.weight.requires_grad,
        )

    def _conv_transpose_forward(self, input, weight, bias):
        if self.padding_mode != "zeros":
            return torch.ops.mkldnn._convolution_transpose_pointwise(
                F.pad(
                    input, self._reversed_padding_repeated_twice, mode=self.padding_mode
                ),
                weight,
                bias,
                _pair(0),
                self.output_padding,
                self.stride,
                self.dilation,
                self.groups,
                self.attr,
                self.scalars,
                self.algorithm,
            )
        return torch.ops.mkldnn._convolution_transpose_pointwise(
            input,
            weight,
            bias,
            self.padding,
            self.output_padding,
            self.stride,
            self.dilation,
            self.groups,
            self.attr,
            self.scalars,
            self.algorithm,
        )

    def forward(self, input):
        if torch.is_autocast_cpu_enabled():
            input = input.bfloat16()
        return self._conv_transpose_forward(input, self.weight, self.bias)


def packed_conv_eval(conv: nn.Module, input_size: list):
    assert not (conv.training), "Perform Fusion only when we are in eval!"
    return ConvUnary2d(
        conv,
        None,
        input_size,
    )


def packed_conv_transpose_eval(conv_transpose: nn.Module, input_size: list):
    assert not (conv_transpose.training), "Fusion only for eval!"
    return ConvTransposeUnary2d(
        conv_transpose,
        None,
        input_size,
    )


def fused_conv_unary_eval(conv: nn.Module, unary: nn.Module, input_size: list):
    assert not (conv.training), "Fusion only for eval!"
    return ConvUnary2d(
        conv,
        unary,
        input_size,
    )


def fused_conv_binary_eval(conv: nn.Module, binary_op_name: str, input_size: list):
    assert not (conv.training), "Fusion only for eval!"
    return ConvBinary2d(
        conv,
        binary_op_name,
        input_size,
    )


def fused_conv_binary_unary_eval(
    conv_binary: nn.Module, unary: nn.Module, input_size: list
):
    assert not (conv_binary.training), "Fusion only for eval!"
    # reuse origin conv module, and just update its' unary attr.
    conv_binary._update_unary_params(unary)
    return conv_binary


def fused_conv_transpose_unary_eval(
    conv_transpose: nn.Module, unary: nn.Module, input_size: list
):
    assert not (conv_transpose.training), "Fusion only for eval!"
    return ConvTransposeUnary2d(
        conv_transpose,
        unary,
        input_size,
    )


def fake_mode_from_tensors(inputs: List[Any]):
    # Takes a list of anything, unflattened is fine, returns a fake_mode
    # if any are fake. All fake modes on all fake tensors must be identical.
    # Returns None if no fake_mode.
    flat_inputs, _ = tree_flatten(inputs)
    fake_mode = None
    for flat_input in flat_inputs:
        if isinstance(flat_input, torch._subclasses.FakeTensor):
            if fake_mode is None:
                fake_mode = flat_input.fake_mode
            else:
                assert fake_mode is flat_input.fake_mode
    return fake_mode


def replace_conv(gm):
    modules = dict(gm.named_modules())
    for node in gm.graph.nodes:
        if node.target in modules and type(modules[node.target]) in [nn.Conv2d]:
            computation_node = modules[node.target]

            if computation_node.training:
                continue
            if isinstance(
                computation_node.padding, str
            ):
                continue
            if is_group_depthwise_conv_transpose(computation_node):
                continue
            computation_node_input_size = (
                node.args[0].meta.get("tensor_meta").shape
            )
            fused_module = fused_conv_unary_eval(
                computation_node, None, computation_node_input_size
            )
            replace_node_module(node, modules, fused_module)
    gm.graph.lint()
    gm.recompile()
    return gm


def mkldnn_fuse_fx(gm: torch.fx.GraphModule, example_inputs):
    is_cpu = all(
        example_input.device == torch.device("cpu")
        for example_input in example_inputs
        if isinstance(example_input, torch.Tensor)
    )

    # make sure the autograd is disabled.
    if torch.is_grad_enabled():
        return gm
    if not (torch.backends.mkldnn.enabled and torch.backends.mkldnn.is_available()):
        return gm
    if not is_cpu:
        return gm
    # For binary fusion, we need to check inputs info to make sure
    # the binary inputs have same tensor info(device, dtype, and layout).

    fake_mode = fake_mode_from_tensors(example_inputs)
    ShapeProp(gm, fake_mode=fake_mode).propagate(*example_inputs)
    gm = fuse_unary(gm)
    if config.cpp.weight_prepack:
        gm = pack_module(gm)

    # Replacing leftover native convolutions into mkldnn convolutions.
    # fuse unary looks for patterns (conv+relu), hence convs that are not
    # followed by unary nodes will be left over.
    replace_conv(gm)
    return gm


def create_unary_module(node: torch.fx.node):
    assert (
        node.op == "call_function" or node.op == "call_method"
    ), "The current node should be a function/method node"
    unary_map = {
        F.relu: nn.ReLU,
        F.sigmoid: nn.Sigmoid,
        F.tanh: nn.Tanh,
        F.hardswish: nn.Hardswish,
        F.leaky_relu: nn.LeakyReLU,
        F.hardtanh: nn.Hardtanh,
        F.gelu: nn.GELU,
        F.relu6: nn.ReLU6,
        F.silu: nn.SiLU,
        F.hardsigmoid: nn.Hardsigmoid,
        torch.relu: nn.ReLU,
        torch.sigmoid: nn.Sigmoid,
        torch.tanh: nn.Tanh,
        "relu": nn.ReLU,
        "sigmoid": nn.Sigmoid,
        "tanh": nn.Tanh,
    }
    return unary_map[node.target](*(node.args[1:]), **(node.kwargs))


def fuse_unary(gm: torch.fx.GraphModule):
    # This function is designed to fuse unary operations with computation
    # nodes in a PyTorch computational graph, optimizing the graph structure
    # for improved performance during inference.
    modules = dict(gm.named_modules())

    for unary_op, (
        computation_module,
        fuse_func,
    ) in itertools.product(unary_ops, computation_op_unary_op_fusion_map.items()):
        pattern = (computation_module, unary_op)
        for node in gm.graph.nodes:
            if matches_module_pattern(
                pattern, node, modules
            ) or matches_module_function_pattern(pattern, node, modules):
                if (
                    len(node.args[0].users) > 1
                ):  # Output of computation_node is used by other nodes
                    continue
                computation_node = modules[node.args[0].target]
                if node.op == "call_function" or node.op == "call_method":
                    # make sure unary function's inputs only one
                    # fx.node(others should be constant value).
                    if any(isinstance(v, torch.fx.Node) for v in node.args[1:]) or any(
                        isinstance(v, torch.fx.Node) for _, v in node.kwargs.items()
                    ):
                        continue
                    unary_node = create_unary_module(node)
                    unary_node.eval()
                else:
                    unary_node = modules[node.target]
                eval_mode = all(not n.training for n in [computation_node, unary_node])
                if not eval_mode:
                    continue
                # TODO: support padding str input("valid", "same").
                if type(computation_node) in [nn.Conv2d] and isinstance(
                    computation_node.padding, str
                ):
                    continue
                # TODO: support more conv+binary+unary fusion.
                if type(computation_node) in [ConvBinary2d] and type(
                    unary_node
                ) not in [nn.ReLU]:
                    continue
                # TODO: remove this when group depthwise ConvTranspose is supported
                if is_group_depthwise_conv_transpose(computation_node):
                    continue
                computation_node_input_size = (
                    node.args[0].args[0].meta.get("tensor_meta").shape
                )
                fused_module = fuse_func(
                    computation_node, unary_node, computation_node_input_size
                )
                replace_node_module(node.args[0], modules, fused_module)

                node.replace_all_uses_with(node.args[0])
                gm.graph.erase_node(node)
    gm.graph.lint()
    gm.recompile()
    return gm


def pack_module(gm: torch.fx.GraphModule):
    # This function is designed to optimize computational graphs by
    # replacing certain types of modules with more efficient packed
    # versions, contributing to improved performance during inference.
    modules = dict(gm.named_modules())
    for node in gm.graph.nodes:
        if node.op == "call_module":
            assert isinstance(node.target, str)
            cur_module = modules[node.target]
            if type(cur_module) in computation_op_packed_map:
                if cur_module.training:
                    continue
                computation_node_input_meta = node.args[0].meta.get("tensor_meta")
                if computation_node_input_meta.dtype != torch.float32:
                    continue
                computation_node_input_size = computation_node_input_meta.shape
                if type(cur_module) in [nn.Conv2d] and isinstance(
                    cur_module.padding, str
                ):
                    continue
                # TODO: remove this when group depthwise ConvTranspose is supported
                if is_group_depthwise_conv_transpose(cur_module):
                    continue
                new_module = computation_op_packed_map[type(cur_module)](
                    cur_module, computation_node_input_size
                )
                assert isinstance(new_module, nn.Module)
                replace_node_module(node, modules, new_module)
    gm.graph.lint()
    gm.recompile()
    return gm


computation_op_unary_op_fusion_map = {
    nn.Conv2d: fused_conv_unary_eval,
    ConvBinary2d: fused_conv_binary_unary_eval,
    nn.ConvTranspose2d: fused_conv_transpose_unary_eval,
}


unary_modules_map = {
    nn.ReLU: UnaryAttr("relu"),
    nn.Sigmoid: UnaryAttr("sigmoid"),
    nn.Tanh: UnaryAttr("tanh"),
    nn.Hardswish: UnaryAttr("hardswish"),
    nn.LeakyReLU: UnaryAttr("leaky_relu", scalars_attr=["negative_slope"]),
    nn.Hardtanh: UnaryAttr("hardtanh", scalars_attr=["min_val", "max_val"]),
    nn.GELU: UnaryAttr("gelu", algorithm_attr="approximate"),
    nn.ReLU6: UnaryAttr("hardtanh", scalars_attr=["min_val", "max_val"]),
    nn.SiLU: UnaryAttr("swish"),
    nn.Hardsigmoid: UnaryAttr("hardsigmoid"),
}

unary_ops = [
    # modules
    nn.ReLU,
    nn.Sigmoid,
    nn.Tanh,
    nn.Hardswish,
    nn.LeakyReLU,
    nn.Hardtanh,
    nn.GELU,
    nn.ReLU6,
    nn.SiLU,
    nn.Hardsigmoid,
    # functional
    F.relu,
    F.sigmoid,
    F.tanh,
    F.hardswish,
    F.leaky_relu,
    F.hardtanh,
    F.gelu,
    F.relu6,
    F.silu,
    F.hardsigmoid,
    torch.relu,
    torch.sigmoid,
    torch.tanh,
    # methods (torch.Tensor.xxx)
    "relu",
    "sigmoid",
    "tanh",
]


binary_attr = {
    torch.add: "add",  # node.op == "call_function"
    "add": "add",  # node.op == "call_method"
    "add_": "iadd",  # node.op == "call_method"
    operator.add: "add",  # node.op == "call_function"
    operator.iadd: "iadd",  # node.op == "call_function"
    torch.sub: "sub",  # node.op == "call_function"
    "sub": "sub",  # node.op == "call_method"
    "sub_": "sub",  # node.op == "call_method"
    operator.sub: "sub",  # node.op == "call_function"
    operator.isub: "sub",  # node.op == "call_function"
}


computation_op_packed_map = {
    nn.Conv2d: packed_conv_eval,
    nn.ConvTranspose2d: packed_conv_transpose_eval,
}
