# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import os

import torch
from torch._inductor.ir import (
    ExternKernelAlloc,
    FixedLayout,
    FlexibleLayout,
    get_device_type,
    Layout,
    MutationLayoutSHOULDREMOVE,
    MultiOutput,
    MultiOutputLayout,
    TensorBox,
)
from torch._inductor.lowering import (
    add_needs_realized_inputs,
    register_lowering,
)

_ZENTORCH_HEADER = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "include", "shim_cpu_zentorch.hpp")
)

add_needs_realized_inputs(
    [
        torch.ops.zentorch.zentorch_linear_unary_binary.default,
        torch.ops.zentorch.zentorch_linear_binary_binary.default,
        torch.ops.zentorch.zentorch_qlinear.default,
        torch.ops.zentorch.zentorch_qlinear_relu.default,
        torch.ops.zentorch.zentorch_qlinear_sigmoid.default,
        torch.ops.zentorch.zentorch_qlinear_mul_add.default,
        torch.ops.zentorch.zentorch_qlinear.out,
        torch.ops.zentorch.zentorch_qlinear_relu.out,
    ]
)


def _create_output_node(packed):
    output_ir = MultiOutput(
        packed.get_layout(),
        packed,
        [],
    )
    packed.layout = MultiOutputLayout(device=packed.get_device())
    packed.outputs = [output_ir]
    return output_ir


def _qlinear_codegen_args(self):
    """Reconstruct codegen args in schema order for qlinear ops.

    ExternKernel.codegen_args concatenates [*self.inputs, *self.constant_args]
    which breaks schema ordering when optional tensor args are split between
    inputs (present) and constant_args (None). This override reconstructs the
    argument list in the correct schema order using _optional_tensor_presence.
    """
    from torch._inductor.virtualized import V

    required = list(self.inputs[: self._num_required_tensors])
    opt_iter = iter(self.inputs[self._num_required_tensors :])

    ordered = list(required)
    for present in self._optional_tensor_presence:
        ordered.append(next(opt_iter) if present else None)

    if V.graph.cpp_wrapper and self.op_overload is not None:
        ordered.extend(self.constant_args)
        ordered = list(self.fill_non_provided_args(ordered, self.kwargs))
        return [
            V.graph.wrapper_code.val_to_arg_str(
                x, self.arg_properties[i].get("type")
            )
            for i, x in enumerate(ordered)
        ]
    else:
        args = [V.graph.wrapper_code.val_to_arg_str(x) for x in ordered]
        args.extend(self.codegen_const_args())
        return args


class zentorch_LinearUnary(ExternKernelAlloc):
    def __init__(
        self,
        layout,
        inputs,
        constant_args=(),
        kwargs=None,
    ) -> None:
        self.device_type = get_device_type(inputs[0])
        super().__init__(
            layout,
            inputs,
            constant_args,
            kwargs,
            op_overload=torch.ops.zentorch.zentorch_linear_unary.default,
            cpp_kernel_name="aoti_torch_cpu_zentorch_linear_unary",
        )

    def codegen(self, wrapper):
        wrapper.include_extra_header(_ZENTORCH_HEADER)
        super().codegen(wrapper)

        if isinstance(self.layout, Layout):
            self.codegen_size_asserts(wrapper)

    @classmethod
    def create(cls, x, w, B, is_weight_prepacked, post_op, name):
        x = cls.require_contiguous(cls.realize_input(x))
        w = cls.require_contiguous(cls.realize_input(w))

        *m, _ic = x.get_size()
        oc, _ic = w.get_size()
        output_size = list(m) + [oc]
        inputs = [x, w]

        if B is not None:
            B = cls.require_contiguous(cls.realize_input(B))
            inputs.append(B)

        kwargs = {
            "is_weight_prepacked": is_weight_prepacked,
            "post_op": post_op,
            "zentorch_op_name": name,
        }

        device = x.get_device()
        assert device is not None

        packed = zentorch_LinearUnary(
            layout=FixedLayout(
                device=device,
                dtype=x.get_dtype(),
                size=output_size,
            ),
            inputs=inputs,
            constant_args=(),
            kwargs=kwargs,
        )
        return _create_output_node(packed)

    def apply_constraint(self):
        pass


@register_lowering(torch.ops.zentorch.zentorch_linear_unary)
def zentorch_linear_unary_lowering(
    input: TensorBox,
    weight: TensorBox,
    bias: TensorBox = None,
    is_weight_prepacked=False,
    post_op="none",
    zentorch_op_name="zentorch_linear_unary",
):
    return TensorBox.create(
        zentorch_LinearUnary.create(
            input,
            weight,
            bias,
            is_weight_prepacked,
            post_op,
            zentorch_op_name,
        )
    )


class zentorch_LinearUnaryBinary(ExternKernelAlloc):
    def __init__(
        self,
        layout,
        inputs,
        constant_args=(),
        kwargs=None,
    ) -> None:
        self.device_type = get_device_type(inputs[0])
        super().__init__(
            layout,
            inputs,
            constant_args,
            kwargs,
            op_overload=torch.ops.zentorch.zentorch_linear_unary_binary.default,
            cpp_kernel_name="aoti_torch_cpu_zentorch_linear_unary_binary",
        )

    def codegen(self, wrapper):
        wrapper.include_extra_header(_ZENTORCH_HEADER)
        super().codegen(wrapper)

        if isinstance(self.layout, Layout):
            self.codegen_size_asserts(wrapper)

    @classmethod
    def create(cls, x, w, binary_input, B, is_weight_prepacked,
               post_op_1, post_op_2, name):
        x = cls.require_contiguous(cls.realize_input(x))
        w = cls.require_contiguous(cls.realize_input(w))
        binary_input = cls.require_contiguous(cls.realize_input(binary_input))

        *m, _ic = x.get_size()
        oc, _ic = w.get_size()
        output_size = list(m) + [oc]
        inputs = [x, w, binary_input]

        if B is not None:
            B = cls.require_contiguous(cls.realize_input(B))
            inputs.append(B)

        kwargs = {
            "is_weight_prepacked": is_weight_prepacked,
            "post_op_1": post_op_1,
            "post_op_2": post_op_2,
            "zentorch_op_name": name,
        }

        device = x.get_device()
        assert device is not None

        packed = zentorch_LinearUnaryBinary(
            layout=FixedLayout(
                device=device,
                dtype=x.get_dtype(),
                size=output_size,
            ),
            inputs=inputs,
            constant_args=(),
            kwargs=kwargs,
        )
        return _create_output_node(packed)

    def apply_constraint(self):
        pass


@register_lowering(torch.ops.zentorch.zentorch_linear_unary_binary)
def zentorch_linear_unary_binary_lowering(
    input: TensorBox,
    weight: TensorBox,
    binary_input: TensorBox,
    bias: TensorBox = None,
    is_weight_prepacked=False,
    post_op_1="none",
    post_op_2="none",
    zentorch_op_name="zentorch_linear_unary_binary",
):
    return TensorBox.create(
        zentorch_LinearUnaryBinary.create(
            input,
            weight,
            binary_input,
            bias,
            is_weight_prepacked,
            post_op_1,
            post_op_2,
            zentorch_op_name,
        )
    )


class zentorch_LinearBinaryBinary(ExternKernelAlloc):
    def __init__(
        self,
        layout,
        inputs,
        constant_args=(),
        kwargs=None,
    ) -> None:
        self.device_type = get_device_type(inputs[0])
        super().__init__(
            layout,
            inputs,
            constant_args,
            kwargs,
            op_overload=torch.ops.zentorch.zentorch_linear_binary_binary.default,
            cpp_kernel_name="aoti_torch_cpu_zentorch_linear_binary_binary",
        )

    def codegen(self, wrapper):
        wrapper.include_extra_header(_ZENTORCH_HEADER)
        super().codegen(wrapper)

        if isinstance(self.layout, Layout):
            self.codegen_size_asserts(wrapper)

    @classmethod
    def create(cls, x, w, binary_input_1, binary_input_2, B,
               is_weight_prepacked, post_op_1, post_op_2, name):
        x = cls.require_contiguous(cls.realize_input(x))
        w = cls.require_contiguous(cls.realize_input(w))
        binary_input_1 = cls.require_contiguous(cls.realize_input(binary_input_1))
        binary_input_2 = cls.require_contiguous(cls.realize_input(binary_input_2))

        *m, _ic = x.get_size()
        oc, _ic = w.get_size()
        output_size = list(m) + [oc]
        inputs = [x, w, binary_input_1, binary_input_2]

        if B is not None:
            B = cls.require_contiguous(cls.realize_input(B))
            inputs.append(B)

        kwargs = {
            "is_weight_prepacked": is_weight_prepacked,
            "post_op_1": post_op_1,
            "post_op_2": post_op_2,
            "zentorch_op_name": name,
        }

        device = x.get_device()
        assert device is not None

        packed = zentorch_LinearBinaryBinary(
            layout=FixedLayout(
                device=device,
                dtype=x.get_dtype(),
                size=output_size,
            ),
            inputs=inputs,
            constant_args=(),
            kwargs=kwargs,
        )
        return _create_output_node(packed)

    def apply_constraint(self):
        pass


@register_lowering(torch.ops.zentorch.zentorch_linear_binary_binary)
def zentorch_linear_binary_binary_lowering(
    input: TensorBox,
    weight: TensorBox,
    binary_input_1: TensorBox,
    binary_input_2: TensorBox,
    bias: TensorBox = None,
    is_weight_prepacked=False,
    post_op_1="none",
    post_op_2="none",
    zentorch_op_name="zentorch_linear_binary_binary",
):
    return TensorBox.create(
        zentorch_LinearBinaryBinary.create(
            input,
            weight,
            binary_input_1,
            binary_input_2,
            bias,
            is_weight_prepacked,
            post_op_1,
            post_op_2,
            zentorch_op_name,
        )
    )


class zentorch_QlinearUnary(ExternKernelAlloc):
    _num_required_tensors = 2
    _optional_tensor_presence = [True] * 7
    codegen_args = _qlinear_codegen_args

    def __init__(
        self,
        layout,
        inputs,
        constant_args=(),
        kwargs=None,
    ) -> None:
        self.device_type = get_device_type(inputs[0])
        super().__init__(
            layout,
            inputs,
            constant_args,
            kwargs,
            op_overload=torch.ops.zentorch.zentorch_qlinear.default,
            cpp_kernel_name="aoti_torch_cpu_zentorch_qlinear",
        )

    def codegen(self, wrapper):
        wrapper.include_extra_header(_ZENTORCH_HEADER)
        super().codegen(wrapper)

    @classmethod
    def create(
        cls, input, weight, input_scales, input_zero_points,
        weight_scales, weight_zero_points, bias, output_scales,
        output_zero_points, output_dtype, name,
    ):
        input.realize()
        weight.realize()

        *m, _ic = input.get_size()
        oc, _ic = weight.get_size()
        output_size = list(m) + [oc]

        inputs = [input, weight]

        if input_scales is not None:
            input_scales.realize()
            inputs.append(input_scales)
        if input_zero_points is not None:
            input_zero_points.realize()
            inputs.append(input_zero_points)
        if weight_scales is not None:
            weight_scales.realize()
            inputs.append(weight_scales)
        if weight_zero_points is not None:
            weight_zero_points.realize()
            inputs.append(weight_zero_points)
        if bias is not None:
            bias.realize()
            inputs.append(bias)
        if output_scales is not None:
            output_scales.realize()
            inputs.append(output_scales)
        if output_zero_points is not None:
            output_zero_points.realize()
            inputs.append(output_zero_points)

        constant_args = [output_dtype]

        device = input.get_device()
        assert device is not None

        output_stride = FlexibleLayout.contiguous_strides(output_size)
        kernel_layout = FixedLayout(
            device, input.get_dtype(), output_size, output_stride,
        )
        if output_dtype is None:
            kernel_layout.dtype = torch.float32
        else:
            kernel_layout.dtype = output_dtype

        packed = zentorch_QlinearUnary(
            layout=FixedLayout(
                device=device, dtype=kernel_layout.dtype, size=output_size,
            ),
            inputs=inputs,
            constant_args=constant_args,
            kwargs=None,
        )
        packed._optional_tensor_presence = [
            input_scales is not None,
            input_zero_points is not None,
            weight_scales is not None,
            weight_zero_points is not None,
            bias is not None,
            output_scales is not None,
            output_zero_points is not None,
        ]
        return packed

    def apply_constraint(self):
        pass


class zentorch_QlinearUnaryRelu(ExternKernelAlloc):
    _num_required_tensors = 2
    _optional_tensor_presence = [True] * 7
    codegen_args = _qlinear_codegen_args

    def __init__(
        self, layout, inputs, constant_args=(), kwargs=None,
    ) -> None:
        self.device_type = get_device_type(inputs[0])
        super().__init__(
            layout, inputs, constant_args, kwargs,
            op_overload=torch.ops.zentorch.zentorch_qlinear_relu.default,
            cpp_kernel_name="aoti_torch_cpu_zentorch_qlinear_relu",
        )

    def codegen(self, wrapper):
        wrapper.include_extra_header(_ZENTORCH_HEADER)
        super().codegen(wrapper)

    @classmethod
    def create(
        cls, input, weight, input_scales, input_zero_points,
        weight_scales, weight_zero_points, bias, output_scales,
        output_zero_points, output_dtype, name,
    ):
        input.realize()
        weight.realize()

        *m, _ic = input.get_size()
        oc, _ic = weight.get_size()
        output_size = list(m) + [oc]

        inputs = [input, weight]

        if input_scales is not None:
            input_scales.realize()
            inputs.append(input_scales)
        if input_zero_points is not None:
            input_zero_points.realize()
            inputs.append(input_zero_points)
        if weight_scales is not None:
            weight_scales.realize()
            inputs.append(weight_scales)
        if weight_zero_points is not None:
            weight_zero_points.realize()
            inputs.append(weight_zero_points)
        if bias is not None:
            bias.realize()
            inputs.append(bias)
        if output_scales is not None:
            output_scales.realize()
            inputs.append(output_scales)
        if output_zero_points is not None:
            output_zero_points.realize()
            inputs.append(output_zero_points)

        constant_args = [output_dtype]

        device = input.get_device()
        assert device is not None

        output_stride = FlexibleLayout.contiguous_strides(output_size)
        kernel_layout = FixedLayout(
            device, input.get_dtype(), output_size, output_stride,
        )
        if output_dtype is None:
            kernel_layout.dtype = torch.float32
        else:
            kernel_layout.dtype = output_dtype

        packed = zentorch_QlinearUnaryRelu(
            layout=FixedLayout(
                device=device, dtype=kernel_layout.dtype, size=output_size,
            ),
            inputs=inputs,
            constant_args=constant_args,
            kwargs=None,
        )
        packed._optional_tensor_presence = [
            input_scales is not None,
            input_zero_points is not None,
            weight_scales is not None,
            weight_zero_points is not None,
            bias is not None,
            output_scales is not None,
            output_zero_points is not None,
        ]
        return packed

    def apply_constraint(self):
        pass


class zentorch_QlinearUnarySigmoid(ExternKernelAlloc):
    _num_required_tensors = 2
    _optional_tensor_presence = [True] * 7
    codegen_args = _qlinear_codegen_args

    def __init__(
        self, layout, inputs, constant_args=(), kwargs=None,
    ) -> None:
        self.device_type = get_device_type(inputs[0])
        super().__init__(
            layout, inputs, constant_args, kwargs,
            op_overload=torch.ops.zentorch.zentorch_qlinear_sigmoid.default,
            cpp_kernel_name="aoti_torch_cpu_zentorch_qlinear_sigmoid",
        )

    def codegen(self, wrapper):
        wrapper.include_extra_header(_ZENTORCH_HEADER)
        super().codegen(wrapper)

    @classmethod
    def create(
        cls, input, weight, input_scales, input_zero_points,
        weight_scales, weight_zero_points, bias, output_scales,
        output_zero_points, output_dtype, name,
    ):
        input.realize()
        weight.realize()

        *m, _ic = input.get_size()
        oc, _ic = weight.get_size()
        output_size = list(m) + [oc]

        inputs = [input, weight]

        if input_scales is not None:
            input_scales.realize()
            inputs.append(input_scales)
        if input_zero_points is not None:
            input_zero_points.realize()
            inputs.append(input_zero_points)
        if weight_scales is not None:
            weight_scales.realize()
            inputs.append(weight_scales)
        if weight_zero_points is not None:
            weight_zero_points.realize()
            inputs.append(weight_zero_points)
        if bias is not None:
            bias.realize()
            inputs.append(bias)
        if output_scales is not None:
            output_scales.realize()
            inputs.append(output_scales)
        if output_zero_points is not None:
            output_zero_points.realize()
            inputs.append(output_zero_points)

        constant_args = [output_dtype]

        device = input.get_device()
        assert device is not None

        output_stride = FlexibleLayout.contiguous_strides(output_size)
        kernel_layout = FixedLayout(
            device, input.get_dtype(), output_size, output_stride,
        )
        if output_dtype is None:
            kernel_layout.dtype = torch.float32
        else:
            kernel_layout.dtype = output_dtype

        packed = zentorch_QlinearUnarySigmoid(
            layout=FixedLayout(
                device=device, dtype=kernel_layout.dtype, size=output_size,
            ),
            inputs=inputs,
            constant_args=constant_args,
            kwargs=None,
        )
        packed._optional_tensor_presence = [
            input_scales is not None,
            input_zero_points is not None,
            weight_scales is not None,
            weight_zero_points is not None,
            bias is not None,
            output_scales is not None,
            output_zero_points is not None,
        ]
        return packed

    def apply_constraint(self):
        pass


class zentorch_QlinearMulAdd(ExternKernelAlloc):
    _num_required_tensors = 2
    _optional_tensor_presence = [True] * 9
    codegen_args = _qlinear_codegen_args

    def __init__(
        self, layout, inputs, constant_args=(), kwargs=None,
    ) -> None:
        self.device_type = get_device_type(inputs[0])
        super().__init__(
            layout, inputs, constant_args, kwargs,
            op_overload=torch.ops.zentorch.zentorch_qlinear_mul_add.default,
            cpp_kernel_name="aoti_torch_cpu_zentorch_qlinear_mul_add",
        )

    def codegen(self, wrapper):
        wrapper.include_extra_header(_ZENTORCH_HEADER)
        super().codegen(wrapper)

    @classmethod
    def create(
        cls, input, weight, input_scales, input_zero_points,
        weight_scales, weight_zero_points, mul_input, add_input,
        bias, output_scales, output_zero_points, output_dtype, name,
    ):
        input.realize()
        weight.realize()
        mul_input.realize()
        add_input.realize()

        *m, _ic = input.get_size()
        oc, _ic = weight.get_size()
        output_size = list(m) + [oc]

        inputs = [input, weight]

        if input_scales is not None:
            input_scales.realize()
            inputs.append(input_scales)
        if input_zero_points is not None:
            input_zero_points.realize()
            inputs.append(input_zero_points)
        if weight_scales is not None:
            weight_scales.realize()
            inputs.append(weight_scales)
        if weight_zero_points is not None:
            weight_zero_points.realize()
            inputs.append(weight_zero_points)

        inputs.append(mul_input)
        inputs.append(add_input)

        if bias is not None:
            bias.realize()
            inputs.append(bias)
        if output_scales is not None:
            output_scales.realize()
            inputs.append(output_scales)
        if output_zero_points is not None:
            output_zero_points.realize()
            inputs.append(output_zero_points)

        constant_args = [output_dtype]

        device = input.get_device()
        assert device is not None

        output_stride = FlexibleLayout.contiguous_strides(output_size)
        kernel_layout = FixedLayout(
            device, input.get_dtype(), output_size, output_stride,
        )
        if output_dtype is None:
            kernel_layout.dtype = torch.float32
        else:
            kernel_layout.dtype = output_dtype

        packed = zentorch_QlinearMulAdd(
            layout=FixedLayout(
                device=device, dtype=kernel_layout.dtype, size=output_size,
            ),
            inputs=inputs,
            constant_args=constant_args,
            kwargs=None,
        )
        packed._optional_tensor_presence = [
            input_scales is not None,
            input_zero_points is not None,
            weight_scales is not None,
            weight_zero_points is not None,
            True,
            True,
            bias is not None,
            output_scales is not None,
            output_zero_points is not None,
        ]
        return packed

    def apply_constraint(self):
        pass


@register_lowering(torch.ops.zentorch.zentorch_qlinear.default, type_promotion_kind=None)
def zentorch_qlinear_lowering(
    input: TensorBox,
    weight: TensorBox,
    input_scales: TensorBox,
    input_zero_points: TensorBox,
    weight_scales: TensorBox,
    weight_zero_points: TensorBox,
    bias: TensorBox,
    output_scales: TensorBox,
    output_zero_points: TensorBox,
    output_dtype=None,
    zentorch_op_name="zentorch_qlinear",
):
    return TensorBox.create(
        zentorch_QlinearUnary.create(
            input, weight, input_scales, input_zero_points,
            weight_scales, weight_zero_points, bias,
            output_scales, output_zero_points, output_dtype,
            "zentorch_qlinear",
        )
    )


@register_lowering(torch.ops.zentorch.zentorch_qlinear_relu.default, type_promotion_kind=None)
def zentorch_qlinear_relu_lowering(
    input: TensorBox,
    weight: TensorBox,
    input_scales: TensorBox,
    input_zero_points: TensorBox,
    weight_scales: TensorBox,
    weight_zero_points: TensorBox,
    bias: TensorBox,
    output_scales: TensorBox,
    output_zero_points: TensorBox,
    output_dtype=None,
    zentorch_op_name="zentorch_qlinear_relu",
):
    return TensorBox.create(
        zentorch_QlinearUnaryRelu.create(
            input, weight, input_scales, input_zero_points,
            weight_scales, weight_zero_points, bias,
            output_scales, output_zero_points, output_dtype,
            "zentorch_qlinear_relu",
        )
    )


@register_lowering(torch.ops.zentorch.zentorch_qlinear.out, type_promotion_kind=None)
def zentorch_qlinear_out_lowering(
    out: TensorBox,
    input: TensorBox,
    weight: TensorBox,
    input_scales: TensorBox,
    input_zero_points: TensorBox,
    weight_scales: TensorBox,
    weight_zero_points: TensorBox,
    bias: TensorBox,
    output_scales: TensorBox,
    output_zero_points: TensorBox,
    output_dtype=None,
    zentorch_op_name="zentorch_qlinear",
):
    result = zentorch_QlinearUnary.create(
        input, weight, input_scales, input_zero_points,
        weight_scales, weight_zero_points, bias,
        output_scales, output_zero_points, output_dtype,
        "zentorch_qlinear",
    )
    result = TensorBox.create(result)
    MutationLayoutSHOULDREMOVE.realize_into(result, out)


@register_lowering(torch.ops.zentorch.zentorch_qlinear_relu.out, type_promotion_kind=None)
def zentorch_qlinear_relu_out_lowering(
    out: TensorBox,
    input: TensorBox,
    weight: TensorBox,
    input_scales: TensorBox,
    input_zero_points: TensorBox,
    weight_scales: TensorBox,
    weight_zero_points: TensorBox,
    bias: TensorBox,
    output_scales: TensorBox,
    output_zero_points: TensorBox,
    output_dtype=None,
    zentorch_op_name="zentorch_qlinear_relu",
):
    result = zentorch_QlinearUnaryRelu.create(
        input, weight, input_scales, input_zero_points,
        weight_scales, weight_zero_points, bias,
        output_scales, output_zero_points, output_dtype,
        "zentorch_qlinear_relu",
    )
    result = TensorBox.create(result)
    MutationLayoutSHOULDREMOVE.realize_into(result, out)


@register_lowering(torch.ops.zentorch.zentorch_qlinear_sigmoid.default, type_promotion_kind=None)
def zentorch_qlinear_sigmoid_lowering(
    input: TensorBox,
    weight: TensorBox,
    input_scales: TensorBox,
    input_zero_points: TensorBox,
    weight_scales: TensorBox,
    weight_zero_points: TensorBox,
    bias: TensorBox,
    output_scales: TensorBox,
    output_zero_points: TensorBox,
    output_dtype=None,
    zentorch_op_name="zentorch_qlinear_sigmoid",
):
    return TensorBox.create(
        zentorch_QlinearUnarySigmoid.create(
            input, weight, input_scales, input_zero_points,
            weight_scales, weight_zero_points, bias,
            output_scales, output_zero_points, output_dtype,
            "zentorch_qlinear_sigmoid",
        )
    )


@register_lowering(torch.ops.zentorch.zentorch_qlinear_mul_add.default, type_promotion_kind=None)
def zentorch_qlinear_mul_add_lowering(
    input: TensorBox,
    weight: TensorBox,
    input_scales: TensorBox,
    input_zero_points: TensorBox,
    weight_scales: TensorBox,
    weight_zero_points: TensorBox,
    mul_input: TensorBox,
    add_input: TensorBox,
    bias: TensorBox,
    output_scales: TensorBox,
    output_zero_points: TensorBox,
    output_dtype=None,
    zentorch_op_name="zentorch_qlinear_mul_add",
):
    return TensorBox.create(
        zentorch_QlinearMulAdd.create(
            input, weight, input_scales, input_zero_points,
            weight_scales, weight_zero_points, mul_input, add_input,
            bias, output_scales, output_zero_points, output_dtype,
            "zentorch_qlinear_mul_add",
        )
    )


# -----------------------------------------------------------------------------
# WOQ Linear ExternKernel classes and lowerings
#
# WOQ schemas (see src/cpu/cpp/WOQ_Linear.cpp):
#   zentorch_woq_linear*        : (input, weight, weight_scales,
#                                  weight_zero_points?, bias?, *, op_name)
#   zentorch_woq_linear_add     : (input, weight, weight_scales,
#                                  weight_zero_points?, add_input,
#                                  bias?, *, op_name)
#   zentorch_woq_linear_mul_add : (input, weight, weight_scales,
#                                  weight_zero_points?, mul_input, add_input,
#                                  bias?, *, op_name)
#   zentorch_woq_linear_add_add : (input, weight, weight_scales,
#                                  weight_zero_points?, add_input, add_input_2,
#                                  bias?, *, op_name)
#
# Required tensors (always present): input, weight -> _num_required_tensors = 2.
# weight_scales is required by schema but is appended to inputs and tracked via
# _optional_tensor_presence so that _qlinear_codegen_args can interleave the
# truly optional weight_zero_points/bias into the right schema slots.
# -----------------------------------------------------------------------------


def _make_woq_linear_unary_class(class_name, op_overload, cpp_kernel_name):
    """Factory for the 5 woq_linear unary variants. They differ only in the
    op_overload and cpp_kernel_name; the create() body is identical."""

    class _WoqLinearUnary(ExternKernelAlloc):
        _num_required_tensors = 2
        _optional_tensor_presence = [True, True, True]
        codegen_args = _qlinear_codegen_args

        def __init__(
            self, layout, inputs, constant_args=(), kwargs=None,
        ) -> None:
            self.device_type = get_device_type(inputs[0])
            super().__init__(
                layout, inputs, constant_args, kwargs,
                op_overload=op_overload,
                cpp_kernel_name=cpp_kernel_name,
            )

        def codegen(self, wrapper):
            wrapper.include_extra_header(_ZENTORCH_HEADER)
            super().codegen(wrapper)

        @classmethod
        def create(cls, input, weight, weight_scales, weight_zero_points,
                   bias, name):
            # Enforce contiguous activations at the IR level so the kernel's
            # get_contiguous_view becomes a no-op. weight is in WOQ packed
            # int32 layout and must NOT be touched.
            input = cls.require_contiguous(cls.realize_input(input))
            weight.realize()
            weight_scales.realize()

            *m, _ic = input.get_size()
            # WOQ packed weight is laid out (K_packed, N); out_features = N
            _kpacked, oc = weight.get_size()
            output_size = list(m) + [oc]

            inputs = [input, weight, weight_scales]
            if weight_zero_points is not None:
                weight_zero_points.realize()
                inputs.append(weight_zero_points)
            if bias is not None:
                bias.realize()
                inputs.append(bias)

            device = input.get_device()
            assert device is not None

            packed = cls(
                layout=FixedLayout(
                    device=device, dtype=input.get_dtype(), size=output_size,
                ),
                inputs=inputs,
                constant_args=(),
                kwargs=None,
            )
            packed._optional_tensor_presence = [
                True,
                weight_zero_points is not None,
                bias is not None,
            ]
            return packed

        def apply_constraint(self):
            pass

    _WoqLinearUnary.__name__ = class_name
    _WoqLinearUnary.__qualname__ = class_name
    return _WoqLinearUnary


zentorch_WoqLinear = _make_woq_linear_unary_class(
    "zentorch_WoqLinear",
    torch.ops.zentorch.zentorch_woq_linear.default,
    "aoti_torch_cpu_zentorch_woq_linear",
)

zentorch_WoqLinearRelu = _make_woq_linear_unary_class(
    "zentorch_WoqLinearRelu",
    torch.ops.zentorch.zentorch_woq_linear_relu.default,
    "aoti_torch_cpu_zentorch_woq_linear_relu",
)

zentorch_WoqLinearSigmoid = _make_woq_linear_unary_class(
    "zentorch_WoqLinearSigmoid",
    torch.ops.zentorch.zentorch_woq_linear_sigmoid.default,
    "aoti_torch_cpu_zentorch_woq_linear_sigmoid",
)

zentorch_WoqLinearGeluTanh = _make_woq_linear_unary_class(
    "zentorch_WoqLinearGeluTanh",
    torch.ops.zentorch.zentorch_woq_linear_gelu_tanh.default,
    "aoti_torch_cpu_zentorch_woq_linear_gelu_tanh",
)

zentorch_WoqLinearGeluErf = _make_woq_linear_unary_class(
    "zentorch_WoqLinearGeluErf",
    torch.ops.zentorch.zentorch_woq_linear_gelu_erf.default,
    "aoti_torch_cpu_zentorch_woq_linear_gelu_erf",
)


class zentorch_WoqLinearAdd(ExternKernelAlloc):
    _num_required_tensors = 2
    _optional_tensor_presence = [True, True, True, True]
    codegen_args = _qlinear_codegen_args

    def __init__(
        self, layout, inputs, constant_args=(), kwargs=None,
    ) -> None:
        self.device_type = get_device_type(inputs[0])
        super().__init__(
            layout, inputs, constant_args, kwargs,
            op_overload=torch.ops.zentorch.zentorch_woq_linear_add.default,
            cpp_kernel_name="aoti_torch_cpu_zentorch_woq_linear_add",
        )

    def codegen(self, wrapper):
        wrapper.include_extra_header(_ZENTORCH_HEADER)
        super().codegen(wrapper)

    @classmethod
    def create(cls, input, weight, weight_scales, weight_zero_points,
               add_input, bias, name):
        # Enforce contiguous activations + post-op buffers at the IR level
        # so the kernel's get_contiguous_view becomes a no-op.
        input = cls.require_contiguous(cls.realize_input(input))
        weight.realize()
        weight_scales.realize()
        add_input = cls.require_contiguous(cls.realize_input(add_input))

        *m, _ic = input.get_size()
        # WOQ packed weight is laid out (K_packed, N); out_features = N
        _kpacked, oc = weight.get_size()
        output_size = list(m) + [oc]

        inputs = [input, weight, weight_scales]
        if weight_zero_points is not None:
            weight_zero_points.realize()
            inputs.append(weight_zero_points)
        inputs.append(add_input)
        if bias is not None:
            bias.realize()
            inputs.append(bias)

        device = input.get_device()
        assert device is not None

        packed = zentorch_WoqLinearAdd(
            layout=FixedLayout(
                device=device, dtype=input.get_dtype(), size=output_size,
            ),
            inputs=inputs,
            constant_args=(),
            kwargs=None,
        )
        packed._optional_tensor_presence = [
            True,
            weight_zero_points is not None,
            True,
            bias is not None,
        ]
        return packed

    def apply_constraint(self):
        pass


def _make_woq_linear_binary_binary_class(class_name, op_overload,
                                         cpp_kernel_name):
    """Factory for the woq_linear_mul_add and woq_linear_add_add variants."""

    class _WoqLinearBinaryBinary(ExternKernelAlloc):
        _num_required_tensors = 2
        _optional_tensor_presence = [True, True, True, True, True]
        codegen_args = _qlinear_codegen_args

        def __init__(
            self, layout, inputs, constant_args=(), kwargs=None,
        ) -> None:
            self.device_type = get_device_type(inputs[0])
            super().__init__(
                layout, inputs, constant_args, kwargs,
                op_overload=op_overload,
                cpp_kernel_name=cpp_kernel_name,
            )

        def codegen(self, wrapper):
            wrapper.include_extra_header(_ZENTORCH_HEADER)
            super().codegen(wrapper)

        @classmethod
        def create(cls, input, weight, weight_scales, weight_zero_points,
                   binary1_input, binary2_input, bias, name):
            # Enforce contiguous activations + post-op buffers at the IR level
            # so the kernel's get_contiguous_view becomes a no-op.
            input = cls.require_contiguous(cls.realize_input(input))
            weight.realize()
            weight_scales.realize()
            binary1_input = cls.require_contiguous(
                cls.realize_input(binary1_input)
            )
            binary2_input = cls.require_contiguous(
                cls.realize_input(binary2_input)
            )

            *m, _ic = input.get_size()
            # WOQ packed weight is laid out (K_packed, N); out_features = N
            _kpacked, oc = weight.get_size()
            output_size = list(m) + [oc]

            inputs = [input, weight, weight_scales]
            if weight_zero_points is not None:
                weight_zero_points.realize()
                inputs.append(weight_zero_points)
            inputs.append(binary1_input)
            inputs.append(binary2_input)
            if bias is not None:
                bias.realize()
                inputs.append(bias)

            device = input.get_device()
            assert device is not None

            packed = cls(
                layout=FixedLayout(
                    device=device, dtype=input.get_dtype(), size=output_size,
                ),
                inputs=inputs,
                constant_args=(),
                kwargs=None,
            )
            packed._optional_tensor_presence = [
                True,
                weight_zero_points is not None,
                True,
                True,
                bias is not None,
            ]
            return packed

        def apply_constraint(self):
            pass

    _WoqLinearBinaryBinary.__name__ = class_name
    _WoqLinearBinaryBinary.__qualname__ = class_name
    return _WoqLinearBinaryBinary


zentorch_WoqLinearMulAdd = _make_woq_linear_binary_binary_class(
    "zentorch_WoqLinearMulAdd",
    torch.ops.zentorch.zentorch_woq_linear_mul_add.default,
    "aoti_torch_cpu_zentorch_woq_linear_mul_add",
)

zentorch_WoqLinearAddAdd = _make_woq_linear_binary_binary_class(
    "zentorch_WoqLinearAddAdd",
    torch.ops.zentorch.zentorch_woq_linear_add_add.default,
    "aoti_torch_cpu_zentorch_woq_linear_add_add",
)


@register_lowering(
    torch.ops.zentorch.zentorch_woq_linear.default, type_promotion_kind=None
)
def zentorch_woq_linear_lowering(
    input: TensorBox,
    weight: TensorBox,
    weight_scales: TensorBox,
    weight_zero_points: TensorBox = None,
    bias: TensorBox = None,
    zentorch_op_name="zentorch_woq_linear",
):
    return TensorBox.create(
        zentorch_WoqLinear.create(
            input, weight, weight_scales, weight_zero_points, bias,
            "zentorch_woq_linear",
        )
    )


@register_lowering(
    torch.ops.zentorch.zentorch_woq_linear_relu.default,
    type_promotion_kind=None,
)
def zentorch_woq_linear_relu_lowering(
    input: TensorBox,
    weight: TensorBox,
    weight_scales: TensorBox,
    weight_zero_points: TensorBox = None,
    bias: TensorBox = None,
    zentorch_op_name="zentorch_woq_linear_relu",
):
    return TensorBox.create(
        zentorch_WoqLinearRelu.create(
            input, weight, weight_scales, weight_zero_points, bias,
            "zentorch_woq_linear_relu",
        )
    )


@register_lowering(
    torch.ops.zentorch.zentorch_woq_linear_sigmoid.default,
    type_promotion_kind=None,
)
def zentorch_woq_linear_sigmoid_lowering(
    input: TensorBox,
    weight: TensorBox,
    weight_scales: TensorBox,
    weight_zero_points: TensorBox = None,
    bias: TensorBox = None,
    zentorch_op_name="zentorch_woq_linear_sigmoid",
):
    return TensorBox.create(
        zentorch_WoqLinearSigmoid.create(
            input, weight, weight_scales, weight_zero_points, bias,
            "zentorch_woq_linear_sigmoid",
        )
    )


@register_lowering(
    torch.ops.zentorch.zentorch_woq_linear_gelu_tanh.default,
    type_promotion_kind=None,
)
def zentorch_woq_linear_gelu_tanh_lowering(
    input: TensorBox,
    weight: TensorBox,
    weight_scales: TensorBox,
    weight_zero_points: TensorBox = None,
    bias: TensorBox = None,
    zentorch_op_name="zentorch_woq_linear_gelu_tanh",
):
    return TensorBox.create(
        zentorch_WoqLinearGeluTanh.create(
            input, weight, weight_scales, weight_zero_points, bias,
            "zentorch_woq_linear_gelu_tanh",
        )
    )


@register_lowering(
    torch.ops.zentorch.zentorch_woq_linear_gelu_erf.default,
    type_promotion_kind=None,
)
def zentorch_woq_linear_gelu_erf_lowering(
    input: TensorBox,
    weight: TensorBox,
    weight_scales: TensorBox,
    weight_zero_points: TensorBox = None,
    bias: TensorBox = None,
    zentorch_op_name="zentorch_woq_linear_gelu_erf",
):
    return TensorBox.create(
        zentorch_WoqLinearGeluErf.create(
            input, weight, weight_scales, weight_zero_points, bias,
            "zentorch_woq_linear_gelu_erf",
        )
    )


@register_lowering(
    torch.ops.zentorch.zentorch_woq_linear_add.default,
    type_promotion_kind=None,
)
def zentorch_woq_linear_add_lowering(
    input: TensorBox,
    weight: TensorBox,
    weight_scales: TensorBox,
    weight_zero_points: TensorBox,
    add_input: TensorBox,
    bias: TensorBox = None,
    zentorch_op_name="zentorch_woq_linear_add",
):
    return TensorBox.create(
        zentorch_WoqLinearAdd.create(
            input, weight, weight_scales, weight_zero_points, add_input, bias,
            "zentorch_woq_linear_add",
        )
    )


@register_lowering(
    torch.ops.zentorch.zentorch_woq_linear_mul_add.default,
    type_promotion_kind=None,
)
def zentorch_woq_linear_mul_add_lowering(
    input: TensorBox,
    weight: TensorBox,
    weight_scales: TensorBox,
    weight_zero_points: TensorBox,
    mul_input: TensorBox,
    add_input: TensorBox,
    bias: TensorBox = None,
    zentorch_op_name="zentorch_woq_linear_mul_add",
):
    return TensorBox.create(
        zentorch_WoqLinearMulAdd.create(
            input, weight, weight_scales, weight_zero_points,
            mul_input, add_input, bias,
            "zentorch_woq_linear_mul_add",
        )
    )


@register_lowering(
    torch.ops.zentorch.zentorch_woq_linear_add_add.default,
    type_promotion_kind=None,
)
def zentorch_woq_linear_add_add_lowering(
    input: TensorBox,
    weight: TensorBox,
    weight_scales: TensorBox,
    weight_zero_points: TensorBox,
    add_input: TensorBox,
    add_input_2: TensorBox,
    bias: TensorBox = None,
    zentorch_op_name="zentorch_woq_linear_add_add",
):
    return TensorBox.create(
        zentorch_WoqLinearAddAdd.create(
            input, weight, weight_scales, weight_zero_points,
            add_input, add_input_2, bias,
            "zentorch_woq_linear_add_add",
        )
    )
