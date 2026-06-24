# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import os

import torch
import torch._inductor.config as _inductor_config
import torch._inductor.ir as _inductor_ir
from torch._inductor.ir import (
    ExternKernelAlloc,
    FixedLayout,
    FlexibleLayout,
    get_device_type,
    Layout,
    MultiOutput,
    MultiOutputLayout,
    NoneLayout,
    TensorBox,
)
from torch._inductor.virtualized import V
from torch._inductor.lowering import (
    add_needs_realized_inputs,
    fallbacks,
    register_lowering,
)
from torch.utils import _pytree as pytree

_ZENTORCH_HEADER = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "include", "shim_cpu_zentorch.hpp")
)

# When zentorch ops are registered into
# `torch._inductor.config.aot_inductor.custom_ops_to_c_shims` (so cpp_wrapper
# emits a direct C-ABI call instead of the slow `custom_op_wrapper` Python
# fallback), the dict ends up keyed by `torch._ops.OpOverload` objects. Those
# objects reference pybind11-wrapped C++ functions which Python's `pickle`
# cannot serialize (`RuntimeError: <pybind11_builtins....> is not pickleable.`).
# `FxGraphHashDetails` pickles the full `inductor_config` to compute the cache
# key, so without intervention every JIT compile triggers a `BypassFxGraphCache`
# warning and we lose all FxGraphCache hits.
#
# Tell PyTorch to ignore that config key when computing the cache hash. This
# is correct behaviour for our use: the dict's contents are populated
# deterministically at zentorch import time, so they're invariant across runs
# and don't need to participate in the cache key.
_CACHE_IGNORE_KEY = "aot_inductor.custom_ops_to_c_shims"
# Use getattr so we degrade gracefully if PyTorch ever renames or removes
# `_cache_config_ignore_prefix` (it's a private API). Without the guard,
# `import zentorch` would raise AttributeError on those builds. If the
# attribute isn't there we silently skip; the only downside is that
# FxGraphCache will keep bypassing on every compile (same regression we
# saw before this knob was added) -- the rest of zentorch still works.
_cache_ignore_prefix = getattr(
    _inductor_config, "_cache_config_ignore_prefix", None
)
if (
    _cache_ignore_prefix is not None
    and _CACHE_IGNORE_KEY not in _cache_ignore_prefix
):
    _cache_ignore_prefix.append(_CACHE_IGNORE_KEY)

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
        torch.ops.zentorch.zentorch_quant_embedding_bag.default,
        torch.ops.zentorch.zentorch_quant_embedding_bag.out,
        torch.ops.zentorch.zentorch_horizontal_quant_embedding_bag_group.default,
        torch.ops.zentorch.zentorch_horizontal_quant_embedding_bag_group.out,
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
        # Pin input contiguity at the IR level so the kernel doesn't pay an
        # internal at::contiguous() copy on every call under cpp_wrapper.
        input = cls.require_contiguous(cls.realize_input(input))
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
            # Route the caller-supplied op name into the IR node's kwargs so
            # `codegen_kwargs` emits it instead of falling back to the schema
            # default. Matches what the linear / .out lowerings already do.
            kwargs={"zentorch_op_name": name},
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
        input = cls.require_contiguous(cls.realize_input(input))
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
            kwargs={"zentorch_op_name": name},
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
        input = cls.require_contiguous(cls.realize_input(input))
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
            kwargs={"zentorch_op_name": name},
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
        input = cls.require_contiguous(cls.realize_input(input))
        weight.realize()
        mul_input = cls.require_contiguous(cls.realize_input(mul_input))
        add_input = cls.require_contiguous(cls.realize_input(add_input))

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
            kwargs={"zentorch_op_name": name},
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


class _zentorch_QlinearOutBase(ExternKernelAlloc):
    """Base for `.out` variants of qlinear ops.

    The kernel writes directly into the user-provided ``out`` tensor and
    returns nothing. Subclasses provide the bound ``op_overload`` and
    ``cpp_kernel_name`` for the corresponding AOTI shim.

    Layout is ``NoneLayout``; mutation of ``out`` is exposed to the scheduler
    via ``mark_buffer_mutated`` and ``get_mutation_names``.
    """

    _num_required_tensors = 3   # out, input, weight
    _optional_tensor_presence = [True] * 7
    codegen_args = _qlinear_codegen_args
    _op_overload = None
    _cpp_kernel_name = None

    def __init__(
        self, layout, inputs, constant_args=(), kwargs=None,
    ) -> None:
        out = inputs[0]
        self.device_type = get_device_type(out)
        super().__init__(
            layout, inputs, constant_args, kwargs,
            op_overload=self._op_overload,
            cpp_kernel_name=self._cpp_kernel_name,
        )
        V.graph.mark_buffer_mutated(out.get_name())

    def get_mutation_names(self):
        return [self.input_name(0)]

    def codegen(self, wrapper):
        wrapper.include_extra_header(_ZENTORCH_HEADER)
        self.codegen_comment(wrapper)
        args = [*self.codegen_args(), *self.codegen_kwargs()]
        # `super().codegen()` would route to `generate_extern_kernel_alloc`,
        # which in cpp_wrapper appends `&out_handle` for a tensor-returning
        # shim. Our `.out` shim returns void, so emit the call directly.
        if V.graph.cpp_wrapper:
            device = d.type if (d := self.get_device()) else V.graph.device_type
            wrapper.generate_c_shim_extern_kernel_call(
                self.get_kernel_name(),
                args,
                device,
                stack_traces=self.get_stack_traces(),
            )
        else:
            wrapper.writeline(
                f"{self.get_kernel_name()}({', '.join(args)}){wrapper.ending}"
            )

    def apply_constraint(self):
        pass

    @classmethod
    def _build_inputs(
        cls, out, input, weight, input_scales, input_zero_points,
        weight_scales, weight_zero_points, bias,
        output_scales, output_zero_points,
    ):
        # Pin contiguity on `input` so the kernel takes its `input.view()`
        # fast-path instead of falling into `input.contiguous().view()`.
        # `out` comes from upstream `torch.empty(...)` and is already
        # contiguous; enforcing it via require_contiguous would interfere
        # with mark_buffer_mutated tracking.
        out.realize()
        input = cls.require_contiguous(cls.realize_input(input))
        weight.realize()
        inputs = [out, input, weight]
        optional_tensors = [
            input_scales, input_zero_points,
            weight_scales, weight_zero_points,
            bias, output_scales, output_zero_points,
        ]
        for t in optional_tensors:
            if t is not None:
                t.realize()
                inputs.append(t)
        return inputs, [t is not None for t in optional_tensors]

    @classmethod
    def create(
        cls, out, input, weight, input_scales, input_zero_points,
        weight_scales, weight_zero_points, bias,
        output_scales, output_zero_points, output_dtype, zentorch_op_name,
    ):
        inputs, presence = cls._build_inputs(
            out, input, weight, input_scales, input_zero_points,
            weight_scales, weight_zero_points, bias,
            output_scales, output_zero_points,
        )
        device = out.get_device()
        assert device is not None
        # Thread `zentorch_op_name` through to the IR node's kwargs so the
        # shim call carries the caller-supplied name (rather than silently
        # falling back to the schema default). This matters for any caller
        # that customizes the name for profiling/counters; without this it
        # would be a no-op parameter.
        packed = cls(
            layout=NoneLayout(device=device),
            inputs=inputs,
            constant_args=[output_dtype],
            kwargs={"zentorch_op_name": zentorch_op_name},
        )
        packed._optional_tensor_presence = presence
        return packed


class zentorch_QlinearOut(_zentorch_QlinearOutBase):
    _op_overload = torch.ops.zentorch.zentorch_qlinear.out
    _cpp_kernel_name = "aoti_torch_cpu_zentorch_qlinear_out"


class zentorch_QlinearReluOut(_zentorch_QlinearOutBase):
    _op_overload = torch.ops.zentorch.zentorch_qlinear_relu.out
    _cpp_kernel_name = "aoti_torch_cpu_zentorch_qlinear_relu_out"


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
            zentorch_op_name,
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
            zentorch_op_name,
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
    zentorch_op_name="zentorch_qlinear.out",
):
    # Routes to the C++ `.out` shim which writes directly into `out`,
    # avoiding the temp buffer + copy that `realize_into` would generate.
    zentorch_QlinearOut.create(
        out, input, weight, input_scales, input_zero_points,
        weight_scales, weight_zero_points, bias,
        output_scales, output_zero_points, output_dtype,
        zentorch_op_name,
    )


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
    zentorch_op_name="zentorch_qlinear_relu.out",
):
    zentorch_QlinearReluOut.create(
        out, input, weight, input_scales, input_zero_points,
        weight_scales, weight_zero_points, bias,
        output_scales, output_zero_points, output_dtype,
        zentorch_op_name,
    )


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
            zentorch_op_name,
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
            zentorch_op_name,
        )
    )


# ============================================================================
# Quantized embedding bag lowerings.
#
# These ops have schemas with `Tensor[]`, `Tensor?[]`, `int[]` and `str` args
# (see `QuantEmbedBag.cpp` for definitions). None of these types are
# representable via StableIValue, so the default `make_fallback` route makes
# `cpp_wrapper` fall back to `torch._inductor.codecache.custom_op_wrapper`,
# which costs ~25us/call (microbenchmarked) due to GIL acquire/release plus a
# Python dispatch.
#
# We instead route them to dedicated AOTI shims (see `shim_cpu_zentorch.{hpp,
# cpp}`). The implementation reuses Inductor's `FallbackKernel` (so we get the
# schema-aware `Tensor[]`/`Tensor?[]`/`int[]` codegen for free via
# `_generate_temporary_array_pointer` in `cpp_wrapper_cpu.py`) and:
#   1. Overrides `set_cpp_kernel_name` so each overload (`.default` vs `.out`)
#      maps to its own shim. By default `FallbackKernel.set_cpp_kernel_name`
#      uses `kernel._schema.name` for non-aten ops which is the same string for
#      every overload of a given op, so we'd otherwise collide.
#   2. Registers the op into `config.aot_inductor.custom_ops_to_c_shims`. This
#      flips `use_runtime_dispatch` to `False` in `FallbackKernel.codegen`, so
#      the call is emitted as a direct shim call instead of routed through
#      `aoti_torch_call_dispatcher` (StableIValue) or `custom_op_wrapper`
#      (Python).
#   3. For `.out` variants (void-returning, kernel mutates `Tensor(a!)[]`
#      output buffers), overrides `codegen()` so we don't append the spurious
#      `&out_handle` that `generate_c_shim_extern_kernel_alloc` would otherwise
#      add for tensor-returning shims.
# ============================================================================


class _ZentorchEmbBagFallbackBase(_inductor_ir.FallbackKernel):
    """Base for FallbackKernels that route to a hand-written zentorch shim."""

    _zen_shim_name = ""  # subclasses set this

    def set_cpp_kernel_name(self, cpp_kernel_name=None):
        # Forward an explicit name if provided (callers typically don't), else
        # use the per-subclass shim name. This is what differentiates `.out`
        # from `.default` -- the parent would otherwise use `_schema.name`
        # which is identical across overloads for non-aten ops.
        super().set_cpp_kernel_name(cpp_kernel_name or self._zen_shim_name)

    def codegen(self, wrapper):
        # Make the zentorch shim function declarations available in the
        # generated main.cpp. The parent codegen path emits a direct call to
        # `aoti_torch_cpu_zentorch_*` which would otherwise be undeclared.
        if V.graph.cpp_wrapper:
            wrapper.include_extra_header(_ZENTORCH_HEADER)
        super().codegen(wrapper)


class _ZentorchEmbBagFallbackOutBase(_ZentorchEmbBagFallbackBase):
    """Base for `.out` variants. The kernel writes through user-provided
    `Tensor(a!)[]` buffers and returns nothing, so we skip the `&out_handle`
    that `generate_c_shim_extern_kernel_alloc` appends for tensor returns."""

    def codegen(self, wrapper):
        if not V.graph.cpp_wrapper:
            return super().codegen(wrapper)

        wrapper.include_extra_header(_ZENTORCH_HEADER)

        kernel = self.op_overload
        # Mirror the `use_runtime_dispatch` decision the parent makes for
        # non-aten cpp_wrapper ops; if our op is registered in
        # `custom_ops_to_c_shims` we take the fast path and emit a direct
        # shim call. Otherwise fall back to the parent (slow Python path).
        if kernel in _inductor_config.aot_inductor.custom_ops_to_c_shims:
            self.use_runtime_dispatch = False
        else:
            self.use_runtime_dispatch = True

        if self.use_runtime_dispatch:
            return super().codegen(wrapper)

        self.codegen_comment(wrapper)
        args = [*self.codegen_args(), *self.codegen_kwargs()]
        device = d.type if (d := self.get_device()) else V.graph.device_type
        wrapper.generate_c_shim_extern_kernel_call(
            self.cpp_kernel_name, args, device,
            stack_traces=self.get_stack_traces(),
        )
        self.codegen_unbacked_symbol_defs(wrapper)


class _ZentorchQuantEmbBag(_ZentorchEmbBagFallbackBase):
    _zen_shim_name = "aoti_torch_cpu_zentorch_quant_embedding_bag"


class _ZentorchQuantEmbBagOut(_ZentorchEmbBagFallbackOutBase):
    _zen_shim_name = "aoti_torch_cpu_zentorch_quant_embedding_bag_out"


class _ZentorchHorizontalQuantEmbBagGroupOut(_ZentorchEmbBagFallbackOutBase):
    _zen_shim_name = (
        "aoti_torch_cpu_zentorch_horizontal_quant_embedding_bag_group_out"
    )


class _ZentorchHorizontalQuantEmbBagGroupDefault(_ZentorchEmbBagFallbackBase):
    """Lowering for `zentorch_horizontal_quant_embedding_bag_group.default`.

    The op returns `Tensor[]` -- a variable-length list of N output tensors,
    where N is the number of embedding bags fused into the group call (known
    at lowering time but not at shim-compile time).

    Inductor's standard multi-output cpp_wrapper codegen
    (`generate_c_shim_fallback_kernel`) would emit one `&handle_i` per
    output and append all N as separate args at the end of the shim call.
    That doesn't match our shim signature, which takes a single
    `(AtenTensorHandle* ret0_handles, int64_t ret0_len_)` pair instead --
    the only way to express a variable-length return in a fixed C ABI.

    We override `codegen` to allocate an array of N handles, pass
    `(array, N)` to the shim, then wrap each filled-in handle in a
    `RAIIAtenTensorHandle` named after the corresponding `MultiOutput`
    so downstream IR can reference it the usual way.
    """

    _zen_shim_name = (
        "aoti_torch_cpu_zentorch_horizontal_quant_embedding_bag_group"
    )

    def codegen(self, wrapper):
        if not V.graph.cpp_wrapper:
            return super().codegen(wrapper)

        wrapper.include_extra_header(_ZENTORCH_HEADER)

        kernel = self.op_overload
        # Same `use_runtime_dispatch` decision logic as the parent uses for
        # non-aten cpp_wrapper ops -- if our op is in `custom_ops_to_c_shims`
        # we take the direct-shim path; otherwise punt back to the parent.
        if kernel in _inductor_config.aot_inductor.custom_ops_to_c_shims:
            self.use_runtime_dispatch = False
        else:
            self.use_runtime_dispatch = True

        if self.use_runtime_dispatch:
            return super().codegen(wrapper)

        self.codegen_comment(wrapper)
        args = [*self.codegen_args(), *self.codegen_kwargs()]

        # Allocate an array of N output handles and pass (array, N) to the
        # shim instead of N separate `&handle` args. Guard the degenerate
        # `n_outputs == 0` case: `AtenTensorHandle name[0];` is a
        # gcc-extension (UB by ISO C++) and emitting `nullptr` is what the
        # shim already expects when there's nothing to write back.
        outputs = list(self.outputs)
        n_outputs = len(outputs)
        if n_outputs:
            arr_var = f"{self.get_name()}_handles"
            wrapper.writeline(f"AtenTensorHandle {arr_var}[{n_outputs}];")
            args.append(arr_var)
        else:
            args.append("nullptr")
        args.append(f"{n_outputs}L")

        device = d.type if (d := self.get_device()) else V.graph.device_type
        wrapper.generate_c_shim_extern_kernel_call(
            self.cpp_kernel_name, args, device,
            stack_traces=self.get_stack_traces(),
        )

        # Wrap each returned handle in RAII so it gets freed on scope exit.
        # The MultiOutput names are what downstream IR (`getitem(group, i)`)
        # codegens against, so we must name the RAII wrappers accordingly.
        for idx, output in enumerate(outputs):
            wrapper.writeline(
                f"RAIIAtenTensorHandle {output.get_name()}({arr_var}[{idx}]);"
            )

        self.codegen_unbacked_symbol_defs(wrapper)


def _shim_routed_handler(kernel, fk_class):
    """Build a `register_lowering` handler that creates `fk_class` (a
    FallbackKernel subclass with our explicit shim name) and registers the op
    into `custom_ops_to_c_shims` so the cpp_wrapper codegen takes the direct
    shim path."""
    fallbacks.add(kernel)
    _inductor_config.aot_inductor.custom_ops_to_c_shims.setdefault(kernel, [])

    def handler(*args, **kwargs):
        def wrap_tensors(x):
            return TensorBox.create(x) if isinstance(x, _inductor_ir.IRNode) else x

        return pytree.tree_map(
            wrap_tensors, fk_class.create(kernel, *args, **kwargs)
        )

    handler._is_fallback_handler = True  # type: ignore[attr-defined]
    return handler


# Register the actual lowerings. This *replaces* the corresponding
# `make_fallback` entries in `_meta_registrations.py`, which must be removed.

register_lowering(
    torch.ops.zentorch.zentorch_quant_embedding_bag.default,
    type_promotion_kind=None,
)(
    _shim_routed_handler(
        torch.ops.zentorch.zentorch_quant_embedding_bag.default,
        _ZentorchQuantEmbBag,
    )
)

register_lowering(
    torch.ops.zentorch.zentorch_quant_embedding_bag.out,
    type_promotion_kind=None,
)(
    _shim_routed_handler(
        torch.ops.zentorch.zentorch_quant_embedding_bag.out,
        _ZentorchQuantEmbBagOut,
    )
)

register_lowering(
    torch.ops.zentorch.zentorch_horizontal_quant_embedding_bag_group.out,
    type_promotion_kind=None,
)(
    _shim_routed_handler(
        torch.ops.zentorch.zentorch_horizontal_quant_embedding_bag_group.out,
        _ZentorchHorizontalQuantEmbBagGroupOut,
    )
)

register_lowering(
    torch.ops.zentorch.zentorch_horizontal_quant_embedding_bag_group.default,
    type_promotion_kind=None,
)(
    _shim_routed_handler(
        torch.ops.zentorch.zentorch_horizontal_quant_embedding_bag_group.default,
        _ZentorchHorizontalQuantEmbBagGroupDefault,
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
                kwargs={"zentorch_op_name": name},
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
            kwargs={"zentorch_op_name": name},
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
                kwargs={"zentorch_op_name": name},
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
            zentorch_op_name,
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
            zentorch_op_name,
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
            zentorch_op_name,
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
            zentorch_op_name,
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
            zentorch_op_name,
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
            zentorch_op_name,
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
            zentorch_op_name,
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
            zentorch_op_name,
        )
    )


# -----------------------------------------------------------------------------
# Dynamic QLinear ExternKernel + lowering
#
# Schema (see src/cpu/cpp/DynamicQLinear.cpp):
#   zentorch_dynamic_qlinear(Tensor input, Tensor weight, Tensor weight_scales,
#                            Tensor? bias=None, *, str zentorch_op_name)
#                            -> Tensor
#
# Weight is in original nn.Linear layout [N, K] (NOT a packed WOQ layout), so
# out_features = weight.size(0). Required tensors (always present): input,
# weight -> _num_required_tensors = 2. weight_scales is required by schema but
# tracked via _optional_tensor_presence (always True) so _qlinear_codegen_args
# interleaves the truly-optional bias into its correct schema slot. Routing
# through this ExternKernelAlloc (with op_overload + cpp_kernel_name) makes
# cpp_wrapper emit a direct `aoti_torch_cpu_zentorch_dynamic_qlinear` C-shim
# call instead of the slow `custom_op_wrapper` Python fallback.
# -----------------------------------------------------------------------------


class zentorch_DynamicQlinear(ExternKernelAlloc):
    _num_required_tensors = 2
    _optional_tensor_presence = [True, True]
    codegen_args = _qlinear_codegen_args

    def __init__(
        self, layout, inputs, constant_args=(), kwargs=None,
    ) -> None:
        self.device_type = get_device_type(inputs[0])
        super().__init__(
            layout, inputs, constant_args, kwargs,
            op_overload=torch.ops.zentorch.zentorch_dynamic_qlinear.default,
            cpp_kernel_name="aoti_torch_cpu_zentorch_dynamic_qlinear",
        )

    def codegen(self, wrapper):
        wrapper.include_extra_header(_ZENTORCH_HEADER)
        super().codegen(wrapper)

    @classmethod
    def create(cls, input, weight, weight_scales, bias, name):
        # Pin contiguity at the IR level for every tensor the kernel reads
        # through a raw data_ptr() with hardcoded leading dims. The kernel
        # (DynamicQLinear.cpp) assumes contiguous storage for weight (ldb=K),
        # weight_scales and bias and never calls .contiguous() on them -- only
        # `input` is guarded inside the kernel. require_contiguous is a no-op
        # when the buffer is already contiguous (the common case for a frozen
        # [N, K] weight / per-channel scales / 1-D bias), so this only inserts
        # a clone in the rare non-contiguous case. The weight keeps its logical
        # [N, K] layout -- we pin storage contiguity, we do NOT transpose or
        # repack it.
        input = cls.require_contiguous(cls.realize_input(input))
        weight = cls.require_contiguous(cls.realize_input(weight))
        weight_scales = cls.require_contiguous(cls.realize_input(weight_scales))

        *m, _ = input.get_size()
        # weight is [N, K] (original nn.Linear layout); out_features = N
        oc, _ic = weight.get_size()
        output_size = list(m) + [oc]

        inputs = [input, weight, weight_scales]
        if bias is not None:
            bias = cls.require_contiguous(cls.realize_input(bias))
            inputs.append(bias)

        device = input.get_device()
        assert device is not None

        packed = zentorch_DynamicQlinear(
            layout=FixedLayout(
                device=device, dtype=input.get_dtype(), size=output_size,
            ),
            inputs=inputs,
            constant_args=(),
            kwargs={"zentorch_op_name": name},
        )
        packed._optional_tensor_presence = [
            True,            # weight_scales (required by schema, always present)
            bias is not None,
        ]
        return packed

    def apply_constraint(self):
        pass


@register_lowering(
    torch.ops.zentorch.zentorch_dynamic_qlinear.default,
    type_promotion_kind=None,
)
def zentorch_dynamic_qlinear_lowering(
    input: TensorBox,
    weight: TensorBox,
    weight_scales: TensorBox,
    bias: TensorBox = None,
    zentorch_op_name="zentorch_dynamic_qlinear",
):
    return TensorBox.create(
        zentorch_DynamicQlinear.create(
            input, weight, weight_scales, bias, zentorch_op_name,
        )
    )
