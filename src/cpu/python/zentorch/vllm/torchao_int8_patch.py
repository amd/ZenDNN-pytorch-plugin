# ****************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ****************************************************************************
"""Override torchao Int8Tensor aten dispatch for selected ops + zentorch linear.

Copies ``aten.view`` from
``torchao.quantization.quantize_.workflows.int8.int8_tensor``,
adds ``aten.permute`` and registers ``aten.linear`` / ``F.linear`` to use
``zentorch.zentorch_dynamic_qlinear``.
"""

from __future__ import annotations

import torch
from torch.utils._python_dispatch import return_and_correct_aliasing
from zentorch._utils import counters
from zentorch._logging import get_logger

logger = get_logger(__name__)


def _register_int8_tensor_handlers(Int8Tensor) -> None:
    aten = torch.ops.aten
    implements = Int8Tensor.implements
    implements_torch_function = Int8Tensor.implements_torch_function

    @implements(aten.view.default)
    def _int8_view(func, types, args, kwargs):
        self, size = args
        original_shape = self.shape
        if len(original_shape) == 3 and len(size) == 2:
            assert (
                original_shape[-1] == size[-1]
            ), f"Only support reshaping when last dimension matches, requested: reshaping from {original_shape} to {size}"
            bs = self.block_size
            block_size = [bs[0] * bs[1], bs[2]]
        elif len(original_shape) == 2 and len(size) == 3:
            assert (
                original_shape[-1] == size[-1]
            ), f"Only support reshaping when last dimension matches, requested: reshaping from {original_shape} to {size}"
            bs = self.block_size
            block_size = [1, bs[0], bs[1]]
        elif len(original_shape) == len(size):
            assert all(
                x == y or y == -1 for x, y in zip(original_shape, size, strict=True)
            ), f"Only support viewing with match dimensions or -1, got: {original_shape}, {size}"
            block_size = list(self.block_size)
        else:
            raise NotImplementedError(
                "Int8Tensor aten.view: only rank-preserving reshapes or 2D<->3D "
                f"rank changes are supported, got {original_shape} -> {size}"
            )

        qdata = self.qdata.reshape(*size)
        scale_shape = [qdata.shape[i] // block_size[i] for i in range(len(size))]
        scale = (
            self.scale if self.scale.numel() == 1 else self.scale.reshape(*scale_shape)
        )

        zero_point = (
            self.zero_point.reshape(scale.shape)
            if self.zero_point is not None
            else None
        )

        new = Int8Tensor(
            qdata,
            scale,
            block_size,
            self.dtype,
            zero_point=zero_point,
            act_quant_scale=self.act_quant_scale,
            act_quant_zero_point=self.act_quant_zero_point,
            act_pre_scale=self.act_pre_scale,
            act_quant_kwargs=self.act_quant_kwargs,
        )
        return return_and_correct_aliasing(func, args, kwargs, new)

    @implements(aten.permute.default)
    def _int8_permute(func, types, args, kwargs):
        self = args[0]
        dims_tuple = tuple(int(d) for d in args[1])
        q_ndim = self.qdata.ndim

        for name, t in (
            ("scale", self.scale),
            ("zero_point", self.zero_point),
            ("act_quant_scale", self.act_quant_scale),
            ("act_quant_zero_point", self.act_quant_zero_point),
            ("act_pre_scale", self.act_pre_scale),
        ):
            if t is not None and t.ndim != q_ndim:
                raise NotImplementedError(
                    "Int8Tensor aten.permute: expected "
                    f"{name} rank to match qdata, got qdata ndim {q_ndim}, "
                    f"{name} shape {tuple(t.shape)}."
                )

        new_qdata = self.qdata.permute(dims_tuple)
        new_block_size = [self.block_size[i] for i in dims_tuple]
        new_scale = self.scale.permute(dims_tuple)
        new_zero_point = (
            self.zero_point.permute(dims_tuple) if self.zero_point is not None else None
        )
        new_act_quant_scale = (
            self.act_quant_scale.permute(dims_tuple)
            if self.act_quant_scale is not None
            else None
        )
        new_act_quant_zero_point = (
            self.act_quant_zero_point.permute(dims_tuple)
            if self.act_quant_zero_point is not None
            else None
        )
        new_act_pre_scale = (
            self.act_pre_scale.permute(dims_tuple)
            if self.act_pre_scale is not None
            else None
        )

        return return_and_correct_aliasing(
            func,
            args,
            kwargs,
            Int8Tensor(
                new_qdata,
                new_scale,
                new_block_size,
                self.dtype,
                zero_point=new_zero_point,
                act_quant_kwargs=self.act_quant_kwargs,
                act_quant_scale=new_act_quant_scale,
                act_quant_zero_point=new_act_quant_zero_point,
                act_pre_scale=new_act_pre_scale,
            ),
        )

    @implements(torch.ops.aten.linear.default)
    @implements_torch_function(torch.nn.functional.linear)
    def _zentorch_int8_linear(func, types, args, kwargs):
        activation_tensor = args[0]
        weight_tensor = args[1]
        bias = args[2] if len(args) > 2 else None

        if (
            isinstance(weight_tensor, Int8Tensor)
            and weight_tensor.act_quant_kwargs is not None
        ):
            weight_int8 = weight_tensor.qdata
            weight_scales = weight_tensor.scale.contiguous()
            if weight_scales.dim() == 2 and weight_scales.shape[-1] == 1:
                weight_scales = weight_scales.squeeze(-1)
            counters["zentorch"]["zentorch_dynamic_qlinear"] += 1
            return torch.ops.zentorch.zentorch_dynamic_qlinear(
                activation_tensor,
                weight_int8,
                weight_scales,
                bias,
            )
        # TODO: Will add fallback to default linear later
        return func(*args, **(kwargs or {}))


def _apply_torchao_int8_tensor_patch_impl() -> bool:
    """Re-register Int8Tensor ``view`` + ``permute`` + zentorch dynamic linear.

    Caller must verify ``importlib.util.find_spec("torchao")`` is not None first
    (see ``zentorch.vllm.TorchAOPatch.apply``).
    """
    try:
        from torchao.quantization.quantize_.workflows.int8.int8_tensor import Int8Tensor

        _register_int8_tensor_handlers(Int8Tensor)
        logger.info(
            "[zentorch] Patched torchao Int8Tensor: aten.view, aten.permute, "
            "F.linear dynamic path -> zentorch_dynamic_qlinear"
        )
        return True
    except Exception:
        logger.warning("[zentorch] torchao Int8Tensor patch FAILED", exc_info=True)
        return False
