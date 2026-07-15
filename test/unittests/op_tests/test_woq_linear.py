# ******************************************************************************
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import unittest
import torch
from torch import nn
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from unittest_utils import (  # noqa: 402
    WOQTestCase,
    has_zentorch,
    run_tests,
    compiled_frozen_reference,
    woq_dtypes,
    batch_opt,
    in_features_opt,
    out_features_opt,
    woq_bias_opt,
    input_dim_opt,
    DataTypes,
    Range,
)


# NOTE: Zentorch fused ops are compared against a torch.compile reference model so that
# torch fuses the linear with its post-ops, giving a fused baseline to validate against.
class WOQ_Linear_Gelu_Model(nn.Module):
    """Reference model for WOQ linear + GELU.
    Args:
        dq_weight:   Dequantized weight, shape ``(out_features, in_features)``.
        bias:        Optional additive bias, shape ``(out_features,)``.
        approximate: GELU approximation mode, ``"none"`` (erf) or ``"tanh"``.
    """

    def __init__(self, dq_weight, bias=None, approximate="none"):
        super().__init__()
        self.register_buffer("dq_weight", dq_weight)
        self.register_buffer("bias", bias)
        self.approximate = approximate

    def forward(self, x):
        out = torch.nn.functional.linear(x, self.dq_weight, self.bias)
        return torch.nn.functional.gelu(out, approximate=self.approximate)


class WOQ_Linear_Add_Add_Model(nn.Module):
    """Reference model for WOQ linear + add + add.
    Args:
        dq_weight: Dequantized weight, shape ``(out_features, in_features)``.
        bias:      Optional additive bias, shape ``(out_features,)``.
    """

    def __init__(self, dq_weight, bias=None):
        super().__init__()
        self.register_buffer("dq_weight", dq_weight)
        self.register_buffer("bias", bias)

    def forward(self, x, add_input, add_input_2):
        out = torch.nn.functional.linear(x, self.dq_weight, self.bias)
        return out + add_input + add_input_2


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_WOQLinear(WOQTestCase):
    qmin, qmax = -8, 7

    def quantize_weight(self, weight):
        """
        Quantize weight to int4 format with per-channel symmetric quantization.

        Args:
            weight: Original weight tensor (out_features, in_features)

        Returns:
            quantized_weight: Quantized weight (int8)
            scale: Scale tensor for dequantization (out_features, 1) - float32
            zero_point: Zero point tensor (out_features, 1) - int8, all zeros for symmetric
        """
        # Calculate per-channel max absolute value for symmetric quantization
        weight_absmax = weight.abs().max(dim=1, keepdim=True)[0]

        # Calculate scale for symmetric quantization.
        # Scale maps [-weight_absmax, weight_absmax] to [qmin, qmax]
        scale = weight_absmax / max(abs(self.qmin), abs(self.qmax))
        scale = torch.clamp(scale, min=1e-8).to(torch.float32)

        # Quantize the weight
        quantized_weight = torch.clamp(
            torch.round(weight / scale), self.qmin, self.qmax
        ).to(torch.int8)
        zero_point = torch.zeros_like(scale, dtype=torch.int8)
        return quantized_weight, scale, zero_point

    @WOQTestCase.hypothesis_params_woq_itr(
        dtype_opt_list=woq_dtypes,
        batch_opt_list=batch_opt,
        in_features_opt_list=in_features_opt,
        out_features_opt_list=out_features_opt,
        bias_opt_list=woq_bias_opt,
        input_dim_opt_list=input_dim_opt,
    )
    @torch.inference_mode()
    def test_woq_linear_accuracy(self):
        """Test accuracy of WOQ linear against PyTorch linear with dequantized weights."""
        # Get all pre-created tensors from hypothesis
        input = self.data.woq_input
        original_weight = self.data.woq_weight
        bias = self.data.woq_bias

        # Quantize weight
        quantized_weight, scale, _ = self.quantize_weight(original_weight)

        # Pack the quantized weight
        packed_weight = torch.ops.zentorch.zentorch_woq_repack_weight(
            quantized_weight
        )
        woq_dtype = DataTypes.get_torch_type(self.data.dtype)
        # Dequantize weight for reference (scale is already (out_features, 1))
        dq_weight = quantized_weight.to(woq_dtype) * scale

        # Run zentorch WOQ linear (expects scale as (out_features, 1))
        zentorch_result = torch.ops.zentorch.zentorch_woq_linear(
            input, packed_weight.transpose(0, 1), scale.transpose(0, 1), None, bias
        )
        # Run PyTorch linear with dequantized weight
        pytorch_result = torch.nn.functional.linear(input, dq_weight.to(woq_dtype), bias)

        # Compare results
        self.assertTrue(
            torch.allclose(zentorch_result, pytorch_result, rtol=1e-3, atol=1e-3),
            "Zentorch WOQ result does not match PyTorch DQ result.",
        )

    def _woq_setup(self, bias=False):
        """Create packed weight, scales, and reference dq_weight from pre-created tensors."""
        input_t = self.data.woq_input
        original_weight = self.data.woq_weight
        bias_t = self.data.woq_bias if bias else None

        quantized_weight, scale, _ = self.quantize_weight(original_weight)
        packed_weight = torch.ops.zentorch.zentorch_woq_repack_weight(
            quantized_weight
        )
        woq_dtype = DataTypes.get_torch_type(self.data.dtype)
        packed_weight_t = packed_weight.transpose(0, 1)
        scale_t = scale.transpose(0, 1).contiguous()
        dq_weight = quantized_weight.to(woq_dtype) * scale.to(woq_dtype)
        return input_t, packed_weight_t, scale_t, dq_weight, bias_t

    def _assert_woq_fused_output_sanity(self, result, op_name):
        """Assert fused WOQ op output has correct shape, dtype, and finite values."""
        # Derive expected shape from input shape plus out_features as trailing dim
        expected_shape = self.data.woq_input.shape[:-1] + (self.data.out_features,)
        woq_dtype = DataTypes.get_torch_type(self.data.dtype)
        self.assertEqual(result.shape, expected_shape, f"{op_name}: wrong output shape")
        self.assertEqual(result.dtype, woq_dtype, f"{op_name}: wrong output dtype")
        self.assertTrue(torch.isfinite(result).all(), f"{op_name}: output contains inf/nan")

    # Fused post-ops (GELU, binary) can have slightly larger numerical error than
    # plain WOQ linear; use rtol/atol 1e-2 for comparison with dq_output reference.
    _fused_rtol, _fused_atol = 1e-2, 1e-2

    @WOQTestCase.hypothesis_params_woq_itr(
        dtype_opt_list=woq_dtypes,
        batch_opt_list=batch_opt,
        in_features_opt_list=in_features_opt,
        out_features_opt_list=out_features_opt,
        bias_opt_list=woq_bias_opt,
        input_dim_opt_list=input_dim_opt,
    )
    @torch.inference_mode()
    def test_woq_linear_gelu_tanh_accuracy(self):
        """Test accuracy of WOQ linear + GELU(tanh) against dq linear + gelu(tanh)."""
        input_t, packed_weight_t, scale_t, dq_weight, bias_t = self._woq_setup(bias=self.data.with_bias)
        zentorch_result = torch.ops.zentorch.zentorch_woq_linear_gelu_tanh(
            input_t, packed_weight_t, scale_t, None, bias_t
        )
        woq_dtype = DataTypes.get_torch_type(self.data.dtype)

        ref_model = WOQ_Linear_Gelu_Model(
            dq_weight.to(woq_dtype), bias_t, approximate="tanh"
        )
        pytorch_result = compiled_frozen_reference(ref_model, input_t)

        self._assert_woq_fused_output_sanity(zentorch_result, "zentorch_woq_linear_gelu_tanh")
        self.assertTrue(
            torch.allclose(zentorch_result, pytorch_result, rtol=self._fused_rtol, atol=self._fused_atol),
            "zentorch_woq_linear_gelu_tanh does not match dq linear + gelu(tanh).",
        )

    @WOQTestCase.hypothesis_params_woq_itr(
        dtype_opt_list=woq_dtypes,
        batch_opt_list=batch_opt,
        in_features_opt_list=in_features_opt,
        out_features_opt_list=out_features_opt,
        bias_opt_list=woq_bias_opt,
        input_dim_opt_list=input_dim_opt,
    )
    @torch.inference_mode()
    def test_woq_linear_gelu_erf_accuracy(self):
        """Test accuracy of WOQ linear + GELU(erf) against dq linear + gelu(none)."""
        input_t, packed_weight_t, scale_t, dq_weight, bias_t = self._woq_setup(bias=self.data.with_bias)
        zentorch_result = torch.ops.zentorch.zentorch_woq_linear_gelu_erf(
            input_t, packed_weight_t, scale_t, None, bias_t
        )
        woq_dtype = DataTypes.get_torch_type(self.data.dtype)

        ref_model = WOQ_Linear_Gelu_Model(
            dq_weight.to(woq_dtype), bias_t, approximate="none"
        )
        pytorch_result = compiled_frozen_reference(ref_model, input_t)

        self._assert_woq_fused_output_sanity(zentorch_result, "zentorch_woq_linear_gelu_erf")
        self.assertTrue(
            torch.allclose(zentorch_result, pytorch_result, rtol=self._fused_rtol, atol=self._fused_atol),
            "zentorch_woq_linear_gelu_erf does not match dq linear + gelu(erf).",
        )

    # Test Fails while generalising test
    # Bug has been reported Jira ID: ZENAI-3714
    # @WOQTestCase.hypothesis_params_woq_itr(
    #     dtype_opt_list=woq_dtypes,
    #     batch_opt_list=batch_opt,
    #     in_features_opt_list=in_features_opt,
    #     out_features_opt_list=out_features_opt,
    #     bias_opt_list=woq_bias_opt,
    #     input_dim_opt_list=input_dim_opt,
    # )

    @WOQTestCase.hypothesis_params_woq_itr(
        dtype_opt_list=woq_dtypes,
        batch_opt_list=[2, 3],
        in_features_opt_list=[64],
        out_features_opt_list=[48],
        bias_opt_list=[False],
        input_dim_opt_list=input_dim_opt,
        pRange=Range(1, 3),
        qRange=Range(1, 3),
    )
    @torch.inference_mode()
    def test_woq_linear_mul_add_accuracy(self):
        """Test accuracy of WOQ linear + mul + add against dq linear then (out * mul) + add."""
        input_t, packed_weight_t, scale_t, dq_weight, bias_t = self._woq_setup(bias=self.data.with_bias)
        mul_input = self.data.woq_mul_input
        add_input = self.data.woq_add_input
        zentorch_result = torch.ops.zentorch.zentorch_woq_linear_mul_add(
            input_t, packed_weight_t, scale_t, None, mul_input, add_input, bias_t
        )
        woq_dtype = DataTypes.get_torch_type(self.data.dtype)
        dq_output = torch.nn.functional.linear(input_t, dq_weight.to(woq_dtype), bias_t)
        pytorch_result = dq_output * mul_input + add_input
        self._assert_woq_fused_output_sanity(zentorch_result, "zentorch_woq_linear_mul_add")
        self.assertTrue(
            torch.allclose(zentorch_result, pytorch_result, rtol=self._fused_rtol, atol=self._fused_atol),
            "zentorch_woq_linear_mul_add does not match dq linear then (out * mul) + add.",
        )

    @WOQTestCase.hypothesis_params_woq_itr(
        dtype_opt_list=woq_dtypes,
        batch_opt_list=batch_opt,
        in_features_opt_list=in_features_opt,
        out_features_opt_list=out_features_opt,
        bias_opt_list=woq_bias_opt,
        input_dim_opt_list=input_dim_opt,
        time_out=20000,
    )
    @torch.inference_mode()
    def test_woq_linear_add_add_accuracy(self):
        """Test accuracy of WOQ linear + add + add against dq linear then out + add + add_2."""
        input_t, packed_weight_t, scale_t, dq_weight, bias_t = self._woq_setup(bias=self.data.with_bias)
        add_input = self.data.woq_add_input
        add_input_2 = self.data.woq_add_input_2
        zentorch_result = torch.ops.zentorch.zentorch_woq_linear_add_add(
            input_t, packed_weight_t, scale_t, None, add_input, add_input_2, bias_t
        )
        woq_dtype = DataTypes.get_torch_type(self.data.dtype)

        ref_model = WOQ_Linear_Add_Add_Model(dq_weight.to(woq_dtype), bias_t)
        pytorch_result = compiled_frozen_reference(
            ref_model, input_t, add_input, add_input_2
        )
        self._assert_woq_fused_output_sanity(zentorch_result, "zentorch_woq_linear_add_add")
        self.assertTrue(
            torch.allclose(zentorch_result, pytorch_result, rtol=self._fused_rtol, atol=self._fused_atol),
            "zentorch_woq_linear_add_add does not match dq linear then out + add + add_2.",
        )


if __name__ == "__main__":
    run_tests()
