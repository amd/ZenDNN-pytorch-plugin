# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import unittest
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).parent.parent))
from unittest_utils import (  # noqa: E402
    GroupMatmulTestCase,
    has_zentorch,
    run_tests,
    supported_dtypes,
)

FFN_ACTIVATIONS = []
if has_zentorch:
    from zentorch._utils import _SUPPORTED_MOE_ACTIVATIONS
    FFN_ACTIVATIONS = _SUPPORTED_MOE_ACTIVATIONS

# Tolerance per dtype (mirrors test_group_matmul.py). The fused W13 -> act -> W2
# chain accumulates error at reduced precision, so bf16/fp16 compare against an
# fp32 reference through the looser "fused_bf16" band while fp32 stays tight.
TOLERANCES = {
    torch.float32: {"atol": 1e-3, "rtol": 1e-3},
    torch.bfloat16: {"atol": 3e-2, "rtol": 3e-2},
    "fused_bf16": {"atol": 5e-1, "rtol": 5e-1},
    torch.float16: {"atol": 1e-2, "rtol": 1e-2},
}


def ffn_output_buffer(x):
    """ NaN-filled output buffer shaped like ``x`` """
    return torch.full_like(x, float("nan"))


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_FusedFFNConcat(GroupMatmulTestCase):
    """Single-expert (non-MoE) ``zentorch_fused_ffn_concat.out`` tests."""

    # ------------------------------------------------------------------
    # Reference helpers (mirror test_group_matmul.py).
    # ------------------------------------------------------------------

    def _reference_ffn(
        self, x, w13, w2, w13_bias, w2_bias, activation, compute_in_fp32
    ):
        x_s = [x]
        w13_s = [w13]
        w2_s = [w2]
        w13_bias_s = [w13_bias]
        w2_bias_s = [w2_bias]
        return self._reference_expert_outputs(x_s, w13_s, w13_bias_s, activation, w2_s, w2_bias_s, compute_in_fp32)[0]

    def _single_expert_weights(self, with_bias):
        """First-expert gated weights as the single-expert FFN's W13 / W2.

        ``w13_weights_gated[e]`` is ``[2*I, H]`` and ``w2_weights_gated[e]`` is
        ``[H, I]`` -- exactly the layout ``fused_ffn_concat`` expects.
        """
        w13 = self.data.w13_weights_gated[0]
        w2 = self.data.w2_weights_gated[0]
        w13_bias = self.data.w13_bias_gated[0] if with_bias else None
        w2_bias = self.data.w2_bias_gated[0] if with_bias else None
        return w13, w2, w13_bias, w2_bias

    def _run_op(self, x, w13, w2, w13_bias, w2_bias, activation):
        """Run the out-variant op; return (output, input_after_call)."""
        output = ffn_output_buffer(x)
        # Clone the input so an accidental in-place write can be detected.
        op_input = x.clone()
        torch.ops.zentorch.zentorch_fused_ffn_concat.out(
            output,
            op_input,
            w13,
            w2,
            w13_bias=w13_bias,
            w2_bias=w2_bias,
            activation=activation,
        )
        return output, op_input

    # ------------------------------------------------------------------
    # Accuracy: 2D input, all supported gated activations, +/- bias.
    # ------------------------------------------------------------------
    @GroupMatmulTestCase.hypothesis_params_group_matmul_itr(
        dtype_list=supported_dtypes,
    )
    @torch.inference_mode()
    def test_fused_ffn_accuracy_2d(self, dtype, with_bias):
        torch_dtype = self.data.get_torch_type(dtype)
        is_reduced_precision = torch_dtype in (torch.bfloat16, torch.float16)
        tol = (
            TOLERANCES["fused_bf16"]
            if is_reduced_precision
            else TOLERANCES[torch_dtype]
        )

        x = self.data.inputs[0]  # [M, H]
        w13, w2, w13_bias, w2_bias = self._single_expert_weights(with_bias)

        for activation in FFN_ACTIVATIONS:
            ref = self._reference_ffn(
                x, w13, w2, w13_bias, w2_bias, activation,
                compute_in_fp32=is_reduced_precision,
            )
            output, _ = self._run_op(
                x, w13, w2, w13_bias, w2_bias, activation
            )

            self.assertEqual(output.shape, ref.shape)
            self.assertFalse(
                torch.isnan(output).any(),
                f"output left uninitialized (activation={activation})",
            )
            actual = output.float() if is_reduced_precision else output
            self.assertEqual(actual, ref, **tol)

    # ------------------------------------------------------------------
    # Accuracy: 3D input [B, S, H] -- the op flattens to 2D internally.
    # ------------------------------------------------------------------
    @GroupMatmulTestCase.hypothesis_params_group_matmul_itr(
        dtype_list=supported_dtypes,
    )
    @torch.inference_mode()
    def test_fused_ffn_accuracy_3d(self, dtype, with_bias):
        torch_dtype = self.data.get_torch_type(dtype)
        is_reduced_precision = torch_dtype in (torch.bfloat16, torch.float16)
        tol = (
            TOLERANCES["fused_bf16"]
            if is_reduced_precision
            else TOLERANCES[torch_dtype]
        )

        # Reshape the [M, H] expert input to a 3D [1, M, H] activation.
        x = self.data.inputs[0].unsqueeze(0).contiguous()  # [1, M, H]
        w13, w2, w13_bias, w2_bias = self._single_expert_weights(with_bias)
        activation = "silu"

        ref = self._reference_ffn(
            x, w13, w2, w13_bias, w2_bias, activation,
            compute_in_fp32=is_reduced_precision,
        )
        output, _ = self._run_op(x, w13, w2, w13_bias, w2_bias, activation)

        self.assertEqual(output.shape, ref.shape)
        self.assertEqual(output.dim(), 3)
        self.assertFalse(torch.isnan(output).any())
        actual = output.float() if is_reduced_precision else output
        self.assertEqual(actual, ref, **tol)

    # ------------------------------------------------------------------
    # Unsupported activation strings must raise.
    # ------------------------------------------------------------------
    @GroupMatmulTestCase.hypothesis_params_group_matmul_itr(
        dtype_list=supported_dtypes,
    )
    @torch.inference_mode()
    def test_unsupported_activation(self, dtype):
        x = self.data.inputs[0]
        w13, w2, _, _ = self._single_expert_weights(with_bias=False)
        output = ffn_output_buffer(x)

        for bad_activation in ("relu", "tanh"):
            with self.assertRaisesRegex(RuntimeError, "unsupported activation"):
                torch.ops.zentorch.zentorch_fused_ffn_concat.out(
                    output,
                    x.clone(),
                    w13,
                    w2,
                    activation=bad_activation,
                )


if __name__ == "__main__":
    run_tests()
