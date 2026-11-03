# ******************************************************************************
# Copyright (c) 2024-2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import unittest
import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from unittest_utils import (  # noqa: 402
    K_RANGE,
    MMTestCase,
    Zentorch_TestCase,
    _clear_weight_cache,
    has_zentorch,
    reset_dynamo,
    run_tests,
    skip_test_pt_2_0,
    supported_dtypes,
    update_supported_dtypes,
)
supported_dtypes = update_supported_dtypes(supported_dtypes, "zentorch_linear")


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_Linear_Unary(MMTestCase):

    def get_and_compare_linear_output(
        self, ref_output, input, weight, bias=None, post_op="none"
    ):
        # The callers pass a column-major [N, K] weight (y.t()); the row-major
        # copy covers the nn.Linear layout.
        row_major_weight = weight.contiguous()
        for layout, layout_weight in (
            ("column_major", weight),
            ("row_major", row_major_weight),
        ):
            zen_output = torch.ops.zentorch.zentorch_linear_unary(
                input, layout_weight, bias, post_op=post_op
            )
            self.assertEqual(
                ref_output,
                zen_output,
                atol=1e-2,
                rtol=1e-2,
                msg=f"weight_layout={layout}",
            )
        prepacked_weight = torch.ops.zentorch.zentorch_weight_prepack_for_linear(weight)
        zen_output_prepacked = torch.ops.zentorch.zentorch_linear_unary(
            input, prepacked_weight, bias, post_op=post_op, is_weight_prepacked=True
        )
        self.assertEqual(ref_output, zen_output_prepacked, atol=1e-2, rtol=1e-2)

    @MMTestCase.hypothesis_params_mm_itr(dtype_list=supported_dtypes)
    @unittest.skipIf(skip_test_pt_2_0, "Skipping test due to PT2.0 instability")
    def test_linear_unary_with_bias(self):
        reset_dynamo()
        output_1 = torch.nn.functional.linear(
            self.data.x, self.data.y.t(), self.data.input1d
        )
        self.get_and_compare_linear_output(
            output_1, self.data.x, self.data.y.t(), self.data.input1d
        )

    @MMTestCase.hypothesis_params_mm_itr(dtype_list=supported_dtypes)
    @unittest.skipIf(skip_test_pt_2_0, "Skipping test due to PT2.0 instability")
    def test_linear_unary_with_no_bias(self):
        reset_dynamo()
        output_1 = torch.nn.functional.linear(self.data.x, self.data.y.t())
        self.get_and_compare_linear_output(output_1, self.data.x, self.data.y.t())

    @MMTestCase.hypothesis_params_mm_itr(dtype_list=supported_dtypes)
    @unittest.skipIf(skip_test_pt_2_0, "Skipping test due to PT2.0 instability")
    def test_linear_unary_with_bias_relu(self):
        reset_dynamo()
        output_1 = torch.relu(
            torch.nn.functional.linear(self.data.x, self.data.y.t(), self.data.input1d)
        )
        self.get_and_compare_linear_output(
            output_1, self.data.x, self.data.y.t(), self.data.input1d, post_op="relu"
        )

    @MMTestCase.hypothesis_params_mm_itr(dtype_list=supported_dtypes)
    @unittest.skipIf(skip_test_pt_2_0, "Skipping test due to PT2.0 instability")
    def test_linear_unary_with_no_bias_relu(self):
        reset_dynamo()
        output_1 = torch.relu(torch.nn.functional.linear(self.data.x, self.data.y.t()))
        self.get_and_compare_linear_output(
            output_1, self.data.x, self.data.y.t(), post_op="relu"
        )

    @MMTestCase.hypothesis_params_mm_itr(dtype_list=supported_dtypes)
    @unittest.skipIf(skip_test_pt_2_0, "Skipping test due to PT2.0 instability")
    def test_linear_unary_with_bias_gelu_tanh(self):
        reset_dynamo()
        output_1 = torch.nn.GELU(approximate="tanh")(
            torch.nn.functional.linear(self.data.x, self.data.y.t(), self.data.input1d)
        )
        self.get_and_compare_linear_output(
            output_1,
            self.data.x,
            self.data.y.t(),
            self.data.input1d,
            post_op="gelu_tanh",
        )

    @MMTestCase.hypothesis_params_mm_itr(dtype_list=supported_dtypes)
    @unittest.skipIf(skip_test_pt_2_0, "Skipping test due to PT2.0 instability")
    def test_linear_unary_with_no_bias_gelu_tanh(self):
        reset_dynamo()
        output_1 = torch.nn.GELU(approximate="tanh")(
            torch.nn.functional.linear(self.data.x, self.data.y.t())
        )
        self.get_and_compare_linear_output(
            output_1, self.data.x, self.data.y.t(), post_op="gelu_tanh"
        )

    @MMTestCase.hypothesis_params_mm_itr(dtype_list=supported_dtypes)
    @unittest.skipIf(skip_test_pt_2_0, "Skipping test due to PT2.0 instability")
    def test_linear_unary_with_bias_gelu_erf(self):
        reset_dynamo()
        output_1 = torch.nn.GELU()(
            torch.nn.functional.linear(self.data.x, self.data.y.t(), self.data.input1d)
        )
        self.get_and_compare_linear_output(
            output_1,
            self.data.x,
            self.data.y.t(),
            self.data.input1d,
            post_op="gelu_erf",
        )

    @MMTestCase.hypothesis_params_mm_itr(dtype_list=supported_dtypes)
    @unittest.skipIf(skip_test_pt_2_0, "Skipping test due to PT2.0 instability")
    def test_linear_unary_with_no_bias_gelu_erf(self):
        reset_dynamo()
        output_1 = torch.nn.GELU()(
            torch.nn.functional.linear(self.data.x, self.data.y.t())
        )
        self.get_and_compare_linear_output(
            output_1, self.data.x, self.data.y.t(), post_op="gelu_erf"
        )

    @MMTestCase.hypothesis_params_mm_itr(dtype_list=supported_dtypes)
    @unittest.skipIf(skip_test_pt_2_0, "Skipping test due to PT2.0 instability")
    def test_linear_unary_with_bias_silu(self):
        reset_dynamo()
        output_1 = torch.nn.functional.silu(
            torch.nn.functional.linear(self.data.x, self.data.y.t(), self.data.input1d)
        )
        self.get_and_compare_linear_output(
            output_1, self.data.x, self.data.y.t(), self.data.input1d, post_op="silu"
        )

    @MMTestCase.hypothesis_params_mm_itr(dtype_list=supported_dtypes)
    @unittest.skipIf(skip_test_pt_2_0, "Skipping test due to PT2.0 instability")
    def test_linear_unary_with_no_bias_silu(self):
        reset_dynamo()
        output_1 = torch.nn.functional.silu(
            torch.nn.functional.linear(self.data.x, self.data.y.t())
        )
        self.get_and_compare_linear_output(
            output_1, self.data.x, self.data.y.t(), post_op="silu"
        )

    @MMTestCase.hypothesis_params_mm_itr(dtype_list=supported_dtypes)
    @unittest.skipIf(skip_test_pt_2_0, "Skipping test due to PT2.0 instability")
    def test_linear_unary_with_bias_sigmoid(self):
        reset_dynamo()
        output_1 = torch.nn.functional.sigmoid(
            torch.nn.functional.linear(self.data.x, self.data.y.t(), self.data.input1d)
        )
        self.get_and_compare_linear_output(
            output_1, self.data.x, self.data.y.t(), self.data.input1d, post_op="sigmoid"
        )

    @MMTestCase.hypothesis_params_mm_itr(dtype_list=supported_dtypes)
    @unittest.skipIf(skip_test_pt_2_0, "Skipping test due to PT2.0 instability")
    def test_linear_unary_with_no_bias_sigmoid(self):
        reset_dynamo()
        output_1 = torch.nn.functional.sigmoid(
            torch.nn.functional.linear(self.data.x, self.data.y.t())
        )
        self.get_and_compare_linear_output(
            output_1, self.data.x, self.data.y.t(), post_op="sigmoid"
        )

    @MMTestCase.hypothesis_params_mm_itr(dtype_list=supported_dtypes)
    @unittest.skipIf(skip_test_pt_2_0, "Skipping test due to PT2.0 instability")
    def test_linear_unary_with_bias_tanh(self):
        reset_dynamo()
        output_1 = torch.nn.functional.tanh(
            torch.nn.functional.linear(self.data.x, self.data.y.t(), self.data.input1d)
        )
        self.get_and_compare_linear_output(
            output_1, self.data.x, self.data.y.t(), self.data.input1d, post_op="tanh"
        )

    @MMTestCase.hypothesis_params_mm_itr(dtype_list=supported_dtypes)
    @unittest.skipIf(skip_test_pt_2_0, "Skipping test due to PT2.0 instability")
    def test_linear_unary_with_no_bias_tanh(self):
        reset_dynamo()
        output_1 = torch.nn.functional.tanh(
            torch.nn.functional.linear(self.data.x, self.data.y.t())
        )
        self.get_and_compare_linear_output(
            output_1, self.data.x, self.data.y.t(), post_op="tanh"
        )

    @MMTestCase.hypothesis_params_mm_itr(dtype_list=supported_dtypes)
    def test_linear_unary_out_variant(self):
        reset_dynamo()
        tol = 1e-2
        expected = torch.ops.zentorch.zentorch_linear_unary(
            self.data.x, self.data.y.t(), self.data.input1d, post_op="relu"
        )
        row_major_weight = self.data.y.t().contiguous()
        for layout, weight in (
            ("column_major", self.data.y.t()),
            ("row_major", row_major_weight),
        ):
            out = torch.empty_like(expected)
            torch.ops.zentorch.zentorch_linear_unary.out(
                self.data.x, weight, self.data.input1d, post_op="relu", out=out
            )
            self.assertEqual(
                expected, out, atol=tol, rtol=tol, msg=f"weight_layout={layout}"
            )


# A square weight has the same shape as its transpose, so only the values show
# whether input @ weight^T was computed. Hypothesis draws dtype, m and k; each
# test builds a k x k weight from the generated k.
def square_weight_params():
    return MMTestCase.hypothesis_params_mm_itr(
        dtype_list=supported_dtypes, kRange=K_RANGE
    )


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_Linear_Square_Weight(MMTestCase):

    def tolerances(self, dtype):
        if dtype == torch.float32:
            return {"atol": 1e-3, "rtol": 1e-3}
        return {"atol": 5e-2, "rtol": 5e-2}

    def layout_weight(self, weight, layout):
        if layout == "column_major":
            return weight.t().contiguous().t()
        if layout == "prepacked":
            return torch.ops.zentorch.zentorch_weight_prepack_for_linear(weight)
        return weight

    def cases(self, n_extra=0):
        # The square size comes from the generated data (k); the weight is
        # built here because the generated y is [k, n]. ZenDNN caches weights
        # by data pointer, so the cache is cleared before every case.
        x = self.data.x
        dtype = x.dtype
        m, k = x.shape
        torch.manual_seed(100 * m + k)
        for layout in ("row_major", "column_major", "prepacked"):
            for with_bias in (False, True):
                _clear_weight_cache()
                weight = (torch.randn(k, k) / k**0.5).to(dtype)
                bias = torch.randn(k).to(dtype) if with_bias else None
                extra = [torch.randn(m, k).to(dtype) for _ in range(n_extra)]
                ref = torch.nn.functional.linear(
                    x.float(), weight.float(), None if bias is None else bias.float()
                )
                yield {"layout": layout, "bias": with_bias}, (
                    dtype,
                    x,
                    self.layout_weight(weight, layout),
                    layout == "prepacked",
                    bias,
                    extra,
                    ref,
                )

    @square_weight_params()
    @unittest.skipIf(skip_test_pt_2_0, "Skipping test due to PT2.0 instability")
    def test_linear_unary_square_weight(self):
        reset_dynamo()
        for label, (dtype, x, weight, prepacked, bias, _, ref) in self.cases():
            with self.subTest(**label):
                out = torch.ops.zentorch.zentorch_linear_unary(
                    x, weight, bias, is_weight_prepacked=prepacked, post_op="relu"
                )
                self.assertEqual(
                    out.float(), torch.relu(ref), **self.tolerances(dtype)
                )

    @square_weight_params()
    @unittest.skipIf(skip_test_pt_2_0, "Skipping test due to PT2.0 instability")
    def test_linear_unary_out_square_weight(self):
        reset_dynamo()
        for label, (dtype, x, weight, prepacked, bias, _, ref) in self.cases():
            with self.subTest(**label):
                out = torch.empty(ref.shape, dtype=dtype)
                torch.ops.zentorch.zentorch_linear_unary.out(
                    x, weight, bias, is_weight_prepacked=prepacked, out=out
                )
                self.assertEqual(out.float(), ref, **self.tolerances(dtype))

    @square_weight_params()
    @unittest.skipIf(skip_test_pt_2_0, "Skipping test due to PT2.0 instability")
    def test_linear_unary_binary_square_weight(self):
        reset_dynamo()
        for label, (dtype, x, weight, prepacked, bias, extra, ref) in self.cases(1):
            with self.subTest(**label):
                (residual,) = extra
                out = torch.ops.zentorch.zentorch_linear_unary_binary(
                    x,
                    weight,
                    residual,
                    bias,
                    is_weight_prepacked=prepacked,
                    post_op_1="relu",
                    post_op_2="add",
                )
                self.assertEqual(
                    out.float(),
                    torch.relu(ref) + residual.float(),
                    **self.tolerances(dtype),
                )

    @square_weight_params()
    @unittest.skipIf(skip_test_pt_2_0, "Skipping test due to PT2.0 instability")
    def test_linear_binary_binary_square_weight(self):
        reset_dynamo()
        for label, (dtype, x, weight, prepacked, bias, extra, ref) in self.cases(2):
            with self.subTest(**label):
                scale, residual = extra
                out = torch.ops.zentorch.zentorch_linear_binary_binary(
                    x,
                    weight,
                    scale,
                    residual,
                    bias,
                    is_weight_prepacked=prepacked,
                    post_op_1="mul",
                    post_op_2="add",
                )
                self.assertEqual(
                    out.float(),
                    ref * scale.float() + residual.float(),
                    **self.tolerances(dtype),
                )

    @square_weight_params()
    @unittest.skipIf(skip_test_pt_2_0, "Skipping test due to PT2.0 instability")
    def test_mm_square_weight(self):
        # The same square tensor is multiplied untransposed by mm and addmm.
        reset_dynamo()
        x = self.data.x
        dtype = x.dtype
        m, k = x.shape
        torch.manual_seed(100 * m + k)
        weight = (torch.randn(k, k) / k**0.5).to(dtype)
        bias = torch.randn(k).to(dtype)
        ref = x.float() @ weight.float()
        self.assertEqual(
            torch.ops.zentorch.zentorch_mm(x, weight).float(),
            ref,
            **self.tolerances(dtype),
        )
        self.assertEqual(
            torch.ops.zentorch.zentorch_addmm_1dbias(bias, x, weight).float(),
            ref + bias.float(),
            **self.tolerances(dtype),
        )

    def test_weight_without_unit_stride_raises(self):
        # ZenDNN takes one leading dimension per matrix, so a 2-d weight with
        # no unit-stride dimension must be rejected, not computed wrongly.
        x = torch.randn(7, 40)
        base = torch.randn(96, 80)
        with self.assertRaises(RuntimeError):
            torch.ops.zentorch.zentorch_linear_unary(x, base[:48, ::2], None)
        with self.assertRaises(RuntimeError):
            torch.ops.zentorch.zentorch_mm(x, base[:40, ::2])


if __name__ == "__main__":
    run_tests()
