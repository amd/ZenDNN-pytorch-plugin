# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

"""Model test for the qlinear pattern-matcher pass (`replace_with_zentorch_qops`).

The pass folds a PT2E `dequantize -> zentorch_addmm/mm` chain into a single
`zentorch_qlinear`. A Linear is quantized with X86InductorQuantizer or
XNNPACKQuantizer and run through the real compile flow with
`backend="inductor"` (int8-GEMM reference) and `backend="zentorch"`, asserting
the counter, numeric parity (Pillar 1), and -- on cpp_wrapper runs -- the
qlinear AOTI shim with no `aten.full`-derived `*full*` kernel (Pillar 2).
"""

import copy
import unittest
import torch
from torch.testing import FileCheck
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from unittest_utils import (  # noqa: 402
    QLinearTestCase,
    counters,
    has_zentorch,
    zentorch,
    run_tests,
    reset_dynamo,
    freeze_opt,
    cpp_wrapper_opt,
    bias_opt,
    test_with_freeze_opt_and_cpp_wrapper,
)


def quantize_linear_pt2e(model, example_inputs, quantizer_type="x86"):
    """PT2E-quantize a Linear.

    ``x86`` is the X86InductorQuantizer recipe the pass matches: uint8
    per-tensor activations and int8 per-channel weights. ``xnnpack`` uses
    dynamic int8 activations and static per-tensor int8 weights. A bfloat16
    Linear makes both dequants set out_dtype to bfloat16.
    """
    from torchao.quantization.pt2e import move_exported_model_to_eval
    from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e

    if quantizer_type == "x86":
        from torchao.quantization.pt2e.quantizer.x86_inductor_quantizer import (
            X86InductorQuantizer,
            get_default_x86_inductor_quantization_config,
        )

        quantizer = X86InductorQuantizer()
        config = get_default_x86_inductor_quantization_config(
            is_qat=False, is_dynamic=False, reduce_range=False
        )
    elif quantizer_type == "xnnpack":
        from torchao.testing.pt2e._xnnpack_quantizer import (
            XNNPACKQuantizer,
            get_symmetric_quantization_config,
        )

        quantizer = XNNPACKQuantizer()
        config = get_symmetric_quantization_config(
            is_per_channel=False, is_dynamic=True
        )
    else:
        raise ValueError(f"Unsupported quantizer type: {quantizer_type!r}")

    quantizer.set_global(config)

    with torch.no_grad():
        exported_model = torch.export.export(
            model, example_inputs, strict=True
        ).module()

    prepared_model = prepare_pt2e(exported_model, quantizer)
    move_exported_model_to_eval(prepared_model)

    with torch.no_grad():
        prepared_model(*[torch.randn_like(t) for t in example_inputs])

    return convert_pt2e(prepared_model)


class Model(torch.nn.Module):
    def __init__(self, in_features, out_features, bias):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        return self.linear(x)


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
@unittest.skipIf(
    has_zentorch and not zentorch._C.is_avx512_supported(),
    "INT8 qlinear kernels require AVX512",
)
class Test_Qlinear_Pattern_Matcher_Model(QLinearTestCase):
    @torch.inference_mode()
    @QLinearTestCase.hypothesis_params_qlinear_itr(
        input_dim_opt_list=[2, 3],
        dtype_list=["float32", "bfloat16"],
        bias_opt_list=bias_opt,
        freeze_list=freeze_opt,
        cpp_wrapper_opt_list=cpp_wrapper_opt,
        time_out=60000,  # headroom for cold cpp_wrapper compiles
    )
    def test_qlinear_pattern_matcher_model(
        self, dtype, input_dim, bias_opt_idx, freeze_opt, cpp_wrapper
    ):
        torch_dtype = self.data.get_torch_type(dtype)
        M, N = max(2, self.data.m), max(2, self.data.n)
        B = max(2, self.data.b)
        # 3-D exercises the view-wrapped roots.
        shape = (B, M) if input_dim == 2 else (B, max(2, self.data.p), M)

        model = Model(M, N, bias=bool(bias_opt_idx))
        if dtype == "bfloat16":
            model = model.to(torch.bfloat16)
        model.eval()

        example = torch.randn(*shape, dtype=torch_dtype)
        native_qmodel = quantize_linear_pt2e(model, (example,))
        zentorch_qmodel = copy.deepcopy(native_qmodel)

        inputs = torch.randn(*shape, dtype=torch_dtype)

        # Reference: inductor int8 GEMM (eager would use a diverging kernel).
        reset_dynamo()
        native_output = torch.compile(native_qmodel, backend="inductor")(inputs)

        counters.clear()
        self.assertEqual(counters["zentorch"]["zentorch_qlinear"], 0)

        reset_dynamo()
        zentorch_compiled = torch.compile(zentorch_qmodel, backend="zentorch")
        # Helper returns (output, generated C++ or None).
        zentorch_output, cpp_code = test_with_freeze_opt_and_cpp_wrapper(
            zentorch_compiled, (inputs,), freeze_opt, cpp_wrapper
        )

        self.assertEqual(counters["zentorch"]["zentorch_qlinear"], 1)
        # TODO: align with ZenDNN library on tensor gen and tolerances.
        self.assertEqual(native_output, zentorch_output, atol=1e-2, rtol=1e-2)

        # Scalar qparams must stay interned constants, never an aten.full that
        # Inductor materialises into a per-call *full* kernel.
        if cpp_wrapper:
            FileCheck().check("aoti_torch_cpu_zentorch_qlinear").run(cpp_code)
            self.assertNotRegex(cpp_code, r"fused\w*full")

    # f32 activations into a bf16-quantized Linear: the input dequant casts to
    # bf16, so the qlinear output must follow out_dtype, not the input dtype.
    @torch.inference_mode()
    @QLinearTestCase.hypothesis_params_qlinear_itr(
        input_dim_opt_list=[2, 3],
        dtype_list=["bfloat16"],
        bias_opt_list=bias_opt,
    )
    def test_qlinear_f32_input_bf16_dequant(self, input_dim, bias_opt_idx):
        M, N = max(2, self.data.m), max(2, self.data.n)
        B = max(2, self.data.b)
        shape = (B, M) if input_dim == 2 else (B, max(2, self.data.p), M)

        model = Model(M, N, bias=bool(bias_opt_idx)).to(torch.bfloat16).eval()
        example = torch.randn(*shape, dtype=torch.bfloat16)
        native_qmodel = quantize_linear_pt2e(model, (example,))
        zentorch_qmodel = copy.deepcopy(native_qmodel)

        inputs = torch.randn(*shape, dtype=torch.float32)

        reset_dynamo()
        native_output = torch.compile(native_qmodel, backend="inductor")(inputs)

        counters.clear()
        reset_dynamo()
        zentorch_output = torch.compile(zentorch_qmodel, backend="zentorch")(inputs)

        self.assertEqual(counters["zentorch"]["zentorch_qlinear"], 1)
        self.assertEqual(zentorch_output.dtype, torch.bfloat16)
        self.assertEqual(native_output, zentorch_output, atol=1e-2, rtol=1e-2)

    # Same dtype check on the XNNPACK graph. The x86 test never builds it:
    # int8 dynamic activations, a per-tensor weight dequant, and a permute.
    @torch.inference_mode()
    def test_qlinear_xnnpack_f32_input_bf16_dequant(self):
        shape = (2, 16)
        model = Model(16, 16, bias=True).to(torch.bfloat16).eval()
        example = torch.randn(*shape, dtype=torch.bfloat16)
        native_qmodel = quantize_linear_pt2e(
            model, (example,), quantizer_type="xnnpack"
        )
        zentorch_qmodel = copy.deepcopy(native_qmodel)

        inputs = torch.randn(*shape, dtype=torch.float32)

        reset_dynamo()
        native_output = torch.compile(native_qmodel, backend="inductor")(inputs)

        counters.clear()
        reset_dynamo()
        zentorch_output = torch.compile(zentorch_qmodel, backend="zentorch")(inputs)

        self.assertEqual(counters["zentorch"]["zentorch_qlinear"], 1)
        self.assertEqual(zentorch_output.dtype, torch.bfloat16)
        self.assertEqual(native_output, zentorch_output, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    run_tests()
