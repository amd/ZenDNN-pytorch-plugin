# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

"""Model test for the qlinear pattern-matcher pass (`replace_with_zentorch_qops`).

The pass folds a PT2E `dequantize -> zentorch_addmm/mm -> ...` chain into a
single `zentorch_qlinear`. This drives it through the real compile flow: a Linear
is X86Inductor-quantized, then run with `backend="inductor"` (the int8-GEMM
reference) and `backend="zentorch"`. Hypothesis parameterises dtype,
`input_dim` {2, 3} (2-D addmm/mm roots and 3-D view-wrapped roots), bias,
freezing, and cpp_wrapper, and each example asserts:

  * exactly one `zentorch_qlinear` replaced the chain (counter),
  * numeric parity with the inductor reference, and
  * no `aten.full`-derived `*fused*full*` kernel -- the rewrite interns scalar
    qparams as `get_attr` constants instead of materialising them per call.
"""

import copy
import unittest
import torch
import sys
from pathlib import Path

from torch._inductor.utils import run_and_get_code

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


def quantize_linear_pt2e(model, example_inputs):
    """PT2E-quantize `model` with the X86InductorQuantizer (uint8 per-tensor
    activations, int8 per-channel weights) -- the recipe that produces the
    `dequantize -> addmm/mm` chain the pass matches."""
    from torchao.quantization.pt2e import move_exported_model_to_eval
    from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e
    from torchao.quantization.pt2e.quantizer.x86_inductor_quantizer import (
        X86InductorQuantizer,
        get_default_x86_inductor_quantization_config,
    )

    with torch.no_grad():
        exported_model = torch.export.export(
            model, example_inputs, strict=True
        ).module()

    quantizer = X86InductorQuantizer()
    quantizer.set_global(
        get_default_x86_inductor_quantization_config(
            is_qat=False, is_dynamic=False, reduce_range=False
        )
    )
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
        # 3-D adds the view-wrapped roots that fold into the same N-D qlinear.
        shape = (B, M) if input_dim == 2 else (B, max(2, self.data.p), M)

        model = Model(M, N, bias=bool(bias_opt_idx))
        if dtype == "bfloat16":
            model = model.to(torch.bfloat16)
        model.eval()

        example = torch.randn(*shape, dtype=torch_dtype)
        quantized_model = quantize_linear_pt2e(model, (example,))
        native_qmodel = copy.deepcopy(quantized_model)
        zentorch_qmodel = copy.deepcopy(quantized_model)

        inputs = torch.randn(*shape, dtype=torch_dtype)

        # Reference: inductor int8 GEMM (eager would use a diverging kernel).
        reset_dynamo()
        native_output = torch.compile(native_qmodel, backend="inductor")(inputs)

        counters.clear()
        self.assertEqual(counters["zentorch"]["zentorch_qlinear"], 0)

        reset_dynamo()
        zentorch_compiled = torch.compile(zentorch_qmodel, backend="zentorch")
        # Apply freezing/cpp_wrapper via the shared helper, wrapped in
        # run_and_get_code to capture the generated kernels.
        zentorch_output, codes = run_and_get_code(
            test_with_freeze_opt_and_cpp_wrapper,
            zentorch_compiled,
            (inputs,),
            freeze_opt,
            cpp_wrapper,
        )

        # The dq -> zentorch_addmm/mm chain folds into exactly one qlinear
        # (bias -> addmm root, no bias -> mm root; 3-D adds the view wrappers).
        self.assertEqual(counters["zentorch"]["zentorch_qlinear"], 1)
        # Scalar qparams must intern as get_attr constants, never an aten.full
        # that Inductor fuses into a per-call *full* kernel.
        self.assertTrue(codes, "no generated code captured")
        self.assertNotRegex("\n".join(codes), r"fused\w*full")
        # TODO: align with ZenDNN library on tensor gen and tolerances.
        self.assertEqual(native_output, zentorch_output, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    run_tests()
