# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import copy
import unittest
import torch
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
)


def quantize_model_pt2e(model, example_inputs):

    # TorchAO quantization
    from torchao.quantization.pt2e import move_exported_model_to_eval
    from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e
    from torchao.quantization.pt2e.quantizer.x86_inductor_quantizer import (
        X86InductorQuantizer,
        get_default_x86_inductor_quantization_config,
        default_quantizable_ops,
    )
    from torchao.quantization.pt2e.quantizer.composable_quantizer import (
        ComposableQuantizer,
    )

    def calibrate(model):
        move_exported_model_to_eval(model)
        # example_inputs is a tuple of 3 tensors, where first tensor is the input,
        # second tensor is the cat output, third tensor is the add tensor.
        with torch.no_grad():
            inputs = torch.randn_like(example_inputs[0])
            cat_output = torch.randn_like(example_inputs[1])
            add_tensor = torch.randn_like(example_inputs[2])
            model(inputs, cat_output, add_tensor)

    # Export model
    with torch.no_grad():
        exported_model = torch.export.export(
            model,
            example_inputs,
            strict=True,
        ).module()

    # Create quantizers
    quantizers = []

    original_default_quantizable_ops = default_quantizable_ops.copy()
    # We are not quantizing the mul op for DLRM-V2 model
    default_quantizable_ops.discard(torch.ops.aten.mul.Tensor)
    linear_quantizer = X86InductorQuantizer()
    linear_quantizer.set_global(
        get_default_x86_inductor_quantization_config(
            is_qat=False, is_dynamic=False, reduce_range=False
        )
    )
    quantizers.append(linear_quantizer)

    composable_quantizer = ComposableQuantizer(quantizers)

    # Prepare, calibrate, convert
    prepared_model = prepare_pt2e(exported_model, composable_quantizer)
    calibrate(prepared_model)
    quantized_model = convert_pt2e(prepared_model)

    default_quantizable_ops.clear()
    default_quantizable_ops.update(original_default_quantizable_ops)

    return quantized_model


class Model(torch.nn.Module):
    def __init__(self, input_features, output_features, mul_arg_pos, add_arg_pos):
        super().__init__()

        self.linear = torch.nn.Linear(input_features, output_features)
        self.mul_arg_pos = mul_arg_pos
        self.add_arg_pos = add_arg_pos

    def forward(self, x, cat_output, add_tensor):
        x = self.linear(x)
        if self.mul_arg_pos == 0:
            mul_output = torch.mul(x, cat_output)
        else:
            mul_output = torch.mul(cat_output, x)
        if self.add_arg_pos == 0:
            add_output = torch.add(mul_output, add_tensor)
        else:
            add_output = torch.add(add_tensor, mul_output)

        return add_output


@unittest.skipIf(not zentorch._C.is_avx512_supported(), "No bf16 support on hardware")
@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_Qlinear_Mul_Add_Model(QLinearTestCase):
    @torch.inference_mode()
    @QLinearTestCase.hypothesis_params_qlinear_itr(
        dtype_list=["float32", "bfloat16"], time_out=15000
    )
    def test_qlinear_mul_add_model(self, dtype):
        # Define position combinations for mul/add operands
        MUL_ADD_POSITIONS = {
            "mul_arg_first_add_arg_first": {"mul_arg_pos": 0, "add_arg_pos": 0},
            "mul_arg_first_add_arg_second": {"mul_arg_pos": 0, "add_arg_pos": 1},
            "mul_arg_second_add_arg_first": {"mul_arg_pos": 1, "add_arg_pos": 0},
            "mul_arg_second_add_arg_second": {"mul_arg_pos": 1, "add_arg_pos": 1},
        }
        for _, pos_config in MUL_ADD_POSITIONS.items():
            torch_dtype = self.data.get_torch_type(dtype)

            B, M, N = max(2, self.data.b), max(2, self.data.m), max(2, self.data.n)

            model = Model(
                M, N, pos_config["mul_arg_pos"], pos_config["add_arg_pos"]
            )
            if dtype == "bfloat16":
                model = model.to(torch.bfloat16)
            inputs = torch.randn(B, M, dtype=torch_dtype)
            cat_output = torch.randn(B, N, dtype=torch_dtype)
            add_tensor = torch.randn(B, N, dtype=torch_dtype)

            quantized_model = quantize_model_pt2e(
                model, (inputs, cat_output, add_tensor)
            )
            zentorch_qmodel = copy.deepcopy(quantized_model)

            inputs = torch.randn(B, M, dtype=torch_dtype)
            cat_output = torch.randn(B, N, dtype=torch_dtype)
            add_tensor = torch.randn(B, N, dtype=torch_dtype)

            native_output = quantized_model(inputs, cat_output, add_tensor)

            counters.clear()
            self.assertEqual(counters["zentorch"]["qlinear_mul_add"], 0)

            reset_dynamo()
            zentorch_qmodel = torch.compile(zentorch_qmodel, backend="zentorch")

            zentorch_output = zentorch_qmodel(inputs, cat_output, add_tensor)

            # TODO: Remove this once we have a proper fix for bfloat16 performance issue.
            # Hence doing the fusion only when the dtype is float32.
            if dtype == "float32":
                self.assertEqual(counters["zentorch"]["qlinear_mul_add"], 1)
            elif dtype == "bfloat16":
                self.assertEqual(counters["zentorch"]["qlinear_mul_add"], 0)

            # TODO: to be aligned with ZenDNN library on tensor generation and tolerances
            self.assertEqual(native_output, zentorch_output, atol=2e-2, rtol=1e-2)


if __name__ == "__main__":
    run_tests()
