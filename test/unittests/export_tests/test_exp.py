# ******************************************************************************
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import torch
from torch import nn
import zentorch
import sys
import unittest
from pathlib import Path
import os  # noqa: E402
import tempfile
from torch._inductor import config

sys.path.append(str(Path(__file__).parent.parent))
from unittest_utils import (  # noqa: 402
    AddmmTestCase,
    DataTypes,
    has_zentorch_full,
    run_tests,
    supported_dtypes,
    counters,
)

# export_optimize_pass lives in libzentorch.so, so it is absent when only the
# portable library loaded. Evaluated here at import time, before the class-level
# skip below can take effect.
ind_conf = (
    {"joint_custom_post_pass": zentorch.export_optimize_pass}
    if has_zentorch_full
    else {}
)


# create simple linear model
class SimpleLinearModel(nn.Module):
    def __init__(self):
        super(SimpleLinearModel, self).__init__()
        self.linear = nn.Linear(10, 10)
        self.silu = nn.SiLU()
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()
        self.gelu_0 = nn.GELU(approximate="tanh")
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.silu(self.linear(x))  # linear + silu
        x = x * self.silu(self.linear(x))  # linear + silu + mul
        x = x * self.linear(x) + x  # linear + mul + add
        x = self.relu(self.linear(x))  # linear + relu
        x = self.gelu(self.linear(x))  # linear + gelu
        x = self.gelu_0(self.linear(x))  # linear + gelu_tanh
        x = self.sigmoid(self.linear(x))  # linear + sigmoid
        x = self.tanh(self.linear(x))  # linear + tanh
        x = self.linear(x) + x  # linear + add
        x = self.linear(x) * x  # linear + mul
        x = self.linear(x) + x + x  # linear + add + add
        return x


# create qkv fusion model
class QKVFusionModel(nn.Module):
    def __init__(self, dtype, input_dim, hidden_dim):
        super(QKVFusionModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Three parallel linear layers representing Query, Key, Value projections
        self.linear_q = nn.Linear(input_dim, hidden_dim, bias=True, dtype=dtype)
        self.linear_k = nn.Linear(input_dim, hidden_dim, bias=True, dtype=dtype)
        self.linear_v = nn.Linear(input_dim, hidden_dim, bias=True, dtype=dtype)

    def forward(self, x):
        # Q = X @ Wq
        q = self.linear_q(x)
        # K = X @ Wk
        k = self.linear_k(x)
        # V = X @ Wv
        v = self.linear_v(x)

        # Return concatenated for easy comparison
        return torch.cat([q, k, v], dim=-1)


# Minimal single-fusion models used to exercise each linear `.out` variant
# (lowered to its AOTI out-variant shim) on the export path.
class LinearUnaryNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, x):
        return torch.nn.functional.relu(self.linear(x))


class LinearUnaryBinaryNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, x, binary):
        return torch.nn.functional.silu(self.linear(x)) * binary


class LinearBinaryBinaryNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, x, binary_1, binary_2):
        return self.linear(x) * binary_1 + binary_2


@unittest.skipUnless(
    has_zentorch_full, "Export pass requires the full zentorch backend"
)
class TestExport(AddmmTestCase):
    @torch.inference_mode()
    def test_export(self):
        my_model = SimpleLinearModel().eval()
        example_arg_1 = torch.rand(6, 10, device="cpu")
        exp_model = torch.export.export(my_model, args=(example_arg_1,))
        output_path = torch._inductor.aoti_compile_and_package(
            exp_model,
            package_path=os.path.join(os.getcwd(), "model.pt2"),
        )
        counters.clear()
        self.assertEqual(counters["zentorch"]["zentorch_linear_gelu_erf"], 0)
        self.assertEqual(counters["zentorch"]["zentorch_linear_gelu_tanh"], 0)
        self.assertEqual(counters["zentorch"]["zentorch_linear_relu"], 0)
        self.assertEqual(counters["zentorch"]["zentorch_linear_silu"], 0)
        self.assertEqual(counters["zentorch"]["zentorch_linear_tanh"], 0)
        self.assertEqual(counters["zentorch"]["zentorch_linear_sigmoid"], 0)
        self.assertEqual(counters["zentorch"]["zentorch_linear_add"], 0)
        self.assertEqual(counters["zentorch"]["zentorch_linear_mul"], 0)
        self.assertEqual(counters["zentorch"]["zentorch_linear_silu_mul"], 0)
        self.assertEqual(counters["zentorch"]["zentorch_linear_add_add"], 0)
        self.assertEqual(counters["zentorch"]["zentorch_linear_mul_add"], 0)
        output_path_z = torch._inductor.aoti_compile_and_package(
            exp_model,
            package_path=os.path.join(os.getcwd(), "model_z.pt2"),
            inductor_configs=ind_conf,
        )
        exported_model = torch._inductor.aoti_load_package(output_path)
        exported_model_z = torch._inductor.aoti_load_package(output_path_z)
        inp_1 = torch.rand(6, 10, device="cpu")
        output = exported_model(inp_1)
        output_z = exported_model_z(inp_1)
        self.assertEqual(counters["zentorch"]["zentorch_linear_gelu_erf"], 1)
        self.assertEqual(counters["zentorch"]["zentorch_linear_gelu_tanh"], 1)
        self.assertEqual(counters["zentorch"]["zentorch_linear_relu"], 1)
        self.assertEqual(counters["zentorch"]["zentorch_linear_silu"], 1)
        self.assertEqual(counters["zentorch"]["zentorch_linear_tanh"], 1)
        self.assertEqual(counters["zentorch"]["zentorch_linear_sigmoid"], 1)
        self.assertEqual(counters["zentorch"]["zentorch_linear_add"], 1)
        self.assertEqual(counters["zentorch"]["zentorch_linear_mul"], 1)
        self.assertEqual(counters["zentorch"]["zentorch_linear_silu_mul"], 1)
        self.assertEqual(counters["zentorch"]["zentorch_linear_add_add"], 1)
        self.assertEqual(counters["zentorch"]["zentorch_linear_mul_add"], 1)
        self.assertEqual(output, output_z, atol=1e-2, rtol=1e-2)

    # Added higher time_out for this test as it was failing with deadline exceeded error frequently
    # TODO: Investigate why it requires higher deadline: ZENAI-3638
    @AddmmTestCase.hypothesis_params_addmm_itr(
        dtype_list=supported_dtypes,
        time_out=25000,
    )
    @torch.inference_mode()
    def test_export_qkv(self, dtype):
        config.freezing = True
        my_model = QKVFusionModel(
            dtype=DataTypes.get_torch_type(dtype),
            input_dim=self.data.k,
            hidden_dim=self.data.n,
        ).eval()
        example_args_1 = torch.randn(
            self.data.b, self.data.m, self.data.k, dtype=DataTypes.get_torch_type(dtype)
        )
        exp_model = torch.export.export(my_model, args=(example_args_1,))
        output_path = torch._inductor.aoti_compile_and_package(
            exp_model,
            package_path=os.path.join(os.getcwd(), "model.pt2"),
        )
        counters.clear()
        self.assertEqual(counters["zentorch"]["qkv_fusion_linear"], 0)
        output_path_z = torch._inductor.aoti_compile_and_package(
            exp_model,
            package_path=os.path.join(os.getcwd(), "model_z.pt2"),
            inductor_configs=ind_conf,
        )
        exported_model = torch._inductor.aoti_load_package(output_path)
        exported_model_z = torch._inductor.aoti_load_package(output_path_z)
        inp_1 = torch.randn(
            self.data.b, self.data.m, self.data.k, dtype=DataTypes.get_torch_type(dtype)
        )
        output = exported_model(inp_1)
        output_z = exported_model_z(inp_1)
        self.assertEqual(counters["zentorch"]["qkv_fusion_linear"], 1)
        self.assertEqual(output, output_z, atol=1e-2, rtol=1e-2)
        config.freezing = False

    # Asserts the linear was lowered to its `.out` shim on the AOTI path. Caching
    # is disabled since the lowering counter is short-circuited on a cache hit.
    @config.patch(force_disable_caches=True)
    @torch.inference_mode()
    def test_linear_unary_out_variant_export(self):
        model = LinearUnaryNet().eval()
        inp = torch.rand(6, 10)
        native_output = model(inp)

        exp_model = torch.export.export(model, args=(inp,))
        counters.clear()
        self.assertEqual(counters["zentorch"]["zentorch_linear_unary_out"], 0)
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = torch._inductor.aoti_compile_and_package(
                exp_model,
                package_path=os.path.join(tmp_dir, "linear_unary_out.pt2"),
                inductor_configs=ind_conf,
            )
            self.assertEqual(counters["zentorch"]["zentorch_linear_unary_out"], 1)
            exported_model = torch._inductor.aoti_load_package(output_path)
            export_output = exported_model(inp)
        self.assertEqual(native_output, export_output, atol=1e-2, rtol=1e-2)

    @config.patch(force_disable_caches=True)
    @torch.inference_mode()
    def test_linear_unary_binary_out_variant_export(self):
        model = LinearUnaryBinaryNet().eval()
        inp = torch.rand(6, 10)
        binary = torch.rand(6, 10)
        native_output = model(inp, binary)

        exp_model = torch.export.export(model, args=(inp, binary))
        counters.clear()
        self.assertEqual(
            counters["zentorch"]["zentorch_linear_unary_binary_out"], 0
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = torch._inductor.aoti_compile_and_package(
                exp_model,
                package_path=os.path.join(tmp_dir, "linear_unary_binary_out.pt2"),
                inductor_configs=ind_conf,
            )
            self.assertEqual(
                counters["zentorch"]["zentorch_linear_unary_binary_out"], 1
            )
            exported_model = torch._inductor.aoti_load_package(output_path)
            export_output = exported_model(inp, binary)
        self.assertEqual(native_output, export_output, atol=1e-2, rtol=1e-2)

    @config.patch(force_disable_caches=True)
    @torch.inference_mode()
    def test_linear_binary_binary_out_variant_export(self):
        model = LinearBinaryBinaryNet().eval()
        inp = torch.rand(6, 10)
        binary_1 = torch.rand(6, 10)
        binary_2 = torch.rand(6, 10)
        native_output = model(inp, binary_1, binary_2)

        exp_model = torch.export.export(model, args=(inp, binary_1, binary_2))
        counters.clear()
        self.assertEqual(
            counters["zentorch"]["zentorch_linear_binary_binary_out"], 0
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = torch._inductor.aoti_compile_and_package(
                exp_model,
                package_path=os.path.join(tmp_dir, "linear_binary_binary_out.pt2"),
                inductor_configs=ind_conf,
            )
            self.assertEqual(
                counters["zentorch"]["zentorch_linear_binary_binary_out"], 1
            )
            exported_model = torch._inductor.aoti_load_package(output_path)
            export_output = exported_model(inp, binary_1, binary_2)
        self.assertEqual(
            native_output, export_output, atol=1e-2, rtol=1e-2
        )


if __name__ == "__main__":
    run_tests()
