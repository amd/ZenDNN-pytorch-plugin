# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import torch
from torch.testing._internal.common_utils import TestCase, run_tests
import zentorch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from unittest_utils import counters  # noqa: E402
import os  # noqa: E402

ind_conf = {"joint_custom_post_pass": zentorch.export_optimize_pass}


# create simple linear model
class SimpleLinearModel(torch.nn.Module):
    def __init__(self):
        super(SimpleLinearModel, self).__init__()
        self.linear = torch.nn.Linear(10, 10)
        self.silu = torch.nn.SiLU()
        self.relu = torch.nn.ReLU()
        self.gelu = torch.nn.GELU()
        self.gelu_0 = torch.nn.GELU(approximate="tanh")
        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()

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


class TestExport(TestCase):
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


if __name__ == "__main__":
    run_tests()
