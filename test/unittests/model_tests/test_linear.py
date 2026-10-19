# ******************************************************************************
# Copyright (c) 2024-2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import unittest
import torch
from torch.testing import FileCheck
from torch import nn
import sys
import copy
from pathlib import Path
from torch._inductor import config as inductor_config

sys.path.append(str(Path(__file__).parent.parent))
from unittest_utils import (  # noqa: 402
    AddmmTestCase,
    has_zentorch,
    reset_dynamo,
    run_tests,
    supported_dtypes,
    update_supported_dtypes,
    zentorch,
    freeze_opt,
    cpp_wrapper_opt,
    default_tolerance,
    test_with_freeze_opt,
    test_with_freeze_opt_and_cpp_wrapper,
    counters,
)

supported_dtypes = update_supported_dtypes(supported_dtypes, "zentorch_linear")


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Custom_Model_Linear(nn.Module):
    def __init__(self, k, dtype) -> None:
        super(Custom_Model_Linear, self).__init__()
        self.mlp_0 = torch.nn.Linear(k, 512, dtype=dtype)
        self.mlp_1 = torch.nn.Linear(512, 256, dtype=dtype)
        self.mlp_2 = torch.nn.Linear(256, 64, dtype=dtype)

    def forward(self, inputs):
        outputs = self.mlp_0(inputs)
        outputs = self.mlp_1(outputs)
        outputs_final = self.mlp_2(outputs)

        return outputs_final


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_Linear_Model(AddmmTestCase):

    @AddmmTestCase.hypothesis_params_addmm_itr(
        dtype_list=supported_dtypes, freeze_list=freeze_opt,
        cpp_wrapper_opt_list=cpp_wrapper_opt,
        # cold cpp_wrapper compile exceeds the default deadline; see
        # test_with_freeze_opt_and_cpp_wrapper in zentorch_test_utils.
        time_out=60000,
    )
    @torch.inference_mode()
    def test_linear_model(self, dtype, freeze_opt, cpp_wrapper):

        model = Custom_Model_Linear(self.data.k, self.data.get_torch_type(dtype))
        inductor_model = copy.deepcopy(model)

        reset_dynamo()
        counters.clear()
        self.assertEqual(counters["zentorch"]["zentorch_linear"], 0)

        # Compiled reference under the same freezing setting, so zentorch's pass
        # is the only difference between the two graphs. zentorch_compile sets
        # that pass on the Inductor config and never restores it, so without the
        # patch this leg replaces the linears too -- and the counter below would
        # see 6 rather than 3.
        with inductor_config.patch(joint_custom_post_pass=None):
            inductor_graph = torch.compile(inductor_model, backend="inductor")
            inductor_output = test_with_freeze_opt(
                inductor_graph, (self.data.x), freeze_opt
            )

        reset_dynamo()
        compiled_graph = torch.compile(model, backend="zentorch")
        compiled_output, cpp_code = test_with_freeze_opt_and_cpp_wrapper(
            compiled_graph, (self.data.x), freeze_opt, cpp_wrapper
        )
        self.assertEqual(counters["zentorch"]["zentorch_linear"], 3)
        if freeze_opt:
            self.assertEqual(
                counters["zentorch"]["zentorch_weight_prepack_for_linear"], 3
            )
        # bfloat16 keeps 8 significant bits, so the two compiled GEMMs may land
        # an ULP apart -- measured at 2**-12 on a few elements, while float32's
        # 1e-5 demands bit-exactness. assertEqual (not allclose) so a failure
        # reports the magnitude.
        atol, rtol = default_tolerance(self.data.get_torch_type(dtype))
        self.assertEqual(inductor_output, compiled_output, atol=atol, rtol=rtol)
        # Pillar 2 (codegen): op lowers to its AOTI C-shim (see helper docstring).
        if cpp_wrapper:
            FileCheck().check("aoti_torch_cpu_zentorch").run(cpp_code)


if __name__ == "__main__":
    run_tests()
