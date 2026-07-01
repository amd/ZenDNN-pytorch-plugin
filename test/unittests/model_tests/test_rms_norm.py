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
    RmsNormTestCase,
    supported_dtypes,
    has_zentorch,
    reset_dynamo,
    run_tests,
    test_with_freeze_opt_and_cpp_wrapper,
)

_EPSILON = 1e-6


class _RmsNormModule(torch.nn.Module):
    """Returns the freshly-allocated normalized output of the tensor-returning
    zentorch_rms_norm. Under backend='zentorch' + cpp_wrapper this exercises the
    aoti_torch_cpu_zentorch_rms_norm shim + FallbackKernel lowering; under
    backend='inductor' it provides the reference compiled output."""

    def __init__(self, weight, epsilon):
        super().__init__()
        self.register_buffer("weight", weight)
        self.epsilon = epsilon

    def forward(self, input):
        return torch.ops.zentorch.zentorch_rms_norm(
            input, self.weight, self.epsilon
        )


class _AddRmsNormModule(torch.nn.Module):
    """Clones its inputs and invokes the void-returning zentorch_add_rms_norm_
    (which mutates `input` Tensor(a!) and `residual` Tensor(b!) in place),
    returning both. Under backend='zentorch' + cpp_wrapper this exercises the
    aoti_torch_cpu_zentorch_add_rms_norm_ shim + FallbackKernel lowering (void
    return + dual mutation); under backend='inductor' it is the reference."""

    def __init__(self, weight, epsilon):
        super().__init__()
        self.register_buffer("weight", weight)
        self.epsilon = epsilon

    def forward(self, input, residual):
        out = input.clone()
        res = residual.clone()
        torch.ops.zentorch.zentorch_add_rms_norm_(
            out, self.weight, res, self.epsilon
        )
        return out, res


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_RMS_Norm_Model(RmsNormTestCase):
    """Compiles the RMS-norm models and checks that the backend='zentorch'
    output (including the cpp_wrapper AOTI-shim path) matches the
    backend='inductor' reference. RmsNormTestCase's Hypothesis decorator does
    not expose a cpp_wrapper flag, so cpp_wrapper is set manually to True (to
    exercise the shim) and freeze is fixed False (RMS-norm has no weight-prepack
    path)."""

    def _compare_inductor_vs_zentorch(self, model, inputs):
        reset_dynamo()
        inductor_graph = torch.compile(copy.deepcopy(model), backend="inductor")
        inductor_out = inductor_graph(*inputs)

        reset_dynamo()
        zentorch_graph = torch.compile(model, backend="zentorch")
        zentorch_out = test_with_freeze_opt_and_cpp_wrapper(
            zentorch_graph, inputs, freeze_opt=False, cpp_wrapper=True
        )
        return inductor_out, zentorch_out

    @RmsNormTestCase.hypothesis_params_rms_norm_itr(
        dtype_list=supported_dtypes,
        # _compare_inductor_vs_zentorch fixes freeze_opt=False (RMS-norm has no
        # weight-prepack/freeze path), so pin freeze_list to match what's
        # actually exercised and not waste Hypothesis search space.
        freeze_list=[False],
        # A fresh cpp_wrapper compile far exceeds the default per-example
        # deadline; raise it so the deadline reflects compile cost.
        time_out=300000,
    )
    @torch.inference_mode()
    def test_rms_norm_model(self, dtype):
        model = _RmsNormModule(self.data.rms_weight, _EPSILON).eval()
        inductor_out, zentorch_out = self._compare_inductor_vs_zentorch(
            model, (self.data.rms_input,)
        )
        self.assertEqual(zentorch_out.dtype, inductor_out.dtype)
        self.assertEqual(zentorch_out, inductor_out, atol=1e-3, rtol=1e-3)

    @RmsNormTestCase.hypothesis_params_rms_norm_itr(
        dtype_list=supported_dtypes,
        # freeze_opt fixed False in _compare_inductor_vs_zentorch; pin it.
        freeze_list=[False],
        time_out=300000,
    )
    @torch.inference_mode()
    def test_add_rms_norm_model(self, dtype):
        model = _AddRmsNormModule(self.data.rms_weight, _EPSILON).eval()
        inductor_out, zentorch_out = self._compare_inductor_vs_zentorch(
            model, (self.data.rms_input, self.data.rms_residual)
        )
        # Both the normalized output and the running residual are returned.
        self.assertEqual(zentorch_out[0], inductor_out[0], atol=1e-3, rtol=1e-3)
        self.assertEqual(zentorch_out[1], inductor_out[1], atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    run_tests()
