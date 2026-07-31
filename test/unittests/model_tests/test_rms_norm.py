# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import unittest
import torch
from torch.testing import FileCheck
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
    aoti_torch_cpu_zentorch_rms_norm shim + FallbackKernel lowering. The eager
    forward runs the same op and serves as the reference."""

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
    return + dual mutation). The eager forward runs the same op and serves as
    the reference."""

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
    """AOTI test for the RMS-norm ops, following PyTorch's two pillars:
      Pillar 1 (numerical): the backend='zentorch' output matches the eager
                            reference.
      Pillar 2 (codegen):   under cpp_wrapper the op lowers through its
                            aoti_torch_cpu_zentorch_*rms_norm* shim.
    RmsNormTestCase's Hypothesis decorator does not expose a cpp_wrapper flag,
    so cpp_wrapper is set manually to True (to exercise the shim) and freeze is
    fixed False (RMS-norm has no weight-prepack path)."""

    def _check_aoti(self, model, inputs):
        # Pillar 1 reference: eager (the raw op runs the same kernel as the
        # shim), captured before compilation.
        eager_out = model(*inputs)

        reset_dynamo()
        zentorch_graph = torch.compile(model, backend="zentorch")
        # Pillar 2 (codegen) needs only one cpp_wrapper build per test method;
        # the rest of the dtype sweep runs numerically on the Python-wrapper
        # path (this decorator does not expose a cpp_wrapper axis to pin, so the
        # once-per-method guard lives here).
        do_codegen = not getattr(self, "_rms_codegen_done", False)
        self._rms_codegen_done = True
        zentorch_out, cpp_code = test_with_freeze_opt_and_cpp_wrapper(
            zentorch_graph, inputs, freeze_opt=False, cpp_wrapper=do_codegen
        )

        # Pillar 2 (codegen): op lowers to its AOTI C-shim (see helper docstring).
        if do_codegen:
            FileCheck().check("aoti_torch_cpu_zentorch").run(cpp_code)
        return eager_out, zentorch_out

    @RmsNormTestCase.hypothesis_params_rms_norm_itr(
        dtype_list=supported_dtypes,
        # _check_aoti fixes freeze_opt=False (RMS-norm has no weight-prepack/
        # freeze path), so pin freeze_list to match what's actually exercised
        # and not waste Hypothesis search space.
        freeze_list=[False],
        # cold cpp_wrapper compile exceeds the default deadline; see
        # test_with_freeze_opt_and_cpp_wrapper in zentorch_test_utils.
        time_out=60000,
    )
    @torch.inference_mode()
    def test_rms_norm_model(self, dtype):
        model = _RmsNormModule(self.data.rms_weight, _EPSILON).eval()
        eager_out, zentorch_out = self._check_aoti(
            model, (self.data.rms_input,)
        )
        self.assertEqual(zentorch_out.dtype, eager_out.dtype)
        self.assertEqual(zentorch_out, eager_out, atol=1e-3, rtol=1e-3)

    @RmsNormTestCase.hypothesis_params_rms_norm_itr(
        dtype_list=supported_dtypes,
        # freeze_opt fixed False in _check_aoti; pin it.
        freeze_list=[False],
        # cold cpp_wrapper compile exceeds the default deadline; see
        # test_with_freeze_opt_and_cpp_wrapper in zentorch_test_utils.
        time_out=60000,
    )
    @torch.inference_mode()
    def test_add_rms_norm_model(self, dtype):
        model = _AddRmsNormModule(self.data.rms_weight, _EPSILON).eval()
        eager_out, zentorch_out = self._check_aoti(
            model, (self.data.rms_input, self.data.rms_residual)
        )
        # Both the normalized output and the running residual are returned.
        self.assertEqual(zentorch_out[0], eager_out[0], atol=1e-3, rtol=1e-3)
        self.assertEqual(zentorch_out[1], eager_out[1], atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    run_tests()
