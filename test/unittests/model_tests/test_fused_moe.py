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
    GroupMatmulTestCase,
    has_zentorch,
    zentorch,
    reset_dynamo,
    run_tests,
    supported_dtypes,
    update_supported_dtypes,
    test_with_freeze_opt_and_cpp_wrapper,
)

supported_dtypes = update_supported_dtypes(supported_dtypes)


class _FusedMoeModule(torch.nn.Module):
    """Allocates the (uninitialized) output inside forward, invokes the
    void-returning zentorch_fused_moe (which mutates it in place), and returns
    it -- so a torch.compile of this module puts the op into the graph. Under
    backend='zentorch' + cpp_wrapper this exercises the
    aoti_torch_cpu_zentorch_fused_moe shim + FallbackKernel lowering. The eager
    forward runs the same op and serves as the reference."""

    def __init__(
        self, w13, w2, w2_bias, topk_weights, topk_id, num_tokens, k_out, act
    ):
        super().__init__()
        self.register_buffer("w13", w13)
        self.register_buffer("w2", w2)
        self.register_buffer("w2_bias", w2_bias)
        self.register_buffer("topk_weights", topk_weights)
        self.register_buffer("topk_id", topk_id)
        self.num_tokens = num_tokens
        self.k_out = k_out
        self.act = act

    def forward(self, hidden_states):
        out = torch.empty(
            self.num_tokens, self.k_out, dtype=hidden_states.dtype
        )
        torch.ops.zentorch.zentorch_fused_moe(
            out,
            hidden_states,
            self.w13,
            self.w2,
            None,  # w13_bias
            self.w2_bias,
            self.topk_weights,
            self.topk_id,
            False,  # skip_weighted
            self.act,
        )
        return out


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_FusedMoe_Model(GroupMatmulTestCase):
    """AOTI test for zentorch_fused_moe, following PyTorch's two pillars:
      Pillar 1 (numerical): backend='zentorch' output matches the eager
                            reference (the void-returning op mutates `out`).
      Pillar 2 (codegen):   under cpp_wrapper the generated C++ calls the
                            aoti_torch_cpu_zentorch_fused_moe shim (routed
                            through a FallbackKernel since the op is void-
                            returning and mutates its `output` Tensor(a!)).
    A single Hypothesis test sweeps the (dtype x cpp_wrapper) combinations."""

    def _check_aoti(self, model, hidden_states, cpp_wrapper):
        # Pillar 1 reference: eager (flush the MoE weight cache first).
        torch.ops.zentorch.zentorch_flush_moe_weight_cache()
        eager_out = model(hidden_states)

        reset_dynamo()
        torch.ops.zentorch.zentorch_flush_moe_weight_cache()
        zentorch_graph = torch.compile(model, backend="zentorch")
        # MoE has no weight-prepack/freeze path, so freeze_opt is fixed False;
        # cpp_wrapper is swept by Hypothesis to exercise the AOTI-shim path.
        zentorch_out, cpp_code = test_with_freeze_opt_and_cpp_wrapper(
            zentorch_graph, (hidden_states,), freeze_opt=False,
            cpp_wrapper=cpp_wrapper,
        )

        # Pillar 1: numerical equivalence vs eager.
        self.assertEqual(zentorch_out.dtype, eager_out.dtype)
        self.assertEqual(zentorch_out, eager_out, atol=1e-3, rtol=1e-3)

        # Pillar 2: codegen assertion -- the op must lower to its AOTI shim.
        if cpp_wrapper:
            FileCheck().check("aoti_torch_cpu_zentorch_fused_moe").run(cpp_code)

    @GroupMatmulTestCase.hypothesis_params_group_matmul_itr(
        dtype_list=supported_dtypes,
        # cold cpp_wrapper compile exceeds the default deadline; see
        # test_with_freeze_opt_and_cpp_wrapper in zentorch_test_utils.
        time_out=60000,
    )
    @torch.inference_mode()
    def test_fused_moe_model(self, dtype, cpp_wrapper):
        num_tokens = self.data.num_tokens
        K_out = self.data.K
        activation = "silu"

        w13_3d = torch.stack(self.data.w13_weights_gated, dim=0)
        w2_3d = torch.stack(self.data.w2_weights_gated, dim=0)
        w2_bias_3d = torch.stack(self.data.w2_bias_gated, dim=0)
        topk_weights_t = self.data.topk_weights_routing
        topk_id = self.data.topk_indices.to(torch.int32)
        hidden_states = self.data.hidden_states

        model = _FusedMoeModule(
            w13_3d, w2_3d, w2_bias_3d, topk_weights_t, topk_id,
            num_tokens, K_out, activation,
        ).eval()

        self._check_aoti(model, hidden_states, cpp_wrapper)


if __name__ == "__main__":
    run_tests()
