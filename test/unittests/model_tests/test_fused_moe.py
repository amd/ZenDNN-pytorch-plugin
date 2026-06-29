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
    """Allocates the (zero-init) output inside forward, invokes the
    void-returning zentorch_fused_moe (which mutates it in place), and returns
    it -- so a torch.compile of this module puts the op into the graph. Under
    backend='zentorch' + cpp_wrapper this exercises the
    aoti_torch_cpu_zentorch_fused_moe shim + FallbackKernel lowering; under
    backend='inductor' it provides the reference compiled output."""

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
        out = torch.zeros(
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
    """Compiles a zentorch_fused_moe model and checks that the
    backend='zentorch' output (including the cpp_wrapper AOTI-shim path, which
    fused_moe routes through a FallbackKernel lowering since it is void-
    returning and mutates its `output` Tensor(a!)) matches the
    backend='inductor' reference. A single Hypothesis test sweeps the
    (dtype x cpp_wrapper) combinations."""

    def _compare_inductor_vs_zentorch(self, model, hidden_states, cpp_wrapper):
        reset_dynamo()
        torch.ops.zentorch.zentorch_flush_moe_weight_cache()
        inductor_graph = torch.compile(copy.deepcopy(model), backend="inductor")
        inductor_out = inductor_graph(hidden_states)

        reset_dynamo()
        torch.ops.zentorch.zentorch_flush_moe_weight_cache()
        zentorch_graph = torch.compile(model, backend="zentorch")
        # MoE has no weight-prepack/freeze path, so freeze_opt is fixed False;
        # cpp_wrapper is swept by Hypothesis to exercise the AOTI-shim path.
        zentorch_out = test_with_freeze_opt_and_cpp_wrapper(
            zentorch_graph, (hidden_states,), freeze_opt=False,
            cpp_wrapper=cpp_wrapper,
        )

        self.assertEqual(zentorch_out.dtype, inductor_out.dtype)
        self.assertEqual(zentorch_out, inductor_out, atol=1e-3, rtol=1e-3)

    @GroupMatmulTestCase.hypothesis_params_group_matmul_itr(
        dtype_list=supported_dtypes,
        # A fresh cpp_wrapper compile far exceeds the default 10s per-example
        # deadline; raise it so the deadline reflects compile cost.
        time_out=300000,
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

        self._compare_inductor_vs_zentorch(model, hidden_states, cpp_wrapper)


if __name__ == "__main__":
    run_tests()
