# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import unittest

import torch
from torch import nn
from torch._inductor import config
from torch._inductor.virtualized import V
from torch._guards import detect_fake_mode

import zentorch
from zentorch._compile_backend import zentorch_compile
from zentorch._optimize_for_export import optimize_for_export
from zentorch._utils import counters


class FactorizedExpert(nn.Module):
    def __init__(self, width=32, hidden=4):
        super().__init__()
        self.left = nn.Linear(width, hidden, bias=False)
        self.right = nn.Linear(width, hidden, bias=False)

    def forward(self, x):
        return self.left(x) * self.right(x)


class ResidualFactorizedBlock(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.cross = FactorizedExpert(width=32, hidden=hidden)
        self.output = nn.Linear(hidden, 32, bias=False)

    def forward(self, x):
        return x + self.output(self.cross(x))


class MMoEHead(nn.Module):
    def __init__(self, width=32, hidden=4):
        super().__init__()
        self.experts = nn.ModuleList(
            [FactorizedExpert(width, hidden) for _ in range(4)]
        )
        self.ctr_gate = nn.Linear(width, 4)
        self.cvr_gate = nn.Linear(width, 4)

    def forward(self, x):
        outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        ctr = self.ctr_gate(x).softmax(-1).unsqueeze(-1)
        cvr = self.cvr_gate(x).softmax(-1).unsqueeze(-1)
        return (outputs * ctr).sum(1), (outputs * cvr).sum(1)


class CombinedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = MMoEHead()
        self.outside = nn.ModuleList([nn.Linear(32, 4) for _ in range(4)])

    def forward(self, x):
        return self.head(x), [layer(x) for layer in self.outside]


class UnevenHead(nn.Module):
    def __init__(
        self,
        num_tasks=2,
        softmax=True,
        num_experts=4,
        use_bmm=False,
        expert_bias=True,
        gate_bias=True,
    ):
        super().__init__()
        self.experts = nn.ModuleList(
            [nn.Linear(32, 8, bias=expert_bias) for _ in range(num_experts)]
        )
        self.gates = nn.ModuleList(
            [nn.Linear(32, num_experts, bias=gate_bias) for _ in range(num_tasks)]
        )
        self.softmax = softmax
        self.use_bmm = use_bmm

    def forward(self, x):
        outputs = torch.stack([expert(x) for expert in self.experts], 1)
        results = []
        for gate in self.gates:
            scores = gate(x)
            weights = scores.softmax(-1) if self.softmax else scores.sigmoid()
            if self.use_bmm:
                results.append(torch.bmm(weights.unsqueeze(1), outputs).squeeze(1))
            else:
                results.append((outputs * weights.unsqueeze(-1)).sum(1))
        return tuple(results)


class TwoHeads(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = MMoEHead()
        self.second = MMoEHead()

    def forward(self, x):
        return self.first(x), self.second(x)


@unittest.skipUnless(zentorch._C.is_avx512_supported(), "AVX-512 required")
class TestMMoEFusion(unittest.TestCase):
    def setUp(self):
        torch._dynamo.reset()
        counters.clear()
        torch.manual_seed(10)

    def tearDown(self):
        torch._dynamo.reset()
        counters.clear()

    def _run(self, model, *, freezing=True, remove_scope=False, dtype=torch.float32):
        model.eval().to(dtype).requires_grad_(False)

        def without_module_metadata(gm, inputs):
            for node in gm.graph.nodes:
                node.meta.pop("nn_module_stack", None)
            return zentorch_compile(gm, inputs)

        backend = without_module_metadata if remove_scope else "zentorch"
        with torch.inference_mode(), config.patch(
            freezing=freezing, force_disable_caches=True
        ):
            compiled = torch.compile(
                model, backend=backend, fullgraph=True, dynamic=True
            )
            for batch in (2, 7):
                x = torch.randn(batch, 32, dtype=dtype)
                tol = 1e-5 if dtype == torch.float32 else 1e-2
                torch.testing.assert_close(compiled(x), model(x), atol=tol, rtol=tol)

    def test_only_mmoe_region_fuses(self):
        self._run(CombinedModel())
        self.assertEqual(counters["zentorch"]["mmoe_fusion_linear"], 2)
        self.assertEqual(counters["zentorch"]["mmoe_fusion_members"], 10)
        self.assertEqual(counters["zentorch"]["qkv_fusion_linear"], 0)

    def test_parallel_linears_alone_do_not_fuse(self):
        class Parallel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linears = nn.ModuleList([nn.Linear(32, 4) for _ in range(4)])

            def forward(self, x):
                return tuple(layer(x) for layer in self.linears)

        self._run(Parallel())
        self.assertEqual(counters["zentorch"]["mmoe_fusion_linear"], 0)
        self.assertEqual(counters["zentorch"]["qkv_fusion_linear"], 0)

    def test_does_not_need_module_provenance(self):
        self._run(CombinedModel(), remove_scope=True)
        self.assertEqual(counters["zentorch"]["mmoe_fusion_linear"], 2)

    def test_freezing_required(self):
        self._run(CombinedModel(), freezing=False)
        self.assertEqual(counters["zentorch"]["mmoe_fusion_linear"], 0)

    def test_distinct_heads_are_not_merged(self):
        self._run(TwoHeads())
        self.assertEqual(counters["zentorch"]["mmoe_fusion_linear"], 4)

    def test_uneven_expert_and_gate_outputs(self):
        self._run(UnevenHead())
        self.assertEqual(counters["zentorch"]["mmoe_fusion_linear"], 1)
        self.assertEqual(counters["zentorch"]["mmoe_fusion_members"], 6)

    def test_both_experts_and_gates_without_bias(self):
        self._run(UnevenHead(expert_bias=False, gate_bias=False))
        self.assertEqual(counters["zentorch"]["mmoe_fusion_linear"], 1)
        self.assertEqual(counters["zentorch"]["mmoe_fusion_members"], 6)

    def test_expert_bias_and_bias_free_gates(self):
        self._run(UnevenHead(expert_bias=True, gate_bias=False))
        self.assertEqual(counters["zentorch"]["mmoe_fusion_linear"], 2)
        self.assertEqual(counters["zentorch"]["mmoe_fusion_members"], 6)

    def test_three_tasks_six_experts_with_bmm(self):
        self._run(UnevenHead(num_tasks=3, num_experts=6, use_bmm=True))
        self.assertEqual(counters["zentorch"]["mmoe_fusion_linear"], 1)
        self.assertEqual(counters["zentorch"]["mmoe_fusion_members"], 9)

    def test_bfloat16(self):
        self._run(CombinedModel(), dtype=torch.bfloat16)
        self.assertEqual(counters["zentorch"]["mmoe_fusion_linear"], 2)

    def test_export_graph(self):
        model = CombinedModel().eval().requires_grad_(False)
        x = torch.randn(2, 32)
        with torch.inference_mode(), config.patch(freezing=True):
            expected = model(x)
            exported = torch.export.export(model, (x,))
            fake_mode = detect_fake_mode(
                [
                    node.meta.get("val")
                    for node in exported.graph.nodes
                    if node.op == "placeholder"
                ]
            )
            with V.set_fake_mode(fake_mode):
                optimize_for_export(exported.graph)
            exported.graph_module.recompile()
            torch.testing.assert_close(exported.module()(x), expected)
        self.assertEqual(counters["zentorch"]["mmoe_fusion_linear"], 2)

    def test_single_gate_is_not_mmoe(self):
        self._run(UnevenHead(num_tasks=1))
        self.assertEqual(counters["zentorch"]["mmoe_fusion_linear"], 0)

    def test_sigmoid_mixture_is_not_matched(self):
        self._run(UnevenHead(softmax=False))
        self.assertEqual(counters["zentorch"]["mmoe_fusion_linear"], 0)

    def test_unrelated_narrowing_pair_is_not_matched(self):
        self._run(FactorizedExpert(hidden=4))
        self.assertEqual(counters["zentorch"]["mmoe_fusion_linear"], 0)
        self.assertEqual(counters["zentorch"]["qkv_fusion_linear"], 0)

    def test_widening_mmoe_preserves_binary_post_ops(self):
        self._run(MMoEHead(hidden=32))
        # Only the two task gates fuse; the wide expert group stays split.
        self.assertEqual(counters["zentorch"]["mmoe_fusion_linear"], 1)
        self.assertEqual(counters["zentorch"]["mmoe_fusion_members"], 2)

    def _run_deep_experts(self, hidden):
        model = MMoEHead()
        model.experts = nn.ModuleList(
            [
                nn.Sequential(*(ResidualFactorizedBlock(hidden) for _ in range(3)))
                for _ in range(4)
            ]
        )
        self._run(model)

    def test_deep_experts_fuse_internal_narrowing_pairs(self):
        self._run_deep_experts(hidden=4)
        # First-block projections + task gates, then one pair per expert in
        # each of the two deeper blocks, whose inputs are no longer shared.
        self.assertEqual(counters["zentorch"]["mmoe_fusion_linear"], 2 + 4 * 2)
        self.assertEqual(counters["zentorch"]["mmoe_fusion_members"], 10 + 4 * 2 * 2)

    def test_deep_experts_keep_wide_internal_pairs(self):
        self._run_deep_experts(hidden=24)
        # H=0.75K passes for the first eight-member group, but the deeper
        # two-member groups fail 2H<K and retain their binary post-ops.
        self.assertEqual(counters["zentorch"]["mmoe_fusion_linear"], 2)
        self.assertEqual(counters["zentorch"]["mmoe_fusion_members"], 10)


if __name__ == "__main__":
    unittest.main()
