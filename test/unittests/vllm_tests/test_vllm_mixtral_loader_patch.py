# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import sys
import types
import unittest
import unittest.mock

from ._test_constants import VLLM_AVAILABLE

if VLLM_AVAILABLE:
    from zentorch.vllm._mixtral_moe_loader_patch import (
        _TARGET_MODULE,
        _do_patch_mixtral_loader,
        _remap_mixtral_expert_names,
    )


@unittest.skipUnless(VLLM_AVAILABLE, "vLLM not installed")
class TestMixtralLoaderPatch(unittest.TestCase):
    def test_remaps_only_alternate_expert_projection_names(self):
        weights = [
            (
                "model.layers.0.block_sparse_moe.experts.2."
                "gate_proj.weight_packed",
                object(),
            ),
            (
                "model.layers.0.block_sparse_moe.experts.2."
                "up_proj.weight_scale",
                object(),
            ),
            (
                "model.layers.0.block_sparse_moe.experts.2."
                "down_proj.weight_shape",
                object(),
            ),
            (
                "model.layers.0.block_sparse_moe.experts.2.w1.weight",
                object(),
            ),
            ("model.layers.0.self_attn.q_proj.weight", object()),
        ]

        remapped = list(_remap_mixtral_expert_names(weights))

        self.assertEqual(
            [name for name, _ in remapped],
            [
                "model.layers.0.block_sparse_moe.experts.2.w1.weight_packed",
                "model.layers.0.block_sparse_moe.experts.2.w3.weight_scale",
                "model.layers.0.block_sparse_moe.experts.2.w2.weight_shape",
                "model.layers.0.block_sparse_moe.experts.2.w1.weight",
                "model.layers.0.self_attn.q_proj.weight",
            ],
        )
        for (_, original), (_, mapped) in zip(
            weights, remapped, strict=True
        ):
            self.assertIs(mapped, original)

    def test_wraps_load_weights_once(self):
        calls = []

        class FakeMixtralModel:
            def load_weights(self, weights):
                calls.append(list(weights))
                return {"loaded"}

        module = types.SimpleNamespace(MixtralModel=FakeMixtralModel)
        with unittest.mock.patch.dict(sys.modules, {_TARGET_MODULE: module}):
            self.assertTrue(_do_patch_mixtral_loader())
            wrapped = FakeMixtralModel.load_weights
            self.assertTrue(_do_patch_mixtral_loader())
            self.assertIs(FakeMixtralModel.load_weights, wrapped)

            result = FakeMixtralModel().load_weights(
                [
                    (
                        "layers.0.block_sparse_moe.experts.0.gate_proj.weight",
                        object(),
                    )
                ]
            )

        self.assertEqual(result, {"loaded"})
        self.assertEqual(
            [name for name, _ in calls[0]],
            ["layers.0.block_sparse_moe.experts.0.w1.weight"],
        )
        self.assertTrue(FakeMixtralModel._zentorch_mixtral_loader_patched)


if __name__ == "__main__":
    unittest.main()
