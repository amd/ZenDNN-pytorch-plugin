# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

"""Tests for the Whisper W4A16 packed k_proj bias patch."""

import sys
import types
import unittest
import unittest.mock

import torch

from ._test_constants import VLLM_AVAILABLE

if VLLM_AVAILABLE:
    from zentorch.vllm._whisper_w4a16_patch import (
        _PATCH_MARKER,
        _TARGET_MODULE,
        _create_fake_bias_for_k_proj,
        _do_patch_whisper_loader,
    )


@unittest.skipUnless(VLLM_AVAILABLE, "vLLM not installed")
class TestWhisperW4A16PatchWiring(unittest.TestCase):
    def test_wired_in_patches(self):
        import zentorch.vllm as zv
        from zentorch.vllm._whisper_w4a16_patch import _apply_whisper_w4a16_patch

        names = [name for name, _ in zv._PATCHES]
        self.assertIn("WhisperW4A16", names)
        self.assertIs(dict(zv._PATCHES)["WhisperW4A16"], _apply_whisper_w4a16_patch)


@unittest.skipUnless(VLLM_AVAILABLE, "vLLM not installed")
class TestCreateFakeBiasForKProj(unittest.TestCase):
    def test_injects_zeros_bias_for_unpacked_weight(self):
        weight = torch.ones(8, 16)
        names = [
            n
            for n, _ in _create_fake_bias_for_k_proj(
                [("model.encoder.layers.0.self_attn.k_proj.weight", weight)],
                ".k_proj.weight",
            )
        ]
        self.assertEqual(
            names,
            [
                "model.encoder.layers.0.self_attn.k_proj.weight",
                "model.encoder.layers.0.self_attn.k_proj.bias",
            ],
        )
        bias = dict(
            _create_fake_bias_for_k_proj(
                [("model.encoder.layers.0.self_attn.k_proj.weight", weight)],
                ".k_proj.weight",
            )
        )["model.encoder.layers.0.self_attn.k_proj.bias"]
        self.assertEqual(tuple(bias.shape), (8,))
        self.assertTrue(torch.equal(bias, torch.zeros(8)))

    def test_packed_suffix_uses_out_features_not_packed_dim0(self):
        packed = torch.ones(2, 16)  # packed dim != d_model
        result = dict(
            _create_fake_bias_for_k_proj(
                [("model.encoder.layers.0.self_attn.k_proj.weight_packed", packed)],
                ".k_proj.weight_packed",
                out_features=8,
            )
        )
        bias_name = "model.encoder.layers.0.self_attn.k_proj.bias"
        self.assertIn(bias_name, result)
        self.assertNotIn("model.encoder.layers.0.self_attn.k_proj.bias_packed", result)
        self.assertEqual(tuple(result[bias_name].shape), (8,))

    def test_forwards_real_bias_and_skips_fake(self):
        weight = torch.ones(8, 16)
        real_bias = torch.ones(8)
        items = list(
            _create_fake_bias_for_k_proj(
                [
                    ("layers.0.self_attn.k_proj.weight", weight),
                    ("layers.0.self_attn.k_proj.bias", real_bias),
                ],
                ".k_proj.weight",
                out_features=8,
            )
        )
        names = [n for n, _ in items]
        self.assertEqual(
            names,
            ["layers.0.self_attn.k_proj.weight", "layers.0.self_attn.k_proj.bias"],
        )
        self.assertIs(items[1][1], real_bias)

    def test_real_bias_before_weight_still_skips_fake(self):
        weight = torch.ones(8, 16)
        real_bias = torch.ones(8)
        items = list(
            _create_fake_bias_for_k_proj(
                [
                    ("layers.0.self_attn.k_proj.bias", real_bias),
                    ("layers.0.self_attn.k_proj.weight", weight),
                ],
                ".k_proj.weight",
                out_features=8,
            )
        )
        names = [n for n, _ in items]
        self.assertEqual(
            names,
            ["layers.0.self_attn.k_proj.bias", "layers.0.self_attn.k_proj.weight"],
        )

    def test_does_not_match_unrelated_weights(self):
        items = list(
            _create_fake_bias_for_k_proj(
                [("layers.0.self_attn.q_proj.weight", torch.ones(8, 16))],
                ".k_proj.weight",
            )
        )
        self.assertEqual([n for n, _ in items], ["layers.0.self_attn.q_proj.weight"])


@unittest.skipUnless(VLLM_AVAILABLE, "vLLM not installed")
class TestWhisperLoaderPatch(unittest.TestCase):
    def test_wraps_load_weights_once_and_injects_packed_bias(self):
        loaded = []

        class FakeConfig:
            d_model = 8

        module = types.SimpleNamespace()

        class WhisperForConditionalGeneration:
            def __init__(self):
                self.config = FakeConfig()
                self.hf_to_vllm_mapper = object()

            def load_weights(self, weights):
                # Mimic stock: only ``.k_proj.weight``.
                weights = module._create_fake_bias_for_k_proj(
                    weights, ".k_proj.weight"
                )
                loaded.append(list(weights))
                return {"loaded"}

        module.WhisperForConditionalGeneration = WhisperForConditionalGeneration
        module._create_fake_bias_for_k_proj = lambda w, k: w

        with unittest.mock.patch.dict(sys.modules, {_TARGET_MODULE: module}):
            self.assertTrue(_do_patch_whisper_loader())
            wrapped = WhisperForConditionalGeneration.load_weights
            self.assertTrue(_do_patch_whisper_loader())
            self.assertIs(WhisperForConditionalGeneration.load_weights, wrapped)

            packed = torch.ones(2, 16)
            result = WhisperForConditionalGeneration().load_weights(
                [
                    ("model.encoder.layers.0.self_attn.k_proj.weight_packed", packed),
                    ("model.encoder.layers.0.self_attn.q_proj.weight", torch.ones(8, 16)),
                ]
            )

        self.assertEqual(result, {"loaded"})
        names = [n for n, _ in loaded[0]]
        self.assertEqual(
            names,
            [
                "model.encoder.layers.0.self_attn.k_proj.weight_packed",
                "model.encoder.layers.0.self_attn.q_proj.weight",
                "model.encoder.layers.0.self_attn.k_proj.bias",
            ],
        )
        bias = dict(loaded[0])["model.encoder.layers.0.self_attn.k_proj.bias"]
        self.assertEqual(tuple(bias.shape), (8,))
        self.assertTrue(
            getattr(WhisperForConditionalGeneration, _PATCH_MARKER)
        )

    def test_stock_weight_walk_still_injects_unpacked_bias(self):
        loaded = []

        class FakeConfig:
            d_model = 8

        module = types.SimpleNamespace()

        class WhisperForConditionalGeneration:
            def __init__(self):
                self.config = FakeConfig()

            def load_weights(self, weights):
                weights = module._create_fake_bias_for_k_proj(
                    weights, ".k_proj.weight"
                )
                loaded.append(list(weights))
                return {"loaded"}

        module.WhisperForConditionalGeneration = WhisperForConditionalGeneration
        module._create_fake_bias_for_k_proj = lambda w, k: w

        with unittest.mock.patch.dict(sys.modules, {_TARGET_MODULE: module}):
            self.assertTrue(_do_patch_whisper_loader())
            # Bias size must be d_model (8), not packed dim 0 (4).
            result = WhisperForConditionalGeneration().load_weights(
                [("layers.0.self_attn.k_proj.weight", torch.ones(4, 16))]
            )

        self.assertEqual(result, {"loaded"})
        names = [n for n, _ in loaded[0]]
        self.assertEqual(
            names,
            [
                "layers.0.self_attn.k_proj.weight",
                "layers.0.self_attn.k_proj.bias",
            ],
        )
        bias = dict(loaded[0])["layers.0.self_attn.k_proj.bias"]
        self.assertEqual(tuple(bias.shape), (8,))


if __name__ == "__main__":
    unittest.main()
