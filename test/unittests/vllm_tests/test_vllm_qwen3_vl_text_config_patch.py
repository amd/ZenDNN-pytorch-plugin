# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

"""Tests for the vLLM 0.29 Qwen3-VL inner-text architecture backport."""

import sys
import types
import unittest
import unittest.mock


class TestQwen3VLTextConfigPatch(unittest.TestCase):
    @staticmethod
    def _patch():
        from zentorch.vllm import _qwen3_vl_text_config_patch

        return _qwen3_vl_text_config_patch

    def test_injects_dense_and_moe_text_architectures(self):
        patch = self._patch()
        calls = []

        def original(instance, hf_config, architectures=None):
            calls.append((instance, hf_config, architectures))
            return architectures

        wrapped = patch._make_patched_with_hf_config(original)
        instance = object()
        cases = {
            "qwen3_vl_text": ["Qwen3ForCausalLM"],
            "qwen3_vl_moe_text": ["Qwen3MoeForCausalLM"],
        }

        for model_type, expected in cases.items():
            with self.subTest(model_type=model_type):
                config = types.SimpleNamespace(
                    model_type=model_type,
                    architectures=None,
                )
                self.assertEqual(wrapped(instance, config), expected)
                self.assertEqual(calls[-1], (instance, config, expected))

    def test_preserves_explicit_or_existing_architecture(self):
        patch = self._patch()

        def original(instance, hf_config, architectures=None):
            return architectures

        wrapped = patch._make_patched_with_hf_config(original)
        qwen_config = types.SimpleNamespace(
            model_type="qwen3_vl_text",
            architectures=None,
        )
        explicit = ["CallerSelectedForCausalLM"]
        self.assertIs(wrapped(object(), qwen_config, explicit), explicit)

        checkpoint_arch = ["CheckpointForCausalLM"]
        configured = types.SimpleNamespace(
            model_type="qwen3_vl_text",
            architectures=checkpoint_arch,
        )
        self.assertIsNone(wrapped(object(), configured))

    def test_leaves_unrelated_config_unchanged(self):
        patch = self._patch()
        sentinel = object()

        def original(instance, hf_config, architectures=None):
            return sentinel, architectures

        wrapped = patch._make_patched_with_hf_config(original)
        config = types.SimpleNamespace(
            model_type="llama",
            architectures=None,
        )
        self.assertEqual(wrapped(object(), config), (sentinel, None))

    def test_applies_only_to_vllm_029(self):
        patch = self._patch()

        for version, expected in (
            ("0.28.0+cpu", False),
            ("0.29.0+cpu", True),
        ):
            with self.subTest(version=version):
                fake_vllm = types.ModuleType("vllm")
                fake_vllm.__version__ = version
                with (
                    unittest.mock.patch.dict(
                        sys.modules,
                        {"vllm": fake_vllm},
                    ),
                    unittest.mock.patch.object(
                        patch,
                        "patch_now_or_on_import",
                        return_value=True,
                    ) as arm,
                ):
                    self.assertEqual(
                        patch._apply_qwen3_vl_text_config_patch(),
                        expected,
                    )

                if expected:
                    arm.assert_called_once_with(
                        patch._TARGET_MODULE,
                        patch._do_patch_qwen3_vl_text_config,
                    )
                else:
                    arm.assert_not_called()

    def test_patch_is_idempotent(self):
        patch = self._patch()

        class VllmConfig:
            def with_hf_config(self, hf_config, architectures=None):
                return architectures

        target = types.SimpleNamespace(VllmConfig=VllmConfig)
        original = VllmConfig.with_hf_config
        with unittest.mock.patch.dict(
            sys.modules,
            {patch._TARGET_MODULE: target},
        ):
            self.assertTrue(patch._do_patch_qwen3_vl_text_config())
            self.assertTrue(patch._do_patch_qwen3_vl_text_config())

        self.assertIs(
            VllmConfig._zentorch_orig_with_hf_config,
            original,
        )
        self.assertTrue(getattr(VllmConfig, patch._MARKER))
        config = types.SimpleNamespace(
            model_type="qwen3_vl_text",
            architectures=None,
        )
        self.assertEqual(
            VllmConfig().with_hf_config(config),
            ["Qwen3ForCausalLM"],
        )


if __name__ == "__main__":
    unittest.main()
