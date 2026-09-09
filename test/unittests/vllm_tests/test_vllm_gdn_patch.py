# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

"""Tests for the Qwen Gated Delta Net fp16 vLLM patch."""

import os
import subprocess
import sys
import types
import unittest
import unittest.mock

from ._test_constants import VLLM_AVAILABLE


@unittest.skipUnless(VLLM_AVAILABLE, "vLLM not installed")
class TestGDNPatch(unittest.TestCase):
    @staticmethod
    def _gdn_patch():
        from zentorch.vllm import _gdn_patch

        return _gdn_patch

    def test_wired_in_patches(self):
        import zentorch.vllm as zv

        gdn_patch = self._gdn_patch()
        self.assertIs(
            dict(zv._PATCHES)["GatedDeltaNet"],
            gdn_patch._apply_gdn_patch,
        )

    def test_enabled_by_default_and_disabled_via_env(self):
        gdn_patch = self._gdn_patch()

        with unittest.mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ZENTORCH_GDN", None)
            self.assertTrue(gdn_patch._gdn_enabled())

        with unittest.mock.patch.dict(os.environ, {"ZENTORCH_GDN": "0"}):
            self.assertFalse(gdn_patch._gdn_enabled())
            with unittest.mock.patch.object(
                gdn_patch, "patch_now_or_on_import"
            ) as hook:
                self.assertFalse(gdn_patch._apply_gdn_patch())
                hook.assert_not_called()

    def test_apply_uses_deferred_import_hook(self):
        gdn_patch = self._gdn_patch()

        with unittest.mock.patch.dict(
            os.environ, {"ZENTORCH_GDN": "1"}
        ), unittest.mock.patch.object(
            gdn_patch,
            "patch_now_or_on_import",
            return_value=True,
        ) as hook:
            self.assertTrue(gdn_patch._apply_gdn_patch())

        hook.assert_called_once_with(
            gdn_patch._TARGET_MODULE,
            gdn_patch._do_patch_gdn,
        )

    def test_custom_op_schema_can_be_inferred(self):
        """Regression test for a missing/unresolvable LayerNameType annotation."""
        from torch._library.infer_schema import infer_schema

        gdn_patch = self._gdn_patch()
        schema = infer_schema(
            gdn_patch._gdn_attention_core_cpu,
            mutates_args=["core_attn_out"],
        )

        self.assertIn("LayerName layer_name", schema)
        self.assertIn("Tensor(a3!) core_attn_out", schema)

    def test_patch_swaps_forward_and_is_idempotent(self):
        gdn_patch = self._gdn_patch()
        target = types.ModuleType(gdn_patch._TARGET_MODULE)

        class QwenGatedDeltaNetAttention:
            def forward_cpu(self, hidden_states):
                return hidden_states

        original = QwenGatedDeltaNetAttention.forward_cpu
        target.QwenGatedDeltaNetAttention = QwenGatedDeltaNetAttention

        with unittest.mock.patch.dict(
            sys.modules, {gdn_patch._TARGET_MODULE: target}
        ), unittest.mock.patch.object(gdn_patch, "_register_core_op") as register:
            self.assertTrue(gdn_patch._do_patch_gdn())
            self.assertTrue(gdn_patch._do_patch_gdn())

        register.assert_called_once_with()
        self.assertIs(
            QwenGatedDeltaNetAttention._zentorch_orig_forward_cpu,
            original,
        )
        self.assertIs(
            QwenGatedDeltaNetAttention.forward_cpu,
            gdn_patch.forward_cpu_zen,
        )
        self.assertTrue(QwenGatedDeltaNetAttention._zentorch_gdn_patched)

    def test_clean_vllm_startup_has_no_plugin_import_failure(self):
        """Catch imports from a partially initialized vllm.utils.torch_utils."""
        result = subprocess.run(
            [sys.executable, "-c", "import vllm"],
            capture_output=True,
            check=False,
            text=True,
            timeout=120,
        )
        output = result.stdout + result.stderr

        self.assertEqual(result.returncode, 0, output)
        self.assertNotIn("Failed to load plugin zentorch", output)
        self.assertNotIn("partially initialized module", output)


if __name__ == "__main__":
    unittest.main()
