# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import os
import sys
import types
import unittest
import unittest.mock

from ._test_utils import load_source_vllm_module


class TestCPUSdpaPatch(unittest.TestCase):
    """_do_patch_cpu_sdpa wraps CPUAttentionBackendImpl.forward and is idempotent."""

    @staticmethod
    def _make_fake_cpu_attn_module():
        module = types.ModuleType("vllm.v1.attention.backends.cpu_attn")

        class CPUAttentionBackendImpl:
            def forward(self, *args, **kwargs):
                raise AssertionError("original forward must be replaced")

        module.CPUAttentionBackendImpl = CPUAttentionBackendImpl
        return module

    def test_patch_swaps_forward_and_is_idempotent(self):
        spec, plugin = load_source_vllm_module()
        with unittest.mock.patch.dict(
            sys.modules, {"zentorch.vllm": plugin}
        ):
            spec.loader.exec_module(plugin)

        fake = self._make_fake_cpu_attn_module()
        with unittest.mock.patch.dict(
            sys.modules,
            {"vllm.v1.attention.backends.cpu_attn": fake},
        ):
            self.assertTrue(plugin._do_patch_cpu_sdpa())
            self.assertTrue(fake.CPUAttentionBackendImpl._zentorch_sdpa_patched)
            self.assertTrue(
                hasattr(fake.CPUAttentionBackendImpl, "_zentorch_orig_forward")
            )
            self.assertTrue(plugin._do_patch_cpu_sdpa())

    def test_disabled_via_env(self):
        spec, plugin = load_source_vllm_module()
        with unittest.mock.patch.dict(
            sys.modules, {"zentorch.vllm": plugin}
        ):
            spec.loader.exec_module(plugin)

        with unittest.mock.patch.dict(os.environ, {"ZENTORCH_SDPA": "0"}):
            self.assertFalse(plugin._apply_cpu_sdpa_patch())

    def test_deferred_via_import_hook(self):
        spec, plugin = load_source_vllm_module()
        with unittest.mock.patch.dict(
            sys.modules, {"zentorch.vllm": plugin}
        ):
            spec.loader.exec_module(plugin)

        import zentorch.vllm._import_hook as import_hook

        import_hook._handled.discard(plugin._CPU_ATTN_MODULE)
        with unittest.mock.patch.object(
            plugin, "patch_now_or_on_import", wraps=plugin.patch_now_or_on_import
        ) as hook:
            self.assertTrue(plugin._apply_cpu_sdpa_patch())
            hook.assert_called_once_with(
                plugin._CPU_ATTN_MODULE, plugin._do_patch_cpu_sdpa
            )


if __name__ == "__main__":
    unittest.main()
