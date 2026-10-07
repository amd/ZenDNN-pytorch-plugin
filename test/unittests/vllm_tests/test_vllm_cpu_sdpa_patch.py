# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import os
import sys
import types
import unittest
import unittest.mock

import torch

from ._test_constants import VLLM_AVAILABLE
from ._test_utils import load_source_vllm_module


@unittest.skipUnless(VLLM_AVAILABLE, "vLLM not installed")
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

    def test_native_zentorch_sdpa_impl_keeps_the_original_forward(self):
        """vLLM 0.31 calls zentorch SDPA for encoder layers itself and upcasts
        bf16 + ALiBi to fp32 first; the wrapper must hand those layers back."""
        import torch
        from vllm.v1.attention.backend import AttentionType

        spec, plugin = load_source_vllm_module()
        with unittest.mock.patch.dict(
            sys.modules, {"zentorch.vllm": plugin}
        ):
            spec.loader.exec_module(plugin)

        q = torch.zeros(4, 2, 8, dtype=torch.bfloat16)
        metadata = types.SimpleNamespace(num_actual_tokens=4)
        for native in (True, False):
            with self.subTest(use_zentorch_sdpa=native):
                impl = types.SimpleNamespace(
                    attn_type=AttentionType.ENCODER_ONLY,
                    use_zentorch_sdpa=native,
                    _zentorch_orig_forward=unittest.mock.Mock(),
                )
                with unittest.mock.patch.object(
                    plugin, "_run_encoder_sdpa_zentorch"
                ) as run_zentorch, unittest.mock.patch.object(
                    plugin, "zentorch_sdpa_supports_dtype", return_value=True
                ):
                    plugin._forward_cpu_attn_zentorch(
                        impl, None, q, q, q, None, metadata, torch.empty_like(q)
                    )
                self.assertEqual(impl._zentorch_orig_forward.called, native)
                self.assertEqual(run_zentorch.called, not native)

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


@unittest.skipUnless(VLLM_AVAILABLE, "vLLM not installed")
class TestVllmSlidingWindowMapping(unittest.TestCase):
    """The encoder path forwards vLLM's width as flash-attn radii."""

    @classmethod
    def setUpClass(cls):
        spec, plugin = load_source_vllm_module()
        with unittest.mock.patch.dict(sys.modules, {"zentorch.vllm": plugin}):
            spec.loader.exec_module(plugin)
        cls.plugin = plugin

    def _forwarded_window(self, sliding_window):
        captured = {}

        def _capture(*args, **kwargs):
            captured["kwargs"] = kwargs

        layer = types.SimpleNamespace(
            sliding_window=sliding_window, scale=0.125, alibi_slopes=None
        )
        tokens, heads, dim = 4, 2, 8
        metadata = types.SimpleNamespace(
            causal=False, query_start_loc=torch.tensor([0, tokens])
        )
        with unittest.mock.patch(
            "torch.ops.zentorch.zentorch_sdpa_attn", side_effect=_capture
        ):
            self.plugin._run_encoder_sdpa_zentorch(
                layer,
                torch.randn(tokens, heads, dim),
                torch.randn(tokens, 1, dim),
                torch.randn(tokens, 1, dim),
                torch.empty(tokens, heads, dim),
                metadata,
            )
        kwargs = captured["kwargs"]
        return kwargs["left_window_size"], kwargs["right_window_size"]

    def test_window_forwarded(self):
        self.assertEqual(self._forwarded_window(257), (256, 256))

    def test_disabled_window_is_zero(self):
        self.assertEqual(self._forwarded_window(-1), (-1, -1))


if __name__ == "__main__":
    unittest.main()
