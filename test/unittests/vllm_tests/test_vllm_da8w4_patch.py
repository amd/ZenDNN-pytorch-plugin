# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

"""
Unit tests for the out-of-tree DA8W4 (W4A8) vLLM kernel patch.

The plugin adds the W4A8 fast path onto vLLM's in-tree W4A16
``ZentorchWNA16LinearKernel``. Registration is via the flat
``_PATCHES`` tuple, so version gating is covered by the shared
version-contract tests; the DA8W4-specific surface is the
``VLLM_CPU_INT4_W4A8`` toggle.
"""

import os
import types
import unittest
import unittest.mock

import torch
import zentorch  # noqa: F401 - ensures zentorch native extension is loaded

from ._test_constants import VLLM_AVAILABLE


@unittest.skipUnless(VLLM_AVAILABLE, "vLLM not installed")
class TestDa8w4KernelPatch(unittest.TestCase):
    """DA8W4 is wired into _PATCHES and respects VLLM_CPU_INT4_W4A8."""

    def test_wired_in_patches(self):
        import zentorch.vllm as zv

        names = [name for name, _ in zv._PATCHES]
        self.assertIn("Da8w4Kernel", names)

    def test_enabled_by_default(self):
        from zentorch.vllm._da8w4_kernel_patch import _da8w4_enabled

        with unittest.mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VLLM_CPU_INT4_W4A8", None)
            self.assertTrue(_da8w4_enabled())

    def test_disabled_via_env(self):
        from zentorch.vllm._da8w4_kernel_patch import _da8w4_enabled

        with unittest.mock.patch.dict(os.environ, {"VLLM_CPU_INT4_W4A8": "0"}):
            self.assertFalse(_da8w4_enabled())

    def test_apply_is_noop_when_disabled(self):
        # Disabled -> returns False and installs nothing, so vLLM's in-tree
        # W4A16 ZentorchWNA16LinearKernel is used unchanged.
        from zentorch.vllm._da8w4_kernel_patch import _apply_da8w4_patch

        with unittest.mock.patch.dict(os.environ, {"VLLM_CPU_INT4_W4A8": "0"}):
            self.assertFalse(_apply_da8w4_patch())

    @staticmethod
    def _kernel_and_layer(with_g_idx):
        from vllm.scalar_type import scalar_types

        out_features, in_features = 8, 16
        config = types.SimpleNamespace(
            zero_points=False, weight_type=scalar_types.uint4b8
        )
        kernel = types.SimpleNamespace(
            config=config,
            w_q_name="weight_packed",
            w_s_name="weight_scale",
            w_zp_name=None,
        )
        layer = torch.nn.Module()
        weight = torch.nn.Parameter(
            torch.zeros(out_features, in_features // 8, dtype=torch.int32),
            requires_grad=False,
        )
        weight.packed_dim = 1
        layer.weight_packed = weight
        layer.weight_scale = torch.nn.Parameter(
            torch.ones(out_features, 1, dtype=torch.bfloat16),
            requires_grad=False,
        )
        if with_g_idx:
            config.has_g_idx = False
            kernel.w_gidx_name = "weight_g_idx"
            layer.weight_g_idx = torch.nn.Parameter(
                torch.zeros(in_features, dtype=torch.int32),
                requires_grad=False,
            )
        return kernel, layer

    def test_process_weights_without_g_idx_fields(self):
        # vLLM 0.30 dropped has_g_idx / w_gidx_name with GPTQ act-order.
        from zentorch.vllm._da8w4_kernel_patch import _process_da8w4_weights

        kernel, layer = self._kernel_and_layer(with_g_idx=False)
        _process_da8w4_weights(kernel, layer)
        self.assertTrue(layer._zentorch_da8w4)
        self.assertEqual(layer._zentorch_da8w4_scale.shape, (1, 8))

    def test_process_weights_clears_unused_g_idx(self):
        from zentorch.vllm._da8w4_kernel_patch import _process_da8w4_weights

        kernel, layer = self._kernel_and_layer(with_g_idx=True)
        _process_da8w4_weights(kernel, layer)
        self.assertIsNone(layer.weight_g_idx)
        self.assertTrue(layer._zentorch_da8w4)


if __name__ == "__main__":
    unittest.main()
