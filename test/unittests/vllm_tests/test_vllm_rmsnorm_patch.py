# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import sys
import types
import unittest
import unittest.mock

import torch

from ._test_constants import VLLM_AVAILABLE
from ._test_utils import load_source_vllm_module


@unittest.skipUnless(VLLM_AVAILABLE, "vLLM not installed")
class TestRMSNormPatch(unittest.TestCase):
    """_do_patch_rmsnorm swaps RMSNorm.forward and is idempotent."""

    @staticmethod
    def _make_fake_layernorm_module():
        module = types.ModuleType("vllm.model_executor.layers.layernorm")

        class RMSNorm:
            def __init__(self):
                self.variance_size_override = None
                self.variance_epsilon = 1e-6
                self.weight = torch.nn.Parameter(torch.ones(8))

            def forward_native(self, x, residual=None):
                if residual is None:
                    return x
                return x, residual

        module.RMSNorm = RMSNorm
        return module

    def test_patch_swaps_forward_and_is_idempotent(self):
        spec, plugin = load_source_vllm_module()
        with unittest.mock.patch.dict(
            sys.modules, {"zentorch.vllm": plugin}
        ):
            spec.loader.exec_module(plugin)

        fake = self._make_fake_layernorm_module()
        with unittest.mock.patch.dict(
            sys.modules,
            {"vllm.model_executor.layers.layernorm": fake},
        ):
            self.assertTrue(plugin._do_patch_rmsnorm())
            self.assertTrue(fake.RMSNorm._zentorch_rmsnorm_patched)
            self.assertTrue(plugin._do_patch_rmsnorm())

    def test_patched_forward_uses_zentorch_for_residual(self):
        spec, plugin = load_source_vllm_module()
        with unittest.mock.patch.dict(
            sys.modules, {"zentorch.vllm": plugin}
        ):
            spec.loader.exec_module(plugin)

        fake = self._make_fake_layernorm_module()
        with unittest.mock.patch.dict(
            sys.modules,
            {"vllm.model_executor.layers.layernorm": fake},
        ):
            plugin._do_patch_rmsnorm()

        layer = fake.RMSNorm()
        x = torch.randn(4, 8)
        residual = torch.randn(4, 8)
        with unittest.mock.patch.object(
            torch.ops.zentorch, "zentorch_add_rms_norm_"
        ) as fused:
            out_x, out_residual = layer.forward(x, residual)
            fused.assert_called_once()
        self.assertIs(out_x, x)
        self.assertIs(out_residual, residual)


if __name__ == "__main__":
    unittest.main()
