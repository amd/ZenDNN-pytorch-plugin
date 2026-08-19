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


class TestFusedMoEPatch(unittest.TestCase):
    """_do_patch_fused_moe swaps CPUFusedMOE.__init__ and is idempotent."""

    @staticmethod
    def _make_fake_cpu_fused_moe_module():
        module = types.ModuleType(
            "vllm.model_executor.layers.fused_moe.cpu_fused_moe"
        )

        class CPUFusedMOE:
            def __init__(self, layer):
                raise AssertionError("original __init__ must be replaced")

        module.CPUFusedMOE = CPUFusedMOE
        return module

    def test_patch_swaps_init_and_is_idempotent(self):
        spec, plugin = load_source_vllm_module()
        with unittest.mock.patch.dict(
            sys.modules, {"zentorch.vllm": plugin}
        ):
            spec.loader.exec_module(plugin)

        fake = self._make_fake_cpu_fused_moe_module()
        with unittest.mock.patch.dict(
            sys.modules,
            {"vllm.model_executor.layers.fused_moe.cpu_fused_moe": fake},
        ):
            self.assertTrue(plugin._do_patch_fused_moe())
            self.assertTrue(fake.CPUFusedMOE._zentorch_fused_moe_patched)
            self.assertTrue(hasattr(fake.CPUFusedMOE, "_zentorch_forward"))
            self.assertTrue(plugin._do_patch_fused_moe())

    def test_disabled_via_env(self):
        spec, plugin = load_source_vllm_module()
        with unittest.mock.patch.dict(
            sys.modules, {"zentorch.vllm": plugin}
        ):
            spec.loader.exec_module(plugin)

        with unittest.mock.patch.dict(
            os.environ, {"ZENTORCH_FUSED_MOE": "0"}
        ):
            self.assertFalse(plugin._apply_fused_moe_patch())


if __name__ == "__main__":
    unittest.main()
