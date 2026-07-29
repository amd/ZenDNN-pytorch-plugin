# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

"""
Unit tests for the out-of-tree DA8W4 (W4A8) vLLM kernel patch
(``Da8w4KernelPatch`` in ``zentorch.vllm``): patch registration and version
gating (vLLM >= 0.22.1).
"""

import unittest
import unittest.mock

import zentorch  # noqa: F401 - ensures zentorch native extension is loaded

try:
    import vllm  # noqa: F401

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False


@unittest.skipUnless(VLLM_AVAILABLE, "vLLM not installed")
class TestDa8w4KernelPatch(unittest.TestCase):
    """Da8w4KernelPatch is registered and gated to vLLM >= 0.22.1."""

    def test_patch_is_registered(self):
        from zentorch.vllm import register
        from zentorch.vllm._core import manager

        register()
        self.assertIn("Da8w4Kernel", manager.patches)

    def test_patch_targets_v22_1_through_v25_1(self):
        """Gated to exactly {0.22.1, 0.23, 0.24, 0.25, 0.25.1}."""
        from zentorch.vllm import Da8w4KernelPatch
        from zentorch.vllm._core import (
            VLLM_V22_1,
            VLLM_V23,
            VLLM_V24,
            VLLM_V25,
            VLLM_V25_1,
        )

        self.assertTrue(hasattr(Da8w4KernelPatch, "_target_versions"))
        self.assertEqual(
            Da8w4KernelPatch._target_versions,
            {VLLM_V22_1, VLLM_V23, VLLM_V24, VLLM_V25, VLLM_V25_1},
        )

    def test_apply_skips_below_v22_1(self):
        """apply() must be a no-op (False) on vLLM versions preceding 0.22.1."""
        from zentorch.vllm import Da8w4KernelPatch
        from zentorch.vllm import _core as zv_core

        with unittest.mock.patch.object(
            zv_core, "get_vllm_version", return_value="0.22.0"
        ):
            self.assertFalse(Da8w4KernelPatch.apply())


if __name__ == "__main__":
    unittest.main()
