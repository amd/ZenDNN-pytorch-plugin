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
import unittest
import unittest.mock

import zentorch  # noqa: F401 - ensures zentorch native extension is loaded


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


if __name__ == "__main__":
    unittest.main()
