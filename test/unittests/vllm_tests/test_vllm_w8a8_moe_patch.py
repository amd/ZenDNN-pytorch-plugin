# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

"""
Unit tests for the out-of-tree W8A8 INT8 fused-MoE vLLM patches in
``zentorch.vllm``: patch registration and version gating (vLLM >= 0.22.1).
"""

import unittest
import unittest.mock

import zentorch  # noqa: F401 - ensures zentorch native extension is loaded

try:
    import vllm  # noqa: F401
    import zentorch.vllm  # noqa: F401 - load OOT vLLM patch module for access

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False


# manager registration key -> patch class attribute name in ``zentorch.vllm``.
_W8A8_MOE_PATCHES = {
    "Int8MoE": "Int8MoEPatch",
    "GptOssMoELoader": "GptOssMoELoaderPatch",
    "MixtralMoELoader": "MixtralMoELoaderPatch",
    "MoERunnerCompile": "MoERunnerCompilePatch",
    "MoETopkCpu": "MoETopkCpuPatch",
    "CpuWorkerWorkspace": "CpuWorkerWorkspacePatch",
}


@unittest.skipUnless(VLLM_AVAILABLE, "vLLM not installed")
class TestW8A8MoEPatches(unittest.TestCase):
    """W8A8 INT8 fused-MoE patches are registered and gated to vLLM >= 0.22.1."""

    def test_patches_are_registered(self):
        zv = zentorch.vllm
        manager = zentorch.vllm._core.manager

        zv._INITIALIZED = False
        with unittest.mock.patch(
            "zentorch._C.is_avx512_supported", return_value=True
        ):
            zv.register()
        for key in _W8A8_MOE_PATCHES:
            self.assertIn(
                key,
                manager.patches,
                f"{key} patch should be registered by _register_patches()",
            )

    def test_patches_target_v22_1_through_max(self):
        """Each patch is gated to every known vLLM version in
        [VLLM_V22_1, VLLM_MAX_VERSION]"""
        zv = zentorch.vllm
        core = zentorch.vllm._core

        expected = {
            v
            for v in core._VERSION_MAP
            if core.VLLM_V22_1 <= v <= core.VLLM_MAX_VERSION
        }
        self.assertTrue(
            {core.VLLM_V22_1, core.VLLM_MAX_VERSION} <= expected,
            f"derived range {sorted(expected)} should span "
            f"{core.VLLM_V22_1}..{core.VLLM_MAX_VERSION}",
        )
        for cls_name in _W8A8_MOE_PATCHES.values():
            patch_cls = getattr(zv, cls_name)
            self.assertTrue(
                hasattr(patch_cls, "_target_versions"),
                f"{cls_name} should expose _target_versions",
            )
            self.assertEqual(
                patch_cls._target_versions,
                expected,
                f"{cls_name} should be gated to {expected}",
            )

    def test_apply_skips_below_v22_1(self):
        """Each patch's apply() must be a no-op (False) on vLLM < 0.22.1."""
        zv = zentorch.vllm
        zv_core = zentorch.vllm._core

        with unittest.mock.patch.object(
            zv_core, "get_vllm_version", return_value="0.22.0"
        ):
            for cls_name in _W8A8_MOE_PATCHES.values():
                patch_cls = getattr(zv, cls_name)
                self.assertFalse(
                    patch_cls.apply(),
                    f"{cls_name}.apply() should return False on vLLM 0.22.0",
                )


if __name__ == "__main__":
    unittest.main()
