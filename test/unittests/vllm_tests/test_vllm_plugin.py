# ******************************************************************************
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

"""Runtime, registration, wiring, and platform tests for zentorch.vllm."""

import sys
import types
import unittest
import unittest.mock

import torch

from ._test_constants import VLLM_AVAILABLE, vllm
from ._test_utils import load_source_vllm_module


EXPECTED_PATCHES = [
    "Gemma4HeteroConfig",
    "TorchAO",
    "Int8MoE",
    "Wna16MoE",
    "MixtralMoELoader",
    "RMSNorm",
    "FusedMoE",
    "FusedMLP",
    "CPUSdpa",
    "Da8w4Kernel",
    "SWBlockSize",
    "WhisperW4A16",
    "GptOssStreamedExpert",
    "GatedDeltaNet",
]

REMOVED_PATCHES = [
    "CppIndirectAssert",
    "CPURunnerShutdown",
    "CpuZeroBlockIds",
    "TorchcodecImportGuard",
    "CPUProfiler",
    "CompilationConfigRepr",
    "GptOssMoEWeightRemap",
    "GptOssMoELoader",
]


@unittest.skipUnless(VLLM_AVAILABLE, "vLLM not installed")
class TestVersionContract(unittest.TestCase):
    """Accept only the validated vLLM window on PyTorch 2.13 or newer."""

    def test_base_version_strips_suffixes(self):
        from zentorch.vllm import _base_version

        self.assertEqual(_base_version("0.27.0rc1+cpu"), "0.27.0rc1")
        self.assertEqual(
            _base_version("0.27.0.dev1+gabc.d20260101.cpu"),
            "0.27.0.dev1",
        )
        self.assertEqual(_base_version("0.27.1+cpu"), "0.27.1")
        self.assertEqual(_base_version("0.27.0"), "0.27.0")

    def test_is_supported_vllm_accepts_validated_window(self):
        from zentorch.vllm import is_supported_vllm

        for version in [
            "0.27.0",
            "0.27.0+cpu",
            "0.27.1",
            "0.27.1+cpu",
            "0.27.2",
            "0.27.5+cpu",
            "0.27.99",
            "0.28.0",
            "0.28.0+cpu",
        ]:
            self.assertTrue(
                is_supported_vllm(version),
                f"{version} should be supported",
            )

    def test_is_supported_vllm_rejects_other_versions(self):
        from zentorch.vllm import is_supported_vllm

        for version in [
            None,
            "",
            "0.26.0",
            "0.26.9+cpu",
            "0.25.1",
            "0.27.0rc1+cpu",
            "0.27.0.dev123+cpu",
            "0.28.0rc1+cpu",
            "0.28.1",
            "0.28.1+cpu",
            "1.0.0",
            "not-a-version",
        ]:
            self.assertFalse(
                is_supported_vllm(version),
                f"{version} should NOT be supported",
            )

    def test_is_supported_torch_enforces_213(self):
        from zentorch.vllm import is_supported_torch

        for version, expected in [
            ("2.11.0+cpu", False),
            ("2.12.1+cpu", False),
            ("2.13.0+cpu", True),
            ("2.13.1", True),
            ("2.14.0+cpu", True),
        ]:
            with unittest.mock.patch.object(torch, "__version__", version):
                self.assertEqual(
                    is_supported_torch(),
                    expected,
                    f"torch {version} -> {expected}",
                )

    def test_installed_vllm_is_supported(self):
        from zentorch.vllm import is_supported_vllm

        self.assertTrue(
            is_supported_vllm(vllm.__version__),
            f"Installed vLLM {vllm.__version__} must be supported",
        )


@unittest.skipUnless(VLLM_AVAILABLE, "vLLM not installed")
class TestRegisterContract(unittest.TestCase):
    """register() gates on vLLM, PyTorch, and AVX-512."""

    @staticmethod
    def _fresh_source_module():
        spec, plugin = load_source_vllm_module()
        with unittest.mock.patch.dict(
            sys.modules, {"zentorch.vllm": plugin}
        ):
            spec.loader.exec_module(plugin)
        plugin._INITIALIZED = False
        return plugin

    @staticmethod
    def _register_with(
        plugin,
        vllm_version,
        torch_ok=True,
        avx512=True,
    ):
        fake_vllm = types.ModuleType("vllm")
        fake_vllm.__version__ = vllm_version
        with (
            unittest.mock.patch.dict(sys.modules, {"vllm": fake_vllm}),
            unittest.mock.patch.object(
                plugin, "is_supported_torch", return_value=torch_ok
            ),
            unittest.mock.patch.object(
                plugin, "_apply_all_patches"
            ) as apply_all,
            unittest.mock.patch(
                "zentorch._C.is_avx512_supported", return_value=avx512
            ),
        ):
            result = plugin.register()
        return result, apply_all

    def test_accepts_supported_runtime(self):
        plugin = self._fresh_source_module()
        result, apply_all = self._register_with(plugin, "0.27.0+cpu")
        self.assertEqual(result, "zentorch.vllm._platform.ZenCPUPlatform")
        apply_all.assert_called_once_with()

    def test_rejects_old_vllm(self):
        plugin = self._fresh_source_module()
        result, apply_all = self._register_with(plugin, "0.26.0+cpu")
        self.assertIsNone(result)
        apply_all.assert_not_called()

    def test_accepts_supported_runtime_028(self):
        plugin = self._fresh_source_module()
        result, apply_all = self._register_with(plugin, "0.28.0+cpu")
        self.assertEqual(result, "zentorch.vllm._platform.ZenCPUPlatform")
        apply_all.assert_called_once_with()

    def test_rejects_future_vllm(self):
        plugin = self._fresh_source_module()
        result, apply_all = self._register_with(plugin, "0.28.1+cpu")
        self.assertIsNone(result)
        apply_all.assert_not_called()

    def test_rejects_unsupported_torch(self):
        plugin = self._fresh_source_module()
        result, apply_all = self._register_with(
            plugin, "0.27.0+cpu", torch_ok=False
        )
        self.assertIsNone(result)
        apply_all.assert_not_called()

    def test_falls_back_without_avx512(self):
        plugin = self._fresh_source_module()
        result, apply_all = self._register_with(
            plugin, "0.27.0+cpu", avx512=False
        )
        self.assertIsNone(result)
        apply_all.assert_not_called()

    def test_patches_applied_only_once(self):
        plugin = self._fresh_source_module()
        fake_vllm = types.ModuleType("vllm")
        fake_vllm.__version__ = "0.27.0+cpu"
        with (
            unittest.mock.patch.dict(sys.modules, {"vllm": fake_vllm}),
            unittest.mock.patch.object(
                plugin, "is_supported_torch", return_value=True
            ),
            unittest.mock.patch.object(
                plugin, "_apply_all_patches"
            ) as apply_all,
            unittest.mock.patch(
                "zentorch._C.is_avx512_supported", return_value=True
            ),
        ):
            first = plugin.register()
            second = plugin.register()
        self.assertEqual(first, "zentorch.vllm._platform.ZenCPUPlatform")
        self.assertEqual(second, "zentorch.vllm._platform.ZenCPUPlatform")
        apply_all.assert_called_once_with()

    def test_register_returns_platform_for_installed_vllm(self):
        from zentorch.vllm import is_supported_vllm, register

        if not is_supported_vllm(vllm.__version__):
            self.skipTest(f"Installed vLLM {vllm.__version__} is not supported")

        with unittest.mock.patch(
            "zentorch._C.is_avx512_supported", return_value=True
        ):
            self.assertEqual(
                register(), "zentorch.vllm._platform.ZenCPUPlatform"
            )


@unittest.skipUnless(VLLM_AVAILABLE, "vLLM not installed")
class TestPatchWiring(unittest.TestCase):
    """register() wires only the Zen-specific vLLM 0.27 hooks."""

    def test_expected_patches_are_wired(self):
        from zentorch.vllm import _PATCHES

        names = [name for name, _ in _PATCHES]
        self.assertEqual(names, EXPECTED_PATCHES)

    def test_removed_backports_are_gone(self):
        from zentorch import vllm as plugin

        names = {name for name, _ in plugin._PATCHES}
        for removed in REMOVED_PATCHES:
            self.assertNotIn(removed, names)

        self.assertFalse(hasattr(plugin, "manager"))
        self.assertFalse(hasattr(plugin, "PatchManager"))
        self.assertFalse(hasattr(plugin, "vllm_version"))

    def test_core_module_deleted(self):
        with self.assertRaises(ImportError):
            import zentorch.vllm._core  # noqa: F401

    def test_apply_all_records_applied_patches(self):
        from zentorch import vllm as plugin

        if not plugin.is_supported_vllm(vllm.__version__):
            self.skipTest(f"Installed vLLM {vllm.__version__} is not supported")

        plugin._INITIALIZED = False
        with unittest.mock.patch(
            "zentorch._C.is_avx512_supported", return_value=True
        ):
            plugin.register()

        for name in ("RMSNorm", "FusedMoE"):
            self.assertIn(name, plugin.APPLIED_PATCHES)


@unittest.skipUnless(VLLM_AVAILABLE, "vLLM not installed")
class TestFusedMLPPatch(unittest.TestCase):
    """FusedMLP dense-MLP fusion is wired in and opt-in via ZENTORCH_FUSED_FFN."""

    def test_wired_right_after_fused_moe(self):
        from zentorch.vllm import _PATCHES

        names = [name for name, _ in _PATCHES]
        self.assertIn("FusedMLP", names)
        self.assertEqual(names[names.index("FusedMoE") + 1], "FusedMLP")

    def test_disabled_by_default(self):
        import os

        from zentorch.vllm import _fused_mlp_patch as fmp

        with unittest.mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ZENTORCH_FUSED_FFN", None)
            self.assertFalse(fmp._apply_fused_mlp_patch_impl())

    def test_enabled_via_env_arms_hook(self):
        import os

        from zentorch.vllm import _fused_mlp_patch as fmp

        with (
            unittest.mock.patch.dict(os.environ, {"ZENTORCH_FUSED_FFN": "1"}),
            unittest.mock.patch.object(
                fmp, "patch_now_or_on_import", return_value=True
            ) as arm,
        ):
            self.assertTrue(fmp._apply_fused_mlp_patch_impl())
            arm.assert_called_once_with(
                fmp._TARGET_MODULE, fmp._do_patch_fused_mlp
            )


@unittest.skipUnless(VLLM_AVAILABLE, "vLLM not installed")
class TestPlatformConfiguration(unittest.TestCase):
    """ZenCPUPlatform identity."""

    def test_platform_device_name_and_type(self):
        from zentorch.vllm._platform import ZenCPUPlatform

        self.assertEqual(ZenCPUPlatform.device_name, "cpu")
        self.assertEqual(ZenCPUPlatform.device_type, "cpu")

    def test_platform_is_zen_cpu(self):
        from zentorch.vllm._platform import ZenCPUPlatform

        instance = ZenCPUPlatform.__new__(ZenCPUPlatform)
        self.assertTrue(ZenCPUPlatform.is_zen_cpu(instance))


@unittest.skipUnless(VLLM_AVAILABLE, "vLLM not installed")
class TestZentorchOptimizePass(unittest.TestCase):
    """zentorch optimize_pass must be importable and callable."""

    def test_optimize_pass_is_callable(self):
        from zentorch._compile_backend import optimize_pass

        self.assertIsNotNone(optimize_pass)
        self.assertTrue(callable(optimize_pass))


def run_tests():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for test_case in (
        TestVersionContract,
        TestRegisterContract,
        TestPatchWiring,
        TestFusedMLPPatch,
        TestPlatformConfiguration,
        TestZentorchOptimizePass,
    ):
        suite.addTests(loader.loadTestsFromTestCase(test_case))
    return unittest.TextTestRunner(verbosity=2).run(suite)


if __name__ == "__main__":
    run_tests()
