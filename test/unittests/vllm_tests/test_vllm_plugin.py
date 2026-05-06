# ******************************************************************************
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

"""
Unit tests for zentorch.vllm plugin.

Tests verify:
- Version compatibility checks for supported versions (0.15.0 - 0.20.1)
- Version parsing logic
- Patch registration and application
- Individual patch functionality (oneDNN disable, CompilationConfig repr, etc.)
- Platform configuration

Supported vLLM versions: 0.15.0, 0.15.1, 0.16.0, 0.17.0, 0.17.1, 0.18.0, 0.18.1,
0.19.0, 0.19.1, 0.20.0, 0.20.1
"""

import os
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

TORCHAO_AVAILABLE = importlib.util.find_spec("torchao") is not None

# vLLM 0.11+ uses Python 3.10+ type syntax (e.g., `X | None`)
# which fails at import time on Python 3.9
IS_PYTHON_3_10_OR_ABOVE = sys.version_info >= (3, 10)

try:
    import vllm  # NoQA: F401

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False


def _load_source_vllm_module():
    import zentorch  # noqa: F401 - ensures native extension is available

    plugin_root = Path(__file__).resolve().parents[3]
    vllm_init = os.path.join(
        plugin_root, "src", "cpu", "python", "zentorch", "vllm", "__init__.py"
    )
    spec = importlib.util.spec_from_file_location("zentorch.vllm", vllm_init)
    zv = importlib.util.module_from_spec(spec)
    return spec, zv


# =============================================================================
# Version Parsing Tests
# =============================================================================


class TestVersionParsing(unittest.TestCase):
    """Test version parsing logic in core.py."""

    def test_base_version_strips_suffixes(self):
        """_base_version should strip dev/rc/local suffixes."""
        from zentorch.vllm.core import _base_version

        self.assertEqual(
            _base_version("0.12.0.dev1+gb8b302cde.d20251203.cpu"), "0.12.0"
        )
        self.assertEqual(_base_version("0.13.0rc1+cpu"), "0.13.0")
        self.assertEqual(_base_version("0.13.0rc0"), "0.13.0")
        self.assertEqual(_base_version("0.14.0rc1+cpu"), "0.14.0")
        self.assertEqual(_base_version("0.17.1rc1+cpu"), "0.17.1")
        self.assertEqual(_base_version("0.18.0.dev1+cpu"), "0.18.0")

    def test_version_map_contains_supported_versions(self):
        """VERSION_MAP should contain all supported base versions."""
        from zentorch.vllm.core import _VERSION_MAP

        expected_versions = [
            "0.15.0",
            "0.15.1",
            "0.16.0",
            "0.17.0",
            "0.17.1",
            "0.18.0",
            "0.18.1",
            "0.19.0",
            "0.19.1",
            "0.20.0",
            "0.20.1",
        ]
        for ver in expected_versions:
            self.assertIn(ver, _VERSION_MAP, f"{ver} should be in VERSION_MAP")

    def test_version_family_detection_supported(self):
        """VERSION_MAP should return correct family for supported versions."""
        from zentorch.vllm.core import _base_version, _VERSION_MAP

        # v15 family
        self.assertEqual(_VERSION_MAP.get(_base_version("0.15.0")), "v15")
        self.assertEqual(_VERSION_MAP.get(_base_version("0.15.0+cpu")), "v15")

        # v15_1 family
        self.assertEqual(_VERSION_MAP.get(_base_version("0.15.1")), "v15_1")
        self.assertEqual(_VERSION_MAP.get(_base_version("0.15.1+cpu")), "v15_1")

        # v16 family
        self.assertEqual(_VERSION_MAP.get(_base_version("0.16.0")), "v16")
        self.assertEqual(_VERSION_MAP.get(_base_version("0.16.0+cpu")), "v16")

        # v17 family
        self.assertEqual(_VERSION_MAP.get(_base_version("0.17.0")), "v17")
        self.assertEqual(_VERSION_MAP.get(_base_version("0.17.0+cpu")), "v17")
        self.assertEqual(_VERSION_MAP.get(_base_version("0.17.1")), "v17")
        self.assertEqual(_VERSION_MAP.get(_base_version("0.17.1+cpu")), "v17")

        # v18 family
        self.assertEqual(_VERSION_MAP.get(_base_version("0.18.0")), "v18")
        self.assertEqual(_VERSION_MAP.get(_base_version("0.18.0+cpu")), "v18")
        self.assertEqual(_VERSION_MAP.get(_base_version("0.18.1")), "v18")
        self.assertEqual(_VERSION_MAP.get(_base_version("0.18.1+cpu")), "v18")

        # v19 family
        self.assertEqual(_VERSION_MAP.get(_base_version("0.19.0")), "v19")
        self.assertEqual(_VERSION_MAP.get(_base_version("0.19.0+cpu")), "v19")
        self.assertEqual(_VERSION_MAP.get(_base_version("0.19.1")), "v19")
        self.assertEqual(_VERSION_MAP.get(_base_version("0.19.1+cpu")), "v19")

        # v20 family
        self.assertEqual(_VERSION_MAP.get(_base_version("0.20.0")), "v20")
        self.assertEqual(_VERSION_MAP.get(_base_version("0.20.0+cpu")), "v20")
        self.assertEqual(_VERSION_MAP.get(_base_version("0.20.0rc1+cpu")), "v20")
        self.assertEqual(_VERSION_MAP.get(_base_version("0.20.1")), "v20")
        self.assertEqual(_VERSION_MAP.get(_base_version("0.20.1+cpu")), "v20")
        self.assertEqual(_VERSION_MAP.get(_base_version("0.20.1rc0+cpu")), "v20")

    def test_version_family_detection_unsupported(self):
        """VERSION_MAP should return None for unsupported versions."""
        from zentorch.vllm.core import _base_version, _VERSION_MAP

        unsupported = [
            "0.9.1",
            "0.10.0",
            "0.10.5",
            "0.11.0",
            "0.11.1",
            "0.11.2",
            "0.12.0",
            "0.13.0",
            "0.14.0",
            "0.14.1",
            "0.19.2",
            "0.20.2",
            "0.21.0",
            "1.0.0",
        ]
        for ver in unsupported:
            self.assertIsNone(
                _VERSION_MAP.get(_base_version(ver)),
                f"{ver} should not be in VERSION_MAP",
            )


# =============================================================================
# Plugin Registration Tests
# =============================================================================


class TestVllmPluginVersionCheck(unittest.TestCase):
    """Test version compatibility with installed vLLM."""

    @unittest.skipUnless(VLLM_AVAILABLE, "vLLM not installed")
    @unittest.skipUnless(IS_PYTHON_3_10_OR_ABOVE, "vLLM 0.11+ requires Python 3.10+")
    def test_register_returns_platform_for_installed_vllm(self):
        """register() should return platform path for the installed vLLM version."""
        from zentorch.vllm import register
        from zentorch.vllm.core import get_version_family

        family = get_version_family()
        if family is None:
            self.skipTest(f"Installed vLLM {vllm.__version__} is not supported")

        result = register()
        self.assertEqual(result, "zentorch.vllm.platform.ZenCPUPlatform")

    @unittest.skipUnless(VLLM_AVAILABLE, "vLLM not installed")
    @unittest.skipUnless(IS_PYTHON_3_10_OR_ABOVE, "vLLM 0.11+ requires Python 3.10+")
    def test_installed_vllm_version_is_supported(self):
        """Installed vLLM version should be in supported list."""
        from zentorch.vllm.core import get_version_family, _base_version

        base_ver = _base_version(vllm.__version__)
        family = get_version_family()

        self.assertIsNotNone(
            family,
            f"vLLM {vllm.__version__} (base: {base_ver}) should be supported",
        )


class TestPatchRegistration(unittest.TestCase):
    """Test that patches are registered and applied correctly."""

    @unittest.skipUnless(VLLM_AVAILABLE, "vLLM not installed")
    @unittest.skipUnless(IS_PYTHON_3_10_OR_ABOVE, "vLLM 0.11+ requires Python 3.10+")
    def test_patches_are_registered(self):
        """All expected patches should be registered with manager."""
        from zentorch.vllm import register
        from zentorch.vllm.core import manager

        register()  # Ensure patches are registered

        expected_patches = [
            "CompilationConfigRepr",
            "CPUProfiler",
        ]
        for patch_name in expected_patches:
            self.assertIn(
                patch_name,
                manager.patches,
                f"Patch {patch_name!r} should be registered",
            )

        self.assertNotIn("OneDNNDisable", manager.patches)
        self.assertNotIn("DispatchCPUUnquantizedGemm", manager.patches)

    @unittest.skipUnless(VLLM_AVAILABLE, "vLLM not installed")
    @unittest.skipUnless(IS_PYTHON_3_10_OR_ABOVE, "vLLM 0.11+ requires Python 3.10+")
    def test_version_appropriate_patches_applied(self):
        """Patches appropriate for installed vLLM version should be applied."""
        from zentorch.vllm import register
        from zentorch.vllm.core import manager, get_version_family

        register()
        family = get_version_family()

        universal_patches = ["CompilationConfigRepr"]
        for patch_name in universal_patches:
            self.assertIn(
                patch_name,
                manager.applied,
                f"Universal patch {patch_name!r} should be applied for {family}",
            )

        self.assertNotIn("OneDNNDisable", manager.applied)
        self.assertNotIn("DispatchCPUUnquantizedGemm", manager.applied)


# =============================================================================
# Individual Patch Tests
# =============================================================================


class TestPreV18DispatchHook(unittest.TestCase):
    """Test shared dispatch hook behavior for v0.15.0-v0.17.x."""

    def test_register_installs_dispatch_hooks_only_for_pre_v18_families(self):
        """register() should install dispatch hooks only for v15-v17 families."""
        pre_v18_families = {"v15", "v15_1", "v16", "v17"}

        for family in ("v15", "v15_1", "v16", "v17", "v18", "v19", "v20"):
            with self.subTest(family=family):
                spec, zv = _load_source_vllm_module()
                fake_vllm = types.ModuleType("vllm")
                fake_vllm.__version__ = "0.15.0"

                with mock.patch.dict(
                    sys.modules, {"zentorch.vllm": zv, "vllm": fake_vllm}
                ):
                    spec.loader.exec_module(zv)
                    zv._INITIALIZED = False

                    with (
                        mock.patch.object(zv, "get_version_family", return_value=family),
                        mock.patch.object(zv, "_apply_faketensor_subclass_patch"),
                        mock.patch.object(zv, "_apply_fxgraphcache_pickle_patch"),
                        mock.patch.object(zv, "_register_zentorch_linear_dispatch"),
                        mock.patch.object(zv, "_register_patches"),
                        mock.patch.object(zv.manager, "apply_all"),
                        mock.patch.object(
                            zv, "_install_pre_v18_dispatch_hooks"
                        ) as install_hooks,
                    ):
                        result = zv.register()

                    self.assertEqual(result, "zentorch.vllm.platform.ZenCPUPlatform")
                    if family in pre_v18_families:
                        install_hooks.assert_called_once_with()
                    else:
                        install_hooks.assert_not_called()

    def test_pre_v18_dispatch_patch_preserves_weight_removal(self):
        """The shared dispatch hook should preserve remove_weight semantics."""
        spec, zv = _load_source_vllm_module()

        vllm_pkg = types.ModuleType("vllm")
        vllm_pkg.__path__ = []
        model_executor_pkg = types.ModuleType("vllm.model_executor")
        model_executor_pkg.__path__ = []
        layers_pkg = types.ModuleType("vllm.model_executor.layers")
        layers_pkg.__path__ = []
        utils_module = types.ModuleType("vllm.model_executor.layers.utils")

        def original_dispatch(layer, remove_weight):
            raise AssertionError("dispatch patch did not replace original function")

        utils_module.dispatch_cpu_unquantized_gemm = original_dispatch
        vllm_pkg.model_executor = model_executor_pkg
        model_executor_pkg.layers = layers_pkg
        layers_pkg.utils = utils_module

        with mock.patch.dict(
            sys.modules,
            {
                "zentorch.vllm": zv,
                "vllm": vllm_pkg,
                "vllm.model_executor": model_executor_pkg,
                "vllm.model_executor.layers": layers_pkg,
                "vllm.model_executor.layers.utils": utils_module,
            },
        ):
            spec.loader.exec_module(zv)
            zv._do_patch_pre_v18_gemm_dispatch()

            patched_dispatch = utils_module.dispatch_cpu_unquantized_gemm
            self.assertTrue(hasattr(patched_dispatch, "_zentorch_patched"))

            layer = torch.nn.Linear(4, 3).eval()
            original_weight = layer.weight.detach().clone()
            x = torch.randn(2, 4)

            patched_dispatch(layer, remove_weight=True)

            self.assertEqual(layer.weight.numel(), 0)
            actual = layer.cpu_linear(x, layer.weight, layer.bias)
            expected = torch.nn.functional.linear(x, original_weight, layer.bias)
            self.assertTrue(torch.allclose(actual, expected))


class TestCompilationConfigPatch(unittest.TestCase):
    """Test CompilationConfig repr patch."""

    @unittest.skipUnless(VLLM_AVAILABLE, "vLLM not installed")
    @unittest.skipUnless(IS_PYTHON_3_10_OR_ABOVE, "vLLM 0.11+ requires Python 3.10+")
    def test_compilation_config_repr_is_patched(self):
        """CompilationConfig.__repr__ should have _zentorch_patched attribute."""
        from zentorch.vllm import register

        register()

        from vllm.config import CompilationConfig

        self.assertTrue(
            hasattr(CompilationConfig.__repr__, "_zentorch_patched"),
            "CompilationConfig.__repr__ should be patched",
        )

    @unittest.skipUnless(VLLM_AVAILABLE, "vLLM not installed")
    @unittest.skipUnless(IS_PYTHON_3_10_OR_ABOVE, "vLLM 0.11+ requires Python 3.10+")
    def test_compilation_config_repr_handles_custom_pass(self):
        """Patched repr should not raise errors with zentorch optimize_pass."""
        from zentorch.vllm import register

        register()

        from vllm.config import CompilationConfig
        from zentorch._compile_backend import optimize_pass

        config = CompilationConfig()
        config.inductor_compile_config["joint_custom_post_pass"] = optimize_pass

        # Should not raise
        repr_str = repr(config)
        self.assertIsInstance(repr_str, str)
        # Should be valid (not error fallback)
        self.assertNotIn("<error", repr_str)


# =============================================================================
# Platform Version Guard Tests
# =============================================================================


class TestPlatformProfilerPatchVersionRange(unittest.TestCase):
    """Test profiler patch version gating in platform.py."""

    def test_profiler_patch_range_uses_normalized_versions(self):
        """Profiler patch should use a normalized 0.15.0-0.20.1 version range."""
        from zentorch.vllm import platform

        cases = [
            (None, False),
            ("0.12.0", False),
            ("0.13.0", False),
            ("0.14.1", False),
            ("0.15.0", True),
            ("0.15.0rc1+cpu", True),
            ("0.17.1+cpu", True),
            ("0.18.0.dev1+cpu", True),
            ("0.18.1", True),
            ("0.19.0", True),
            ("0.19.0+cpu", True),
            ("0.19.1", True),
            ("0.20.0", True),
            ("0.20.0+cpu", True),
            ("0.20.0rc1+cpu", True),
            ("0.20.1", True),
            ("0.20.1+cpu", True),
            ("0.20.1rc0+cpu", True),
            ("0.20.2", False),
            ("0.21.0", False),
        ]

        for version_str, expected in cases:
            with self.subTest(version_str=version_str), mock.patch.object(
                platform, "get_vllm_version", return_value=version_str
            ):
                self.assertEqual(platform._is_profiler_patch_version(), expected)


# =============================================================================
# Platform Configuration Tests
# =============================================================================


class TestPlatformConfiguration(unittest.TestCase):
    """Test ZenCPUPlatform configuration."""

    @unittest.skipUnless(VLLM_AVAILABLE, "vLLM not installed")
    @unittest.skipUnless(IS_PYTHON_3_10_OR_ABOVE, "vLLM 0.11+ requires Python 3.10+")
    def test_platform_device_name_is_cpu(self):
        """device_name should be 'cpu'."""
        from zentorch.vllm.platform import ZenCPUPlatform

        self.assertEqual(ZenCPUPlatform.device_name, "cpu")

    @unittest.skipUnless(VLLM_AVAILABLE, "vLLM not installed")
    @unittest.skipUnless(IS_PYTHON_3_10_OR_ABOVE, "vLLM 0.11+ requires Python 3.10+")
    def test_platform_device_type_is_cpu(self):
        """device_type should be 'cpu'."""
        from zentorch.vllm.platform import ZenCPUPlatform

        self.assertEqual(ZenCPUPlatform.device_type, "cpu")


# =============================================================================
# Zentorch Component Tests
# =============================================================================


class TestDynamicQLinearDispatchPatch(unittest.TestCase):
    """Test that Int8Tensor F.linear dispatch is patched to zentorch_dynamic_qlinear."""

    @unittest.skipUnless(VLLM_AVAILABLE, "vLLM not installed")
    @unittest.skipUnless(IS_PYTHON_3_10_OR_ABOVE, "vLLM 0.11+ requires Python 3.10+")
    @unittest.skipUnless(TORCHAO_AVAILABLE, "torchao not installed")
    def test_patch_is_applied_after_register(self):
        """_DYNAMIC_QLINEAR_PATCH_APPLIED should be True after register()."""
        from zentorch.vllm import register

        register()
        from zentorch.vllm import _DYNAMIC_QLINEAR_PATCH_APPLIED

        self.assertTrue(
            _DYNAMIC_QLINEAR_PATCH_APPLIED,
            "Int8Tensor F.linear dispatch should be patched after register()",
        )


class TestDynamicQLinearDispatchNoTorchAO(unittest.TestCase):
    """Int8 linear patch must not import torchao when the package is absent."""

    def test_register_skips_without_torchao(self):
        """Load zentorch.vllm from this repo's sources (native zentorch from site-packages)."""
        spec, zv = _load_source_vllm_module()

        with mock.patch.dict(sys.modules, {"zentorch.vllm": zv}):
            spec.loader.exec_module(zv)
            with mock.patch.object(
                zv.importlib.util,
                "find_spec",
                return_value=None,
            ) as mock_find:
                zv._DYNAMIC_QLINEAR_PATCH_APPLIED = False
                zv._register_zentorch_linear_dispatch()

            mock_find.assert_called_once_with("torchao")
            self.assertFalse(
                zv._DYNAMIC_QLINEAR_PATCH_APPLIED,
                "Patch should not apply when find_spec('torchao') is None",
            )


class TestZentorchOptimizePass(unittest.TestCase):
    """Test that zentorch optimize_pass is available."""

    @unittest.skipUnless(VLLM_AVAILABLE, "vLLM not installed")
    def test_optimize_pass_is_callable(self):
        """optimize_pass should be available and callable."""
        from zentorch._compile_backend import optimize_pass

        self.assertIsNotNone(optimize_pass)
        self.assertTrue(callable(optimize_pass))


class TestCppIndirectAssertPatch(unittest.TestCase):
    """CppIndirectAssertPatch must be registered and gated to vLLM 0.20.0/0.20.1."""

    @unittest.skipUnless(VLLM_AVAILABLE, "vLLM not installed")
    @unittest.skipUnless(IS_PYTHON_3_10_OR_ABOVE, "vLLM 0.11+ requires Python 3.10+")
    def test_patch_is_registered(self):
        """CppIndirectAssertPatch should be registered with the manager."""
        from zentorch.vllm import register
        from zentorch.vllm.core import manager

        register()
        self.assertIn("CppIndirectAssert", manager.patches)

    def test_patch_targets_v20_and_v20_1(self):
        """The @vllm_version decorator should target v0.20.0 and v0.20.1."""
        from zentorch.vllm import CppIndirectAssertPatch
        from zentorch.vllm.core import VLLM_V20, VLLM_V20_1

        self.assertTrue(hasattr(CppIndirectAssertPatch, "_target_versions"))
        self.assertEqual(
            CppIndirectAssertPatch._target_versions, {VLLM_V20, VLLM_V20_1}
        )


class TestCPURunnerShutdownPatch(unittest.TestCase):
    """CPURunnerShutdownPatch must be registered, gated to v0.20.0/v0.20.1,
    and actually replace torch.accelerator.{synchronize,empty_cache} on apply.
    """

    @unittest.skipUnless(VLLM_AVAILABLE, "vLLM not installed")
    @unittest.skipUnless(IS_PYTHON_3_10_OR_ABOVE, "vLLM 0.11+ requires Python 3.10+")
    def test_patch_is_registered(self):
        from zentorch.vllm import register
        from zentorch.vllm.core import manager

        register()
        self.assertIn("CPURunnerShutdown", manager.patches)

    def test_patch_targets_v20_and_v20_1(self):
        from zentorch.vllm import CPURunnerShutdownPatch
        from zentorch.vllm.core import VLLM_V20, VLLM_V20_1

        self.assertTrue(hasattr(CPURunnerShutdownPatch, "_target_versions"))
        self.assertEqual(
            CPURunnerShutdownPatch._target_versions, {VLLM_V20, VLLM_V20_1}
        )

    def test_apply_makes_accelerator_apis_noop(self):
        """After apply(), synchronize and empty_cache must not raise on CPU."""
        import torch

        if not hasattr(torch, "accelerator"):
            self.skipTest("torch.accelerator API not present on this torch build")

        original_sync = torch.accelerator.synchronize
        original_empty = torch.accelerator.empty_cache

        from zentorch.vllm import _apply_torch_accelerator_noop_patch
        import zentorch.vllm as zv

        zv._TORCH_ACCELERATOR_NOOP_APPLIED = False
        applied = _apply_torch_accelerator_noop_patch()
        self.assertTrue(applied)

        self.assertIsNone(torch.accelerator.synchronize())
        self.assertIsNone(torch.accelerator.empty_cache())

        torch.accelerator.synchronize = original_sync
        torch.accelerator.empty_cache = original_empty
        zv._TORCH_ACCELERATOR_NOOP_APPLIED = False


# =============================================================================
# Test Runner
# =============================================================================


def run_tests():
    """Run all vLLM plugin tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestVersionParsing))
    suite.addTests(loader.loadTestsFromTestCase(TestVllmPluginVersionCheck))
    suite.addTests(loader.loadTestsFromTestCase(TestPatchRegistration))
    suite.addTests(loader.loadTestsFromTestCase(TestPreV18DispatchHook))
    suite.addTests(loader.loadTestsFromTestCase(TestCompilationConfigPatch))
    suite.addTests(loader.loadTestsFromTestCase(TestPlatformConfiguration))
    suite.addTests(loader.loadTestsFromTestCase(TestDynamicQLinearDispatchPatch))
    suite.addTests(loader.loadTestsFromTestCase(TestDynamicQLinearDispatchNoTorchAO))
    suite.addTests(loader.loadTestsFromTestCase(TestZentorchOptimizePass))
    suite.addTests(loader.loadTestsFromTestCase(TestCppIndirectAssertPatch))
    suite.addTests(loader.loadTestsFromTestCase(TestCPURunnerShutdownPatch))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result


if __name__ == "__main__":
    run_tests()
