# ******************************************************************************
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

"""
Unit tests for zentorch.vllm plugin.

Tests verify:
- Version compatibility checks for supported versions (0.11.x, 0.12.x, 0.13.x)
- Version parsing logic
- Patch registration and application
- Individual patch functionality (oneDNN disable, CompilationConfig repr, etc.)
- Platform configuration

Supported vLLM versions: 0.11.0, 0.11.1, 0.11.2, 0.12.0, 0.13.0
"""

import unittest

try:
    import vllm  # NoQA: F401

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False


# =============================================================================
# Version Parsing Tests
# =============================================================================


class TestVersionParsing(unittest.TestCase):
    """Test version parsing logic in core.py."""

    def test_base_version_strips_suffixes(self):
        """_base_version should strip dev/rc/local suffixes."""
        from zentorch.vllm.core import _base_version

        self.assertEqual(_base_version("0.11.0"), "0.11.0")
        self.assertEqual(_base_version("0.11.1.dev0+cpu"), "0.11.1")
        self.assertEqual(_base_version("0.12.0.dev1+gb8b302cde.d20251203.cpu"), "0.12.0")
        self.assertEqual(_base_version("0.13.0rc1+cpu"), "0.13.0")
        self.assertEqual(_base_version("0.13.0rc0"), "0.13.0")

    def test_version_map_contains_supported_versions(self):
        """VERSION_MAP should contain all supported base versions."""
        from zentorch.vllm.core import _VERSION_MAP

        expected_versions = ["0.11.0", "0.11.1", "0.11.2", "0.12.0", "0.13.0"]
        for ver in expected_versions:
            self.assertIn(ver, _VERSION_MAP, f"{ver} should be in VERSION_MAP")

    def test_version_family_detection_supported(self):
        """VERSION_MAP should return correct family for supported versions."""
        from zentorch.vllm.core import _base_version, _VERSION_MAP

        # v11 family (0.11.0)
        self.assertEqual(_VERSION_MAP.get(_base_version("0.11.0")), "v11")

        # v11_1 family (0.11.1)
        self.assertEqual(_VERSION_MAP.get(_base_version("0.11.1")), "v11_1")
        self.assertEqual(_VERSION_MAP.get(_base_version("0.11.1.dev0+cpu")), "v11_1")

        # v11_2 family (0.11.2)
        self.assertEqual(_VERSION_MAP.get(_base_version("0.11.2")), "v11_2")

        # v12 family
        self.assertEqual(_VERSION_MAP.get(_base_version("0.12.0")), "v12")
        self.assertEqual(_VERSION_MAP.get(_base_version("0.12.0.dev1+cpu")), "v12")

        # v13 family
        self.assertEqual(_VERSION_MAP.get(_base_version("0.13.0")), "v13")
        self.assertEqual(_VERSION_MAP.get(_base_version("0.13.0rc1+cpu")), "v13")

    def test_version_family_detection_unsupported(self):
        """VERSION_MAP should return None for unsupported versions."""
        from zentorch.vllm.core import _base_version, _VERSION_MAP

        unsupported = ["0.9.1", "0.10.0", "0.10.5", "0.14.0", "1.0.0"]
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
    def test_patches_are_registered(self):
        """All expected patches should be registered with manager."""
        from zentorch.vllm import register
        from zentorch.vllm.core import manager

        register()  # Ensure patches are registered

        expected_patches = [
            "PagedAttention",
            "IPEXFlashAttention",
            "CompilationConfigRepr",
            "OneDNNDisable",
            "InternVLDtype",
            "CPUProfiler",
            "CPUProfilerV12",
            "CPUProfilerV13",
        ]
        for patch_name in expected_patches:
            self.assertIn(
                patch_name,
                manager.patches,
                f"Patch {patch_name!r} should be registered",
            )

    @unittest.skipUnless(VLLM_AVAILABLE, "vLLM not installed")
    def test_version_appropriate_patches_applied(self):
        """Patches appropriate for installed vLLM version should be applied."""
        from zentorch.vllm import register
        from zentorch.vllm.core import manager, get_version_family

        register()
        family = get_version_family()

        # These patches apply to all versions (0.11-0.13)
        universal_patches = ["CompilationConfigRepr", "OneDNNDisable"]
        for patch_name in universal_patches:
            self.assertIn(
                patch_name,
                manager.applied,
                f"Universal patch {patch_name!r} should be applied for {family}",
            )


# =============================================================================
# Individual Patch Tests
# =============================================================================


class TestOneDNNDisablePatch(unittest.TestCase):
    """Test oneDNN GEMM disable patch."""

    @unittest.skipUnless(VLLM_AVAILABLE, "vLLM not installed")
    def test_onednn_gemm_disabled_after_register(self):
        """_supports_onednn should be False after register() is called."""
        from zentorch.vllm import register

        register()

        import vllm._custom_ops as ops

        self.assertFalse(
            ops._supports_onednn,
            "_supports_onednn should be False to use zentorch linear",
        )


class TestCompilationConfigPatch(unittest.TestCase):
    """Test CompilationConfig repr patch."""

    @unittest.skipUnless(VLLM_AVAILABLE, "vLLM not installed")
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
# Platform Configuration Tests
# =============================================================================


class TestPlatformConfiguration(unittest.TestCase):
    """Test ZenCPUPlatform configuration."""

    @unittest.skipUnless(VLLM_AVAILABLE, "vLLM not installed")
    def test_platform_device_name_is_cpu(self):
        """device_name should be 'cpu'."""
        from zentorch.vllm.platform import ZenCPUPlatform

        self.assertEqual(ZenCPUPlatform.device_name, "cpu")

    @unittest.skipUnless(VLLM_AVAILABLE, "vLLM not installed")
    def test_platform_device_type_is_cpu(self):
        """device_type should be 'cpu'."""
        from zentorch.vllm.platform import ZenCPUPlatform

        self.assertEqual(ZenCPUPlatform.device_type, "cpu")


# =============================================================================
# Zentorch Component Tests
# =============================================================================


class TestZentorchOptimizePass(unittest.TestCase):
    """Test that zentorch optimize_pass is available."""

    @unittest.skipUnless(VLLM_AVAILABLE, "vLLM not installed")
    def test_optimize_pass_is_callable(self):
        """optimize_pass should be available and callable."""
        from zentorch._compile_backend import optimize_pass

        self.assertIsNotNone(optimize_pass)
        self.assertTrue(callable(optimize_pass))


class TestPagedAttention(unittest.TestCase):
    """Test PagedAttention implementation."""

    def test_paged_attention_supported_head_sizes(self):
        """PagedAttention should report supported head sizes."""
        from zentorch.vllm.attention import PagedAttention

        supported = PagedAttention.get_supported_head_sizes()
        self.assertIsInstance(supported, list)
        self.assertIn(64, supported)
        self.assertIn(128, supported)

    def test_paged_attention_repr(self):
        """PagedAttention repr should identify zentorch backend."""
        from zentorch.vllm.attention import PagedAttention

        pa = PagedAttention()
        repr_str = repr(pa)
        self.assertIn("zentorch", repr_str)
        self.assertIn("cpu", repr_str)

    def test_paged_attention_has_required_methods(self):
        """PagedAttention should have all required static methods."""
        from zentorch.vllm.attention import PagedAttention

        required_methods = [
            "get_supported_head_sizes",
            "get_kv_cache_shape",
            "split_kv_cache",
            "write_to_paged_cache",
            "flash_attn_varlen_func",
            "forward_decode",
        ]
        for method in required_methods:
            self.assertTrue(
                hasattr(PagedAttention, method),
                f"PagedAttention should have {method}",
            )


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
    suite.addTests(loader.loadTestsFromTestCase(TestOneDNNDisablePatch))
    suite.addTests(loader.loadTestsFromTestCase(TestCompilationConfigPatch))
    suite.addTests(loader.loadTestsFromTestCase(TestPlatformConfiguration))
    suite.addTests(loader.loadTestsFromTestCase(TestZentorchOptimizePass))
    suite.addTests(loader.loadTestsFromTestCase(TestPagedAttention))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result


if __name__ == "__main__":
    run_tests()
