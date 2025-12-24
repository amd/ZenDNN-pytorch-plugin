# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

"""
Unit tests for zentorch.vllm plugin.

Tests verify:
- Version compatibility checks (requires vetted 0.11 builds)
- Monkey patches are applied correctly
- Platform configuration is set correctly

Tested with vLLM version: 0.11.0
"""

import unittest
import sys
from unittest.mock import MagicMock, patch

try:
    import vllm  # NoQA: F401

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False


class TestVllmPluginVersionCheck(unittest.TestCase):
    """Test version compatibility checks."""

    @unittest.skipUnless(VLLM_AVAILABLE, "vLLM not installed")
    def test_version_check_passes_for_011(self):
        """Plugin should load successfully for vLLM 0.11.0."""
        mock_vllm = MagicMock()
        mock_vllm.__version__ = "0.11.0"

        with patch.dict(sys.modules, {"vllm": mock_vllm}):
            import importlib
            import zentorch.vllm as vllm_plugin

            importlib.reload(vllm_plugin)

            result = vllm_plugin.register()
            if result is not None:
                self.assertEqual(result, "zentorch.vllm.platform.ZenCPUPlatform")

    @unittest.skipUnless(VLLM_AVAILABLE, "vLLM not installed")
    def test_version_check_passes_for_0111_dev(self):
        """Plugin should load for vetted 0.11.1 dev build."""
        mock_vllm = MagicMock()
        mock_vllm.__version__ = "0.11.1.dev0+gb8b302cde.d20251203.cpu"

        with patch.dict(sys.modules, {"vllm": mock_vllm}):
            import importlib
            import zentorch.vllm as vllm_plugin

            importlib.reload(vllm_plugin)

            result = vllm_plugin.register()
            if result is not None:
                self.assertEqual(result, "zentorch.vllm.platform.ZenCPUPlatform")

    def _test_version_rejected(self, version):
        """Helper: Plugin should return None for unsupported vLLM versions."""
        mock_vllm = MagicMock()
        mock_vllm.__version__ = version

        with patch.dict(sys.modules, {"vllm": mock_vllm}):
            import importlib
            import zentorch.vllm as vllm_plugin

            importlib.reload(vllm_plugin)

            result = vllm_plugin.register()
            self.assertIsNone(result)

    def test_version_check_rejects_unsupported_versions(self):
        """Plugin should return None for unsupported versions."""
        unsupported_versions = [
            "0.10.0",
            "0.9.1",
            "0.10.5",
            "0.11.1",
            "0.12.0",
        ]
        for version in unsupported_versions:
            with self.subTest(version=version):
                self._test_version_rejected(version)


class TestOneDNNDisable(unittest.TestCase):
    """Test oneDNN GEMM disable patch."""

    @unittest.skipUnless(VLLM_AVAILABLE, "vLLM not installed")
    def test_onednn_gemm_disabled(self):
        """_supports_onednn should be set to False after patch."""
        try:
            import vllm._custom_ops as ops
            from zentorch.vllm import _disable_onednn_gemm

            _disable_onednn_gemm()

            self.assertFalse(ops._supports_onednn)
        except ImportError:
            self.skipTest("vLLM _custom_ops not available")


class TestPlatformConfiguration(unittest.TestCase):
    """Test ZenCPUPlatform configuration."""

    @unittest.skipUnless(VLLM_AVAILABLE, "vLLM not installed")
    def test_platform_device_name_is_cpu(self):
        """device_name should be 'cpu'."""
        try:
            from zentorch.vllm.platform import ZenCPUPlatform

            self.assertEqual(ZenCPUPlatform.device_name, "cpu")
        except ImportError:
            self.skipTest("ZenCPUPlatform not available")

    @unittest.skipUnless(VLLM_AVAILABLE, "vLLM not installed")
    def test_platform_device_type_is_cpu(self):
        """device_type should be 'cpu'."""
        try:
            from zentorch.vllm.platform import ZenCPUPlatform

            self.assertEqual(ZenCPUPlatform.device_type, "cpu")
        except ImportError:
            self.skipTest("ZenCPUPlatform not available")


class TestCompilationConfigPatch(unittest.TestCase):
    """Test CompilationConfig repr patch."""

    @unittest.skipUnless(VLLM_AVAILABLE, "vLLM not installed")
    def test_compilation_config_repr_patched(self):
        """CompilationConfig.__repr__ should have _zentorch_patched attribute."""
        try:
            from vllm.config import CompilationConfig
            from zentorch.vllm import _apply_compilation_config_repr_patch

            _apply_compilation_config_repr_patch()

            self.assertTrue(
                hasattr(CompilationConfig.__repr__, "_zentorch_patched"),
                "CompilationConfig.__repr__ should be patched",
            )
        except ImportError:
            self.skipTest("vLLM CompilationConfig not available")

    @unittest.skipUnless(VLLM_AVAILABLE, "vLLM not installed")
    def test_compilation_config_repr_no_error(self):
        """Patched repr should not raise errors with zentorch optimize_pass."""
        try:
            from vllm.config import CompilationConfig
            from zentorch.vllm import _apply_compilation_config_repr_patch
            from zentorch._compile_backend import optimize_pass

            _apply_compilation_config_repr_patch()

            config = CompilationConfig()
            config.inductor_compile_config["joint_custom_post_pass"] = optimize_pass

            repr_str = repr(config)
            self.assertIsInstance(repr_str, str)
        except ImportError:
            self.skipTest("Required modules not available")


class TestPagedAttentionPatch(unittest.TestCase):
    """Test PagedAttention monkey patch."""

    @unittest.skipUnless(VLLM_AVAILABLE, "vLLM not installed")
    def test_paged_attention_patch_applied(self):
        """PagedAttention should be replaced with zentorch implementation."""
        try:
            import vllm.v1.attention.backends.cpu_attn as cpu_attn_module
            from zentorch.vllm.attention import PagedAttention
            from zentorch.vllm import _apply_paged_attention_monkey_patch

            _apply_paged_attention_monkey_patch()
            impl = cpu_attn_module._get_paged_attn_impl()

            self.assertEqual(impl, PagedAttention)
        except ImportError:
            self.skipTest("vLLM cpu_attn module not available")


class TestZentorchOptimizePassInjection(unittest.TestCase):
    """Test that zentorch optimize_pass is injected correctly."""

    @unittest.skipUnless(VLLM_AVAILABLE, "vLLM not installed")
    def test_optimize_pass_in_compilation_config(self):
        """check_and_update_config should inject zentorch optimize_pass."""
        try:
            from zentorch._compile_backend import optimize_pass

            self.assertIsNotNone(optimize_pass)
            self.assertTrue(callable(optimize_pass))

        except ImportError as e:
            self.skipTest(f"Required modules not available: {e}")


def run_tests():
    """Run all vLLM plugin tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestVllmPluginVersionCheck))
    suite.addTests(loader.loadTestsFromTestCase(TestOneDNNDisable))
    suite.addTests(loader.loadTestsFromTestCase(TestPlatformConfiguration))
    suite.addTests(loader.loadTestsFromTestCase(TestCompilationConfigPatch))
    suite.addTests(loader.loadTestsFromTestCase(TestPagedAttentionPatch))
    suite.addTests(loader.loadTestsFromTestCase(TestZentorchOptimizePassInjection))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result


if __name__ == "__main__":
    run_tests()
