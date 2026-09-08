# ****************************************************************************
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ****************************************************************************

"""
Application tests: prove each patch actually lands on the REAL installed vLLM.

The other suites check register() wiring and per-patch behavior against mocks;
this one imports the genuine vLLM target after register() and asserts the patch's
idempotency marker is present on it. Meaningful only where a supported vLLM is
installed (e.g. the build-verify env) -- it skips cleanly otherwise, and running
it in each supported-version env is how "does it apply on this vLLM variant" is
answered.
"""

import importlib
import unittest
import unittest.mock

from ._test_constants import VLLM_AVAILABLE, vllm


# (patch_name, target_module, target_attr_or_None, marker_attr, required_attr)
# required_attr: the method the patch wraps. When the real target lacks it the
# patch is INAPPLICABLE on this build (the patch itself no-ops), so the test skips
# with a reason instead of failing. None => the patch must always apply.
_PATCH_TARGETS = [
    ("FusedMoE", "vllm.model_executor.layers.fused_moe.cpu_fused_moe",
     "CPUFusedMOE", "_zentorch_fused_moe_patched", None),
    ("CPUSdpa", "vllm.v1.attention.backends.cpu_attn",
     "CPUAttentionBackendImpl", "_zentorch_sdpa_patched", "forward"),
    ("Int8MoE",
     "vllm.model_executor.layers.quantization.compressed_tensors."
     "compressed_tensors_moe.compressed_tensors_moe_w8a8_int8",
     "CompressedTensorsW8A8Int8MoEMethod", "_zentorch_int8_moe_patched", None),
    ("MixtralMoELoader", "vllm.model_executor.models.mixtral", "MixtralModel",
     "_zentorch_mixtral_loader_patched", "load_weights"),
    ("TorchAO", "vllm.model_executor.layers.quantization.torchao",
     "TorchAOConfig", "_zentorch_moe_patched", None),
    ("Da8w4Kernel", "vllm.model_executor.kernels.linear.mixed_precision.zentorch",
     "ZentorchWNA16LinearKernel", "_zentorch_da8w4_patched", None),
    ("WhisperW4A16", "vllm.model_executor.models.whisper",
     "WhisperForConditionalGeneration", "_zentorch_whisper_w4a16_patched",
     "load_weights"),
    ("SWBlockSize",
     "vllm.model_executor.layers.attention.attention",
     "Attention", "_zentorch_sw_blocksize_patched",
     "get_kv_cache_spec"),
    ("GptOssStreamedExpert", "vllm.model_executor.models.gpt_oss", "GptOssModel",
     "_zentorch_gptoss_streamed_patched", "_load_weights_other"),
]

# Gemma-4 wraps get_config by name in each of these; marker sits on the module.
_GEMMA_MODULES = [
    "vllm.transformers_utils.config",
    "vllm.config.model",
    "vllm.tokenizers.registry",
]

# Hook-install state to clear so register() re-applies cleanly in this process.
_OWN_HOOK_FLAGS = [
    ("zentorch.vllm._moe_class", "_MOE_HOOK_INSTALLED"),
]


@unittest.skipUnless(VLLM_AVAILABLE, "vLLM not installed")
class TestPatchApplication(unittest.TestCase):
    """Each patch's marker is present on the real vLLM target after register()."""

    @classmethod
    def setUpClass(cls):
        import zentorch.vllm as zv

        if not zv.is_supported_vllm(vllm.__version__):
            raise unittest.SkipTest(
                f"vLLM {vllm.__version__} outside the supported window"
            )

        # Reset hook state so register() re-installs/re-applies in this process.
        import zentorch.vllm._import_hook as ih

        ih._handled.clear()
        for modname, flag in _OWN_HOOK_FLAGS:
            setattr(importlib.import_module(modname), flag, False)

        # AVX-512 gates register(); mock it True so patch INSTALLATION proceeds
        # on any CI (only kernel EXECUTION needs the hardware).
        zv._INITIALIZED = False
        with unittest.mock.patch(
            "zentorch._C.is_avx512_supported", return_value=True
        ):
            platform = zv.register()
        assert platform is not None, "register() returned None under the AVX-512 mock"

    def _assert_applied(self, name, modname, attr, marker, required_attr=None):
        try:
            mod = importlib.import_module(modname)  # fires the deferred hook
        except Exception as exc:  # feature absent on this build -> not applicable
            self.skipTest(f"{name}: {modname} not importable ({exc})")
        target = getattr(mod, attr) if attr else mod
        # If the method the patch wraps is absent, the patch is inapplicable on
        # this vLLM build (it no-ops); document via skip rather than fail.
        if required_attr is not None and not hasattr(target, required_attr):
            self.skipTest(
                f"{name}: {modname}.{attr or ''} lacks {required_attr} on this "
                f"vLLM -- patch inapplicable"
            )
        self.assertTrue(
            getattr(target, marker, False),
            f"{name}: patch did NOT apply -- {modname}.{attr or ''} lacks {marker}",
        )

    def test_gemma4_get_config_patched_on_every_module(self):
        for modname in _GEMMA_MODULES:
            with self.subTest(module=modname):
                self._assert_applied(
                    "Gemma4HeteroConfig", modname, None,
                    "_zentorch_gemma4_hetero_patched",
                )

    def test_applies_RMSNorm(self):
        """zentorch registers an IR provider for the residual
        ``fused_add_rms_norm`` (the platform raises it above native); the
        non-residual ``rms_norm`` stays on native.

        RMSNorm applies by registering a provider in the ``vllm.ir`` op stack
        (a dict), not by a class-attr marker, so it doesn't fit the generic
        ``_PATCH_TARGETS`` tuple -- hence a dedicated check, like Gemma4.
        """
        from vllm import ir

        self.assertIn(
            "zentorch", ir.ops.fused_add_rms_norm.impls,
            "RMSNorm: zentorch IR provider not registered for fused_add_rms_norm",
        )
        self.assertNotIn(
            "zentorch", ir.ops.rms_norm.impls,
            "RMSNorm: non-residual rms_norm should stay on native",
        )


# One independently-skippable test method per marker-based patch.
def _make_test(name, modname, attr, marker, required_attr):
    def test(self):
        self._assert_applied(name, modname, attr, marker, required_attr)

    test.__name__ = f"test_applies_{name}"
    return test


for _n, _m, _a, _mk, _req in _PATCH_TARGETS:
    setattr(TestPatchApplication, f"test_applies_{_n}",
            _make_test(_n, _m, _a, _mk, _req))


if __name__ == "__main__":
    unittest.main()
