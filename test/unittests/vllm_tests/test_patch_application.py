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

from packaging import version as pkg_version

from ._test_constants import VLLM_AVAILABLE, vllm


# (patch_name, target_module, target_attr_or_None, marker_attr, required_attr)
# required_attr: the method the patch wraps. When the real target lacks it the
# patch is INAPPLICABLE on this build (the patch itself no-ops), so the test skips
# with a reason instead of failing. None => the patch must always apply.
_PATCH_TARGETS = [
    ("CPUSdpa", "vllm.v1.attention.backends.cpu_attn",
     "CPUAttentionBackendImpl", "_zentorch_sdpa_patched", "forward"),
    # Marker and wrapped function both sit on the module, not on a class.
    ("MMEncoderSdpa", "vllm.v1.attention.ops.vit_attn_wrappers",
     None, "_zentorch_mm_encoder_sdpa_patched", "apply_sdpa"),
    ("Int8MoE",
     "vllm.model_executor.layers.quantization.compressed_tensors."
     "compressed_tensors_moe.compressed_tensors_moe_w8a8_int8",
     "CompressedTensorsW8A8Int8MoEMethod", "_zentorch_int8_moe_patched", None),
    ("Wna16MoEOracle",
     "vllm.model_executor.layers.fused_moe.oracle.int_wna16",
     None, "_zentorch_wna16_oracle_patched", "backend_to_kernel_cls"),
    ("Wna16MoEMethod",
     "vllm.model_executor.layers.quantization.compressed_tensors."
     "compressed_tensors_moe.compressed_tensors_moe_wna16",
     "CompressedTensorsWNA16MoEMethod", "_zentorch_wna16_moe_patched",
     "process_weights_after_loading"),
    ("MixtralMoELoader", "vllm.model_executor.models.mixtral", "MixtralModel",
     "_zentorch_mixtral_loader_patched", "load_weights"),
    ("TorchAO", "vllm.model_executor.layers.quantization.torchao",
     "TorchAOConfig", "_zentorch_moe_patched", None),
    ("Da8w4Kernel", "vllm.model_executor.kernels.linear.mixed_precision.zentorch",
     "ZentorchWNA16LinearKernel", "_zentorch_da8w4_patched", None),
    ("Wna16CanImplement",
     "vllm.model_executor.kernels.linear.mixed_precision.zentorch",
     "ZentorchWNA16LinearKernel", "_zentorch_wna16_can_implement_patched",
     "can_implement"),
    ("WhisperW4A16", "vllm.model_executor.models.whisper",
     "WhisperForConditionalGeneration", "_zentorch_whisper_w4a16_patched",
     "load_weights"),
    ("WhisperTruncation", "vllm.model_executor.models.whisper",
     "WhisperMultiModalProcessor", "_zentorch_whisper_truncation_patched",
     ("_preprocess_hf_mm_data", "_get_hf_mm_inputs")),
    ("GptOssStreamedExpert", "vllm.model_executor.models.gpt_oss", "GptOssModel",
     "_zentorch_gptoss_streamed_patched", "_load_weights_other"),
    ("GatedDeltaNet",
     "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn",
     "QwenGatedDeltaNetAttention", "_zentorch_gdn_patched", "forward_cpu"),
]

# Gemma-4 wraps get_config by name in each of these; marker sits on the module.
_GEMMA_MODULES = [
    "vllm.transformers_utils.config",
    "vllm.config.model",
    "vllm.tokenizers.registry",
]

# Pre-0.29 import-hook state to reset when testing those releases.
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
        ih._fns.clear()
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
        # Skip instead of fail when this vLLM build has none of the methods
        # the patch wraps (the patch no-ops). required_attr may be one name or
        # a tuple of alternatives, e.g. Whisper 0.29 `_preprocess_hf_mm_data`
        # vs later `_get_hf_mm_inputs`; any one of them is enough to apply.
        if required_attr is not None:
            needed = (
                required_attr
                if isinstance(required_attr, tuple)
                else (required_attr,)
            )
            if not any(hasattr(target, hook) for hook in needed):
                self.skipTest(
                    f"{name}: {modname}.{attr or ''} lacks {needed} on this "
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

    def test_qwen3_vl_text_config_backport_scope(self):
        import zentorch.vllm as plugin

        current = pkg_version.parse(vllm.__version__.split("+")[0])
        if current == pkg_version.parse("0.29.0"):
            self._assert_applied(
                "Qwen3VLTextConfig",
                "vllm.config.vllm",
                "VllmConfig",
                "_zentorch_qwen3_vl_text_config_patched",
            )
        else:
            self.assertNotIn(
                "Qwen3VLTextConfig",
                plugin.APPLIED_PATCHES,
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

    def test_legacy_fused_moe_patch_scope(self):
        import zentorch.vllm as plugin

        current = pkg_version.parse(vllm.__version__.split("+")[0])
        if current < pkg_version.parse("0.28.0"):
            self._assert_applied(
                "FusedMoE",
                "vllm.model_executor.layers.fused_moe.cpu_fused_moe",
                "CPUFusedMOE",
                "_zentorch_fused_moe_patched",
            )
        elif current < pkg_version.parse("0.29.0"):
            self.assertIn("FusedMoE", plugin.APPLIED_PATCHES)
        else:
            self.assertNotIn("FusedMoE", plugin.APPLIED_PATCHES)

    def test_sliding_window_alignment_scope(self):
        import zentorch.vllm as plugin

        current = pkg_version.parse(vllm.__version__.split("+")[0])
        if current < pkg_version.parse("0.29.0"):
            self._assert_applied(
                "SWBlockSize",
                "vllm.model_executor.layers.attention.attention",
                "Attention",
                "_zentorch_sw_blocksize_patched",
                "get_kv_cache_spec",
            )
        else:
            from vllm.v1.attention.backend import MultipleOf
            from vllm.v1.attention.backends.cpu_attn import (
                CPUAttentionBackend,
            )

            self.assertNotIn("SWBlockSize", plugin.APPLIED_PATCHES)
            supported_sizes = (
                CPUAttentionBackend.get_supported_kernel_block_sizes()
            )
            self.assertTrue(
                any(
                    isinstance(size, MultipleOf) and size.base == 32
                    for size in supported_sizes
                )
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
