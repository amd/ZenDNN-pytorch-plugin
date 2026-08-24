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
    ("RMSNorm", "vllm.model_executor.layers.layernorm", "RMSNorm",
     "_zentorch_rmsnorm_patched", None),
    ("FusedMoE", "vllm.model_executor.layers.fused_moe.cpu_fused_moe",
     "CPUFusedMOE", "_zentorch_fused_moe_patched", None),
    ("Int8MoE",
     "vllm.model_executor.layers.quantization.compressed_tensors."
     "compressed_tensors_moe.compressed_tensors_moe_w8a8_int8",
     "CompressedTensorsW8A8Int8MoEMethod", "_zentorch_int8_moe_patched", None),
    ("GptOssMoELoader", "vllm.model_executor.models.gpt_oss", "GptOssModel",
     "_zentorch_gptoss_loader_patched", "_load_weights_other"),
    ("MixtralMoELoader", "vllm.model_executor.models.mixtral", "MixtralModel",
     "_zentorch_mixtral_loader_patched", "load_weights"),
    ("TorchAO", "vllm.model_executor.layers.quantization.torchao",
     "TorchAOConfig", "_zentorch_moe_patched", None),
    ("Da8w4Kernel", "vllm.model_executor.kernels.linear.mixed_precision.zentorch",
     "ZentorchWNA16LinearKernel", "_zentorch_da8w4_patched", None),
]

# Gemma-4 wraps get_config by name in each of these; marker sits on the module.
_GEMMA_MODULES = [
    "vllm.transformers_utils.config",
    "vllm.config.model",
    "vllm.tokenizers.registry",
]

# Hook-install state to clear so register() re-applies cleanly in this process.
_OWN_HOOK_FLAGS = [
    ("zentorch.vllm._gptoss_moe_loader_patch", "_HOOK_INSTALLED"),
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
