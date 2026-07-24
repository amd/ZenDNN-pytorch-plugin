# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

"""Regression test for the TorchAO MoE quant-method patch in
``zentorch/vllm/_moe_class.py``."""

import unittest
import zentorch  # noqa: F401

from ._test_constants import TORCHAO_AVAILABLE, VLLM_AVAILABLE


@unittest.skipUnless(
    VLLM_AVAILABLE and TORCHAO_AVAILABLE,
    "requires vLLM and torchao",
)
class TestTorchAOMoEQuantMethodPatch(unittest.TestCase):
    """Covers the ``_moe_class.py`` TorchAO MoE quant-method patch."""

    def _install_patch(self, torchao_mod, should_skip):
        """Register the TorchAO MoE patches and confirm they installed.

        Snapshots every attribute the patch mutates and restores it on
        teardown so tests sharing this process cannot leak state into one
        another. ``should_skip`` selects the skip -> unquantized path
        (``True``) or the quantized MoE path (``False``); the patch captures
        ``should_skip`` at registration time, so it is set before registering.
        """
        from zentorch.vllm import _moe_class

        TorchAOConfig = torchao_mod.TorchAOConfig

        orig_get_quant_method = TorchAOConfig.__dict__.get("get_quant_method")
        orig_resolve = TorchAOConfig.__dict__.get(
            "_resolve_torchao_config_for_prefix"
        )
        orig_moe_method = getattr(torchao_mod, "TorchAOFusedMoEMethod", None)
        orig_should_skip = torchao_mod.should_skip

        def restore():
            torchao_mod.should_skip = orig_should_skip

            if orig_get_quant_method is not None:
                TorchAOConfig.get_quant_method = orig_get_quant_method
            elif "get_quant_method" in TorchAOConfig.__dict__:
                del TorchAOConfig.get_quant_method

            if orig_resolve is not None:
                TorchAOConfig._resolve_torchao_config_for_prefix = orig_resolve
            elif "_resolve_torchao_config_for_prefix" in TorchAOConfig.__dict__:
                del TorchAOConfig._resolve_torchao_config_for_prefix

            if orig_moe_method is not None:
                torchao_mod.TorchAOFusedMoEMethod = orig_moe_method
            elif hasattr(torchao_mod, "TorchAOFusedMoEMethod"):
                del torchao_mod.TorchAOFusedMoEMethod

        self.addCleanup(restore)

        torchao_mod.should_skip = lambda prefix, skip_modules: should_skip

        _moe_class._register_torchao_moe_patches(torchao_mod)
        self.assertEqual(
            TorchAOConfig.get_quant_method.__name__,
            "_patched_get_quant_method",
            "patch did not install _patched_get_quant_method",
        )
        return TorchAOConfig

    def test_get_quant_method_routes_linear_layer_to_unquantized(self):
        from vllm.model_executor.layers.linear import (
            LinearBase,
            UnquantizedLinearMethod,
        )
        from vllm.model_executor.layers.quantization import (
            torchao as torchao_mod,
        )

        TorchAOConfig = self._install_patch(torchao_mod, should_skip=True)

        # Non-MoE LinearBase layer and a TorchAOConfig built without invoking
        # version-varying constructors.
        config = TorchAOConfig.__new__(TorchAOConfig)
        config.skip_modules = []
        linear_layer = LinearBase.__new__(LinearBase)
        prefix = "model.layers.0.self_attn.qkv_proj"

        method = config.get_quant_method(linear_layer, prefix)
        self.assertIsInstance(
            method,
            UnquantizedLinearMethod,
            "A skipped LinearBase layer should resolve to "
            "UnquantizedLinearMethod.",
        )

    def test_get_quant_method_routes_moe_layer_to_moe_method(self):
        from vllm.model_executor.layers.quantization import (
            torchao as torchao_mod,
        )
        from zentorch.vllm._moe_class import _resolve_moe_layer_types

        # Single source of truth shared with the source patch.
        moe_layer_types = _resolve_moe_layer_types()
        self.assertTrue(
            moe_layer_types,
            "No MoE layer type resolved from the installed vLLM. The TorchAO "
            "MoE patch likely needs an update for this vLLM version.",
        )
        moe_cls = moe_layer_types[0]

        TorchAOConfig = self._install_patch(torchao_mod, should_skip=False)

        # Stub the MoE quant-method with a sentinel so we prove, by identity,
        # that the MoE branch was taken and this factory was invoked.
        sentinel = object()
        torchao_mod.TorchAOFusedMoEMethod = lambda quant_config, moe: sentinel

        # A plain object as torchao_config makes the prefix resolver return it
        # as-is (not a ModuleFqnToConfig), so resolution is deterministic.
        config = TorchAOConfig.__new__(TorchAOConfig)
        config.skip_modules = []
        config.torchao_config = object()

        moe_layer = moe_cls.__new__(moe_cls)
        moe_layer.moe_config = object()

        result = config.get_quant_method(moe_layer, prefix="model.layers.0.mlp")
        self.assertIs(
            result,
            sentinel,
            "A non-skipped MoE layer should route to TorchAOFusedMoEMethod.",
        )


if __name__ == "__main__":
    unittest.main()
