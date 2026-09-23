# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************
"""Oracle gates for the out-of-tree W8A8 INT8 fused-MoE patch."""

import types
import unittest

import zentorch  # noqa: F401

from ._test_constants import VLLM_AVAILABLE


@unittest.skipUnless(VLLM_AVAILABLE, "vLLM not installed")
class TestInt8OracleSupportGates(unittest.TestCase):
    """CPUInt8Experts is nested in the register function; build it the same way."""

    def _experts_cls(self):
        from zentorch.vllm import _int8_moe_patch as zi

        def _apply_monolithic(self, *a, **k):
            raise NotImplementedError

        mod = types.ModuleType(zi._TARGET_MODULE)
        mod.CompressedTensorsW8A8Int8MoEMethod = type(
            zi._TARGET_CLASS,
            (),
            {
                "__init__": lambda self, *a, **k: None,
                "create_weights": lambda self, *a, **k: None,
                "get_fused_moe_quant_config": lambda self, layer: None,
                "process_weights_after_loading": lambda self, layer: None,
                "apply_monolithic": _apply_monolithic,
            },
        )
        zi._register_int8_moe_patches(mod)
        return mod.CompressedTensorsW8A8Int8MoEMethod().experts_cls

    def test_custom_routing_is_accepted(self):
        """Gemma4 uses RoutingMethodType.Custom; the INT8 oracle must opt in."""
        from vllm.model_executor.layers.fused_moe.config import RoutingMethodType

        cls = self._experts_cls()
        self.assertTrue(
            cls._supports_routing_method(RoutingMethodType.Custom, None, None)
        )

    def test_plugin_experts_reject_expert_parallelism(self):
        from vllm.model_executor.layers.fused_moe.experts import cpu_moe

        cls = self._experts_cls()
        expected = not hasattr(cpu_moe, "ZenCPUExpertsInt8")
        self.assertEqual(
            cls._supports_parallel_config(types.SimpleNamespace(ep_size=2)),
            expected,
        )

    def test_native_zen_experts_are_preserved_for_default_routing(self):
        from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
        from vllm.model_executor.layers.fused_moe.experts import cpu_moe
        from zentorch.vllm import _int8_moe_patch as patch

        if not hasattr(cpu_moe, "ZenCPUExpertsInt8"):
            self.skipTest("native Zen INT8 experts start in vLLM 0.29")
        ZenCPUExpertsInt8 = cpu_moe.ZenCPUExpertsInt8

        sentinel = object()
        processed = []

        class FakeMethod:
            def __init__(self, weight_quant, input_quant, moe):
                self.moe = moe
                self.experts_cls = ZenCPUExpertsInt8

            def create_weights(self, *args, **kwargs):
                return None

            def get_fused_moe_quant_config(self, layer):
                return sentinel

            def process_weights_after_loading(self, layer):
                processed.append(layer)

            def apply_monolithic(self, *args, **kwargs):
                return None

        module = types.ModuleType(patch._TARGET_MODULE)
        setattr(module, patch._TARGET_CLASS, FakeMethod)
        patch._register_int8_moe_patches(module)

        moe = types.SimpleNamespace(
            routing_method=RoutingMethodType.Default
        )
        method = FakeMethod(None, None, moe)
        layer = object()

        self.assertIs(method.experts_cls, ZenCPUExpertsInt8)
        self.assertTrue(method._zentorch_uses_native_int8)
        self.assertIs(method.get_fused_moe_quant_config(layer), sentinel)
        method.process_weights_after_loading(layer)
        self.assertEqual(processed, [layer])


if __name__ == "__main__":
    unittest.main()
