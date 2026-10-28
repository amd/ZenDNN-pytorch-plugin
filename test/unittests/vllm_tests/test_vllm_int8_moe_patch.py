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

    def test_apply_accepts_every_base_apply_parameter(self):
        """The monolithic kernel calls ``apply`` with the base's keywords."""
        import inspect

        import vllm.model_executor.layers.fused_moe.modular_kernel as mk

        base = inspect.signature(mk.FusedMoEExpertsMonolithic.apply).parameters
        ours = inspect.signature(self._experts_cls().apply).parameters
        self.assertEqual([p for p in base if p not in ours], [])

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


@unittest.skipUnless(VLLM_AVAILABLE, "vLLM not installed")
class TestPadMoEWeightsForInplace(unittest.TestCase):
    """The op does the padding; this is the walk over a layer's projections."""

    NAMES = ("w13_weight", "w2_weight")

    def test_both_projections_are_replaced(self):
        import torch
        from zentorch.vllm._int8_moe_patch import _pad_moe_weights_for_inplace

        E, N, K = 3, 64, 32
        layer = torch.nn.Module()
        for name in self.NAMES:
            w = torch.randint(-128, 128, (E, N, K), dtype=torch.int8)
            setattr(layer, name, torch.nn.Parameter(w, requires_grad=False))
        before = {n: getattr(layer, n).data.clone() for n in self.NAMES}

        _pad_moe_weights_for_inplace(layer)

        for name, original in before.items():
            padded = getattr(layer, name).data
            self.assertGreater(padded.stride(0), N * K)
            self.assertTrue(torch.equal(padded, original))

    def test_ineligible_projections_are_left_alone(self):
        import torch
        from zentorch.vllm._int8_moe_patch import _pad_moe_weights_for_inplace

        layer = torch.nn.Module()
        layer.w13_weight = torch.nn.Parameter(
            torch.zeros(2, 4, 8, dtype=torch.bfloat16), requires_grad=False
        )
        kept = layer.w13_weight
        _pad_moe_weights_for_inplace(layer)
        self.assertIs(layer.w13_weight, kept)


if __name__ == "__main__":
    unittest.main()
