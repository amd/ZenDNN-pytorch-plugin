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


if __name__ == "__main__":
    unittest.main()
