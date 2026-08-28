# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import sys
import types
import unittest
import unittest.mock

import torch

from zentorch.vllm import _ir_rms_norm


class _FakeOp:
    """Records providers registered via ``register_impl``."""

    def __init__(self):
        self.impls = {}

    def register_impl(self, provider, **_):
        def deco(fn):
            self.impls[provider] = fn
            return fn

        return deco


class TestRMSNormIRProvider(unittest.TestCase):
    def test_registers_zentorch_for_fused_add_only(self):
        fused, rms = _FakeOp(), _FakeOp()
        ir = types.ModuleType("vllm.ir")
        ir.ops = types.SimpleNamespace(fused_add_rms_norm=fused, rms_norm=rms)
        vllm = types.ModuleType("vllm")
        vllm.ir = ir

        _ir_rms_norm._ZENTORCH_IR_NORM_REGISTERED = False
        with unittest.mock.patch.dict(sys.modules, {"vllm": vllm, "vllm.ir": ir}):
            self.assertTrue(_ir_rms_norm.register_zentorch_ir_norm_impls())
            self.assertTrue(_ir_rms_norm.register_zentorch_ir_norm_impls())  # idempotent

        self.assertIn("zentorch", fused.impls)      # residual op -> zentorch
        self.assertNotIn("zentorch", rms.impls)     # non-residual stays native

    def test_supports_args_gating(self):
        sa = _ir_rms_norm._add_rms_supports_args
        x, r, w = torch.randn(4, 8), torch.randn(4, 8), torch.ones(8)
        self.assertTrue(sa(x, r, w, 1e-6))
        self.assertFalse(sa(x, r, w, 1e-6, 4))             # variance_size override
        self.assertFalse(sa(x, r, None, 1e-6))            # weightless
        self.assertFalse(sa(x, torch.randn(4, 16), w, 1e-6))  # shape mismatch


if __name__ == "__main__":
    unittest.main()
