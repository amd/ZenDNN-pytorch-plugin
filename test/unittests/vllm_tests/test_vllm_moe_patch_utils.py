# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************
"""Shared INT8 / WNA16 MoE patch helpers in ``zentorch.vllm._moe_patch_utils``."""

import types
import unittest

import torch
import zentorch  # noqa: F401

from ._test_constants import VLLM_AVAILABLE


@unittest.skipUnless(VLLM_AVAILABLE, "vLLM not installed")
class TestMoePatchUtils(unittest.TestCase):
    def test_import_select_experts_resolves(self):
        from zentorch.vllm._moe_patch_utils import import_select_experts

        self.assertTrue(callable(import_select_experts()))

    def test_allocate_expert_biases_when_needed(self):
        from zentorch.vllm._moe_patch_utils import allocate_expert_biases

        layer = torch.nn.Module()
        moe = types.SimpleNamespace(has_bias=True, is_act_and_mul=True)
        allocate_expert_biases(
            layer,
            moe,
            num_experts=4,
            hidden_size=16,
            intermediate_size_per_partition=8,
            params_dtype=torch.bfloat16,
            extra_weight_attrs={"intermediate_size_full": 16, "load_tag": "x"},
        )
        self.assertEqual(tuple(layer.w13_bias.shape), (4, 16))
        self.assertEqual(tuple(layer.w2_bias.shape), (4, 16))
        self.assertEqual(layer.w13_bias.dtype, torch.bfloat16)
        self.assertEqual(getattr(layer.w13_bias, "load_tag", None), "x")
        self.assertFalse(hasattr(layer.w13_bias, "intermediate_size_full"))

    def test_allocate_expert_biases_is_a_no_op_without_bias(self):
        from zentorch.vllm._moe_patch_utils import allocate_expert_biases

        layer = torch.nn.Module()
        moe = types.SimpleNamespace(has_bias=False, is_act_and_mul=True)
        allocate_expert_biases(
            layer,
            moe,
            num_experts=4,
            hidden_size=16,
            intermediate_size_per_partition=8,
            params_dtype=torch.bfloat16,
            extra_weight_attrs={},
        )
        self.assertFalse(hasattr(layer, "w13_bias"))

    def test_schedule_patches_already_imported_module(self):
        from zentorch.vllm._moe_patch_utils import schedule_module_patches

        seen = []

        def _apply(mod):
            seen.append(mod.__name__)
            return True

        mod = types.ModuleType("zentorch_test_fake_moe_target_schedule")
        import sys

        import zentorch.vllm._import_hook as ih

        sys.modules[mod.__name__] = mod
        ih._handled.discard(mod.__name__)
        try:
            ok = schedule_module_patches({mod.__name__: _apply})
            self.assertTrue(ok)
            self.assertEqual(seen, [mod.__name__])
        finally:
            del sys.modules[mod.__name__]
            ih._handled.discard(mod.__name__)

    def test_run_moe_patch_apply_success_and_idempotent(self):
        from zentorch.vllm._moe_patch_utils import run_moe_patch_apply

        registered = []
        target = types.SimpleNamespace()
        mod = types.SimpleNamespace()

        def _register(m):
            registered.append(m)

        kwargs = {
            "target": target,
            "flag": "_patched",
            "register_fn": _register,
            "success_log": "ok",
            "fail_log": "fail",
        }
        self.assertTrue(run_moe_patch_apply(mod, **kwargs))
        self.assertTrue(target._patched)
        self.assertEqual(registered, [mod])
        self.assertTrue(run_moe_patch_apply(mod, **kwargs))
        self.assertEqual(registered, [mod])

    def test_run_moe_patch_apply_missing_target(self):
        from zentorch.vllm._moe_patch_utils import run_moe_patch_apply

        registered = []
        kwargs = {
            "target": None,
            "flag": "_patched",
            "register_fn": lambda m: registered.append(m),
            "success_log": "ok",
            "fail_log": "fail",
            "missing_log": "missing",
        }
        self.assertFalse(run_moe_patch_apply(object(), **kwargs))
        self.assertTrue(run_moe_patch_apply(object(), missing_ok=True, **kwargs))
        self.assertEqual(registered, [])

    def test_run_moe_patch_apply_extra_guard_and_exception(self):
        from zentorch.vllm._moe_patch_utils import run_moe_patch_apply

        target = types.SimpleNamespace()
        self.assertFalse(
            run_moe_patch_apply(
                object(),
                target=target,
                flag="_patched",
                register_fn=lambda m: None,
                success_log="ok",
                fail_log="fail",
                extra_guard=lambda: "ops missing",
            )
        )
        self.assertFalse(hasattr(target, "_patched"))

        def _boom(mod):
            raise RuntimeError("register failed")

        self.assertFalse(
            run_moe_patch_apply(
                object(),
                target=target,
                flag="_patched",
                register_fn=_boom,
                success_log="ok",
                fail_log="fail",
            )
        )
        self.assertFalse(hasattr(target, "_patched"))


if __name__ == "__main__":
    unittest.main()
