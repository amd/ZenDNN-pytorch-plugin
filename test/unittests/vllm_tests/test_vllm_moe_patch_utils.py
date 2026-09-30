# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************
"""Shared INT8 / WNA16 MoE patch helpers in ``zentorch.vllm._moe_patch_utils``."""

import types
import unittest
from unittest import mock

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

    @staticmethod
    def _swigluoai_layer(weight, scale, *, activation="swigluoai"):
        layer = torch.nn.Module()
        layer.activation = activation
        layer.w13_weight_scale = torch.nn.Parameter(scale, requires_grad=False)
        layer.w13_bias = torch.nn.Parameter(
            torch.arange(weight.shape[0] * 4, dtype=torch.bfloat16).reshape(
                weight.shape[0], 4
            ),
            requires_grad=False,
        )
        return layer

    def test_permute_swigluoai_w13_interleaves_the_int8_layout(self):
        from zentorch.vllm._moe_patch_utils import permute_swigluoai_w13

        # [E, 2I, K] weight and [E, 2I] scale, so 2I sits at dim 1 on both.
        weight = torch.arange(2 * 4 * 3, dtype=torch.bfloat16).reshape(2, 4, 3)
        scale = torch.arange(2 * 4, dtype=torch.bfloat16).reshape(2, 4)
        layer = self._swigluoai_layer(weight, scale)
        layer.w13_weight = torch.nn.Parameter(weight.clone(), requires_grad=False)
        bias = layer.w13_bias.detach().clone()

        self.assertTrue(permute_swigluoai_w13(layer))

        perm = torch.tensor([0, 2, 1, 3])
        torch.testing.assert_close(layer.w13_weight.data, weight[:, perm, :])
        torch.testing.assert_close(layer.w13_weight_scale.data, scale[:, perm])
        torch.testing.assert_close(layer.w13_bias.data, bias[:, perm])

    def test_permute_swigluoai_w13_interleaves_the_packed_w4_layout(self):
        from zentorch.vllm._moe_patch_utils import permute_swigluoai_w13

        # Packed W4 keeps 2I last: [E, K/8, 2I] weight and [E, groups, 2I] scale.
        weight = torch.arange(2 * 3 * 4, dtype=torch.int32).reshape(2, 3, 4)
        scale = torch.arange(2 * 2 * 4, dtype=torch.bfloat16).reshape(2, 2, 4)
        layer = self._swigluoai_layer(weight, scale)
        layer.w13_weight_packed = torch.nn.Parameter(
            weight.clone(), requires_grad=False
        )
        bias = layer.w13_bias.detach().clone()

        self.assertTrue(
            permute_swigluoai_w13(
                layer,
                weight_attr="w13_weight_packed",
                weight_dim=2,
                scale_dim=2,
            )
        )

        perm = torch.tensor([0, 2, 1, 3])
        torch.testing.assert_close(layer.w13_weight_packed.data, weight[:, :, perm])
        torch.testing.assert_close(layer.w13_weight_scale.data, scale[:, :, perm])
        torch.testing.assert_close(layer.w13_bias.data, bias[:, perm])

    def test_permute_swigluoai_w13_reads_an_enum_activation(self):
        from zentorch.vllm._moe_patch_utils import permute_swigluoai_w13

        weight = torch.arange(2 * 4 * 3, dtype=torch.bfloat16).reshape(2, 4, 3)
        scale = torch.arange(2 * 4, dtype=torch.bfloat16).reshape(2, 4)
        layer = self._swigluoai_layer(
            weight, scale, activation=types.SimpleNamespace(value="SwigluOAI")
        )
        layer.w13_weight = torch.nn.Parameter(weight.clone(), requires_grad=False)

        self.assertTrue(permute_swigluoai_w13(layer))

    def test_permute_swigluoai_w13_leaves_other_activations_alone(self):
        from zentorch.vllm._moe_patch_utils import permute_swigluoai_w13

        weight = torch.arange(2 * 4 * 3, dtype=torch.bfloat16).reshape(2, 4, 3)
        scale = torch.arange(2 * 4, dtype=torch.bfloat16).reshape(2, 4)
        layer = self._swigluoai_layer(weight, scale, activation="silu")
        layer.w13_weight = torch.nn.Parameter(weight.clone(), requires_grad=False)

        self.assertFalse(permute_swigluoai_w13(layer))
        torch.testing.assert_close(layer.w13_weight.data, weight)
        torch.testing.assert_close(layer.w13_weight_scale.data, scale)

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

    def _fake_select_experts(self):
        captured = {}

        def select_experts(**kwargs):
            captured.clear()
            captured.update(kwargs)
            return "weights", "ids"

        return select_experts, captured

    def test_run_select_experts_forwards_custom_callable_and_renormalize(self):
        from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
        from zentorch.vllm import _moe_patch_utils as utils

        select_experts, captured = self._fake_select_experts()
        custom_fn = object()
        moe_config = types.SimpleNamespace(
            routing_method=RoutingMethodType.Custom,
            experts_per_token=8,
        )
        with mock.patch.object(
            utils, "import_select_experts", return_value=select_experts
        ):
            result = utils.run_select_experts(
                "hidden",
                "logits",
                moe_config,
                custom_routing_fn=custom_fn,
            )
            self.assertEqual(result, ("weights", "ids"))
            self.assertIs(captured["custom_routing_function"], custom_fn)
            self.assertTrue(captured["renormalize"])
            self.assertEqual(captured["top_k"], 8)

            utils.run_select_experts(
                "hidden",
                "logits",
                moe_config,
                custom_routing_fn=custom_fn,
                renormalize=False,
            )
            self.assertIs(captured["custom_routing_function"], custom_fn)
            self.assertFalse(captured["renormalize"])

    def test_run_select_experts_custom_without_callable_raises(self):
        from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
        from zentorch.vllm._moe_patch_utils import run_select_experts

        moe_config = types.SimpleNamespace(
            routing_method=RoutingMethodType.Custom,
            experts_per_token=8,
        )
        with self.assertRaisesRegex(RuntimeError, "custom_routing_function"):
            run_select_experts("hidden", "logits", moe_config)


if __name__ == "__main__":
    unittest.main()
