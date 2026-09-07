# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

"""
Unit tests for the out-of-tree compressed-tensors WNA16 fused-MoE patch
(``zentorch.vllm._wna16_moe_patch``): zentorch DA8W4 (W4A8) experts offered
to the WNA16 oracle, per-expert bias allocation, and the checkpoint repack
into the zentorch WOQ layout.
"""

import os
import types
import unittest
from unittest import mock

import zentorch  # noqa: F401 - ensures zentorch native extension is loaded
import torch

from ._test_constants import VLLM_AVAILABLE

# =============================================================================
# Registration
# =============================================================================


@unittest.skipUnless(VLLM_AVAILABLE, "vLLM not installed")
class TestWna16MoEPatchRegistration(unittest.TestCase):
    """The patch is wired into the 0.27 plugin's flat ``_PATCHES`` list.

    Version gating is the shared ``is_supported_vllm`` window, not a
    per-patch ``@vllm_version`` decorator.
    """

    def test_wired_in_patches(self):
        import zentorch.vllm as zv

        names = [name for name, _ in zv._PATCHES]
        self.assertIn("Wna16MoE", names)

    def test_apply_impl_is_the_patch_callable(self):
        import zentorch.vllm as zv
        from zentorch.vllm._wna16_moe_patch import _apply_wna16_moe_patch_impl

        patches = dict(zv._PATCHES)
        self.assertIs(patches["Wna16MoE"], _apply_wna16_moe_patch_impl)


# =============================================================================
# DA8W4 availability gate
# =============================================================================


@unittest.skipUnless(VLLM_AVAILABLE, "vLLM not installed")
class TestDa8w4MoEGate(unittest.TestCase):
    """``use_zentorch_da8w4_moe`` is on by default, opt-out, and op-gated."""

    def test_enabled_by_default(self):
        from vllm.model_executor.kernels.linear import zentorch_utils
        from zentorch.vllm import _wna16_moe_patch as zw

        # Unset variable: on, matching the DA8W4 linear path's default.
        with mock.patch.dict("os.environ"), mock.patch.object(
            zentorch_utils, "has_zentorch_op", return_value=True
        ):
            os.environ.pop("VLLM_CPU_INT4_W4A8", None)
            self.assertTrue(zw.use_zentorch_da8w4_moe())

    def test_explicit_opt_out_is_honoured(self):
        from vllm.model_executor.kernels.linear import zentorch_utils
        from zentorch.vllm import _wna16_moe_patch as zw

        # Even with the ops present, =0 forces the native W4A16 backend.
        with mock.patch.dict(
            "os.environ", {"VLLM_CPU_INT4_W4A8": "0"}
        ), mock.patch.object(zentorch_utils, "has_zentorch_op", return_value=True):
            self.assertFalse(zw.use_zentorch_da8w4_moe())

    def test_enabled_still_requires_the_ops(self):
        from vllm.model_executor.kernels.linear import zentorch_utils
        from zentorch.vllm import _wna16_moe_patch as zw

        cases = [(True, True), (False, False)]
        for has_ops, expected in cases:
            with self.subTest(has_ops=has_ops), mock.patch.dict(
                "os.environ", {"VLLM_CPU_INT4_W4A8": "1"}
            ), mock.patch.object(
                zentorch_utils, "has_zentorch_op", return_value=has_ops
            ):
                self.assertIs(zw.use_zentorch_da8w4_moe(), expected)

    def test_probe_errors_are_swallowed(self):
        from vllm.model_executor.kernels.linear import zentorch_utils
        from zentorch.vllm import _wna16_moe_patch as zw

        with mock.patch.dict("os.environ", {"VLLM_CPU_INT4_W4A8": "1"}), mock.patch.object(
            zentorch_utils,
            "has_zentorch_op",
            side_effect=RuntimeError("probe failed"),
        ):
            self.assertFalse(zw.use_zentorch_da8w4_moe())


# =============================================================================
# Oracle patch
# =============================================================================


@unittest.skipUnless(VLLM_AVAILABLE, "vLLM not installed")
class TestWna16OraclePatch(unittest.TestCase):
    """``backend_to_kernel_cls`` prefers the zentorch experts on CPU."""

    class _Backend:
        CPU = "CPU"
        MARLIN = "MARLIN"

    def _fake_oracle_module(self):
        """Stand-in for ``oracle.int_wna16`` with only what the patch touches."""
        native_cpu_cls = type("CPUExpertsInt4", (), {})
        marlin_cls = type("MarlinExperts", (), {})

        def backend_to_kernel_cls(backend):
            return {
                self._Backend.CPU: [native_cpu_cls],
                self._Backend.MARLIN: [marlin_cls],
            }[backend]

        mod = types.SimpleNamespace(
            WNA16MoEBackend=self._Backend,
            backend_to_kernel_cls=backend_to_kernel_cls,
        )
        return mod, native_cpu_cls, marlin_cls

    def test_zentorch_experts_are_offered_first_on_cpu(self):
        from zentorch.vllm import _wna16_moe_patch as zw

        mod, native_cpu_cls, _ = self._fake_oracle_module()
        zen_cls = type("ZentorchExpertsInt4DA8W4", (), {})

        with mock.patch.object(
            zw, "use_zentorch_da8w4_moe", return_value=True
        ), mock.patch.object(zw, "_get_zentorch_experts_cls", return_value=zen_cls):
            zw._register_oracle_patch(mod)
            self.assertEqual(
                mod.backend_to_kernel_cls(self._Backend.CPU),
                [zen_cls, native_cpu_cls],
                "zentorch DA8W4 experts should be tried before CPUExpertsInt4",
            )

    def test_native_cpu_backend_kept_when_da8w4_unavailable(self):
        from zentorch.vllm import _wna16_moe_patch as zw

        mod, native_cpu_cls, _ = self._fake_oracle_module()
        zen_cls = type("ZentorchExpertsInt4DA8W4", (), {})

        with mock.patch.object(
            zw, "use_zentorch_da8w4_moe", return_value=False
        ), mock.patch.object(zw, "_get_zentorch_experts_cls", return_value=zen_cls):
            zw._register_oracle_patch(mod)
            self.assertEqual(
                mod.backend_to_kernel_cls(self._Backend.CPU), [native_cpu_cls]
            )

    def test_non_cpu_backends_are_untouched(self):
        from zentorch.vllm import _wna16_moe_patch as zw

        mod, _, marlin_cls = self._fake_oracle_module()
        zen_cls = type("ZentorchExpertsInt4DA8W4", (), {})

        with mock.patch.object(
            zw, "use_zentorch_da8w4_moe", return_value=True
        ), mock.patch.object(zw, "_get_zentorch_experts_cls", return_value=zen_cls):
            zw._register_oracle_patch(mod)
            self.assertEqual(
                mod.backend_to_kernel_cls(self._Backend.MARLIN), [marlin_cls]
            )

    def test_apply_is_idempotent(self):
        from zentorch.vllm import _wna16_moe_patch as zw

        mod, native_cpu_cls, _ = self._fake_oracle_module()
        zen_cls = type("ZentorchExpertsInt4DA8W4", (), {})

        with mock.patch.object(
            zw, "use_zentorch_da8w4_moe", return_value=True
        ), mock.patch.object(zw, "_get_zentorch_experts_cls", return_value=zen_cls):
            self.assertTrue(zw._apply_oracle_patch_to_module(mod))
            self.assertTrue(zw._apply_oracle_patch_to_module(mod))
            # A second application would wrap the wrapper and duplicate the
            # zentorch entry.
            self.assertEqual(
                mod.backend_to_kernel_cls(self._Backend.CPU),
                [zen_cls, native_cpu_cls],
            )


@unittest.skipUnless(VLLM_AVAILABLE, "vLLM not installed")
class TestOracleSupportGates(unittest.TestCase):
    """The experts class declines configurations the DA8W4 kernel cannot run.

    Both gates matter for correctness rather than performance: the oracle keeps
    the first candidate that passes, so a gate that wrongly returns True selects
    this class and then fails at model load or on the first forward instead of
    falling through to CPUExpertsInt4.
    """

    def _experts_cls(self):
        from zentorch.vllm import _wna16_moe_patch as zw

        return zw._get_zentorch_experts_cls()

    def test_expert_parallelism_is_declined(self):
        cls = self._experts_cls()

        for ep_size, expected in ((1, True), (2, False), (8, False)):
            with self.subTest(ep_size=ep_size):
                parallel_config = types.SimpleNamespace(ep_size=ep_size)
                self.assertIs(
                    cls._supports_parallel_config(parallel_config),
                    expected,
                )

    def test_custom_routing_is_accepted(self):
        """Gemma4 uses RoutingMethodType.Custom; native CPUExpertsInt4 does not."""
        from vllm.model_executor.layers.fused_moe.config import RoutingMethodType

        cls = self._experts_cls()
        self.assertTrue(
            cls._supports_routing_method(RoutingMethodType.Custom, None, None)
        )

    def test_single_expert_routing_is_declined(self):
        """experts_per_token == 1 cannot reach two active experts."""
        import vllm.model_executor.layers.fused_moe.modular_kernel as mk

        cls = self._experts_cls()

        # Stub the inherited gate so only the added top_k check is under test.
        with mock.patch.object(
            mk.FusedMoEExpertsMonolithic,
            "is_supported_config",
            staticmethod(lambda *args: (True, None)),
        ):
            for top_k, expected in ((1, False), (2, True), (8, True)):
                with self.subTest(experts_per_token=top_k):
                    moe_config = types.SimpleNamespace(experts_per_token=top_k)
                    # Called exactly as the oracle does: cls passed explicitly.
                    supported, reason = cls.is_supported_config(
                        cls, moe_config, None, None, None
                    )
                    self.assertIs(supported, expected)
                    if expected:
                        self.assertIsNone(reason)
                    else:
                        self.assertIn("experts_per_token", reason)

    def test_inherited_rejection_is_passed_through(self):
        import vllm.model_executor.layers.fused_moe.modular_kernel as mk

        cls = self._experts_cls()
        base_reason = "kernel does not support current device cpu"

        with mock.patch.object(
            mk.FusedMoEExpertsMonolithic,
            "is_supported_config",
            staticmethod(lambda *args: (False, base_reason)),
        ):
            # top_k is fine here, so the inherited verdict must survive intact.
            moe_config = types.SimpleNamespace(experts_per_token=4)
            self.assertEqual(
                cls.is_supported_config(cls, moe_config, None, None, None),
                (False, base_reason),
            )

    def test_base_receives_the_class_not_the_config(self):
        """``cls`` must be forwarded to the base, not swallowed or reordered.

        The base declares ``is_supported_config`` as a ``staticmethod`` with an
        explicit ``cls``, which is why every test here calls it the way the
        oracle does. A ``classmethod`` override would bind ``cls`` implicitly
        and shift the arguments along by one, so these calls raise a TypeError.
        """
        import vllm.model_executor.layers.fused_moe.modular_kernel as mk

        cls = self._experts_cls()
        seen = []

        with mock.patch.object(
            mk.FusedMoEExpertsMonolithic,
            "is_supported_config",
            staticmethod(lambda *args: (seen.append(args), (True, None))[1]),
        ):
            moe_config = types.SimpleNamespace(experts_per_token=2)
            cls.is_supported_config(cls, moe_config, None, None, "fmt")

        self.assertEqual(seen[0][0], cls)
        self.assertIs(seen[0][1], moe_config)
        self.assertEqual(seen[0][4], "fmt")

    def test_init_with_missing_bias_reads_none_from_the_property(self):
        """Bias-free checkpoints must not assign the read-only w1_bias property."""
        cls = self._experts_cls()
        experts = cls(
            types.SimpleNamespace(),
            types.SimpleNamespace(w1_bias=None, w2_bias=None),
        )
        self.assertIsNone(experts.w1_bias)
        self.assertIsNone(experts.w2_bias)


# =============================================================================
# Modular quant-method discovery (vLLM 0.27)
# =============================================================================


@unittest.skipUnless(VLLM_AVAILABLE, "vLLM not installed")
class TestModularMethodDiscovery(unittest.TestCase):
    """The modular WNA16 quant method is ``CompressedTensorsWNA16MoEMethod``."""

    def _module(self, name, cls_name=None, with_factory=True):
        mod = types.ModuleType(name)
        if cls_name is not None:
            setattr(mod, cls_name, type(cls_name, (), {}))
        if with_factory:
            mod.make_wna16_moe_kernel = lambda *a, **k: None
        return mod

    def test_wna16_module_is_the_modular_method(self):
        from zentorch.vllm import _wna16_moe_patch as zw

        mod = self._module(zw._METHOD_MODULE, zw._METHOD_CLASS)
        self.assertIsNotNone(zw._modular_method_cls(mod))

    def test_module_without_factory_is_ignored(self):
        from zentorch.vllm import _wna16_moe_patch as zw

        mod = self._module(zw._METHOD_MODULE, zw._METHOD_CLASS, with_factory=False)
        self.assertIsNone(zw._modular_method_cls(mod))

    def test_unknown_module_is_ignored(self):
        from zentorch.vllm import _wna16_moe_patch as zw

        mod = self._module("some.other.module", zw._METHOD_CLASS)
        self.assertIsNone(zw._modular_method_cls(mod))

    def test_oracle_and_method_are_scheduled_as_targets(self):
        from zentorch.vllm import _wna16_moe_patch as zw

        self.assertIn(zw._ORACLE_MODULE, zw._TARGETS)
        self.assertIs(
            zw._TARGETS[zw._METHOD_MODULE], zw._apply_wna16_method_patch_to_module
        )

    def test_select_experts_resolves_on_the_installed_vllm(self):
        from zentorch.vllm._moe_patch_utils import import_select_experts

        self.assertTrue(callable(import_select_experts()))


@unittest.skipUnless(VLLM_AVAILABLE, "vLLM not installed")
class TestKernelSetup(unittest.TestCase):
    """``process_weights_after_loading`` delegates kernel build to ``_setup_kernel``."""

    NUM_EXPERTS, HIDDEN, INTERMEDIATE = 2, 32, 16

    def _layer(self):
        """A minimal stand-in for the FusedMoE layer, real enough for
        ``replace_parameter``."""
        layer = torch.nn.Module()
        packed = torch.zeros(
            self.NUM_EXPERTS, self.HIDDEN // 8, self.INTERMEDIATE, dtype=torch.int32
        )
        scale = torch.ones(self.NUM_EXPERTS, 1, self.INTERMEDIATE)
        for name, tensor in (
            ("w13_weight_packed", packed),
            ("w2_weight_packed", packed.clone()),
            ("w13_weight_scale", scale),
            ("w2_weight_scale", scale.clone()),
        ):
            layer.register_parameter(
                name, torch.nn.Parameter(tensor, requires_grad=False)
            )
        layer._expert_routing_tables = lambda: None
        return layer

    def _stub_module(self, vanilla_calls):
        """A stub vLLM 0.27 WNA16 method module."""
        from zentorch.vllm import _wna16_moe_patch as zw

        mod = types.ModuleType(zw._METHOD_MODULE)
        mod.make_wna16_moe_kernel = lambda *a, **k: vanilla_calls.append((a, k))
        mod.make_wna16_moe_quant_config = lambda **kwargs: "quant_config"
        mod.CompressedTensorsWNA16MoEMethod = type(
            zw._METHOD_CLASS,
            (),
            {
                "create_weights": lambda self, *a, **k: None,
                "process_weights_after_loading": lambda self, layer: None,
                "get_fused_moe_quant_config": lambda self, layer: None,
                "_setup_kernel": lambda self, layer: self._setup_calls.append(layer),
            },
        )
        return mod

    def _method(self, cls):
        method = cls()
        method._setup_calls = []
        method.experts_cls = lambda **kwargs: "experts"
        method.moe = types.SimpleNamespace(has_bias=False, is_act_and_mul=True)
        method.weight_quant = object()
        method.group_size = 16
        method.num_bits = 4
        return method

    def _run(self):
        import vllm.model_executor.layers.fused_moe.modular_kernel as mk
        from vllm.model_executor.layers.fused_moe import all2all_utils
        from zentorch.vllm import _wna16_moe_patch as zw

        vanilla_calls = []
        layer = self._layer()
        repacked = (
            layer.w13_weight_packed.data,
            layer.w2_weight_packed.data,
            layer.w13_weight_scale.data,
            layer.w2_weight_scale.data,
            None,  # w13_bias
            None,  # w2_bias
        )

        # The modular-kernel plumbing needs a real FusedMoEConfig, which this
        # stub layer is not, so stand in for it.
        with mock.patch.object(
            all2all_utils, "maybe_make_prepare_finalize", return_value="prep"
        ), mock.patch.object(
            mk, "FusedMoEKernel", side_effect=lambda *a: ("kernel", a)
        ), mock.patch.object(
            zw, "_is_zentorch_experts", return_value=True
        ), mock.patch.object(
            zw, "_process_weights_zentorch", return_value=repacked
        ):
            mod = self._stub_module(vanilla_calls)
            zw._register_wna16_method_patch(mod)
            cls = mod.CompressedTensorsWNA16MoEMethod
            method = self._method(cls)
            cls.process_weights_after_loading(method, layer)

        self.assertIs(layer.w13_weight, layer.w13_weight_packed)
        self.assertIs(layer.w2_weight, layer.w2_weight_packed)
        # The vanilla kernel factory is only for non-zentorch experts classes.
        self.assertEqual(vanilla_calls, [])
        return method

    def test_process_weights_delegates_to_setup_kernel(self):
        method = self._run()
        self.assertEqual(len(method._setup_calls), 1)

    def test_group_act_ordering_falls_back_to_native_process(self):
        """GROUP act-order is declined before the zentorch repack; native path runs."""
        import sys

        from compressed_tensors.quantization import QuantizationArgs
        from zentorch.vllm import _wna16_moe_patch as zw

        orig_calls = []
        native = type("CPUExpertsInt4", (), {})
        zentorch_cls = type("ZentorchExperts", (), {})

        mod = types.ModuleType(zw._METHOD_MODULE)
        mod.make_wna16_moe_kernel = lambda *a, **k: None
        mod.make_wna16_moe_quant_config = lambda **kwargs: "quant_config"
        mod.CompressedTensorsWNA16MoEMethod = type(
            zw._METHOD_CLASS,
            (),
            {
                "create_weights": lambda self, *a, **k: None,
                "process_weights_after_loading": lambda self, layer: orig_calls.append(
                    layer
                ),
                "get_fused_moe_quant_config": lambda self, layer: None,
                "_setup_kernel": lambda self, layer: None,
            },
        )

        oracle = types.ModuleType(zw._ORACLE_MODULE)
        oracle.WNA16MoEBackend = types.SimpleNamespace(CPU="cpu")
        oracle.backend_to_kernel_cls = lambda backend: [zentorch_cls, native]
        saved_oracle = sys.modules.get(zw._ORACLE_MODULE)
        saved_cls = zw._EXPERTS_CLS
        try:
            sys.modules[zw._ORACLE_MODULE] = oracle
            zw._EXPERTS_CLS = zentorch_cls
            zw._register_wna16_method_patch(mod)
            method = self._method(mod.CompressedTensorsWNA16MoEMethod)
            method.experts_cls = zentorch_cls
            method.weight_quant = QuantizationArgs(
                num_bits=4,
                type="int",
                symmetric=True,
                strategy="group",
                group_size=128,
                actorder="group",
            )
            with mock.patch.object(
                zw,
                "_process_weights_zentorch",
                side_effect=AssertionError("must not repack GROUP act-order"),
            ):
                mod.CompressedTensorsWNA16MoEMethod.process_weights_after_loading(
                    method, self._layer()
                )
            self.assertEqual(len(orig_calls), 1)
            self.assertIs(method.experts_cls, native)
        finally:
            zw._EXPERTS_CLS = saved_cls
            if saved_oracle is None:
                sys.modules.pop(zw._ORACLE_MODULE, None)
            else:
                sys.modules[zw._ORACLE_MODULE] = saved_oracle


# =============================================================================
# Weight post-processing guard rails
# =============================================================================


@unittest.skipUnless(VLLM_AVAILABLE, "vLLM not installed")
class TestProcessWeightsZentorch(unittest.TestCase):
    """``_process_weights_zentorch`` rejects configurations DA8W4 cannot serve."""

    def _quant_args(self, **overrides):
        from compressed_tensors.quantization import QuantizationArgs

        kwargs = {
            "num_bits": 4,
            "type": "int",
            "symmetric": True,
            "strategy": "group",
            "group_size": 128,
        }
        kwargs.update(overrides)
        return QuantizationArgs(**kwargs)

    def _dummy_weights(self):
        # Shapes are irrelevant: every case raises before any repack happens.
        packed = torch.zeros(2, 4, 8, dtype=torch.int32)
        scale = torch.ones(2, 1, 8, dtype=torch.bfloat16)
        return packed, packed.clone(), scale, scale.clone()

    def test_rejects_non_compressed_tensors_config(self):
        from zentorch.vllm._wna16_moe_patch import _process_weights_zentorch

        w13, w2, s13, s2 = self._dummy_weights()
        with self.assertRaises(TypeError):
            _process_weights_zentorch(object(), w13, w2, s13, s2)

    def test_rejects_asymmetric_checkpoints(self):
        from zentorch.vllm._wna16_moe_patch import _process_weights_zentorch

        w13, w2, s13, s2 = self._dummy_weights()
        with self.assertRaises(NotImplementedError):
            _process_weights_zentorch(
                self._quant_args(symmetric=False), w13, w2, s13, s2
            )

    def test_rejects_group_act_ordering(self):
        from zentorch.vllm._wna16_moe_patch import _process_weights_zentorch

        w13, w2, s13, s2 = self._dummy_weights()
        with self.assertRaises(NotImplementedError):
            _process_weights_zentorch(
                self._quant_args(actorder="group"), w13, w2, s13, s2
            )

    def test_repacks_to_transposed_woq_layout(self):
        """[E, K//8, N] int32 -> [E, N, K//8] int32, scales passed through."""
        from vllm.model_executor.kernels.linear import zentorch_utils
        from zentorch.vllm._wna16_moe_patch import _process_weights_zentorch

        if not zentorch_utils.has_zentorch_op(["zentorch_woq_repack_weight"]):
            self.skipTest("zentorch_woq_repack_weight not available")

        num_experts, k, n = 2, 32, 16
        w13 = torch.randint(0, 2**16, (num_experts, k // 8, n), dtype=torch.int32)
        w2 = torch.randint(0, 2**16, (num_experts, n // 8, k), dtype=torch.int32)
        s13 = torch.ones(num_experts, k // 16, n, dtype=torch.bfloat16)
        s2 = torch.ones(num_experts, n // 16, k, dtype=torch.bfloat16)
        bias13 = torch.ones(num_experts, n, dtype=torch.float32)

        (
            w13_out,
            w2_out,
            s13_out,
            s2_out,
            bias13_out,
            bias2_out,
        ) = _process_weights_zentorch(
            self._quant_args(group_size=16),
            w13,
            w2,
            s13,
            s2,
            w13_bias=bias13,
        )

        self.assertEqual(tuple(w13_out.shape), (num_experts, n, k // 8))
        self.assertEqual(tuple(w2_out.shape), (num_experts, k, n // 8))
        self.assertEqual(w13_out.dtype, torch.int32)
        self.assertTrue(torch.equal(s13_out, s13))
        self.assertTrue(torch.equal(s2_out, s2))
        # The W4A8 kernel is bf16-in/bf16-out, so bias is converted at load time.
        self.assertEqual(bias13_out.dtype, torch.bfloat16)
        self.assertIsNone(bias2_out)


if __name__ == "__main__":
    unittest.main()
