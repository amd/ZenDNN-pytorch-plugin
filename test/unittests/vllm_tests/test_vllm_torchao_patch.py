# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import sys
import unittest
import unittest.mock

import torch
from zentorch._utils import counters

from ._test_constants import TORCHAO_AVAILABLE, VLLM_AVAILABLE, vllm
from ._test_utils import load_source_vllm_module


@unittest.skipUnless(VLLM_AVAILABLE, "vLLM not installed")
class TestTorchAODispatchNoTorchAO(unittest.TestCase):
    """_apply_torchao_patch must not import torchao when it is absent."""

    def test_skips_without_torchao(self):
        spec, plugin = load_source_vllm_module()
        with unittest.mock.patch.dict(
            sys.modules, {"zentorch.vllm": plugin}
        ):
            spec.loader.exec_module(plugin)
            with unittest.mock.patch.object(
                plugin.importlib.util, "find_spec", return_value=None
            ) as mock_find:
                applied = plugin._apply_torchao_patch()
            mock_find.assert_called_once_with("torchao")
            self.assertFalse(applied)


@unittest.skipUnless(VLLM_AVAILABLE, "vLLM not installed")
class TestTorchAODispatchApplied(unittest.TestCase):
    """When torchao is installed, register() applies the TorchAO hook."""

    @unittest.skipUnless(TORCHAO_AVAILABLE, "torchao not installed")
    def test_torchao_in_applied_after_register(self):
        from zentorch import vllm as plugin

        if not plugin.is_supported_vllm(vllm.__version__):
            self.skipTest(f"Installed vLLM {vllm.__version__} is not supported")

        plugin._INITIALIZED = False
        with unittest.mock.patch(
            "zentorch._C.is_avx512_supported", return_value=True
        ):
            plugin.register()
        self.assertIn("TorchAO", plugin.APPLIED_PATCHES)


@unittest.skipUnless(VLLM_AVAILABLE, "vLLM not installed")
@unittest.skipUnless(TORCHAO_AVAILABLE, "torchao not installed")
class TestInt8TensorHandlers(unittest.TestCase):
    """End-to-end checks for the Int8Tensor shape and linear handlers."""

    @classmethod
    def setUpClass(cls):
        from torchao.quantization.granularity import PerRow
        from torchao.quantization.quantize_.workflows.int8.int8_tensor import (
            Int8Tensor,
        )
        from zentorch.vllm._torchao_int8_patch import (
            _apply_torchao_int8_tensor_patch_impl,
        )

        _apply_torchao_int8_tensor_patch_impl()
        cls.Int8Tensor = Int8Tensor
        cls.PerRow = PerRow

    def _static_qt(self, shape, seed=0, dtype=torch.bfloat16):
        torch.manual_seed(seed)
        hp = torch.randn(*shape, dtype=dtype)
        qt = self.Int8Tensor.from_hp(hp, granularity=self.PerRow())
        return hp, qt

    @staticmethod
    def _manual_dequant(qt):
        qdata = qt.qdata.to(qt.dtype)
        scale = qt.scale.to(qt.dtype)
        if qt.zero_point is not None:
            qdata = qdata - qt.zero_point.to(qt.dtype)
        return qdata * scale

    def test_view_rank_preserving_2d(self):
        _, qt = self._static_qt((4, 8), seed=0)
        viewed = qt.view(4, 8)
        self.assertEqual(viewed.shape, qt.shape)
        self.assertTrue(
            torch.equal(self._manual_dequant(viewed), self._manual_dequant(qt))
        )

    def test_view_3d_to_2d_flatten(self):
        _, qt = self._static_qt((2, 4, 8), seed=1)
        viewed = qt.view(8, 8)
        dq_view = self._manual_dequant(viewed)
        view_dq = self._manual_dequant(qt).view(8, 8)
        self.assertTrue(torch.equal(dq_view, view_dq))

    def test_permute_2d_transpose(self):
        _, qt = self._static_qt((4, 8), seed=3)
        permuted = qt.permute(1, 0)
        dq_permute = self._manual_dequant(permuted)
        permute_dq = self._manual_dequant(qt).permute(1, 0)
        self.assertEqual(dq_permute.shape, (8, 4))
        self.assertTrue(torch.equal(dq_permute, permute_dq))

    def test_linear_dispatches_to_dynamic_qlinear_when_activation_quantized(self):
        counters.clear()
        x = torch.randn(3, 8, dtype=torch.bfloat16)
        _, qt = self._static_qt((6, 8), seed=4)
        bias = torch.randn(6, dtype=torch.bfloat16)
        expected = torch.randn(3, 6, dtype=torch.bfloat16)
        with (
            unittest.mock.patch.object(
                qt, "act_quant_kwargs", {"dynamic": True}, create=True
            ),
            unittest.mock.patch.object(
                torch.ops.zentorch,
                "zentorch_dynamic_qlinear",
                return_value=expected,
            ) as dynamic_qlinear,
        ):
            result = torch.nn.functional.linear(x, qt, bias)
        self.assertIs(result, expected)
        dynamic_qlinear.assert_called_once()
        self.assertEqual(counters["zentorch"]["zentorch_dynamic_qlinear"], 1)

    def test_linear_hands_contiguous_tensors_to_dynamic_qlinear(self):
        counters.clear()
        _, qt = self._static_qt((8, 6), seed=5)
        qt_nc = qt.permute(1, 0)
        self.assertFalse(qt_nc.qdata.is_contiguous())

        x = torch.randn(3, 8, dtype=torch.bfloat16)
        wide = torch.zeros(12, dtype=torch.bfloat16)
        wide[::2] = torch.randn(6, dtype=torch.bfloat16)
        bias_nc = wide[::2]
        self.assertFalse(bias_nc.is_contiguous())

        captured = {}

        def _capture(activation, weight, scales, bias_arg=None, *args, **kwargs):
            captured["weight"] = weight
            captured["scales"] = scales
            captured["bias"] = bias_arg
            return torch.zeros(x.shape[0], qt_nc.shape[0], dtype=x.dtype)

        with (
            unittest.mock.patch.object(
                qt_nc, "act_quant_kwargs", {"dynamic": True}, create=True
            ),
            unittest.mock.patch.object(
                torch.ops.zentorch,
                "zentorch_dynamic_qlinear",
                side_effect=_capture,
            ),
        ):
            torch.nn.functional.linear(x, qt_nc, bias_nc)

        self.assertIn(
            "weight", captured, "zentorch_dynamic_qlinear was not called"
        )
        self.assertTrue(captured["weight"].is_contiguous())
        self.assertTrue(captured["scales"].is_contiguous())
        self.assertTrue(captured["bias"].is_contiguous())


if __name__ == "__main__":
    unittest.main()
