# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

"""Tests for ``ZentorchWNA16LinearKernel.can_implement`` without N/K % 32."""

import sys
import types
import unittest
import unittest.mock

from ._test_constants import VLLM_AVAILABLE

if VLLM_AVAILABLE:
    from vllm.scalar_type import scalar_types

    from zentorch.vllm._wna16_can_implement_patch import (
        _PATCH_MARKER,
        _TARGET_MODULE,
        _apply_wna16_can_implement_patch,
        _can_implement_without_cpu_align,
        _do_patch_wna16_can_implement,
    )


def _config(
    *,
    has_g_idx=False,
    weight_type=None,
    group_size=-1,
    partition_weight_shape=(4096, 6448),
):
    if weight_type is None:
        weight_type = scalar_types.uint4
    return types.SimpleNamespace(
        has_g_idx=has_g_idx,
        weight_type=weight_type,
        group_size=group_size,
        partition_weight_shape=partition_weight_shape,
    )


@unittest.skipUnless(VLLM_AVAILABLE, "vLLM not installed")
class TestWna16CanImplementPatchWiring(unittest.TestCase):
    def test_wired_in_patches(self):
        import zentorch.vllm as zv

        names = [name for name, _ in zv._PATCHES]
        self.assertIn("Wna16CanImplement", names)
        self.assertIs(
            dict(zv._PATCHES)["Wna16CanImplement"],
            _apply_wna16_can_implement_patch,
        )
        self.assertEqual(
            names[names.index("Da8w4Kernel") + 1],
            "Wna16CanImplement",
        )


@unittest.skipUnless(VLLM_AVAILABLE, "vLLM not installed")
class TestWna16CanImplement(unittest.TestCase):
    def test_rejects_g_idx(self):
        ok, reason = _can_implement_without_cpu_align(
            object(), _config(has_g_idx=True)
        )
        self.assertFalse(ok)
        self.assertIn("activation re-ordering", reason)

    def test_rejects_unsupported_weight_type(self):
        ok, reason = _can_implement_without_cpu_align(
            object(), _config(weight_type=object())
        )
        self.assertFalse(ok)
        self.assertIn("Quant type", reason)

    def test_accepts_group_size_that_divides_k(self):
        ok, reason = _can_implement_without_cpu_align(
            object(),
            _config(weight_type=scalar_types.uint4b8, group_size=128),
        )
        self.assertTrue(ok)
        self.assertIsNone(reason)

    def test_rejects_group_size_that_does_not_divide_k(self):
        ok, reason = _can_implement_without_cpu_align(
            object(),
            _config(group_size=3, partition_weight_shape=(4096, 6448)),
        )
        self.assertFalse(ok)
        self.assertIn("must divide input size", reason)

    def test_accepts_unaligned_n(self):
        ok, reason = _can_implement_without_cpu_align(object(), _config())
        self.assertTrue(ok)
        self.assertIsNone(reason)

    def test_replaces_can_implement_once(self):
        class FakeKernel:
            @classmethod
            def can_implement(cls, config):
                return False, "Output size (6448) not supported by CPUWNA16"

        module = types.SimpleNamespace(ZentorchWNA16LinearKernel=FakeKernel)
        with unittest.mock.patch.dict(sys.modules, {_TARGET_MODULE: module}):
            self.assertTrue(_do_patch_wna16_can_implement())
            patched = FakeKernel.__dict__["can_implement"]
            self.assertTrue(_do_patch_wna16_can_implement())
            self.assertIs(FakeKernel.__dict__["can_implement"], patched)
            ok, reason = FakeKernel.can_implement(_config())

        self.assertTrue(ok)
        self.assertIsNone(reason)
        self.assertTrue(getattr(FakeKernel, _PATCH_MARKER))


if __name__ == "__main__":
    unittest.main()
