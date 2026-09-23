# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

"""Tests for the multimodal-encoder SDPA patch."""

import sys
import types
import unittest
import unittest.mock

import torch

from ._test_constants import VLLM_AVAILABLE

if VLLM_AVAILABLE:
    from zentorch.vllm._mm_encoder_sdpa_patch import (
        _OP_NAME,
        _PATCH_MARKER,
        _TARGET_MODULE,
        _apply_mm_encoder_sdpa_patch,
        _do_patch_mm_encoder_sdpa,
    )

_PATCH_PREFIX = "zentorch.vllm._mm_encoder_sdpa_patch"


def _fake_vit_module():
    """Stand-in for vllm.v1.attention.ops.vit_attn_wrappers.

    Both functions mirror the stock ones: ``torch_sdpa_wrapper`` reads
    ``apply_sdpa`` from the module globals, which is the rebinding the patch
    relies on, and ``apply_sdpa`` takes and returns {B, S, H, D}. Calls that
    reach the stock implementation are recorded in ``module.stock_calls``.
    """
    module = types.ModuleType(_TARGET_MODULE)
    module.stock_calls = []

    def apply_sdpa(q, k, v, scale=None, enable_gqa=False):
        module.stock_calls.append((tuple(q.shape), scale, enable_gqa))
        return q.clone()

    def torch_sdpa_wrapper(q, k, v, scale=None, cu_seqlens=None, enable_gqa=False):
        if cu_seqlens is None:
            return module.apply_sdpa(q, k, v, scale=scale, enable_gqa=enable_gqa)
        # cu_seqlens holds segment boundaries, so the sequences it delimits are
        # the adjacent differences; split Q/K/V on those and concatenate the
        # per-segment results, as the stock wrapper does.
        lens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        outputs = [
            module.apply_sdpa(q_i, k_i, v_i, scale=scale, enable_gqa=enable_gqa)
            for q_i, k_i, v_i in zip(
                torch.split(q, lens, dim=1),
                torch.split(k, lens, dim=1),
                torch.split(v, lens, dim=1),
                strict=True,
            )
        ]
        return torch.cat(outputs, dim=1)

    module.apply_sdpa = apply_sdpa
    module.torch_sdpa_wrapper = torch_sdpa_wrapper
    return module


@unittest.skipUnless(VLLM_AVAILABLE, "vLLM not installed")
class TestMMEncoderSdpaPatchWiring(unittest.TestCase):
    def test_wired_in_patches(self):
        import zentorch.vllm as zv

        names = [name for name, _ in zv._PATCHES]
        self.assertIn("MMEncoderSdpa", names)
        self.assertIs(
            dict(zv._PATCHES)["MMEncoderSdpa"], _apply_mm_encoder_sdpa_patch
        )

    def test_skipped_when_env_opts_out(self):
        with unittest.mock.patch.dict("os.environ", {"ZENTORCH_SDPA": "0"}):
            self.assertFalse(_apply_mm_encoder_sdpa_patch())

    def test_no_target_module_is_not_an_error(self):
        with unittest.mock.patch.dict(sys.modules, {}, clear=False):
            sys.modules.pop(_TARGET_MODULE, None)
            self.assertFalse(_do_patch_mm_encoder_sdpa())


@unittest.skipUnless(VLLM_AVAILABLE, "vLLM not installed")
class TestMMEncoderSdpaDispatch(unittest.TestCase):
    def _patched_module(self, supports_dtype=True, sdpa=None):
        """Install a fake target module and apply the patch to it."""
        module = _fake_vit_module()
        stack = unittest.mock.patch.dict(sys.modules, {_TARGET_MODULE: module})
        stack.start()
        self.addCleanup(stack.stop)

        gate = unittest.mock.patch(
            f"{_PATCH_PREFIX}.zentorch_sdpa_supports_dtype",
            return_value=supports_dtype,
        )
        gate.start()
        self.addCleanup(gate.stop)

        if sdpa is not None:
            op = unittest.mock.patch.object(
                torch.ops.zentorch, _OP_NAME, sdpa, create=True
            )
            op.start()
            self.addCleanup(op.stop)

        self.assertTrue(_do_patch_mm_encoder_sdpa())
        return module

    def test_patch_is_idempotent(self):
        module = self._patched_module()
        patched = module.apply_sdpa
        self.assertTrue(_do_patch_mm_encoder_sdpa())
        self.assertIs(module.apply_sdpa, patched)
        self.assertTrue(getattr(module, _PATCH_MARKER))

    def test_unsupported_dtype_falls_back_to_original(self):
        module = self._patched_module(supports_dtype=False)
        b, s, h, d = 2, 4, 3, 8
        q = torch.randn(b, s, h, d)
        out = module.torch_sdpa_wrapper(q, q, q, scale=0.25)
        # Delegated untouched: same {B, S, H, D} tensor, scale and enable_gqa.
        self.assertEqual(module.stock_calls, [((b, s, h, d), 0.25, False)])
        self.assertTrue(torch.equal(out, q))

    def test_routes_through_zentorch_with_bhsd_layout(self):
        seen = []

        def fake_sdpa(q, k, v, out, scale=None, is_causal=False, **kwargs):
            seen.append(
                {
                    "q": tuple(q.shape),
                    "out": tuple(out.shape),
                    "out_last_stride": out.stride(-1),
                    "scale": scale,
                    "is_causal": is_causal,
                    "extra": kwargs,
                }
            )
            out.copy_(q)

        module = self._patched_module(sdpa=fake_sdpa)

        b, s, h, d = 2, 4, 3, 8
        q = torch.randn(b, s, h, d)
        out = module.torch_sdpa_wrapper(q, q, q, scale=0.25)

        # Q/K/V and the out buffer all go in as {B, H, S, D}; no mask args, and
        # the head dim stays contiguous so the kernel needs no staging copy.
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0]["q"], (b, h, s, d))
        self.assertEqual(seen[0]["out"], (b, h, s, d))
        self.assertEqual(seen[0]["out_last_stride"], 1)
        self.assertEqual(seen[0]["scale"], 0.25)
        self.assertFalse(seen[0]["is_causal"])
        self.assertEqual(seen[0]["extra"], {})

        # The result comes back in the caller's {B, S, H, D} layout, contiguous.
        self.assertEqual(tuple(out.shape), (b, s, h, d))
        self.assertTrue(out.is_contiguous())
        self.assertTrue(torch.equal(out, q))

    def test_ragged_cu_seqlens_path_is_also_routed(self):
        calls = []

        def fake_sdpa(q, k, v, out, scale=None, is_causal=False, **kwargs):
            calls.append(tuple(q.shape))
            out.copy_(q)

        module = self._patched_module(sdpa=fake_sdpa)
        b, s, h, d = 1, 6, 3, 8
        q = torch.randn(b, s, h, d)
        cu_seqlens = torch.tensor([0, 2, 6], dtype=torch.int32)
        out = module.torch_sdpa_wrapper(q, q, q, scale=0.25, cu_seqlens=cu_seqlens)

        # Boundaries [0, 2, 6] delimit a length-2 and a length-4 sequence, and
        # each segment reaches the op on its own, still as {B, H, S, D}.
        self.assertEqual(calls, [(b, h, 2, d), (b, h, 4, d)])
        self.assertEqual(module.stock_calls, [])
        self.assertEqual(tuple(out.shape), (b, s, h, d))
        self.assertTrue(torch.equal(out, q))

    def test_out_buffer_avoids_a_staging_copy(self):
        """zentorch_sdpa transposes the out tensor back to {B, S, H, D} and only
        stages through a temporary when that view is non-contiguous. Allocating
        in the caller's layout keeps it contiguous on both call paths, including
        the ragged one where q is a non-contiguous split chunk.
        """
        transposed_out_contiguous = []

        def fake_sdpa(q, k, v, out, scale=None, is_causal=False, **kwargs):
            transposed_out_contiguous.append(out.transpose(1, 2).is_contiguous())
            out.copy_(q)

        module = self._patched_module(sdpa=fake_sdpa)
        q = torch.randn(2, 6, 3, 8)
        module.torch_sdpa_wrapper(q, q, q, scale=0.25)
        module.torch_sdpa_wrapper(
            q,
            q,
            q,
            scale=0.25,
            cu_seqlens=torch.tensor([0, 2, 6], dtype=torch.int32),
        )
        # One dense call plus the two ragged segments.
        self.assertEqual(transposed_out_contiguous, [True, True, True])

    def test_movedim_matches_the_stock_rearrange(self):
        """The patch swaps S and H the same way stock apply_sdpa does."""
        import einops

        q = torch.randn(2, 4, 3, 8)
        self.assertTrue(
            torch.equal(q.movedim(1, 2), einops.rearrange(q, "b s h d -> b h s d"))
        )
        self.assertTrue(torch.equal(q.movedim(1, 2).movedim(1, 2), q))


if __name__ == "__main__":
    unittest.main()
