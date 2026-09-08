# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

"""
Unit tests for the sliding-window block-size alignment patch.

The patch wraps Attention.get_kv_cache_spec() and realigns
SlidingWindowSpec.block_size to the CPU ISA's BlockSizeAlignment
(e.g. 32 for vec/amx), preventing the 16 % 32 != 0 crash at
cpu_attn.cpp when running sliding-window models on the CPU backend.
"""

import dataclasses
import unittest
import unittest.mock

import zentorch  # noqa: F401 - ensures zentorch native extension is loaded

from ._test_constants import VLLM_AVAILABLE


@unittest.skipUnless(VLLM_AVAILABLE, "vLLM not installed")
class TestSWBlockSizePatchWiring(unittest.TestCase):
    """SWBlockSize is wired into _PATCHES (always-on correctness fix)."""

    def test_wired_in_patches(self):
        import zentorch.vllm as zv

        names = [name for name, _ in zv._PATCHES]
        self.assertIn("SWBlockSize", names)

    def test_always_enabled(self):
        """The patch has no env-var opt-out; it is a correctness fix."""
        import inspect
        from zentorch.vllm._sw_blocksize_patch import (
            _apply_sw_blocksize_patch,
        )

        source = inspect.getsource(_apply_sw_blocksize_patch)
        self.assertNotIn(
            "os.environ",
            source,
            "_apply_sw_blocksize_patch should not have an "
            "env-var opt-out (correctness fix)",
        )


# ---------------------------------------------------------------------------
# Fake types used by the behavior tests below.  They mirror just enough of
# the real vLLM types so _patched_sw_get_kv_cache_spec can operate on them.
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _FakeSlidingWindowSpec:
    """Stand-in for vllm.v1.kv_cache_interface.SlidingWindowSpec."""

    block_size: int
    num_kv_heads: int = 8
    head_size: int = 128
    dtype: str = "auto"
    sliding_window: int = 512
    page_size_padded: int | None = None

    @property
    def real_page_size_bytes(self) -> int:
        """Simplified: 2 * block_size * num_kv_heads * head_size."""
        return 2 * self.block_size * self.num_kv_heads * self.head_size


@dataclasses.dataclass
class _FakeFullAttentionSpec:
    """Stand-in for a non-SW spec (e.g. FullAttentionSpec)."""

    block_size: int = 16


class _FakeCPUAttentionBackend:
    """Sentinel for the CPU attention backend."""

    pass


class _FakeOtherBackend:
    """Sentinel for a non-CPU attention backend."""

    pass


@dataclasses.dataclass
class _FakeCacheConfig:
    block_size: int = None
    cache_dtype: str = "auto"


@dataclasses.dataclass
class _FakeModelConfig:
    dtype: str = "bfloat16"


@dataclasses.dataclass
class _FakeVllmConfig:
    cache_config: _FakeCacheConfig = dataclasses.field(
        default_factory=_FakeCacheConfig
    )
    model_config: _FakeModelConfig = dataclasses.field(
        default_factory=_FakeModelConfig
    )


def _make_attn_self(backend, head_size=128):
    """Build a mock Attention ``self`` with attributes the patch reads."""
    obj = unittest.mock.MagicMock()
    obj.attn_backend = backend
    obj.head_size = head_size
    return obj


@unittest.skipUnless(VLLM_AVAILABLE, "vLLM not installed")
class TestSWBlockSizeAlignment(unittest.TestCase):
    """Exercise _patched_sw_get_kv_cache_spec alignment logic."""

    def _call_patch(
        self, spec, alignment,
        config_block_size=None, backend=None,
    ):
        """Invoke the patched wrapper with controlled mocks."""
        from zentorch.vllm._sw_blocksize_patch import (
            _patched_sw_get_kv_cache_spec,
        )

        if backend is None:
            backend = _FakeCPUAttentionBackend

        attn_self = _make_attn_self(backend, head_size=128)
        attn_self._zentorch_orig_sw_get_kv_cache_spec = (
            lambda cfg: spec
        )

        vllm_config = _FakeVllmConfig(
            cache_config=_FakeCacheConfig(
                block_size=config_block_size
            ),
        )

        with (
            unittest.mock.patch(
                "zentorch.vllm._sw_blocksize_patch"
                "._get_cpu_isa_block_alignment",
                return_value=alignment,
            ),
            unittest.mock.patch(
                "vllm.v1.kv_cache_interface.SlidingWindowSpec",
                _FakeSlidingWindowSpec,
            ),
            unittest.mock.patch(
                "vllm.v1.attention.backends.cpu_attn.CPUAttentionBackend",
                _FakeCPUAttentionBackend,
            ),
        ):
            return _patched_sw_get_kv_cache_spec(
                attn_self, vllm_config
            )

    # --- core alignment tests ---

    def test_misaligned_block_size_is_rounded_up(self):
        """block_size=16 with alignment=32 -> rounded up to 32."""
        spec = _FakeSlidingWindowSpec(block_size=16)
        result = self._call_patch(spec, alignment=32)
        self.assertEqual(result.block_size, 32)

    def test_already_aligned_is_unchanged(self):
        """block_size=32 with alignment=32 -> returned as-is."""
        spec = _FakeSlidingWindowSpec(block_size=32)
        result = self._call_patch(spec, alignment=32)
        self.assertIs(result, spec)

    def test_config_block_size_not_adopted(self):
        """Misaligned spec (16) + aligned config (64) -> round-up 32.

        Upstream picks the smallest legal block for SW groups on
        purpose; adopting the config block_size would waste KV pool.
        """
        spec = _FakeSlidingWindowSpec(block_size=16)
        result = self._call_patch(
            spec, alignment=32, config_block_size=64
        )
        self.assertEqual(result.block_size, 32)

    def test_vec16_alignment_16(self):
        """block_size=16 with alignment=16 -> already aligned, unchanged."""
        spec = _FakeSlidingWindowSpec(block_size=16)
        result = self._call_patch(spec, alignment=16)
        self.assertIs(result, spec)

    # --- passthrough tests ---

    def test_none_spec_passthrough(self):
        """None spec is returned as None."""
        result = self._call_patch(None, alignment=32)
        self.assertIsNone(result)

    def test_non_sliding_window_spec_passthrough(self):
        """Non-SlidingWindowSpec is returned unchanged."""
        spec = _FakeFullAttentionSpec(block_size=16)
        result = self._call_patch(spec, alignment=32)
        self.assertIs(result, spec)
        self.assertEqual(result.block_size, 16)

    def test_non_cpu_backend_passthrough(self):
        """Misaligned block_size on a non-CPU backend is untouched."""
        spec = _FakeSlidingWindowSpec(block_size=16)
        result = self._call_patch(
            spec, alignment=32, backend=_FakeOtherBackend
        )
        self.assertIs(result, spec)
        self.assertEqual(result.block_size, 16)

    def test_page_size_padded_fits_proceeds(self):
        """When page_size_padded is large enough, alignment proceeds."""
        # real_page_size_bytes for block=16: 2*16*8*128 = 32768
        # per_token = 32768 // 16 = 2048
        # new_block_size = 32 -> 32 * 2048 = 65536
        # page_size_padded = 65536 -> fits
        spec = _FakeSlidingWindowSpec(
            block_size=16, page_size_padded=65536
        )
        result = self._call_patch(spec, alignment=32)
        self.assertEqual(result.block_size, 32)

    def test_page_size_padded_overflow_raises(self):
        """When aligned block exceeds padded page, ValueError is raised."""
        # real_page_size_bytes for block=16: 2*16*8*128 = 32768
        # per_token = 32768 // 16 = 2048
        # new_block_size = 32 -> 32 * 2048 = 65536
        # page_size_padded = 40000 < 65536 -> overflow
        spec = _FakeSlidingWindowSpec(
            block_size=16, page_size_padded=40000
        )
        with self.assertRaises(ValueError):
            self._call_patch(spec, alignment=32)


@unittest.skipUnless(VLLM_AVAILABLE, "vLLM not installed")
class TestSWBlockSizeIdempotency(unittest.TestCase):
    """_do_patch_sw_blocksize is idempotent (marker-based guard)."""

    def test_idempotent(self):
        from zentorch.vllm._sw_blocksize_patch import (
            _do_patch_sw_blocksize,
        )

        # First call patches; second call sees the marker and returns True.
        self.assertTrue(_do_patch_sw_blocksize())
        self.assertTrue(_do_patch_sw_blocksize())


def run_tests():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for test_case in (
        TestSWBlockSizePatchWiring,
        TestSWBlockSizeAlignment,
        TestSWBlockSizeIdempotency,
    ):
        suite.addTests(loader.loadTestsFromTestCase(test_case))
    return unittest.TextTestRunner(verbosity=2).run(suite)


if __name__ == "__main__":
    run_tests()
