# ****************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ****************************************************************************

"""SW block-size alignment patch (correctness fix; always enabled).

Wraps ``Attention.get_kv_cache_spec()`` to realign
``SlidingWindowSpec.block_size`` to the CPU ISA's BlockSizeAlignment
(e.g. 32 for vec/amx), preventing the ``block_size % 32 != 0`` crash
at ``cpu_attn.cpp`` when running sliding-window models on the CPU
backend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from zentorch._logging import get_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheSpec

from zentorch.vllm._import_hook import patch_now_or_on_import

logger = get_logger(__name__)

_SW_ATTN_MODULE = (
    "vllm.model_executor.layers.attention.attention"
)
_sw_blocksize_warned = False


def _warn_sw_block_size_aligned(
    old_block_size: int,
    new_block_size: int,
    alignment: int,
) -> None:
    """Log a one-shot warning when the SW block_size is realigned."""
    global _sw_blocksize_warned
    if not _sw_blocksize_warned:
        logger.warning(
            "[zentorch] SW block_size aligned %d -> %d "
            "(ISA BlockSizeAlignment=%d)",
            old_block_size,
            new_block_size,
            alignment,
        )
        _sw_blocksize_warned = True


def _get_cpu_isa_block_alignment(
    attn_self, vllm_config: VllmConfig,
) -> int:
    """Return the BlockSizeAlignment required by the CPU ISA.

    vec16 (head_size % 32 != 0 and head_size % 16 == 0) uses
    alignment 16; all other ISAs (vec, amx, neon, rvv, vxe, vsx)
    use alignment 32.  Falls back to 32 on any import/attribute
    error.
    """
    try:
        from vllm.v1.attention.backends.cpu_attn import (
            _get_attn_isa,
        )

        block_size = (
            vllm_config.cache_config.block_size or 32
        )
        kv_cache_dtype = getattr(
            vllm_config.cache_config, "cache_dtype", "auto"
        )
        isa = _get_attn_isa(
            vllm_config.model_config.dtype,
            block_size,
            attn_self.head_size,
            kv_cache_dtype,
        )
        return 16 if isa == "vec16" else 32
    except (ImportError, AttributeError):
        logger.debug(
            "[zentorch] _get_cpu_isa_block_alignment "
            "failed; falling back to alignment=32",
            exc_info=True,
        )
        return 32  # safe default: vec/amx alignment


def _patched_sw_get_kv_cache_spec(
    self, vllm_config: VllmConfig,
) -> KVCacheSpec | None:
    """Wrapper that aligns SW block_size.

    Calls the original method, then — only for
    ``SlidingWindowSpec`` results on the CPU backend — ensures
    ``block_size`` is a multiple of the ISA's
    BlockSizeAlignment.  Non-SW layers, non-CPU backends, and
    already-aligned specs are returned as-is.
    """
    import dataclasses

    spec = self._zentorch_orig_sw_get_kv_cache_spec(
        vllm_config
    )
    if spec is None:
        return spec

    try:
        from vllm.v1.kv_cache_interface import (
            SlidingWindowSpec,
        )
        from vllm.v1.attention.backends.cpu_attn import (
            CPUAttentionBackend,
        )
    except ImportError:
        return spec

    if not isinstance(spec, SlidingWindowSpec):
        return spec
    if self.attn_backend is not CPUAttentionBackend:
        return spec

    alignment = _get_cpu_isa_block_alignment(
        self, vllm_config
    )
    if spec.block_size % alignment == 0:
        return spec  # already valid, nothing to do

    # Round up to the ISA's alignment rather than adopting
    # cache_config.block_size: upstream picks the smallest
    # legal block for SW groups on purpose (attention.py,
    # "unify scales it up by an integer ratio"), and a
    # larger block only wastes KV pool per request.
    new_block_size = (
        -(-spec.block_size // alignment) * alignment
    )

    if getattr(spec, "page_size_padded", None) is not None:
        # This spec pads up to a page shared with other
        # groups; growing the block without growing that
        # page breaks the invariant asserted in
        # AttentionSpec.page_size_bytes.
        per_token = (
            spec.real_page_size_bytes // spec.block_size
        )
        if (
            new_block_size * per_token
            > spec.page_size_padded
        ):
            raise ValueError(
                f"Sliding-window layer needs block_size "
                f"{new_block_size} for CPU ISA alignment "
                f"{alignment}, but the shared KV page "
                f"({spec.page_size_padded} B) only fits "
                f"{spec.block_size}. Raise --block-size "
                f"or disable KV-cache skip layers."
            )

    _warn_sw_block_size_aligned(
        spec.block_size, new_block_size, alignment
    )
    return dataclasses.replace(
        spec, block_size=new_block_size
    )


def _do_patch_sw_blocksize() -> bool:
    """Monkey-patch ``Attention.get_kv_cache_spec``."""
    try:
        from vllm.model_executor.layers.attention.attention import (  # noqa: E501
            Attention,
        )
    except ImportError:
        return False

    if hasattr(Attention, "_zentorch_sw_blocksize_patched"):
        return True

    Attention._zentorch_orig_sw_get_kv_cache_spec = (
        Attention.get_kv_cache_spec
    )
    Attention.get_kv_cache_spec = (
        _patched_sw_get_kv_cache_spec
    )
    Attention._zentorch_sw_blocksize_patched = True
    logger.info(
        "[zentorch] Patched Attention.get_kv_cache_spec "
        "-> SW block_size ISA alignment"
    )
    return True


def _apply_sw_blocksize_patch() -> bool:
    """Apply the SW block-size alignment patch.

    Correctness fix; always on (no env-var opt-out).
    """
    return patch_now_or_on_import(
        _SW_ATTN_MODULE, _do_patch_sw_blocksize
    )
