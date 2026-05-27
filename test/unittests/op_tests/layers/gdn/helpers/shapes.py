# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

"""Canonical Qwen3.5-4B / Qwen3-Next GDN shapes."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["GDNShape", "Qwen35_4B_GDN", "QwenNext_GDN", "common_seq_lens"]


@dataclass(frozen=True)
class GDNShape:
    name: str
    hidden_size: int
    num_k_heads: int
    num_v_heads: int
    head_k_dim: int
    head_v_dim: int
    conv_kernel_size: int
    rms_norm_eps: float = 1e-6

    @property
    def key_dim(self) -> int:
        return self.num_k_heads * self.head_k_dim

    @property
    def value_dim(self) -> int:
        return self.num_v_heads * self.head_v_dim

    @property
    def conv_dim(self) -> int:
        return 2 * self.key_dim + self.value_dim


QwenNext_GDN = GDNShape(
    name="Qwen3-Next",
    hidden_size=2048,
    num_k_heads=16,
    num_v_heads=32,
    head_k_dim=128,
    head_v_dim=128,
    conv_kernel_size=4,
)

Qwen35_4B_GDN = GDNShape(
    name="Qwen3.5-4B",
    hidden_size=2048,
    num_k_heads=16,
    num_v_heads=32,
    head_k_dim=128,
    head_v_dim=128,
    conv_kernel_size=4,
)


def common_seq_lens() -> tuple[int, ...]:
    return (1, 7, 64, 65, 128, 257)
