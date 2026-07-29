# ****************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ****************************************************************************

"""Out-of-tree DA8W4 (W4A8) patch for vLLM's ``ZentorchWNA16LinearKernel``.

Adds the DA8W4 (dynamic bf16->s8 activation x symmetric s4 weight) fast path to
``ZentorchWNA16LinearKernel`` as a runtime monkey-patch, so vanilla vLLM needs
no source changes.
"""

from __future__ import annotations

import importlib.util
import os
import sys

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_TARGET_MODULE = "vllm.model_executor.kernels.linear.mixed_precision.zentorch"
_HOOK_INSTALLED = False


def _da8w4_enabled() -> bool:
    """DA8W4 is the default; set ``VLLM_CPU_INT4_W4A8=0`` to force the W4A16 path."""
    return os.environ.get("VLLM_CPU_INT4_W4A8", "1") != "0"


# ---------------------------------------------------------------------------
# Methods injected onto ZentorchWNA16LinearKernel (bound as instance methods)
# ---------------------------------------------------------------------------


def _zentorch_da8w4_eligible(self, layer: "torch.nn.Module") -> bool:
    """Eligibility for the zentorch DA8W4 fast path.

    DA8W4 = dynamic bf16->s8 activation x symmetric s4 weight. Requires DA8W4 not
    be disabled (VLLM_CPU_INT4_W4A8=0), a symmetric (zero-point-free) W4
    checkpoint, bf16 activations, and group_size divisible by 4 (kernel
    constraint). When any check fails we leave the layer for the W4A16 WOQ path /
    ``super()``.
    """
    if not _da8w4_enabled():
        return False

    from vllm.model_executor.kernels.linear.zentorch_utils import has_zentorch_op
    from vllm.scalar_type import scalar_types

    if not has_zentorch_op(
        ["zentorch_woq_repack_weight", "zentorch_dynamic_qlinear"]
    ):
        return False

    if self.config.zero_points:
        return False
    if self.config.weight_type == scalar_types.uint4:
        return False

    # The kernel rejects f32 source; activations must be bf16.
    if getattr(self.config, "act_type", None) != torch.bfloat16:
        return False

    # DA8W4 uses the same W4 checkpoint as the W4A16 WOQ path, so the same
    # layer-eligibility checks apply -- reuse _zentorch_woq_eligible here.
    if not self._zentorch_woq_eligible(layer):
        return False

    weight_packed = getattr(layer, self.w_q_name)
    weight_scale = getattr(layer, self.w_s_name)
    in_features = weight_packed.shape[1] * 8
    num_groups = weight_scale.shape[1]
    if num_groups <= 0 or in_features % num_groups != 0:
        return False
    group_size = in_features // num_groups
    # AOCL sym_quant: K/G must be a multiple of 4; K must be even to pack
    # 2 s4 per byte.
    return group_size % 4 == 0 and in_features % 2 == 0


def _process_da8w4_weights(self, layer: "torch.nn.Module") -> None:
    """Repack CT symmetric W4 weights into the ZenDNN DA8W4 s4 layout.

    Produces signed s4 weights packed 2-per-byte in ``[N, K/2]`` (no transpose;
    consumed with ``transB=true``) and per-group ``{G, N}`` scales in bf16.
    Bit-identical weight values to the validated WOQ s4 reference, only the
    packing differs.
    """
    if (not self.config.zero_points) and (self.w_zp_name is not None):
        setattr(layer, self.w_zp_name, None)
    if (not self.config.has_g_idx) and (self.w_gidx_name is not None):
        setattr(layer, self.w_gidx_name, None)

    weight_q = getattr(layer, self.w_q_name)
    weight_s = getattr(layer, self.w_s_name)
    weight_packed = weight_q.data if hasattr(weight_q, "data") else weight_q
    weight_scale = weight_s.data if hasattr(weight_s, "data") else weight_s

    bits = self.config.weight_type.mantissa
    pack_factor = torch.iinfo(weight_packed.dtype).bits // bits
    out_features, num_groups = weight_scale.shape[0], weight_scale.shape[1]
    in_features = weight_packed.shape[1] * pack_factor
    original_shape = torch.Size([out_features, in_features])
    # Reuse vLLM's own cross-version compressed-tensors importer.
    from vllm.model_executor.kernels.linear.mixed_precision.zentorch import (
        _import_unpack_from_int32,
    )

    unpack_from_int32 = _import_unpack_from_int32()

    weight_unpacked = unpack_from_int32(
        weight_packed,
        bits,
        original_shape,
        packed_dim=weight_q.packed_dim,
    )
    # Reuse the WOQ repack (8 int4/int32)
    packed = torch.ops.zentorch.zentorch_woq_repack_weight.default(
        weight_unpacked.to(torch.int8).contiguous()
    ).view(torch.int8)

    layer._zentorch_da8w4_packed = packed
    # CT scale is [N, G]; DA8W4 wants per-group {G, N} in f32/bf16.
    layer._zentorch_da8w4_scale = weight_scale.t().contiguous().to(torch.bfloat16)

    for param_name in (self.w_q_name, self.w_s_name, self.w_zp_name):
        if param_name is None:
            continue
        param = getattr(layer, param_name, None)
        if param is None:
            continue
        if hasattr(param, "data"):
            param.data = torch.empty(0)
        else:
            setattr(layer, param_name, torch.empty(0))

    layer._zentorch_kind = "compressed_tensors_w4a8_da8w4"
    layer._zentorch_da8w4 = True
    layer._zentorch_processed_weights = True
    logger.info_once(
        "[zen_cpu] Using zentorch DA8W4 (W4A8) for symmetric W4 "
        "(weight_type=%s, group_size=%d)",
        self.config.weight_type,
        in_features // num_groups,
    )


# ---------------------------------------------------------------------------
# Patch application
# ---------------------------------------------------------------------------


def _do_patch_da8w4() -> bool:
    """Attach DA8W4 helpers + wrap process/apply on ZentorchWNA16LinearKernel."""
    try:
        from vllm.model_executor.kernels.linear.mixed_precision.zentorch import (
            ZentorchWNA16LinearKernel as _Kernel,
        )
    except ImportError:
        logger.debug("[zentorch] ZentorchWNA16LinearKernel not importable; skip DA8W4")
        return False

    if getattr(_Kernel, "_zentorch_da8w4_patched", False):
        return True

    _Kernel._zentorch_da8w4_eligible = _zentorch_da8w4_eligible
    _Kernel._process_da8w4_weights = _process_da8w4_weights

    _orig_process = _Kernel.process_weights_after_loading

    def _patched_process(self, layer):
        # process_weights_after_loading mutates `layer` in place and returns
        # None; keep every path a bare return (no value) for consistency.
        if getattr(layer, "_zentorch_processed_weights", False):
            return
        if self._zentorch_da8w4_eligible(layer):
            self._process_da8w4_weights(layer)
            return
        _orig_process(self, layer)
        return

    _Kernel.process_weights_after_loading = _patched_process

    _orig_apply = _Kernel.apply_weights

    def _patched_apply(self, layer, x, bias=None):
        if getattr(layer, "_zentorch_da8w4", False):
            # DA8W4: dynamic bf16->s8 activation x symmetric s4 weight.
            x_bf16 = x if x.dtype == torch.bfloat16 else x.to(torch.bfloat16)
            # The kernel reads bias via raw data_ptr(), so it must be contiguous
            # (packed weight/scales are already made contiguous in processing).
            bias_c = bias.contiguous() if bias is not None else None
            # DA8W4 is inferred from the packed int8 [N, K/2] weight dtype/dims;
            # no explicit mode selector is passed.
            return torch.ops.zentorch.zentorch_dynamic_qlinear.default(
                x_bf16,
                layer._zentorch_da8w4_packed,
                layer._zentorch_da8w4_scale,
                bias_c,
            )
        return _orig_apply(self, layer, x, bias)

    _Kernel.apply_weights = _patched_apply

    _Kernel._zentorch_da8w4_patched = True
    logger.info(
        "[zentorch] Patched ZentorchWNA16LinearKernel with DA8W4 (W4A8) fast path"
    )
    return True


class _Da8w4ImportHook:
    """Post-import hook: patch ZentorchWNA16LinearKernel after its module loads."""

    _TARGET_MODULE = _TARGET_MODULE

    def find_spec(self, fullname, path, target=None):
        if fullname != self._TARGET_MODULE:
            return None
        if self in sys.meta_path:
            sys.meta_path.remove(self)

        spec = importlib.util.find_spec(fullname)
        if spec is None or spec.loader is None:
            return None

        original_exec = spec.loader.exec_module

        def _exec_then_patch(module):
            original_exec(module)
            _do_patch_da8w4()

        spec.loader.exec_module = _exec_then_patch
        return spec


def apply_da8w4_patch() -> bool:
    """Install the DA8W4 patch, deferred if the target module isn't loaded yet."""
    global _HOOK_INSTALLED

    if _HOOK_INSTALLED:
        return True

    if _TARGET_MODULE in sys.modules:
        result = _do_patch_da8w4()
    else:
        sys.meta_path.insert(0, _Da8w4ImportHook())
        logger.debug("[zentorch] Installed DA8W4 kernel import hook")
        result = True

    _HOOK_INSTALLED = result
    return result
