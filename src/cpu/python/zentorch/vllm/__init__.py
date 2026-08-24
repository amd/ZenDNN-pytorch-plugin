# ****************************************************************************
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ****************************************************************************

"""vLLM - zentorch integration via the plugin pattern.

Supports released vLLM in the inclusive [VLLM_MIN_VERSION, VLLM_MAX_VERSION]
window on PyTorch 2.13; other versions fall back to the stock CPU platform.

Entry points:
- vllm.platform_plugins -> returns the ZenCPUPlatform class path
- vllm.general_plugins  -> applies the Zen-CPU patches

The patches applied at registration are listed in ``_PATCHES``.

Reference: https://blog.vllm.ai/2025/11/20/vllm-plugin-system.html
"""

from __future__ import annotations

import importlib.util
import os
import sys

import torch
from packaging import version as pkg_version

from zentorch._logging import get_logger
from zentorch._utils import counters, _SUPPORTED_MOE_ACTIVATIONS

# Re-exported at module scope so tests can mock the Int8Tensor dispatch impl
# (see _apply_torchao_patch); the import has no torchao dependency itself.
from zentorch.vllm._torchao_int8_patch import (  # noqa: E402, F401
    _apply_torchao_int8_tensor_patch_impl,
)
from zentorch.vllm._moe_class import (  # noqa: E402, F401
    _apply_torchao_moe_patch_impl,
)
from zentorch.vllm._int8_moe_patch import (  # noqa: E402, F401
    _apply_int8_moe_patch_impl,
)
from zentorch.vllm._wna16_moe_patch import (  # noqa: E402, F401
    _apply_wna16_moe_patch_impl,
)
from zentorch.vllm._gptoss_moe_loader_patch import (  # noqa: E402, F401
    _apply_gptoss_loader_patch_impl,
)
from zentorch.vllm._mixtral_moe_loader_patch import (  # noqa: E402, F401
    _apply_mixtral_loader_patch_impl,
)
from zentorch.vllm._gemma4_hetero_config_patch import (  # noqa: E402, F401
    _apply_gemma4_hetero_patch,
)
from zentorch.vllm._da8w4_kernel_patch import (  # noqa: E402, F401
    _apply_da8w4_patch,
)
from zentorch.vllm._import_hook import patch_now_or_on_import  # noqa: E402

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Supported runtime
# ---------------------------------------------------------------------------
# Inclusive window of released vLLM versions validated against this plugin. The
# upper bound is explicit, not an open "< next minor" range: bump VLLM_MAX_VERSION
# after validating each new patch release.
VLLM_MIN_VERSION = "0.27.0"
VLLM_MAX_VERSION = "0.27.1"
TORCH_MIN_VERSION = (2, 13)


def get_vllm_version() -> str | None:
    """Return the imported vLLM version string, or None if vLLM is absent."""
    if "vllm" not in sys.modules:
        return None
    return getattr(sys.modules["vllm"], "__version__", None)


def _base_version(ver: str) -> str:
    """Strip the local build suffix, e.g. 0.27.0+cpu -> 0.27.0.

    rc/dev suffixes are kept so pre-releases parse below the release and are
    rejected by ``is_supported_vllm``.
    """
    return ver.split("+")[0]


def is_supported_vllm(ver: str | None) -> bool:
    """True iff ``ver`` is a released version in the inclusive
    [VLLM_MIN_VERSION, VLLM_MAX_VERSION] window. Pre-releases and versions above
    the window are rejected.
    """
    if not ver:
        return False
    try:
        parsed = pkg_version.parse(_base_version(ver))
    except pkg_version.InvalidVersion:
        return False
    if parsed.is_prerelease:  # reject rc/dev/alpha/beta (e.g. 0.27.0rc2, 0.28.0rc1)
        return False
    return (
        pkg_version.parse(VLLM_MIN_VERSION)
        <= parsed
        <= pkg_version.parse(VLLM_MAX_VERSION)
    )


def is_supported_torch() -> bool:
    """True iff the running PyTorch is >= 2.13 (the vLLM 0.27 CPU baseline)."""
    from torch.torch_version import TorchVersion

    return TorchVersion(torch.__version__) >= TORCH_MIN_VERSION


# ---------------------------------------------------------------------------
# RMSNorm CPU Forward Patch (deferred via import hook)
# ---------------------------------------------------------------------------

_LAYERNORM_MODULE = "vllm.model_executor.layers.layernorm"


def _do_patch_rmsnorm() -> bool:
    """Patch RMSNorm.forward to use zentorch fused add-RMS-norm kernel."""
    try:
        from vllm.model_executor.layers.layernorm import RMSNorm
    except ImportError:
        return False

    if hasattr(RMSNorm, "_zentorch_rmsnorm_patched"):
        return True

    def patched_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.variance_size_override is not None:
            return self.forward_native(x, residual)
        if residual is not None:
            torch.ops.zentorch.zentorch_add_rms_norm_(
                x, self.weight.data, residual, self.variance_epsilon
            )
            return x, residual
        # The non-residual custom op causes accuracy issues with Qwen models,
        # so fall back to the native path there.
        return self.forward_native(x, residual)

    RMSNorm.forward = patched_forward
    RMSNorm._zentorch_rmsnorm_patched = True
    logger.info("[zentorch] Patched RMSNorm.forward (bf16/fp32/fp16 -> zentorch)")
    return True


def _apply_rmsnorm_patch() -> bool:
    return patch_now_or_on_import(_LAYERNORM_MODULE, _do_patch_rmsnorm)


# ---------------------------------------------------------------------------
# CPUFusedMOE Patch (opt-out via ZENTORCH_FUSED_MOE=0; deferred via import hook)
# ---------------------------------------------------------------------------
#
# Replaces vLLM's CPUFusedMOE with a single call to
# torch.ops.zentorch.zentorch_fused_moe. The op runs the full MoE FFN block
# (token grouping -> W13 GEMM -> gated activation -> W2 GEMM -> weighted
# reduce) inside one ZenDNN group_matmul_direct call.

_CPU_FUSED_MOE_MODULE = "vllm.model_executor.layers.fused_moe.cpu_fused_moe"


def _moe_forward_zentorch(
    self,
    layer,
    input,
    topk_weights,
    topk_ids,
    activation,
    global_num_experts: int = -1,
    apply_router_weight_on_input: bool = False,
):
    """CPUFusedMOE forward replacement dispatching to zentorch_fused_moe.

    Mirrors vLLM's forward_grouped_gemm / forward_torch signature so the
    enclosing CPUFusedMOE.__call__ can hand off to us transparently.
    """
    # Activation may arrive as a MoEActivation enum or a raw string.
    act = activation if isinstance(activation, str) else activation.value
    if act not in _SUPPORTED_MOE_ACTIVATIONS:
        raise ValueError(
            f"[zentorch] Unsupported activation {act!r}. "
            f"Must be one of {_SUPPORTED_MOE_ACTIVATIONS}"
        )

    # Op's reduce post-op writes every element -> no zero-init needed.
    output = torch.empty_like(input)

    if apply_router_weight_on_input:
        # Match vLLM: pre-apply the K=1 router weight to the input then signal
        # the op to skip its own weighted reduce.
        input = input.mul(topk_weights.to(input.dtype))

    torch.ops.zentorch.zentorch_fused_moe(
        output,
        input,
        layer.w13_weight,
        layer.w2_weight,
        getattr(layer, "w13_bias", None),
        getattr(layer, "w2_bias", None),
        topk_weights,
        topk_ids,
        apply_router_weight_on_input,  # skip_weighted
        act,
        getattr(layer, "w13_scale", None),
        getattr(layer, "w2_scale", None),
    )
    return output


def _do_patch_fused_moe() -> bool:
    """Patch CPUFusedMOE.__init__ to dispatch through zentorch_fused_moe."""
    try:
        from vllm.model_executor.layers.fused_moe.cpu_fused_moe import CPUFusedMOE
    except ImportError:
        return False

    if hasattr(CPUFusedMOE, "_zentorch_fused_moe_patched"):
        return True

    CPUFusedMOE._zentorch_forward = _moe_forward_zentorch

    def _patched_init(self, layer):
        # Skip vLLM's prepacking + grouped-gemm path; we use the standard
        # [E, ...] layout that zentorch_fused_moe expects.
        self.isa = "none"
        self.forward_method = self._zentorch_forward
        # zentorch_fused_moe consumes the [E, ...] weights via per-expert
        # .select(0, e). ZenDNN's group_matmul expects each per-expert slice to
        # be row-major contiguous, which only holds when the parent [E, ...]
        # tensor itself is contiguous. Normalize once here (a no-op when already
        # contiguous) to keep the hot path free of per-call .contiguous() copies.
        from vllm.model_executor.layers.quantization.utils.layer_utils import (
            replace_parameter,
        )

        # Extract int8 weight scales from torchao Int8Tensor.
        if importlib.util.find_spec("torchao") is None:
            logger.info(
                "[zentorch] torchao not installed, skipping Int8Tensor scale extraction"
            )
        else:
            from torchao.quantization.quantize_.workflows.int8.int8_tensor import (
                Int8Tensor,
            )

            for weight_attr, scale_attr in [
                ("w13_weight", "w13_scale"),
                ("w2_weight", "w2_scale"),
            ]:
                w = getattr(layer, weight_attr, None)
                if isinstance(w, Int8Tensor):
                    weight_scales = w.scale
                    if weight_scales.shape[-1] == 1:
                        weight_scales = weight_scales.squeeze(-1).contiguous()
                    if weight_scales.dtype not in (
                        torch.float32,
                        torch.bfloat16,
                        torch.float16,
                    ):
                        raise ValueError(
                            f"[zentorch] {weight_attr}.scale must be float32 "
                            f"or bfloat16 or float16, got {weight_scales.dtype}"
                        )
                    setattr(layer, scale_attr, weight_scales)
                    replace_parameter(layer, weight_attr, w.qdata)

        if not layer.w13_weight.is_contiguous():
            replace_parameter(layer, "w13_weight", layer.w13_weight.contiguous())
        if not layer.w2_weight.is_contiguous():
            replace_parameter(layer, "w2_weight", layer.w2_weight.contiguous())

        # Bump the per-replacement counter (matches _custom_op_replacement.py).
        counters["zentorch"]["zentorch_fused_moe"] += 1

    CPUFusedMOE.__init__ = _patched_init
    CPUFusedMOE._zentorch_fused_moe_patched = True
    logger.info("[zentorch] Patched CPUFusedMOE forward -> zentorch_fused_moe")
    return True


def _apply_fused_moe_patch() -> bool:
    """Opt out with ZENTORCH_FUSED_MOE=0 to keep vLLM's stock CPUFusedMOE."""
    if os.environ.get("ZENTORCH_FUSED_MOE", "1") == "0":
        return False
    return patch_now_or_on_import(_CPU_FUSED_MOE_MODULE, _do_patch_fused_moe)


# ---------------------------------------------------------------------------
# TorchAO patch (Int4 opaque tensor config/ops + Int8Tensor + FusedMoE)
# ---------------------------------------------------------------------------


def _apply_torchao_patch() -> bool:
    """Register TorchAO hooks when torchao is installed, else skip."""
    if importlib.util.find_spec("torchao") is None:
        logger.info("[zentorch] TorchAO not installed, skipping TorchAO patch")
        return False
    from ._torchao_int4_opaque_patch import (
        _register_int4_opaque_tensor_config,
        _register_int4_slice_op,
    )

    _register_int4_opaque_tensor_config()
    _register_int4_slice_op()
    _apply_torchao_int8_tensor_patch_impl()
    _apply_torchao_moe_patch_impl()
    logger.info(
        "[zentorch] Registered TorchAO operations (Int4 + Int8Tensor + FusedMoE)."
    )
    return True


# ---------------------------------------------------------------------------
# CPU SDPA patch (encoder-only; opt out with ZENTORCH_SDPA=0)
# ---------------------------------------------------------------------------
# Wraps CPUAttentionBackendImpl.forward and routes encoder / encoder-only
# attention through zentorch_sdpa.out. Decoder and cross-attention still use
# the native cpu_attention_with_kv_cache path. Mask builders are vendored here
# because vLLM no longer ships them. Experimental; enabled by default.

_CPU_ATTN_MODULE = "vllm.v1.attention.backends.cpu_attn"


def _zentorch_sdpa_supports_dtype(dtype: torch.dtype) -> bool:
    """Mirror of the dtype/ISA gate in zentorch_sdpa_common (Sdpa_ref.cpp).

    Below that gate zentorch_sdpa falls back to the native ATen flash kernel,
    which is what vLLM would have run anyway, so the replacement only adds the
    wrapper cost.
    """
    from zentorch._C import (
        is_avx512_supported,
        is_bf16_supported,
        is_fp16_supported,
    )

    if dtype == torch.bfloat16:
        return is_bf16_supported()
    if dtype == torch.float16:
        return is_fp16_supported()
    if dtype == torch.float32:
        return is_avx512_supported()
    return False


def _zen_make_alibi_bias(alibi_slopes, dtype, start_loc):
    """Vendored ALiBi mask builder (no longer shipped by vLLM cpu_attn)."""
    attn_biases = []
    seq_num = start_loc.size(0) - 1
    start_loc = start_loc.numpy()
    for i in range(seq_num):
        seq_len = start_loc[i + 1] - start_loc[i]
        bias = torch.arange(seq_len, dtype=dtype)
        bias = bias[None, :] - bias[:, None]
        num_heads = alibi_slopes.shape[0]
        bias = bias[None, :].repeat((num_heads, 1, 1))
        bias.mul_(alibi_slopes[:, None, None]).unsqueeze_(0)
        inf_mask = (
            torch.empty((1, seq_len, seq_len), dtype=bias.dtype)
            .fill_(-torch.inf)
            .triu_(diagonal=1)
        )
        attn_biases.append((bias + inf_mask).to(dtype))
    return attn_biases


def _zen_make_sliding_window_bias(
    start_loc, left_window_size, right_window_size, dtype
):
    """Vendored sliding-window mask builder (no longer shipped by vLLM cpu_attn)."""
    attn_biases = []
    seq_num = start_loc.size(0) - 1
    start_loc = start_loc.numpy()
    for i in range(seq_num):
        seq_len = start_loc[i + 1] - start_loc[i]
        mask = torch.full((1, seq_len, seq_len), fill_value=1, dtype=dtype)
        if right_window_size != -1:
            mask = torch.tril(mask, diagonal=right_window_size)
        if left_window_size != -1:
            mask = torch.triu(mask, diagonal=-left_window_size)
        mask = torch.log(mask)
        attn_biases.append(mask)
    return attn_biases


def _masks_are_uniform(attn_masks) -> bool:
    """True when every sequence in the batch would get the same mask.

    The dense path folds the whole batch into a single zentorch_sdpa call with
    one broadcast mask, which is only correct if the per-sequence masks are
    interchangeable. The vendored builders derive masks from seq_len alone, so
    equal-length batches produce equal (though distinct) tensors and stay on
    the dense path; masks taken from attn_metadata.sdpa_attn_masks come from
    vLLM and may differ per request.
    """
    first = attn_masks[0]
    for mask in attn_masks[1:]:
        if mask is first:
            continue
        if (mask is None) != (first is None):
            return False
        if mask.shape != first.shape or not torch.equal(mask, first):
            return False
    return True


def _run_encoder_sdpa_zentorch(self, query, key, value, output, attn_metadata):
    """Encoder-only / encoder attention via zentorch_sdpa.

    Uses attn_metadata.query_start_loc for per-sequence offsets, vendored mask
    builders, and a symmetric bidirectional window derived from self.sliding_window.
    """
    start_loc = attn_metadata.query_start_loc

    # sliding_window is a scalar (-1 == disabled); map to a symmetric window.
    sw = self.sliding_window
    if isinstance(sw, (tuple, list)):
        sw_left, sw_right = int(sw[0]), int(sw[1])
    elif sw is None or sw == -1:
        sw_left = sw_right = -1
    else:
        sw_left = sw_right = int(sw) - 1

    attn_masks = getattr(attn_metadata, "sdpa_attn_masks", None)
    if attn_masks is None:
        if self.alibi_slopes is not None:
            attn_masks = _zen_make_alibi_bias(self.alibi_slopes, query.dtype, start_loc)
        elif sw_left != -1 or sw_right != -1:
            attn_masks = _zen_make_sliding_window_bias(
                start_loc, sw_left, sw_right, query.dtype
            )
        else:
            attn_masks = [None] * (start_loc.size(0) - 1)
        # Cache on the (per-group) metadata so sibling layers reuse it.
        attn_metadata.sdpa_attn_masks = attn_masks

    query = query.movedim(0, query.dim() - 2)
    key = key.movedim(0, key.dim() - 2)
    value = value.movedim(0, value.dim() - 2)

    # zentorch_sdpa has no enable_gqa arg but derives the KV head count from
    # the key tensor and maps each query head onto its KV head, so GQA/MQA
    # needs no expansion here.
    start_loc_np = start_loc.numpy()
    seq_lens = start_loc_np[1:] - start_loc_np[:-1]

    # vLLM packs all encoder sequences along the token dimension. When every
    # sequence has the same length and shares one mask, recover a dense BHSD
    # batch and invoke zentorch_sdpa once for the whole scheduler batch. Keep
    # the loop below as the fallback for ragged batches and for batches whose
    # requests carry different masks.
    if (
        len(seq_lens) > 0
        and (seq_lens == seq_lens[0]).all()
        and _masks_are_uniform(attn_masks)
    ):
        batch_size = len(seq_lens)
        seq_len = int(seq_lens[0])

        def _packed_hsd_to_bhsd(tensor):
            # [H, total_tokens, D] -> [B, H, S, D]
            return tensor.unflatten(1, (batch_size, seq_len)).permute(1, 0, 2, 3)

        q = _packed_hsd_to_bhsd(query)
        k = _packed_hsd_to_bhsd(key)
        v = _packed_hsd_to_bhsd(value)

        # Every sequence shares this mask (checked above). Preserve a leading
        # size-1 batch dimension so the C++ operator broadcasts it across the
        # dense batch.
        mask = attn_masks[0]
        if mask is not None and mask.dim() == 3:
            mask = mask.unsqueeze(1)

        # The op writes {B, H, S, D} into `out`. vLLM's buffer is packed
        # {tokens, H, D}, so hand over the matching view and let the kernel
        # store straight into it instead of materializing a second tensor.
        torch.ops.zentorch.zentorch_sdpa.out(
            q,
            k,
            v,
            dropout_p=0.0,
            is_causal=False,  # encoder attention is bidirectional
            attn_mask=mask,
            scale=self.scale,
            out=output.unflatten(0, (batch_size, seq_len)).permute(0, 2, 1, 3),
        )
        return output

    for i in range(len(attn_masks)):
        mask = attn_masks[i]
        # zentorch_sdpa only accepts a 2D or 4D attn_mask; the vendored builders
        # produce 3D [1, S, S], so promote to [1, 1, S, S].
        if mask is not None and mask.dim() == 3:
            mask = mask.unsqueeze(1)

        start_q = start_loc_np[i]
        end_q = start_loc_np[i + 1]

        q = query[None, :, start_q:end_q, :]
        k = key[None, :, start_q:end_q, :]
        v = value[None, :, start_q:end_q, :]

        torch.ops.zentorch.zentorch_sdpa.out(
            q,
            k,
            v,
            dropout_p=0.0,
            is_causal=False,  # encoder attention is bidirectional
            attn_mask=mask,
            scale=self.scale,
            out=output[start_q:end_q, :, :].movedim(0, 1).unsqueeze(0),
        )
    return output


def _forward_cpu_attn_zentorch(
    self,
    layer,
    query,
    key,
    value,
    kv_cache,
    attn_metadata,
    output,
    output_scale=None,
    output_block_scale=None,
):
    """CPUAttentionBackendImpl.forward wrapper.

    Routes ONLY encoder-only / encoder attention through zentorch_sdpa; every
    other case (decoder, cross-attn with cached KV, warm-up) delegates to the
    original native forward.
    """
    from vllm.v1.attention.backend import AttentionType

    is_encoder = self.attn_type in (
        AttentionType.ENCODER_ONLY,
        AttentionType.ENCODER,
    )
    if (
        attn_metadata is None
        or not is_encoder
        or key is None
        or value is None
        or output_scale is not None
        or output_block_scale is not None
        or not _zentorch_sdpa_supports_dtype(query.dtype)
    ):
        return self._zentorch_orig_forward(
            layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output,
            output_scale,
            output_block_scale,
        )

    num_actual_tokens = attn_metadata.num_actual_tokens
    _run_encoder_sdpa_zentorch(
        self,
        query[:num_actual_tokens],
        key[:num_actual_tokens],
        value[:num_actual_tokens],
        output[:num_actual_tokens],
        attn_metadata,
    )
    return output


def _do_patch_cpu_sdpa() -> bool:
    """Wrap CPUAttentionBackendImpl.forward -> zentorch_sdpa (encoder path)."""
    if not hasattr(torch.ops.zentorch, "zentorch_sdpa"):
        logger.debug("[zentorch] zentorch_sdpa op unavailable; skipping CPU SDPA patch")
        return False
    try:
        from vllm.v1.attention.backends.cpu_attn import CPUAttentionBackendImpl
    except ImportError:
        return False

    if hasattr(CPUAttentionBackendImpl, "_zentorch_sdpa_patched"):
        return True

    CPUAttentionBackendImpl._zentorch_orig_forward = CPUAttentionBackendImpl.forward
    CPUAttentionBackendImpl.forward = _forward_cpu_attn_zentorch
    CPUAttentionBackendImpl._zentorch_sdpa_patched = True
    logger.info(
        "[zentorch] Patched CPUAttentionBackendImpl.forward "
        "-> zentorch_sdpa (encoder-only / pooling)"
    )
    return True


def _apply_cpu_sdpa_patch() -> bool:
    """Opt out with ZENTORCH_SDPA=0 to keep vLLM's native encoder attention."""
    if os.environ.get("ZENTORCH_SDPA", "1") == "0":
        logger.debug(
            "[zentorch] CPU SDPA patch disabled (set ZENTORCH_SDPA=0 to disable)"
        )
        return False
    if not hasattr(torch.ops.zentorch, "zentorch_sdpa"):
        logger.debug(
            "[zentorch] zentorch_sdpa op unavailable; skipping CPU SDPA patch"
        )
        return False
    return patch_now_or_on_import(_CPU_ATTN_MODULE, _do_patch_cpu_sdpa)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
#
# Order matters only in that the import hooks must be armed before the target
# vLLM modules load; register() runs at process startup, well before that.
_PATCHES = (
    ("Gemma4HeteroConfig", _apply_gemma4_hetero_patch),
    ("TorchAO", _apply_torchao_patch),
    ("Int8MoE", _apply_int8_moe_patch_impl),
    ("Wna16MoE", _apply_wna16_moe_patch_impl),
    ("GptOssMoELoader", _apply_gptoss_loader_patch_impl),
    ("MixtralMoELoader", _apply_mixtral_loader_patch_impl),
    ("RMSNorm", _apply_rmsnorm_patch),
    ("FusedMoE", _apply_fused_moe_patch),
    ("CPUSdpa", _apply_cpu_sdpa_patch),
    ("Da8w4Kernel", _apply_da8w4_patch),
)

# Names of patches whose apply() returned True in this process (test hook).
APPLIED_PATCHES: list[str] = []


def _apply_all_patches() -> None:
    """Apply every Zen-specific patch, recording the ones that took effect."""
    APPLIED_PATCHES.clear()
    for name, apply_fn in _PATCHES:
        try:
            if apply_fn():
                APPLIED_PATCHES.append(name)
        except Exception:
            logger.warning("[zentorch] Patch %s FAILED", name, exc_info=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_INITIALIZED = False


def register() -> str | None:
    """Entry-point for vllm.platform_plugins and vllm.general_plugins.

    Called multiple times by vLLM (both entry points). Patches are applied once;
    the platform class path is always returned for supported runtimes.
    """
    global _INITIALIZED

    if "vllm" not in sys.modules:
        logger.warning("[zentorch] vllm not loaded")
        return None

    vllm_ver = get_vllm_version()

    if not is_supported_vllm(vllm_ver):
        logger.warning(
            "[zentorch] Unsupported vLLM %s. This zentorch plugin supports "
            "released vLLM %s-%s (PyTorch %d.%d+) only. Falling back to the "
            "stock CPU platform.",
            vllm_ver,
            VLLM_MIN_VERSION,
            VLLM_MAX_VERSION,
            TORCH_MIN_VERSION[0],
            TORCH_MIN_VERSION[1],
        )
        return None

    if not is_supported_torch():
        logger.warning(
            "[zentorch] Unsupported PyTorch %s; vLLM %s requires PyTorch "
            "%d.%d+. Falling back to the stock CPU platform.",
            torch.__version__,
            vllm_ver,
            TORCH_MIN_VERSION[0],
            TORCH_MIN_VERSION[1],
        )
        return None

    # Master hardware gate: every zentorch vLLM optimization requires AVX-512.
    # Without it, return None so vLLM falls back to the stock CpuPlatform.
    from zentorch._C import is_avx512_supported

    if not is_avx512_supported():
        logger.warning(
            "[zentorch] AVX-512 not detected; zentorch optimizations are "
            "disabled. Falling back to the stock vLLM CPU platform."
        )
        return None

    if not _INITIALIZED:
        _INITIALIZED = True
        logger.info("[zentorch] vLLM %s detected (PyTorch %s)", vllm_ver, torch.__version__)
        _apply_all_patches()
        logger.info("[zentorch] Applied patches: %s", APPLIED_PATCHES)

    return "zentorch.vllm._platform.ZenCPUPlatform"
