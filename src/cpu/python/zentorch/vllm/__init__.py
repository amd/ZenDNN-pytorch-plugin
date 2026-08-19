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
                    if weight_scales.dtype not in (torch.float32, torch.bfloat16, torch.float16):
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
# Registration
# ---------------------------------------------------------------------------
#
# Order matters only in that the import hooks must be armed before the target
# vLLM modules load; register() runs at process startup, well before that.
_PATCHES = (
    ("Gemma4HeteroConfig", _apply_gemma4_hetero_patch),
    ("TorchAO", _apply_torchao_patch),
    ("Int8MoE", _apply_int8_moe_patch_impl),
    ("GptOssMoELoader", _apply_gptoss_loader_patch_impl),
    ("MixtralMoELoader", _apply_mixtral_loader_patch_impl),
    ("RMSNorm", _apply_rmsnorm_patch),
    ("FusedMoE", _apply_fused_moe_patch),
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
