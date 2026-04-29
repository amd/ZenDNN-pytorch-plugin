# ****************************************************************************
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ****************************************************************************

"""vLLM - zentorch integration using the plugin pattern.

Entry points:
- vllm.platform_plugins -> returns ZenCPUPlatform class path
- vllm.general_plugins  -> applies early patches

Patches:
- CompilationConfig repr (all versions)
- Import-hook GEMM dispatch patching (v15-v17)
  v18+ GEMM is handled natively via is_zen_cpu() in dispatch_cpu_unquantized_gemm.

Reference: https://blog.vllm.ai/2025/11/20/vllm-plugin-system.html
"""

from __future__ import annotations

import importlib.util
import sys
from typing import Optional
import torch

from zentorch._logging import get_logger
from zentorch.vllm.core import (
    vllm_version,
    vllm_version_range,
    manager,
    get_version_family,
    VLLM_MIN_VERSION,
    VLLM_MAX_VERSION,
    VLLM_V15,
    VLLM_V15_1,
    VLLM_V16,
    VLLM_V17,
    VLLM_V17_1,
    VLLM_V18,
    VLLM_V18_1,
    VLLM_V19,
    VLLM_V19_1,
    VLLM_V20,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# PyTorch mainline backports (fixes present in mainline but not in 2.10)
# ---------------------------------------------------------------------------

_FXGRAPHCACHE_PATCH_APPLIED = False

# ---------------------------------------------------------------------------
# Patch torchao Int8Tensor F.linear dispatch → zentorch_dynamic_qlinear
#
# torchao's Int8Tensor registers a __torch_function__ handler for F.linear
# that decomposes into choose_qparams → quantize → _int_mm → dequant.
# We re-register the handler so the traced graph contains a single
# zentorch_dynamic_qlinear node instead.
# ---------------------------------------------------------------------------

_DYNAMIC_QLINEAR_PATCH_APPLIED = False


def _register_zentorch_linear_dispatch() -> None:
    """Patch Int8Tensor's F.linear dispatch to call zentorch_dynamic_qlinear."""
    global _DYNAMIC_QLINEAR_PATCH_APPLIED

    if _DYNAMIC_QLINEAR_PATCH_APPLIED:
        return

    if importlib.util.find_spec("torchao") is None:
        logger.info(
            "[zentorch] torchao not installed, skipping Int8Tensor F.linear patch"
        )
        return

    try:
        from torchao.quantization.quantize_.workflows.int8.int8_tensor import (
            Int8Tensor,
        )

        implements = Int8Tensor.implements
        implements_torch_function = Int8Tensor.implements_torch_function

        @implements(torch.ops.aten.linear.default)
        @implements_torch_function(torch.nn.functional.linear)
        def _zentorch_int8_linear(func, types, args, kwargs):
            activation_tensor = args[0]
            weight_tensor = args[1]
            bias = args[2] if len(args) > 2 else None

            if (
                isinstance(weight_tensor, Int8Tensor)
                and weight_tensor.act_quant_kwargs is not None
            ):
                weight_int8 = weight_tensor.qdata
                weight_scales = weight_tensor.scale
                if weight_scales.dim() == 2 and weight_scales.shape[-1] == 1:
                    weight_scales = weight_scales.squeeze(-1)
                return torch.ops.zentorch.zentorch_dynamic_qlinear(
                    activation_tensor, weight_int8, weight_scales, bias,
                )

            return func(*args, **(kwargs or {}))

        _DYNAMIC_QLINEAR_PATCH_APPLIED = True
        logger.warning(
            "[zentorch] Patched Int8Tensor F.linear dispatch -> zentorch_dynamic_qlinear"
        )
    except Exception:
        logger.warning(
            "[zentorch] Int8Tensor F.linear patch FAILED", exc_info=True,
        )


def _apply_fxgraphcache_pickle_patch() -> bool:
    """Backport PyTorch mainline fix: add ValueError to FxGraphCachePickler.dumps().

    PyTorch mainline already catches ValueError in FxGraphCachePickler.dumps(),
    but PyTorch 2.10 does not. Without this, pickle fast mode (self.fast = True)
    crashes on cyclic object references instead of gracefully bypassing the cache.

    Returns:
        True if patch was applied, False if not needed or failed.
    """
    global _FXGRAPHCACHE_PATCH_APPLIED

    if _FXGRAPHCACHE_PATCH_APPLIED:
        return True

    # Only needed for PyTorch 2.10.0 which is missing the mainline fix.
    from torch.torch_version import TorchVersion

    if TorchVersion(torch.__version__) < (2, 10):
        logger.debug("[zentorch] FxGraphCache pickle patch only applies to PyTorch 2.10.0")
        return False

    import pickle

    try:
        from torch._inductor.codecache import FxGraphCachePickler

        original_dumps = FxGraphCachePickler.dumps

        # Check if already patched
        if hasattr(original_dumps, "_zentorch_patched"):
            _FXGRAPHCACHE_PATCH_APPLIED = True
            return True

        def patched_dumps(self, obj):
            """Mainline backport: catches ValueError from cyclic Logger refs."""
            try:
                self.dump(obj)
                return self._stream.getvalue()
            except (TypeError, AttributeError, pickle.PicklingError, ValueError) as e:
                from torch._inductor.codecache import BypassFxGraphCache
                import logging

                logging.getLogger("torch._inductor.codecache").warning(
                    "Failed to pickle cache key", exc_info=True
                )
                raise BypassFxGraphCache("Failed to pickle cache key") from e
            finally:
                self._stream.seek(0)
                self._stream.truncate(0)

        patched_dumps._zentorch_patched = True
        FxGraphCachePickler.dumps = patched_dumps

        _FXGRAPHCACHE_PATCH_APPLIED = True
        logger.info(
            "[zentorch] Backported mainline ValueError fix to FxGraphCachePickler.dumps"
        )
        return True

    except (ImportError, AttributeError):
        logger.debug("[zentorch] FxGraphCachePickler patch not applicable")
        return False


# ---------------------------------------------------------------------------
# PyTorch FakeTensorMode Patch (torch version aware)
# ---------------------------------------------------------------------------

_FAKETENSOR_PATCH_APPLIED = False


def _apply_faketensor_subclass_patch() -> bool:
    """Fix _check_for_subclass_arg to recognize Parameter subclasses.

    Uses issubclass(type(x), Parameter) instead of `type(x) is not Parameter`
    so vLLM's ModelWeightParameter is treated as a Parameter by FakeTensorMode.
    Needed for TORCHINDUCTOR_FREEZING=1.

    Returns True if patch was applied, False if not needed or failed.
    """
    global _FAKETENSOR_PATCH_APPLIED

    if _FAKETENSOR_PATCH_APPLIED:
        return True

    # Only apply for PyTorch 2.10+ where freezing with custom ops is used
    from torch.torch_version import TorchVersion

    if TorchVersion(torch.__version__) < (2, 10):
        logger.debug("[zentorch] FakeTensor patch not needed for PyTorch < 2.10")
        return False

    try:
        import torch._subclasses.fake_tensor as fake_tensor_module

        # Check if the function exists and needs patching
        if not hasattr(fake_tensor_module, "_check_for_subclass_arg"):
            logger.debug("[zentorch] _check_for_subclass_arg not found, skipping patch")
            return False

        original_fn = fake_tensor_module._check_for_subclass_arg

        # Check if already patched
        if hasattr(original_fn, "_zentorch_patched"):
            _FAKETENSOR_PATCH_APPLIED = True
            return True

        # Create fixed version
        def _check_for_subclass_arg_fixed(x: object) -> bool:
            """Fixed version: uses issubclass() for Parameter check.

            This allows Parameter subclasses (like ModelWeightParameter) to be
            handled by FakeTensorMode's registered fake implementations rather
            than returning NotImplemented.
            """
            from torch import Tensor
            from torch._subclasses.fake_tensor import FakeTensor

            return (
                not isinstance(x, FakeTensor)
                and isinstance(x, Tensor)
                and type(x) is not Tensor
                and not issubclass(type(x), torch.nn.Parameter)
            )

        _check_for_subclass_arg_fixed._zentorch_patched = True

        # Apply the patch
        fake_tensor_module._check_for_subclass_arg = _check_for_subclass_arg_fixed

        # Also patch _check_for_subclass which uses _check_for_subclass_arg
        # The function iterates over args and calls _check_for_subclass_arg
        # Since we patched the helper, the main function will use our fixed version

        _FAKETENSOR_PATCH_APPLIED = True
        logger.info(
            "[zentorch] Patched FakeTensorMode._check_for_subclass_arg for Parameter subclass support"
        )
        return True

    except Exception:
        logger.warning("[zentorch] Failed to patch FakeTensorMode", exc_info=True)
        return False


@vllm_version_range(min_ver=VLLM_MIN_VERSION, max_ver=VLLM_MAX_VERSION)
class TorchAOPatch:
    """Register TorchAO operations for Int4OpaqueTensor support."""

    @classmethod
    def apply(cls) -> bool:
        """
        Register torchao operations including:
        - Int4WeightOnlyOpaqueTensorConfig to ALLOWED_AO_MODULES
        - slice operation for Int4OpaqueTensor
        """
        if importlib.util.find_spec("torchao") is None:
            logger.info("[zentorch] TorchAO not installed, skipping Int4 registration")
            return False
        from .torchao_register import _register_int4_opaque_tensor_config, _register_int4_slice_op

        _register_int4_opaque_tensor_config()
        _register_int4_slice_op()
        logger.info("[zentorch] Registered TorchAO operations.")
        return True


# ---------------------------------------------------------------------------
# Patches (all versions)
# ---------------------------------------------------------------------------


@vllm_version_range(min_ver=VLLM_MIN_VERSION, max_ver=VLLM_MAX_VERSION)
class CompilationConfigReprPatch:
    """Fix CompilationConfig repr for pydantic serialization."""

    @classmethod
    def apply(cls) -> bool:
        try:
            from vllm.config import CompilationConfig
            from pydantic import TypeAdapter
            from vllm.config.compilation import PassConfig

            if hasattr(CompilationConfig.__repr__, "_zentorch_patched"):
                return True

            def patched_repr(self):
                exclude = {
                    "static_forward_context": True,
                    "enabled_custom_ops": True,
                    "disabled_custom_ops": True,
                    "compilation_time": True,
                    "bs_to_padded_graph_size": True,
                    "traced_files": True,
                    "inductor_compile_config": {
                        "post_grad_custom_post_pass": True,
                        "joint_custom_post_pass": True,
                        "joint_custom_pre_pass": True,
                    },
                }

                pass_config_exclude = {}
                try:
                    for attr, default_val in vars(PassConfig()).items():
                        if getattr(self.pass_config, attr) == default_val:
                            pass_config_exclude[attr] = True
                    if pass_config_exclude:
                        exclude["pass_config"] = pass_config_exclude
                except Exception:
                    pass

                try:
                    return (
                        TypeAdapter(CompilationConfig)
                        .dump_json(self, exclude=exclude, exclude_unset=True)
                        .decode()
                    )
                except Exception:
                    mode = getattr(self, "mode", getattr(self, "level", "?"))
                    return f"CompilationConfig(mode={mode}, backend={self.backend!r})"

            patched_repr._zentorch_patched = True
            CompilationConfig.__repr__ = patched_repr
            CompilationConfig.__str__ = patched_repr
            logger.info("[zentorch] Patched CompilationConfig repr")
            return True
        except ImportError:
            return False


@vllm_version(VLLM_V15, VLLM_V15_1, VLLM_V16, VLLM_V17, VLLM_V17_1, VLLM_V18, VLLM_V18_1, VLLM_V19, VLLM_V19_1, VLLM_V20)
class CPUProfilerPatch:
    """Stub: Actual patching happens in platform.py check_and_update_config.

    CPUWorker properly uses TorchProfilerWrapper natively, so no additional
    patching is required here. The profiler wrapper._stop patch in platform.py
    suppresses meaningless cuda-time table output for CPU-only.
    """

    @classmethod
    def apply(cls) -> bool:
        return True


# ---------------------------------------------------------------------------
# RMSNorm CPU Forward Patch (deferred via import hook)
# ---------------------------------------------------------------------------

_RMSNORM_HOOK_INSTALLED = False


def _do_patch_rmsnorm():
    """Patch RMSNorm.forward to use vLLM's optimized C++ kernels on CPU."""
    try:
        from vllm.model_executor.layers.layernorm import (
            RMSNorm,
        )
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
        # This custom op causes accuracy issue with Qwen models
        # return rms_norm(x, self.weight.data, self.variance_epsilon)
        return self.forward_native(x, residual)

    RMSNorm.forward = patched_forward
    RMSNorm._zentorch_rmsnorm_patched = True
    logger.info("[zentorch] Patched RMSNorm.forward to use optimized C++ kernels on CPU")
    return True


class _RMSNormImportHook:
    """Post-import hook: patch RMSNorm after layernorm module loads.

    Python 3.12 only calls find_spec (silently skips find_module).
    We cannot import the module inside find_spec and return None --
    that causes Python to re-execute the module via the next finder,
    hitting "Duplicate op name" assertions.

    Instead we: remove ourselves, find the *real* spec via the
    remaining finders, wrap its loader.exec_module to append our
    patch, and return the wrapped spec.  The module loads exactly
    once through normal means.
    """

    _LAYERNORM_MODULE = "vllm.model_executor.layers.layernorm"

    def find_spec(self, fullname, path, target=None):
        if fullname != self._LAYERNORM_MODULE:
            return None
        if self in sys.meta_path:
            sys.meta_path.remove(self)

        import importlib.util

        spec = importlib.util.find_spec(fullname)
        if spec is None or spec.loader is None:
            return None

        original_exec = spec.loader.exec_module

        def _exec_then_patch(module):
            original_exec(module)
            _do_patch_rmsnorm()

        spec.loader.exec_module = _exec_then_patch
        return spec


def _apply_rmsnorm_patch() -> bool:
    """Install RMSNorm forward patch, deferred if the module isn't loaded yet.

    Returns True if patch was applied or hook installed, False otherwise.
    """
    global _RMSNORM_HOOK_INSTALLED

    if _RMSNORM_HOOK_INSTALLED:
        return True

    if "vllm.model_executor.layers.layernorm" in sys.modules:
        result = _do_patch_rmsnorm()
    else:
        sys.meta_path.insert(0, _RMSNormImportHook())
        logger.debug("[zentorch] Installed RMSNorm import hook")
        result = True

    _RMSNORM_HOOK_INSTALLED = True
    return result


@vllm_version_range(min_ver=VLLM_V15, max_ver=VLLM_MAX_VERSION)
class RMSNormPatch:
    """Patch RMSNorm.forward to use vLLM's optimized C++ kernels on CPU."""

    @classmethod
    def apply(cls) -> bool:
        return _apply_rmsnorm_patch()


# ---------------------------------------------------------------------------
# vLLM 0.15-0.17 GEMM Dispatch Hook
# ---------------------------------------------------------------------------
#
# On vLLM 0.15-0.17 we patch dispatch_cpu_unquantized_gemm to route
# through torch.nn.functional.linear. optimize_pass then rewrites
# aten.linear -> zentorch_linear_unary in the inductor IR.
#
# The patched dispatch preserves remove_weight semantics by capturing
# the original weights before optionally emptying layer.weight.
#
# Not needed for v18+: dispatch_cpu_unquantized_gemm natively checks
# is_zen_cpu() and routes to zentorch_linear_unary.

_PRE_V18_DISPATCH_FAMILIES = {"v15", "v15_1", "v16", "v17"}
_PRE_V18_DISPATCH_HOOKS_INSTALLED = False


def _do_patch_pre_v18_gemm_dispatch():
    """Patch dispatch_cpu_unquantized_gemm to use F.linear on v15-v17."""
    import vllm.model_executor.layers.utils as utils

    if hasattr(utils.dispatch_cpu_unquantized_gemm, "_zentorch_patched"):
        return

    def _patched(layer, remove_weight):
        if layer.weight.is_meta:
            layer.cpu_linear = torch.nn.functional.linear
            return

        weights_copy = layer.weight.detach()
        layer.cpu_linear = (
            lambda x, weight, bias: torch.nn.functional.linear(x, weights_copy, bias)
        )
        if remove_weight:
            layer.weight = torch.nn.Parameter(torch.empty(0), requires_grad=False)
        return

    _patched._zentorch_patched = True
    utils.dispatch_cpu_unquantized_gemm = _patched
    logger.info(
        "[zentorch] Patched dispatch_cpu_unquantized_gemm (v15-v17, F.linear bypass)"
    )


_PRE_V18_HOOK_PATCHES = {
    "vllm.model_executor.layers.utils": _do_patch_pre_v18_gemm_dispatch,
}


class _PreV18DispatchImportHook:
    """Intercept imports to patch vLLM 0.15-0.17 dispatch functions.

    Implements PEP 302 (find_module/load_module) for Python <= 3.11 and
    PEP 451 (find_spec wrapping the real loader) for Python >= 3.12.
    """

    def __init__(self, pending):
        self._pending = set(pending)
        self._orig_loaders = {}

    def _maybe_readd(self):
        if self._pending and self not in sys.meta_path:
            sys.meta_path.insert(0, self)

    # PEP 302 (Python <= 3.11)
    def find_module(self, fullname, path=None):
        if fullname in self._pending:
            return self
        return None

    def load_module(self, fullname):
        self._pending.discard(fullname)
        if self in sys.meta_path:
            sys.meta_path.remove(self)

        import importlib
        module = importlib.import_module(fullname)

        _PRE_V18_HOOK_PATCHES[fullname]()

        self._maybe_readd()
        return module

    # PEP 451 (Python >= 3.12): wrap the real loader instead of re-importing
    def find_spec(self, fullname, path, target=None):
        if fullname not in self._pending:
            return None
        self._pending.discard(fullname)
        if self in sys.meta_path:
            sys.meta_path.remove(self)

        import importlib.util
        real_spec = importlib.util.find_spec(fullname)
        if real_spec is None:
            self._maybe_readd()
            return None

        self._orig_loaders[fullname] = real_spec.loader
        real_spec.loader = self
        self._maybe_readd()
        return real_spec

    def create_module(self, spec):
        orig = self._orig_loaders.get(spec.name)
        if orig and hasattr(orig, "create_module"):
            return orig.create_module(spec)
        return None

    def exec_module(self, module):
        orig = self._orig_loaders.pop(module.__name__, None)
        if orig:
            orig.exec_module(module)
        _PRE_V18_HOOK_PATCHES[module.__name__]()


def _install_pre_v18_dispatch_hooks():
    """Install import hooks for vLLM 0.15-0.17 GEMM dispatch patching."""
    global _PRE_V18_DISPATCH_HOOKS_INSTALLED
    if _PRE_V18_DISPATCH_HOOKS_INSTALLED:
        return

    pending = []
    for modname, patcher in _PRE_V18_HOOK_PATCHES.items():
        if modname in sys.modules:
            patcher()
        else:
            pending.append(modname)

    if pending:
        sys.meta_path.insert(0, _PreV18DispatchImportHook(pending))
        logger.debug("[zentorch] Installed dispatch import hooks for: %s", pending)

    _PRE_V18_DISPATCH_HOOKS_INSTALLED = True


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

_REGISTERED = False


def _register_patches():
    """Register all patches with the manager (only once)."""
    global _REGISTERED
    if _REGISTERED:
        return

    manager.register("CompilationConfigRepr", CompilationConfigReprPatch)
    manager.register("CPUProfiler", CPUProfilerPatch)
    manager.register("TorchAO", TorchAOPatch)
    manager.register("RMSNorm", RMSNormPatch)

    _REGISTERED = True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_INITIALIZED = False


def register() -> Optional[str]:
    """Entry-point for vllm.platform_plugins and vllm.general_plugins.

    This is called multiple times by vLLM (both entry points). We only
    apply patches once but always return the platform class path.
    """
    global _INITIALIZED

    if "vllm" not in sys.modules:
        logger.warning("[zentorch] vllm not loaded")
        return None

    vllm_ver = getattr(sys.modules["vllm"], "__version__", None)
    family = get_version_family()

    if family is None:
        logger.warning(
            "[zentorch] Unsupported vLLM %s. Supports: %s through %s",
            vllm_ver,
            VLLM_MIN_VERSION,
            VLLM_MAX_VERSION,
        )
        return None

    if not _INITIALIZED:
        _INITIALIZED = True

        logger.info("[zentorch] vLLM %s detected (family: %s)", vllm_ver, family)

        # CRITICAL: Apply PyTorch mainline backports FIRST
        # These fixes exist in mainline but are missing from PyTorch 2.10.
        # They must run before any torch.compile invocation.
        # 1. isinstance() fix for Parameter subclasses in FakeTensorMode
        _apply_faketensor_subclass_patch()
        # 2. ValueError catch for cyclic Logger refs in FxGraphCachePickler
        _apply_fxgraphcache_pickle_patch()
        # 3. Register torchao int8 dynamic quantized linear → zentorch_dynamic_qlinear
        _register_zentorch_linear_dispatch()
        # Register and apply all patches (decorators handle version filtering)
        _register_patches()
        manager.apply_all()

        if family in _PRE_V18_DISPATCH_FAMILIES:
            _install_pre_v18_dispatch_hooks()

        logger.info("[zentorch] Applied patches: %s", manager.applied)

    return "zentorch.vllm.platform.ZenCPUPlatform"
