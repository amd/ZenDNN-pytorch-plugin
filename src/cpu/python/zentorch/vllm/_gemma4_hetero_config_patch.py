# ****************************************************************************
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ****************************************************************************

"""Enable gemma-4 (heterogeneous per-layer configs) under vLLM 0.27.

transformers >= 5.15 marks gemma-4's ``head_dim`` per-layer, so global reads
(``config.head_dim``) raise ``AmbiguousGlobalPerLayerAttributeError``. vLLM reads
it globally in a few places while loading gemma-4, so the model fails to load.

``head_dim`` is in fact uniform (the heterogeneity is in ``layer_types``), so the
fix arms transformers' ``allow_global_per_layer_attribute_access`` escape hatch on
each heterogeneous config, making global reads return that uniform value. The
larger full-attention dimension lives in a separate ``global_head_dim`` attribute
(rebuilt by ``_restore_global_head_dim`` if transformers drops it).

``get_config`` is imported by name into several modules, each capturing its own
binding, so the hook wraps it on the source and every consumer in
``_TARGET_MODULES``.
"""

from __future__ import annotations

import sys

from zentorch.vllm._import_hook import patch_now_or_on_import
from zentorch._logging import get_logger

logger = get_logger(__name__)

# Every module that binds ``get_config`` by name and can load a gemma-4 config.
# The source module is patched too, so any later importer picks up the wrapped
# version. ``_enable_global_per_layer_access`` is idempotent, so the overlap
# between the source and its re-exports is harmless.
_TARGET_MODULES = (
    "vllm.transformers_utils.config",
    "vllm.config.model",
    "vllm.tokenizers.registry",
)


def _restore_global_head_dim(node) -> None:
    """Rebuild ``global_head_dim`` from ``per_layer_config`` if the scalar absent.

    Released configs carry ``global_head_dim`` directly, so this usually no-ops.
    ``per_layer_config`` shape is not guaranteed (list, view, or dict), so every
    lookup is defensive: an unexpected shape skips reconstruction rather than
    raising into vLLM's model loader.
    """
    if getattr(node, "global_head_dim", None) is not None:
        return
    layer_types = getattr(node, "layer_types", None)
    per_layer = getattr(node, "per_layer_config", None)
    if not layer_types or per_layer is None:
        return
    full_dims = []
    for i, layer_type in enumerate(layer_types):
        if layer_type != "full_attention":
            continue
        try:
            layer_cfg = per_layer[i]
        except (IndexError, KeyError, TypeError):
            continue
        full_dims.append(getattr(layer_cfg, "head_dim", 0) or 0)
    if full_dims:
        node.global_head_dim = max(full_dims)


def _push_config_children(value, stack) -> None:
    """Push config-like children of ``value`` onto the walk stack.

    Handles a direct config attribute as well as configs held inside a list,
    tuple, or dict (``per_layer_config`` is itself a collection). A config is any
    object exposing the transformers ``is_heterogeneous`` marker.
    """
    if hasattr(value, "is_heterogeneous"):
        stack.append(value)
    elif isinstance(value, (list, tuple)):
        for item in value:
            if hasattr(item, "is_heterogeneous"):
                stack.append(item)
    elif isinstance(value, dict):
        for item in value.values():
            if hasattr(item, "is_heterogeneous"):
                stack.append(item)


def _enable_global_per_layer_access(cfg) -> None:
    """Arm the escape hatch on every heterogeneous config in the tree.

    Walks the tree via each node's raw ``__dict__`` (never ``getattr``, which
    could itself trip a per-layer access), following direct attributes and
    list/tuple/dict collections. On each heterogeneous node it sets
    ``allow_global_per_layer_attribute_access`` and rebuilds ``global_head_dim``.
    """
    seen: set[int] = set()
    stack = [cfg]
    while stack:
        node = stack.pop()
        if node is None or id(node) in seen:
            continue
        seen.add(id(node))
        if getattr(node, "is_heterogeneous", False):
            node.allow_global_per_layer_attribute_access = True
            _restore_global_head_dim(node)
        # Objects with __slots__ (or C extension types) have no __dict__; nothing
        # to descend into, so skip rather than raise.
        node_dict = getattr(node, "__dict__", None)
        if not node_dict:
            continue
        for value in node_dict.values():
            _push_config_children(value, stack)


def _patch_module_get_config(module) -> bool:
    """Wrap one module's ``get_config`` binding to arm the escape hatch.

    Idempotent per module (guarded by ``_zentorch_gemma4_hetero_patched``), so it
    is safe to call repeatedly as each target module loads.
    """
    if module is None or not hasattr(module, "get_config"):
        return False
    if getattr(module, "_zentorch_gemma4_hetero_patched", False):
        return True

    original_get_config = module.get_config

    def patched_get_config(*args, **kwargs):
        cfg = original_get_config(*args, **kwargs)
        # Fail OPEN: this is a speculative compatibility shim for one model
        # family, wrapping get_config for *every* model. An unexpected config
        # shape must never turn into a model-load failure -- losing gemma-4
        # support is far better than breaking an unrelated model.
        try:
            _enable_global_per_layer_access(cfg)
        except Exception:
            logger.warning(
                "[zentorch] gemma-4 heterogeneous-config hook skipped "
                "(unexpected config shape)",
                exc_info=True,
            )
        return cfg

    module.get_config = patched_get_config
    module._zentorch_gemma4_hetero_patched = True
    logger.info(
        "[zentorch] Patched %s.get_config (gemma-4 heterogeneous head_dim)",
        module.__name__,
    )
    return True


def _do_patch_gemma4_hetero() -> bool:
    """Patch ``get_config`` on every already-loaded target module."""
    patched_any = False
    for name in _TARGET_MODULES:
        if _patch_module_get_config(sys.modules.get(name)):
            patched_any = True
    return patched_any


def _apply_gemma4_hetero_patch() -> bool:
    # A consumer may load after the source module, so each target re-points every
    # binding currently visible; _patch_module_get_config is idempotent per module.
    for name in _TARGET_MODULES:
        patch_now_or_on_import(name, _do_patch_gemma4_hetero)
    return True
