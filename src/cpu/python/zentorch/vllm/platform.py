# ****************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ****************************************************************************

from __future__ import annotations

"""ZenTorch integration with vLLM – Platform definition.

This lightweight wrapper plugs ZenTorch (CPU) into vLLM using the standard
`Platform` extension mechanism.

The goal is to minimise global monkey-patching – anything ZenTorch-specific
should happen inside the worker / model-runner, not via `nn.Module.__call__`
patches that affect unrelated user code.
"""

from typing import Optional, Tuple

import torch
from zentorch._logging import get_logger
from vllm.platforms import Platform, PlatformEnum

logger = get_logger(__name__)


class ZenCPUPlatform(Platform):
    """Out-of-tree CPU platform backed by ZenTorch optimisations."""

    # Mark this platform as CPU so vLLM treats it like the in-tree CPU backend
    # By re-using the CPU enum we still
    # register as an out-of-tree plugin via the entry-point mechanism, but we let
    # the rest of vLLM assume a standard CPU device.
    _enum = PlatformEnum.CPU
    device_name: str = "zentorch-cpu"
    device_type: str = "cpu"
    dispatch_key: str = "CPU"  # torch dispatcher key
    ray_device_key: str = ""  # CPU resources are specified via normal CPUs
    simple_compile_backend: str = "zentorch"

    supported_quantization: list[str] = []

    # ---------------------------------------------------------------------
    # Mandatory `Platform` API implementation
    # ---------------------------------------------------------------------

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:  # noqa: D401 – API
        return f"Zen CPU #{device_id}"

    @classmethod
    def get_device_uuid(cls, device_id: int = 0) -> str:  # noqa: D401 – API
        # CPUs don't expose a universal UUID – use a deterministic placeholder
        return f"zen-cpu-{device_id}"

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:  # noqa: D401
        # Rough approximation – on CPU we fall back to virtual memory size.
        try:
            import psutil  # optional dependency

            vm = psutil.virtual_memory()
            return int(vm.total)
        except Exception:
            return 0

    @classmethod
    def is_async_output_supported(cls, enforce_eager: Optional[bool]) -> bool:
        # CPU tensors are always synchronised – async output makes no sense.
        return False

    @classmethod
    def inference_mode(cls):  # noqa: D401 – override for CPU
        # Standard PyTorch inference mode is fine.
        return torch.inference_mode()

    @classmethod
    def set_device(cls, device: torch.device):  # noqa: D401 – CPU NOP
        # Nothing to do – CPU context is global.
        return None

    @classmethod
    def empty_cache(cls):  # noqa: D401 – CPU NOP
        return None

    @classmethod
    def synchronize(cls):  # noqa: D401 – CPU NOP
        return None

    @classmethod
    def mem_get_info(cls) -> Tuple[int, int]:  # noqa: D401 – approximate
        # We expose (free, total) like CUDA does, but values are from psutil.
        try:
            import psutil

            vm = psutil.virtual_memory()
            return vm.available, vm.total
        except Exception:
            return 0, 0

    # ------------------------------------------------------------------
    # vLLM configuration hooks
    # ------------------------------------------------------------------

    @classmethod
    def check_and_update_config(cls, vllm_config) -> None:
        from vllm.config import CompilationLevel

        # Disable vLLM's own TorchDynamo compile passes – ZenTorch handles it.
        compilation_cfg = vllm_config.compilation_config
        compilation_cfg.level = CompilationLevel.NO_COMPILATION

        # Ensure worker_cls resolves to our custom worker when left as "auto".
        par_cfg = vllm_config.parallel_config
        if par_cfg and par_cfg.worker_cls == "auto":
            par_cfg.worker_cls = "zentorch.vllm.zentorch_worker.ZenWorker"

        # ------------------------------------------------------------------
        # KV-cache defaults – mirror the in-tree CPU platform
        # ------------------------------------------------------------------
        cache_cfg = vllm_config.cache_config
        if cache_cfg and cache_cfg.block_size is None:
            # we fall back to the same 16-token block size that vLLM uses for
            # plain PyTorch CPUs.
            cache_cfg.block_size = 16

        # Ensure cpu_kvcache_space_bytes is set – otherwise vLLM's CPUWorker
        # will error out when computing the number of available KV-cache blocks.
        if cache_cfg and getattr(cache_cfg, "cpu_kvcache_space_bytes", None) is None:
            import vllm.envs as envs
            from vllm.utils import GiB_bytes

            kv_cache_space_gib = envs.VLLM_CPU_KVCACHE_SPACE

            # The built-in CPU platform treats the *unset* (0) case as 4 GiB.
            # We replicate that behaviour.
            if kv_cache_space_gib <= 0:
                kv_cache_space_gib = 4

            cache_cfg.cpu_kvcache_space_bytes = kv_cache_space_gib * GiB_bytes  # type: ignore[attr-defined]

    # ------------------- Future Attention Integration -------------------

    @classmethod
    def get_attn_backend_cls(
        cls,
        selected_backend,  # noqa: ANN001, F841 – upstream type is private, unused
        head_size: int,  # noqa: F841 - unused
        dtype: torch.dtype,  # noqa: F841 - unused
        kv_cache_dtype: Optional[str],  # noqa: F841 - unused
        block_size: int,  # noqa: F841 - unused
        use_v1: bool,
        use_mla: bool,
    ) -> str:
        """Return the fully-qualified attention backend class path.

        ZenTorch currently only supports the Torch SDPA backend for CPU
        execution. This method always returns the path to
        `TorchSDPABackend`.

        Raises:
            NotImplementedError: If use_v1 or use_mla is True, as these
                features are not currently supported by ZenTorch-vLLM plugin.
        """
        # Check for unsupported features - these are not supported by ZenTorch-vLLM plugin.
        if use_v1:
            raise NotImplementedError(
                "ZenTorch-vLLM plugin does not currently support vLLM V1 "
                "attention backend (use_v1=True)"
            )

        if use_mla:
            raise NotImplementedError(
                "ZenTorch-vLLM plugin does not currently support Multi-head "
                "Latent Attention (use_mla=True)."
            )

        # ZenTorch on CPU currently relies solely on TorchSDPA backend.
        # Future ZenTorch-specific attention kernels could be added here.
        logger.info("[zentorch] Using Torch SDPA attention backend.")
        return "vllm.attention.backends.torch_sdpa.TorchSDPABackend"

    # ------------------------------------------------------------------
    # Memory/pinning helpers
    # ------------------------------------------------------------------

    @classmethod
    def is_pin_memory_available(cls) -> bool:  # noqa: D401
        """Pin-memory is not supported for pure-CPU execution.

        Returning *False* ensures vLLM does not request pinned buffers, which
        would otherwise trigger ``RuntimeError: Need to provide pin_memory
        allocator to use pin memory`` in PyTorch CPU builds.
        """
        logger.debug("[zentorch] Pin memory disabled on CPU platform.")
        return False
