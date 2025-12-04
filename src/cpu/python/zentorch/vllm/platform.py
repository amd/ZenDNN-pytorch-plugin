# ****************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ****************************************************************************

"""zentorch integration with vLLM - Platform definition.

This lightweight wrapper plugs zentorch (CPU) into vLLM using the standard
`Platform` extension mechanism.

The goal is to minimise global monkey-patching - anything zentorch-specific
should happen inside the worker / model-runner, not via `nn.Module.__call__`
patches that affect unrelated user code.

Note: CompilationConfig.__repr__ patching is handled in __init__.py via the
general_plugins entry point, which runs early in all vLLM processes.

Tested with vLLM version: 0.11.0
"""

from typing import TYPE_CHECKING, Optional

import torch

from vllm.config import CompilationLevel
from vllm.logger import init_logger
from vllm.platforms.cpu import CpuPlatform
from vllm.platforms.interface import _Backend

if TYPE_CHECKING:
    from vllm.config import VllmConfig
else:
    VllmConfig = None

logger = init_logger(__name__)


class ZenCPUPlatform(CpuPlatform):
    """Out-of-tree CPU platform backed by zentorch optimisations."""

    # Explicitly set to "cpu" to ensure compatibility with vLLM's
    # torch.device() calls and operational checks
    device_name: str = "cpu"
    device_type: str = "cpu"

    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        # First apply standard CPU platform configuration
        super().check_and_update_config(vllm_config)

        compilation_config = vllm_config.compilation_config

        # Enable torch.compile with inductor backend
        compilation_config.level = CompilationLevel.DYNAMO_ONCE
        compilation_config.backend = "inductor"
        compilation_config.use_inductor = True
        compilation_config.inductor_compile_config.update(
            {
                "dce": True,  # Dead code elimination
                "size_asserts": False,  # Skip size assertions for perf
                "nan_asserts": False,  # Skip NaN checks for perf
                "epilogue_fusion": True,  # Fuse epilogue operations
            }
        )

        compilation_config.custom_ops = ["none"]

        # Inject zentorch optimization pass
        from zentorch._compile_backend import optimize_pass
        compilation_config.inductor_compile_config["joint_custom_post_pass"] = (
            optimize_pass
        )

        # Apply CPU profiler patch (deferred to avoid circular imports)
        cls._patch_cpu_profiler()

    @classmethod
    def _patch_cpu_profiler(cls):
        """
        Patch vLLM's Worker profiler for CPU-only profiling.

        This patch is applied during platform initialization to avoid
        circular imports that occur during early plugin loading.

        Raises:
            RuntimeError: If patching fails.
        """
        import vllm.v1.worker.gpu_worker as worker_module
        import vllm.envs as envs

        # Check if already patched
        if hasattr(worker_module.Worker, "_zentorch_profiler_patched"):
            return

        OriginalWorker = worker_module.Worker
        original_init = OriginalWorker.__init__

        def patched_init(self, *args, **kwargs):
            """Patched __init__ that sets up CPU-only profiler."""
            original_init(self, *args, **kwargs)

            # Re-configure profiler for CPU-only if it was created
            if self.profiler is not None:
                self.profiler = torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                    ],
                    record_shapes=envs.VLLM_TORCH_PROFILER_RECORD_SHAPES,
                    profile_memory=envs.VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY,
                    with_stack=envs.VLLM_TORCH_PROFILER_WITH_STACK,
                    with_flops=envs.VLLM_TORCH_PROFILER_WITH_FLOPS,
                )

        def patched_profile(self, is_start: bool = True):
            """Patched profile method that prints CPU stats."""
            if self.profiler is None:
                raise RuntimeError("Profiler is not enabled.")
            if is_start:
                self.profiler.start()
            else:
                self.profiler.stop()
                print(
                    self.profiler.key_averages().table(
                        sort_by="self_cpu_time_total"
                    )
                )

        OriginalWorker.__init__ = patched_init
        OriginalWorker.profile = patched_profile
        OriginalWorker._zentorch_profiler_patched = True

        logger.info("[zentorch] Patched Worker profiler for CPU-only profiling")

    @classmethod
    def get_attn_backend_cls(
        cls,
        selected_backend: _Backend,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: Optional[str],
        block_size: int,
        use_v1: bool,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
    ) -> str:

        # Monkey-patch vLLM's _get_paged_attn_impl to return zentorch PagedAttention
        import vllm.v1.attention.backends.cpu_attn as cpu_attn_module
        from zentorch.vllm.attention import PagedAttention

        def _get_zentorch_paged_attn_impl():
            return PagedAttention

        cpu_attn_module._get_paged_attn_impl = _get_zentorch_paged_attn_impl

        logger.info("Monkey-patched vLLM PagedAttention with zentorch implementation")

        if use_mla:
            raise NotImplementedError(
                "Multi-head Latent Attention is not supported by "
                "vLLM-zentorch plugin."
            )

        if use_sparse:
            raise NotImplementedError(
                "Sparse Attention is not supported by vLLM-zentorch plugin."
            )

        if selected_backend and selected_backend != _Backend.TORCH_SDPA:
            logger.info(
                "Cannot use %s backend on zentorch CPU, " "falling back to Torch SDPA.",
                selected_backend,
            )

        assert use_v1, "vLLM-zentorch only supports V1 backend."
        logger.info("Using vLLM-zentorch V1 backend.")
        return "vllm.v1.attention.backends.cpu_attn.TorchSDPABackend"
