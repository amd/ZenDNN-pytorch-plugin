# ****************************************************************************
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ****************************************************************************

"""zentorch CPU Platform for vLLM.

Supports:
- 0.11.x: CompilationLevel, _Backend, PagedAttention patching
- 0.12.x/0.13.x: CompilationMode, AttentionBackendEnum, native CPU attention
"""

from typing import TYPE_CHECKING

from zentorch.vllm.core import is_v11, is_v13

if TYPE_CHECKING:
    from vllm.config import VllmConfig

_ZenCPUPlatformImpl = None


def _create_platform():
    """Create ZenCPUPlatform class lazily."""
    global _ZenCPUPlatformImpl
    if _ZenCPUPlatformImpl is not None:
        return _ZenCPUPlatformImpl

    from vllm.platforms.cpu import CpuPlatform
    from vllm.logger import init_logger

    logger = init_logger(__name__)

    class ZenCPUPlatformImpl(CpuPlatform):
        """Out-of-tree CPU platform with zentorch optimizations."""

        device_name: str = "cpu"
        device_type: str = "cpu"

        @classmethod
        def check_and_update_config(cls, vllm_config: "VllmConfig") -> None:
            super().check_and_update_config(vllm_config)

            cc = vllm_config.compilation_config

            if is_v11():
                from vllm.config import CompilationLevel

                cc.level = CompilationLevel.DYNAMO_ONCE
                cc.backend = "inductor"
                cc.use_inductor = True
                cc.custom_ops = ["none"]

            cc.inductor_compile_config.update(
                {
                    "dce": True,
                    "size_asserts": False,
                    "nan_asserts": False,
                    "epilogue_fusion": True,
                }
            )

            # Inject zentorch optimize pass
            try:
                from zentorch._compile_backend import optimize_pass

                cc.inductor_compile_config["joint_custom_post_pass"] = optimize_pass
                logger.info("[zentorch] Injected optimize_pass")
            except ImportError:
                logger.warning("[zentorch] optimize_pass not available")

            # Apply profiler patches (version-specific)
            cls._patch_profiler()

        @classmethod
        def get_attn_backend_cls(cls, selected_backend, *args, **kwargs):
            """Version-aware attention backend selection.

            0.11: (selected_backend, head_size, dtype, ..., use_v1=True)
            0.12: (selected_backend, head_size, dtype, ..., attn_type=None)
            0.13: (selected_backend, attn_selector_config)
            """
            if is_v11():
                return cls._attn_backend_v11(selected_backend, *args, **kwargs)
            # 0.12/0.13: delegate to parent CpuPlatform
            return super(ZenCPUPlatformImpl, cls).get_attn_backend_cls(
                selected_backend, *args, **kwargs
            )

        @classmethod
        def _attn_backend_v11(
            cls,
            selected_backend,
            head_size=None,
            dtype=None,
            kv_cache_dtype=None,
            block_size=None,
            use_v1=True,
            use_mla=False,
            has_sink=False,
            use_sparse=False,
        ):
            """0.11: Patch and return TorchSDPABackend."""
            import vllm.v1.attention.backends.cpu_attn as cpu_attn
            from zentorch.vllm.attention import PagedAttention

            cpu_attn._get_paged_attn_impl = lambda: PagedAttention
            logger.info("[zentorch] Applied PagedAttention in get_attn_backend_cls")

            if use_mla:
                raise NotImplementedError("MLA not supported")
            if use_sparse:
                raise NotImplementedError("Sparse attention not supported")

            if not use_v1:
                raise ValueError("zentorch requires V1 engine")
            return "vllm.v1.attention.backends.cpu_attn.TorchSDPABackend"

        @classmethod
        def _patch_profiler(cls):
            """Apply version-specific profiler patches.

            0.11: Patches Worker.__init__ and Worker.profile for CPU-only profiling
            0.12: Patched in __init__.py register() (must run before worker creation)
            0.13: Suppresses redundant cuda-time table output for CPU (single table like 0.14)
            """
            if is_v11():
                cls._patch_profiler_v11()
            elif is_v13():
                cls._patch_profiler_v13()
            # 0.12 is handled via _apply_profiler_patch_v12() in __init__.py register()

        @classmethod
        def _patch_profiler_v11(cls):
            """Fix vLLM 0.11: Patch Worker profiler for CPU-only operation."""
            import torch

            try:
                import vllm.v1.worker.gpu_worker as worker_module
                import vllm.envs as envs
            except ImportError:
                logger.debug(
                    "[zentorch] Worker module not available for profiler patch"
                )
                return

            if hasattr(worker_module.Worker, "_zentorch_profiler_patched"):
                return

            OriginalWorker = worker_module.Worker
            original_init = OriginalWorker.__init__

            def patched_init(self, *args, **kwargs):
                original_init(self, *args, **kwargs)
                if self.profiler is not None:
                    self.profiler = torch.profiler.profile(
                        activities=[torch.profiler.ProfilerActivity.CPU],
                        record_shapes=envs.VLLM_TORCH_PROFILER_RECORD_SHAPES,
                        profile_memory=envs.VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY,
                        with_stack=envs.VLLM_TORCH_PROFILER_WITH_STACK,
                        with_flops=envs.VLLM_TORCH_PROFILER_WITH_FLOPS,
                    )

            def patched_profile(self, is_start: bool = True):
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
            logger.info("[zentorch] Patched Worker profiler for CPU-only (0.11)")

        @classmethod
        def _patch_profiler_v13(cls):
            """Fix vLLM 0.13: Suppress redundant cuda-time table for CPU-only."""
            try:
                from vllm.profiler import wrapper as wrapper_module
            except ImportError:
                logger.debug("[zentorch] profiler.wrapper not available")
                return

            TorchProfilerWrapper = wrapper_module.TorchProfilerWrapper
            if hasattr(TorchProfilerWrapper, "_zentorch_patched"):
                return

            def patched_stop(self):
                self.profiler.stop()
                profiler_config = self.profiler_config
                rank = self.local_rank

                # Only dump cuda time table if NOT cpu-only
                if (
                    profiler_config.torch_profiler_dump_cuda_time_total
                    and not self.dump_cpu_time_total
                ):
                    profiler_dir = profiler_config.torch_profiler_dir
                    profiler_out_file = f"{profiler_dir}/profiler_out_{rank}.txt"
                    table = self.profiler.key_averages().table(
                        sort_by="self_cuda_time_total"
                    )
                    with open(profiler_out_file, "w") as f:
                        print(table, file=f)
                    if rank == 0:
                        print(table)

                # CPU time table for CPU-only activities
                if self.dump_cpu_time_total and rank == 0:
                    wrapper_module.logger.info(
                        self.profiler.key_averages().table(
                            sort_by="self_cpu_time_total", row_limit=50
                        )
                    )

            TorchProfilerWrapper._stop = patched_stop
            TorchProfilerWrapper._zentorch_patched = True
            logger.info("[zentorch] Patched TorchProfilerWrapper._stop (0.13)")

    _ZenCPUPlatformImpl = ZenCPUPlatformImpl
    return _ZenCPUPlatformImpl


def __getattr__(name):
    if name == "ZenCPUPlatform":
        return _create_platform()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
