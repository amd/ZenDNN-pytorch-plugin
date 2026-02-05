# ****************************************************************************
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ****************************************************************************

"""zentorch CPU Platform for vLLM.

Supports:
- 0.12.0/0.13.0/0.14.0/0.14.1/0.15.0/0.15.1: CompilationMode, AttentionBackendEnum, native CPU attention
"""

from typing import TYPE_CHECKING

from zentorch.vllm.core import is_v13, is_v14, is_v14_1, is_v15, is_v15_1

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
        def _patch_profiler(cls):
            """Apply version-specific profiler patches.

            0.12: Patched in __init__.py register() (must run before worker creation)
            0.13/0.14/0.15+: Suppresses redundant cuda-time table output for CPU
            """
            if is_v13() or is_v14() or is_v14_1() or is_v15() or is_v15_1():
                cls._patch_profiler_v13_v14()
            # 0.12 is handled via _apply_profiler_patch_v12() in __init__.py register()

        @classmethod
        def _patch_profiler_v13_v14(cls):
            """Fix vLLM 0.13/0.14: Suppress redundant cuda-time table for CPU-only."""
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
