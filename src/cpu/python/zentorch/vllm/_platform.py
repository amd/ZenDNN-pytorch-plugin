# ****************************************************************************
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ****************************************************************************

"""zentorch CPU Platform for vLLM.

Targets vLLM 0.27.x / PyTorch 2.13. vLLM's stock CpuPlatform already configures
the CPU compile defaults (DYNAMO_TRACE_ONCE + inductor, dce/size_asserts/
nan_asserts/epilogue_fusion) and CPU-only profiler handling, so this subclass
only marks the platform as Zen and injects the zentorch inductor optimize pass.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import VllmConfig

_ZenCPUPlatformImpl = None


def _create_platform():
    """Create the ZenCPUPlatform class lazily (needs vLLM imported)."""
    global _ZenCPUPlatformImpl
    if _ZenCPUPlatformImpl is not None:
        return _ZenCPUPlatformImpl

    from vllm.platforms.cpu import CpuPlatform
    from vllm.logger import init_logger

    logger = init_logger(__name__)

    class ZenCPUPlatformImpl(CpuPlatform):
        """Out-of-tree CPU platform with zentorch optimizations (vLLM 0.27.x)."""

        device_name: str = "cpu"
        device_type: str = "cpu"

        def is_zen_cpu(self) -> bool:
            return True

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

            # Inject the zentorch optimize pass so torch.compile rewrites aten
            # ops into ZenDNN-backed zentorch ops.
            try:
                from zentorch._compile_backend import optimize_pass

                cc.inductor_compile_config["joint_custom_post_pass"] = optimize_pass
                logger.info("[zentorch] Injected optimize_pass")
            except ImportError:
                logger.warning("[zentorch] optimize_pass not available")

    _ZenCPUPlatformImpl = ZenCPUPlatformImpl
    return _ZenCPUPlatformImpl


def __getattr__(name):
    if name == "ZenCPUPlatform":
        return _create_platform()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
