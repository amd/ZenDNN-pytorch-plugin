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
        def import_ir_kernels(cls) -> None:
            """Register zentorch IR providers alongside the built-in kernels."""
            super().import_ir_kernels()
            from zentorch.vllm._ir_rms_norm import register_zentorch_ir_norm_impls

            register_zentorch_ir_norm_impls()

        @classmethod
        def get_default_ir_op_priority(cls, vllm_config: "VllmConfig"):
            """Prefer registered zentorch IR providers over native.

            Only advertise ``zentorch`` for ops that actually got a provider --
            ``IrOp.set_default`` asserts each entry is a registered impl, and
            ``register_zentorch_ir_norm_impls`` returns False on older vLLM
            without ``vllm.ir`` -- so this stays safe and falls back to native.
            """
            from vllm.config.kernel import IrOpPriorityConfig

            from zentorch.vllm._ir_rms_norm import register_zentorch_ir_norm_impls

            priorities: dict[str, list[str]] = {}
            if register_zentorch_ir_norm_impls():
                from vllm import ir

                # Only fused_add_rms_norm has a zentorch provider (non-residual
                # rms_norm stays on native); advertise it iff it registered.
                op = getattr(ir.ops, "fused_add_rms_norm", None)
                if op is not None and "zentorch" in op.impls:
                    priorities["fused_add_rms_norm"] = ["zentorch", "native"]

            return IrOpPriorityConfig.with_default(["native"], **priorities)

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
