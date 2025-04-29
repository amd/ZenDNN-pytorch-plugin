# ****************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ****************************************************************************

from __future__ import annotations

import builtins
import torch
from vllm.logger import init_logger
from vllm.worker.cpu_model_runner import CPUModelRunner
from vllm.model_executor.models.mixtral import MixtralMoE

logger = init_logger(__name__)


class ZenCPUModelRunner(CPUModelRunner):
    """vLLM CPU runner that JIT-compiles the HF model with ZenTorch."""

    def load_model(self) -> None:
        # First, load the model using the standard vLLM process.
        super().load_model()

        model = self.model
        try:
            # Safeguard for newer torch versions.
            if hasattr(torch, "compiler") and hasattr(torch.compiler, "allow_in_graph"):
                torch.compiler.allow_in_graph(builtins.__import__)

            # Disable MixtralMoE.forward
            MixtralMoE.forward = torch.compiler.disable(MixtralMoE.forward)

            # Compile the model's forward pass using the ZenTorch backend.
            logger.info("[zentorch] compiling model.forward with ZenTorch backend")
            model.forward = torch.compile(model.forward, backend="zentorch")
            logger.info("[zentorch] compilation finished")
        except Exception:
            # Fall back to eager execution if compilation fails.
            logger.exception("[zentorch] torch.compile failed – running eager")
