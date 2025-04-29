# ****************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ****************************************************************************

from __future__ import annotations

"""ZenTorch CPU worker – thin wrapper around vLLM *CPUWorker*.

Key responsibilities:
1.  Ensure the base runner is wrapped with a compile helper that performs
    `torch.compile(backend="zentorch")` exactly once at load-time.
2.  Otherwise keep the upstream behaviour unchanged – KV-cache handling,
    distributed initialisation, etc. remain untouched and inside the
    base class.

We achieve this by:

* letting ``CPUWorker`` construct its normal runner and immediately
  wrapping it with :class:`_CompileRunnerWrapper` (transparent proxy that
  adds compilation).

No further monkey-patching is required – the inherited ``load_model`` method
now simply forwards to the new runner.
"""

from vllm.worker.cpu_worker import CPUWorker
from zentorch.vllm.runner import ZenCPUModelRunner

from zentorch._logging import get_logger

logger = get_logger(__name__)


class ZenWorker(CPUWorker):
    """CPU worker that transparently compiles the model via TorchDynamo."""

    # We only need to intercept the constructor so that the *upstream*
    # ``CPUWorker`` builds our custom runner class.  All other behaviour –
    # KV-cache management, distributed init, execute loop, … – is inherited
    # unchanged.

    # ------------------------------------------------------------------
    # ZenWorker constructor
    # ------------------------------------------------------------------

    def __init__(self, *args, **kwargs):
        # Ensure the parent worker uses our ZenCPUModelRunner instead of the
        # default CPUModelRunner. This is done by providing a factory lambda
        # to the `model_runner_cls` argument, which replaces the runner
        # instance immediately after the parent creates it.
        kwargs.setdefault(
            "model_runner_cls",
            lambda base_runner: ZenCPUModelRunner(
                vllm_config=base_runner.vllm_config,
                kv_cache_dtype=base_runner.kv_cache_dtype,
                is_driver_worker=base_runner.is_driver_worker,
                # Pass through any potential extra args from base
                *getattr(base_runner, "_init_args", []),  # noqa: B026
                **getattr(base_runner, "_init_kwargs", {}),  # noqa: B026
            ),
        )
        super().__init__(*args, **kwargs)

    # vLLM-native CPUWorker handles model loading by calling self.model_runner.load_model().
    # Our ZenCPUModelRunner overrides load_model() to perform compilation.
