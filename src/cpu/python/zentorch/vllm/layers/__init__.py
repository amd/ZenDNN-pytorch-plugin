# ****************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ****************************************************************************

# Intentionally empty. This package is a pure namespace for per-layer patch
# modules. Each *Patch.apply() in zentorch.vllm imports its corresponding
# zentorch.vllm.layers.<name>.patch submodule directly, so vLLM submodules
# (e.g. mamba.gdn_linear_attn pulled in transitively by forward.py) are not
# loaded during plugin registration — only after the deferred import hook
# fires.
