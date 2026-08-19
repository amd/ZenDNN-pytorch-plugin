# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import importlib.util
from pathlib import Path


def load_source_vllm_module():
    """Load an isolated zentorch.vllm module from this repository."""
    plugin_root = Path(__file__).resolve().parents[3]
    vllm_init = (
        plugin_root / "src" / "cpu" / "python" / "zentorch" / "vllm" / "__init__.py"
    )
    spec = importlib.util.spec_from_file_location("zentorch.vllm", vllm_init)
    module = importlib.util.module_from_spec(spec)
    return spec, module
