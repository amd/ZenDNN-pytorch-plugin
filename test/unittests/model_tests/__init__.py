# ******************************************************************************
# Copyright (c) 2024-2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from unittest_utils import has_zentorch_full  # noqa: E402

# These tests drive torch.compile(backend="zentorch") and the pattern-matcher
# counters, both of which live in libzentorch.so. That library is not loaded when
# the runtime torch minor version differs from the build: only the stable-ABI ops
# are, out of libzentorch_stable.so (see src/cpu/python/zentorch/__init__.py).
# Discovery turns a SkipTest raised here into one skipped entry for the whole
# directory, without importing any of the test modules.
if not has_zentorch_full:
    raise unittest.SkipTest("Requires the full zentorch backend (libzentorch.so)")
