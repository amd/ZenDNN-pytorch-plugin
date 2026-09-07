# ******************************************************************************
# Copyright (c) 2024-2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from unittest_utils import has_zentorch_full  # noqa: E402

# These tests need torch.compile(backend="zentorch") and the pattern matcher,
# which only exist in the full library. Skip the whole directory when we loaded
# the portable one instead, so unittest does not import the test modules.
if not has_zentorch_full:
    raise unittest.SkipTest("Requires the full zentorch backend (libzentorch.so)")
