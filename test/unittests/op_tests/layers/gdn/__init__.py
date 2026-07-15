# ****************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ****************************************************************************

import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[4]))
from zentorch_test_utils import has_zentorch_ops  # noqa: E402

# GDN_ops.cpp is excluded from libzentorch_stable.so (see
# cmake/modules/StableAbiLibs.cmake), so none of these ops are registered when
# the runtime torch minor differs from the build and only the portable library
# loads. Every test here would then fail on a missing torch.ops.zentorch.gdn_*
# attribute rather than skip. They all come from that one source file, so any
# of them answers for the rest, and gating on the op instead of the library
# means this directory starts running again by itself once GDN_ops.cpp migrates.
if not has_zentorch_ops("gdn_l2norm_fwd"):
    raise unittest.SkipTest("zentorch GDN ops are not registered")
