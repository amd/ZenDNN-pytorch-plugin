# ****************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ****************************************************************************

import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[4]))
from zentorch_test_utils import has_zentorch_ops  # noqa: E402

# GDN ops are not in the portable library, so skip this directory rather than
# fail on a missing torch.ops.zentorch.gdn_* attribute. One op stands in for
# the rest; once they ship in the portable library, these tests run on their own.
if not has_zentorch_ops("gdn_l2norm_fwd"):
    raise unittest.SkipTest("zentorch GDN ops are not registered")
