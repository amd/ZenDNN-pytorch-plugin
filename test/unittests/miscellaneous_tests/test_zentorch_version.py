# ******************************************************************************
# Copyright (c) 2024-2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import unittest
from importlib import metadata
from unittest.mock import patch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from unittest_utils import(  # noqa: 402
    Zentorch_TestCase,
    has_zentorch,
    run_tests,
    zentorch,
)


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_ZenTorch_Version(Zentorch_TestCase):
    def test_zentorch_version(self):
        dist_version = None
        for name in ("zentorch", "zentorch-weekly"):
            try:
                dist_version = metadata.version(name)
                break
            except metadata.PackageNotFoundError:
                continue
        self.assertIsNotNone(
            dist_version, "No package metadata found for zentorch or zentorch-weekly"
        )
        self.assertEqual(zentorch.__version__, dist_version)


class Test_ZenTorch_Dual_Install(Zentorch_TestCase):
    def test_both_dists_installed_raises_import_error(self):
        def mock_version(name):
            if name in ("zentorch", "zentorch-weekly"):
                return "1.0.0"
            raise metadata.PackageNotFoundError(name)

        with patch("importlib.metadata.version", side_effect=mock_version), \
                self.assertRaises(ImportError) as ctx:
            zentorch._check_dual_install()
        error_message = str(ctx.exception)
        self.assertIn("zentorch", error_message)
        self.assertIn("zentorch-weekly", error_message)


if __name__ == "__main__":
    run_tests()
