# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import sys
import types
import unittest
import unittest.mock


class TestGptOssStreamedExpertPatch(unittest.TestCase):
    def test_native_streamed_loader_is_not_replaced(self):
        from zentorch.vllm import _gptoss_streamed_expert_patch as patch

        class GptOssModel:
            @staticmethod
            def _try_load_streamed_expert(*args, **kwargs):
                return True

            def _load_weights_other(self, *args, **kwargs):
                return set()

        class MLPBlock:
            def __init__(self, *args, **kwargs):
                pass

        module = types.SimpleNamespace(
            GptOssModel=GptOssModel,
            GptOssRoutedExperts=type("GptOssRoutedExperts", (), {}),
            MLPBlock=MLPBlock,
        )
        original_loader = GptOssModel._load_weights_other
        original_init = MLPBlock.__init__

        with (
            unittest.mock.patch.dict(
                sys.modules, {patch._TARGET_MODULE: module}
            ),
            unittest.mock.patch.object(
                patch, "_get_routed_experts_cls"
            ) as build_routed_experts,
        ):
            self.assertTrue(patch._do_patch())

        build_routed_experts.assert_not_called()
        self.assertIs(GptOssModel._load_weights_other, original_loader)
        self.assertIs(MLPBlock.__init__, original_init)
        self.assertTrue(
            getattr(GptOssModel, patch._MARKER, False)
        )


if __name__ == "__main__":
    unittest.main()
