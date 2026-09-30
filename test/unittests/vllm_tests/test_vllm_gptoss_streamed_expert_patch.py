# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************
"""Key routing and native-loader detection in
``zentorch.vllm._gptoss_streamed_expert_patch``."""

import sys
import types
import unittest
import unittest.mock

import torch  # noqa: F401
import zentorch  # noqa: F401

from ._test_constants import VLLM_AVAILABLE


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


@unittest.skipUnless(VLLM_AVAILABLE, "vLLM not installed")
class TestStreamedExpertKeyResolution(unittest.TestCase):
    """int4 checkpoints name their expert weights ``*_weight_packed``."""

    PREFIX = "model.layers.0.mlp.experts"
    FUSED = f"{PREFIX}.routed_experts"

    def _resolve(self, suffix):
        from zentorch.vllm._gptoss_streamed_expert_patch import (
            _get_streamed_expert_info,
        )

        params_dict = {f"{self.FUSED}.{suffix}": object()}
        return _get_streamed_expert_info(f"{self.PREFIX}.3.{suffix}", params_dict)

    def test_packed_expert_weights_resolve_to_their_fused_param(self):
        for suffix, expected_shard in (
            ("w13_weight_packed", "gpt_oss_w13"),
            ("w2_weight_packed", "gpt_oss_w2"),
        ):
            with self.subTest(suffix=suffix):
                self.assertEqual(
                    self._resolve(suffix),
                    (3, f"{self.FUSED}.{suffix}", expected_shard),
                )

    def test_unknown_suffix_is_not_claimed(self):
        self.assertIsNone(self._resolve("w13_weight_shape"))


if __name__ == "__main__":
    unittest.main()
