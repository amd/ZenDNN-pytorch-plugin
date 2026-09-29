# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

"""Tests for the Whisper audio truncation=True patch."""

import sys
import types
import unittest
import unittest.mock

from ._test_constants import VLLM_AVAILABLE

if VLLM_AVAILABLE:
    from zentorch.vllm._whisper_truncation_patch import (
        _PATCH_MARKER,
        _TARGET_MODULE,
        _apply_whisper_truncation_patch,
        _do_patch_whisper_truncation,
        _with_truncation,
    )


@unittest.skipUnless(VLLM_AVAILABLE, "vLLM not installed")
class TestWhisperTruncationPatchWiring(unittest.TestCase):
    def test_wired_in_patches(self):
        import zentorch.vllm as zv

        names = [name for name, _ in zv._PATCHES]
        self.assertIn("WhisperTruncation", names)
        self.assertIs(
            dict(zv._PATCHES)["WhisperTruncation"],
            _apply_whisper_truncation_patch,
        )
        self.assertEqual(
            names[names.index("WhisperW4A16") + 1],
            "WhisperTruncation",
        )


@unittest.skipUnless(VLLM_AVAILABLE, "vLLM not installed")
class TestWithTruncation(unittest.TestCase):
    def test_overrides_false_and_copies(self):
        src = {"truncation": False, "sampling_rate": 16000}
        out = _with_truncation(src)
        self.assertIsNot(out, src)
        self.assertEqual(out["truncation"], True)
        self.assertEqual(src["truncation"], False)
        self.assertEqual(out["sampling_rate"], 16000)


@unittest.skipUnless(VLLM_AVAILABLE, "vLLM not installed")
class TestWhisperTruncationPatch(unittest.TestCase):
    def test_returns_false_when_module_missing(self):
        with unittest.mock.patch.dict(sys.modules, {_TARGET_MODULE: None}):
            sys.modules.pop(_TARGET_MODULE, None)
            self.assertFalse(_do_patch_whisper_truncation())

    def test_returns_false_without_processor(self):
        module = types.SimpleNamespace()
        with unittest.mock.patch.dict(sys.modules, {_TARGET_MODULE: module}):
            self.assertFalse(_do_patch_whisper_truncation())

    def test_returns_false_without_hook_methods(self):
        class WhisperMultiModalProcessor:
            pass

        module = types.SimpleNamespace(
            WhisperMultiModalProcessor=WhisperMultiModalProcessor
        )
        with unittest.mock.patch.dict(sys.modules, {_TARGET_MODULE: module}):
            self.assertFalse(_do_patch_whisper_truncation())

    def test_wraps_preprocess_once_and_forces_truncation(self):
        class WhisperMultiModalProcessor:
            def _preprocess_hf_mm_data(self, mm_data, hf_processor_mm_kwargs):
                return mm_data, dict(hf_processor_mm_kwargs)

        module = types.SimpleNamespace(
            WhisperMultiModalProcessor=WhisperMultiModalProcessor
        )
        with unittest.mock.patch.dict(sys.modules, {_TARGET_MODULE: module}):
            self.assertTrue(_do_patch_whisper_truncation())
            wrapped = WhisperMultiModalProcessor._preprocess_hf_mm_data
            self.assertTrue(_do_patch_whisper_truncation())
            self.assertIs(WhisperMultiModalProcessor._preprocess_hf_mm_data, wrapped)

            incoming = {"audios": object()}
            mm_data, kwargs = WhisperMultiModalProcessor()._preprocess_hf_mm_data(
                incoming,
                {"truncation": False, "sampling_rate": 16000},
            )

        self.assertIs(mm_data, incoming)
        self.assertEqual(kwargs["truncation"], True)
        self.assertEqual(kwargs["sampling_rate"], 16000)
        self.assertTrue(getattr(WhisperMultiModalProcessor, _PATCH_MARKER))

    def test_wraps_get_hf_mm_inputs(self):
        class _HFInputs:
            def __init__(self, hf_kwargs):
                self.hf_kwargs = hf_kwargs

            def _replace(self, **kwargs):
                hf_kwargs = kwargs.get("hf_kwargs", self.hf_kwargs)
                return _HFInputs(hf_kwargs)

        class WhisperMultiModalProcessor:
            def _get_hf_mm_inputs(self, mm_items, hf_kwargs):
                return _HFInputs(dict(hf_kwargs))

        module = types.SimpleNamespace(
            WhisperMultiModalProcessor=WhisperMultiModalProcessor
        )
        with unittest.mock.patch.dict(sys.modules, {_TARGET_MODULE: module}):
            self.assertTrue(_do_patch_whisper_truncation())
            out = WhisperMultiModalProcessor()._get_hf_mm_inputs(
                object(), {"truncation": False, "sampling_rate": 16000}
            )

        self.assertEqual(out.hf_kwargs["truncation"], True)
        self.assertEqual(out.hf_kwargs["sampling_rate"], 16000)

    def test_does_not_wrap_inherited_hooks(self):
        class _Base:
            def _get_hf_mm_inputs(self, mm_items, hf_kwargs):
                return hf_kwargs

        class WhisperMultiModalProcessor(_Base):
            def _preprocess_hf_mm_data(self, mm_data, hf_processor_mm_kwargs):
                return mm_data, dict(hf_processor_mm_kwargs)

        inherited = _Base._get_hf_mm_inputs
        module = types.SimpleNamespace(
            WhisperMultiModalProcessor=WhisperMultiModalProcessor
        )
        with unittest.mock.patch.dict(sys.modules, {_TARGET_MODULE: module}):
            self.assertTrue(_do_patch_whisper_truncation())

        self.assertIs(WhisperMultiModalProcessor._get_hf_mm_inputs, inherited)


if __name__ == "__main__":
    unittest.main()
