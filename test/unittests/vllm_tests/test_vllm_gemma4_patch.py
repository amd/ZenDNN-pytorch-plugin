# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import sys
import types
import unittest
import unittest.mock

from ._test_constants import VLLM_AVAILABLE


@unittest.skipUnless(VLLM_AVAILABLE, "vLLM not installed")
class TestGemma4HeteroConfigPatch(unittest.TestCase):
    """Validate gemma-4 heterogeneous head-dimension compatibility."""

    @staticmethod
    def _fake_gemma_config():
        class _Layer:
            def __init__(self, head_dim, num_key_value_heads=8):
                self.head_dim = head_dim
                self.num_key_value_heads = num_key_value_heads

        class _PerLayerView:
            def __init__(self, layers):
                self._layers = [_Layer(*spec) for spec in layers]

            def __getitem__(self, i):
                return self._layers[i]

        class _HeteroTextConfig:
            is_heterogeneous = True

            def __init__(self):
                self.layer_types = [
                    "sliding_attention",
                    "sliding_attention",
                    "sliding_attention",
                    "full_attention",
                ]
                self._plc = _PerLayerView(
                    [(256, 8), (256, 8), (256, 8), (512, 2)]
                )
                self.allow_global_per_layer_attribute_access = False

            @property
            def per_layer_config(self):
                return self._plc

        class _TopConfig:
            is_heterogeneous = False

            def __init__(self, text):
                self.text_config = text

        text = _HeteroTextConfig()
        return _TopConfig(text), text

    def test_reconstructs_global_head_dim_and_arms_escape_hatch(self):
        from zentorch.vllm._gemma4_hetero_config_patch import (
            _enable_global_per_layer_access,
        )

        top, text = self._fake_gemma_config()
        _enable_global_per_layer_access(top)

        self.assertTrue(text.allow_global_per_layer_attribute_access)
        self.assertEqual(text.global_head_dim, 512)
        self.assertEqual(text.num_global_key_value_heads, 2)

    def test_homogeneous_config_is_untouched(self):
        from zentorch.vllm._gemma4_hetero_config_patch import (
            _enable_global_per_layer_access,
        )

        class _Homo:
            is_heterogeneous = False

            def __init__(self):
                self.head_dim = 128

        cfg = _Homo()
        _enable_global_per_layer_access(cfg)
        self.assertFalse(hasattr(cfg, "global_head_dim"))
        self.assertFalse(hasattr(cfg, "num_global_key_value_heads"))
        self.assertFalse(hasattr(cfg, "allow_global_per_layer_attribute_access"))

    def test_patches_get_config_on_every_load_path(self):
        from zentorch.vllm import _gemma4_hetero_config_patch as patch

        def _make_get_config(outer):
            def get_config(*args, **kwargs):
                top, _text = outer._fake_gemma_config()
                return top

            return get_config

        saved = {name: sys.modules.get(name) for name in patch._TARGET_MODULES}

        def _restore():
            for name, module in saved.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module

        self.addCleanup(_restore)

        fakes = {}
        for name in patch._TARGET_MODULES:
            module = types.ModuleType(name)
            module.get_config = _make_get_config(self)
            sys.modules[name] = module
            fakes[name] = module

        self.assertTrue(patch._do_patch_gemma4_hetero())

        for name, module in fakes.items():
            self.assertTrue(
                getattr(module, "_zentorch_gemma4_hetero_patched", False),
                f"{name}.get_config was not patched",
            )
            text = module.get_config().text_config
            self.assertTrue(
                text.allow_global_per_layer_attribute_access,
                f"escape hatch not armed through {name}.get_config",
            )
            self.assertEqual(text.global_head_dim, 512)
            self.assertEqual(text.num_global_key_value_heads, 2)

    def test_patched_get_config_fails_open_on_walk_error(self):
        from zentorch.vllm import _gemma4_hetero_config_patch as patch

        sentinel = object()
        module = types.ModuleType("zentorch_test_fake_get_config_mod")
        module.get_config = lambda *args, **kwargs: sentinel
        self.assertTrue(patch._patch_module_get_config(module))

        with unittest.mock.patch.object(
            patch,
            "_enable_global_per_layer_access",
            side_effect=RuntimeError("boom"),
        ):
            result = module.get_config()
        self.assertIs(result, sentinel)

    def test_arms_hetero_configs_nested_in_containers(self):
        from zentorch.vllm._gemma4_hetero_config_patch import (
            _enable_global_per_layer_access,
        )

        class _Hetero:
            is_heterogeneous = True

        class _Top:
            is_heterogeneous = False

            def __init__(self, in_list, in_dict):
                self.sub_list = in_list
                self.sub_map = in_dict

        in_list, in_dict = _Hetero(), _Hetero()
        _enable_global_per_layer_access(_Top([in_list], {"k": in_dict}))
        self.assertTrue(in_list.allow_global_per_layer_attribute_access)
        self.assertTrue(in_dict.allow_global_per_layer_attribute_access)

    def test_restore_global_head_dim_tolerates_bad_per_layer(self):
        from zentorch.vllm._gemma4_hetero_config_patch import (
            _restore_global_head_dim,
        )

        class _Node:
            def __init__(self, layer_types, per_layer):
                self.layer_types = layer_types
                self.per_layer_config = per_layer

        dict_backed = _Node(
            ["full_attention"], {"full_attention": object()}
        )
        _restore_global_head_dim(dict_backed)
        self.assertIsNone(getattr(dict_backed, "global_head_dim", None))
        self.assertIsNone(getattr(dict_backed, "num_global_key_value_heads", None))

        short = _Node(["sliding_attention", "full_attention"], [])
        _restore_global_head_dim(short)
        self.assertIsNone(getattr(short, "global_head_dim", None))
        self.assertIsNone(getattr(short, "num_global_key_value_heads", None))

    def test_restore_kv_heads_when_global_head_dim_already_set(self):
        from zentorch.vllm._gemma4_hetero_config_patch import (
            _restore_global_head_dim,
        )

        _top, text = self._fake_gemma_config()
        text.global_head_dim = 512
        _restore_global_head_dim(text)
        self.assertEqual(text.global_head_dim, 512)
        self.assertEqual(text.num_global_key_value_heads, 2)

    def test_restore_does_not_overwrite_existing_kv_heads(self):
        from zentorch.vllm._gemma4_hetero_config_patch import (
            _restore_global_head_dim,
        )

        _top, text = self._fake_gemma_config()
        text.num_global_key_value_heads = 4
        _restore_global_head_dim(text)
        self.assertEqual(text.num_global_key_value_heads, 4)
        self.assertEqual(text.global_head_dim, 512)

    def test_walk_tolerates_slots_nodes(self):
        from zentorch.vllm._gemma4_hetero_config_patch import (
            _enable_global_per_layer_access,
        )

        class _Slots:
            __slots__ = ()
            is_heterogeneous = False

        _enable_global_per_layer_access(_Slots())


if __name__ == "__main__":
    unittest.main()
