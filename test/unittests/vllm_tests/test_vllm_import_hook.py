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
class TestImportHookFailure(unittest.TestCase):
    """Optional patch failures must not change target-module import semantics."""

    @staticmethod
    def _deferred_patch(patch_fn, original_exec):
        import zentorch.vllm._import_hook as import_hook

        loader = types.SimpleNamespace(exec_module=original_exec)
        spec = types.SimpleNamespace(loader=loader)
        patcher = import_hook._PostImportPatcher(
            "vllm._zentorch_test_target", patch_fn
        )
        with unittest.mock.patch.object(
            import_hook.importlib.util, "find_spec", return_value=spec
        ):
            return import_hook, patcher.find_spec(
                "vllm._zentorch_test_target", None
            )

    def test_deferred_patch_failure_does_not_break_import(self):
        module = types.ModuleType("vllm._zentorch_test_target")
        original_exec = unittest.mock.Mock(
            side_effect=lambda target: setattr(target, "loaded", True)
        )
        patch_fn = unittest.mock.Mock(
            side_effect=RuntimeError("patch failed")
        )
        import_hook, spec = self._deferred_patch(patch_fn, original_exec)

        with self.assertLogs(import_hook.logger, level="WARNING") as logs:
            spec.loader.exec_module(module)

        self.assertTrue(module.loaded)
        original_exec.assert_called_once_with(module)
        patch_fn.assert_called_once_with()
        self.assertIn("continuing without it", "\n".join(logs.output))

    def test_target_module_failure_still_propagates(self):
        module = types.ModuleType("vllm._zentorch_test_target")
        original_exec = unittest.mock.Mock(
            side_effect=RuntimeError("import failed")
        )
        patch_fn = unittest.mock.Mock()
        _, spec = self._deferred_patch(patch_fn, original_exec)

        with self.assertRaisesRegex(RuntimeError, "import failed"):
            spec.loader.exec_module(module)

        patch_fn.assert_not_called()

    def test_immediate_patch_failure_returns_false(self):
        import zentorch.vllm._import_hook as import_hook

        module_name = "vllm._zentorch_test_loaded_target"
        patch_fn = unittest.mock.Mock(
            side_effect=RuntimeError("patch failed")
        )
        self.addCleanup(import_hook._handled.discard, module_name)

        with (
            unittest.mock.patch.dict(
                sys.modules, {module_name: types.ModuleType(module_name)}
            ),
            self.assertLogs(import_hook.logger, level="WARNING"),
        ):
            applied = import_hook.patch_now_or_on_import(
                module_name, patch_fn
            )

        self.assertFalse(applied)
        patch_fn.assert_called_once_with()

    def test_multiple_deferred_callbacks_run_on_same_module(self):
        """Two deferred apply() callbacks share one import (DA8W4 + WNA16).

        ``Mock(return_value=True)`` is required: the hook calls ``patch_fn()``,
        so a bare ``True``/``False`` is not callable, and two identical
        ``lambda: True`` objects cannot prove both ran. Distinct mocks both
        register (unique identity) and ``assert_called_once_with`` shows each
        ran after the shared module load.
        """
        import zentorch.vllm._import_hook as import_hook

        module_name = "vllm._zentorch_multi_patch_target"
        first = unittest.mock.Mock(return_value=True)
        second = unittest.mock.Mock(return_value=True)
        self.addCleanup(import_hook._handled.discard, module_name)
        self.addCleanup(lambda: import_hook._fns.pop(module_name, None))

        self.assertTrue(
            import_hook.patch_now_or_on_import(module_name, first)
        )
        self.assertTrue(
            import_hook.patch_now_or_on_import(module_name, second)
        )

        created = types.ModuleType(module_name)
        loader = types.SimpleNamespace(
            exec_module=lambda target: setattr(target, "loaded", True)
        )
        spec = types.SimpleNamespace(loader=loader)
        patcher = next(
            finder
            for finder in sys.meta_path
            if isinstance(finder, import_hook._PostImportPatcher)
            and finder._module_name == module_name
        )
        with unittest.mock.patch.object(
            import_hook.importlib.util, "find_spec", return_value=spec
        ):
            wrapped = patcher.find_spec(module_name, None)
            wrapped.loader.exec_module(created)

        self.assertTrue(created.loaded)
        first.assert_called_once_with()
        second.assert_called_once_with()

    def test_false_callback_does_not_skip_later_callbacks(self):
        import zentorch.vllm._import_hook as import_hook

        module_name = "vllm._zentorch_multi_patch_false_first"
        first = unittest.mock.Mock(return_value=False)
        second = unittest.mock.Mock(return_value=True)
        self.addCleanup(lambda: import_hook._fns.pop(module_name, None))

        import_hook._register_fn(module_name, first)
        import_hook._register_fn(module_name, second)

        self.assertFalse(import_hook._run_registered(module_name))
        first.assert_called_once_with()
        second.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
