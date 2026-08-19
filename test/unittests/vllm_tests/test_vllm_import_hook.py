# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import sys
import types
import unittest
import unittest.mock


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


if __name__ == "__main__":
    unittest.main()
