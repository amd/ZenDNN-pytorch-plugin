Copyright &copy; 2026 Advanced Micro Devices, Inc. All rights reserved.

# zentorch unit-test flow

This flow mirrors
[`scripts/test.sh`](scripts/test.sh) and the unit-test procedure in
[README.md section 3](../../../README.md).

The script first rejects a missing or `base` environment, then maps the
requested scope. Unknown scopes and nonexistent files stop before zentorch or
package checks, so they cannot mutate the environment. A direct file uses
`python -m unittest <file>`; directory scopes use discovery.

After scope validation, the script checks that zentorch is installed, disables
all required ZenDNN caches, and installs the matching test dependencies:

```bash
python -c "import zentorch"
export ZENDNNL_MATMUL_WEIGHT_CACHE=0
export ZENDNNL_ZP_COMP_CACHE=0
export ZENDNNL_ENABLE_POSTOP_CACHE=0
python test/install_requirements.py
```

The dependency installer covers `transformers`, `expecttest`, `parameterized`,
`hypothesis`, `deprecated`, and PyTorch-matched `torchvision` and `torchao`.
With no argument, the script runs all unit tests under `./test/unittests`.
Other supported scopes map to `./test`, a named test category, or one existing
test file. Export tests may write `model.pt2` and `model_z.pt2` in the checkout;
the script does not delete files by name because their ownership is unknown.

```mermaid
flowchart TD
    start(["START: Run zentorch<br/>tests"])
    env{"Dedicated env<br/>active?"}
    envFail["STOP: Activate a<br/>non-base environment"]
    scope{"1. Test scope?"}
    default["Default:<br/>all unit tests"]
    all["All test suites"]
    category["Named category:<br/>op_tests, model_tests<br/>miscellaneous_tests<br/>export_tests, vllm_tests<br/>llm, pre_trained"]
    file["Existing test<br/>file"]
    invalid["STOP: Unknown scope<br/>Choose scope or file"]
    installed{"zentorch<br/>installed?"}
    installFail["STOP: Build and<br/>install zentorch"]
    cache["2. Disable ZenDNN<br/>test caches"]
    deps["3. Install test<br/>dependencies"]
    depsOk{"Dependencies<br/>ready?"}
    depsFail["STOP: Dependency<br/>install failed"]
    mode{"Directory scope<br/>or test file?"}
    discover["4. Run unittest<br/>discovery"]
    direct["4. Run file without<br/>discovery"]
    passed{"All selected<br/>tests passed?"}
    failed["STOP: Report failures<br/>and non-zero exit"]
    done(["END: All selected<br/>tests passed"])

    start --> env
    env -- No --> envFail
    env -- Yes --> scope
    scope -- No argument --> default --> installed
    scope -- all --> all --> installed
    scope -- category --> category --> installed
    scope -- test file --> file --> installed
    scope -- unknown --> invalid
    installed -- No --> installFail
    installed -- Yes --> cache --> deps --> depsOk
    depsOk -- No --> depsFail
    depsOk -- Yes --> mode
    mode -- Directory --> discover --> passed
    mode -- File --> direct --> passed
    passed -- No --> failed
    passed -- Yes --> done
```
