# zentorch unit-test flow

This flow mirrors
[`scripts/test.sh`](scripts/test.sh) and the unit-test procedure in
[README.md section 3](../../../README.md).

Before running tests, the script disables all required ZenDNN caches and
installs the matching test dependencies:

```bash
export ZENDNNL_MATMUL_WEIGHT_CACHE=0
export ZENDNNL_ZP_COMP_CACHE=0
export ZENDNNL_ENABLE_POSTOP_CACHE=0
python test/install_requirements.py
```

With no argument, the script runs all unit tests under `./test/unittests`.
Other supported scopes map to `./test`, a named test category, or one existing
test file.

```mermaid
flowchart TD
    start(["START: Run zentorch<br/>tests"])
    env["1. Active environment<br/>zentorch installed"]
    cache["2. Disable ZenDNN<br/>test caches"]
    deps["3. Install test<br/>dependencies"]
    scope{"4. Test scope?"}
    default["Default:<br/>all unit tests"]
    all["All test suites"]
    category["Named category:<br/>op_tests, model_tests<br/>miscellaneous_tests<br/>export_tests, vllm_tests<br/>llm, pre_trained"]
    file["Existing test<br/>file"]
    invalid["STOP: Unknown scope<br/>Choose scope or file"]
    run["5. Run Python<br/>unittest discovery"]
    passed{"All selected<br/>tests passed?"}
    failed["STOP: Report failures<br/>and non-zero exit"]
    done(["END: All selected<br/>tests passed"])

    start --> env --> cache --> deps --> scope
    scope -- No argument --> default --> run
    scope -- all --> all --> run
    scope -- category --> category --> run
    scope -- test file --> file --> run
    scope -- unknown --> invalid
    run --> passed
    passed -- No --> failed
    passed -- Yes --> done
```
