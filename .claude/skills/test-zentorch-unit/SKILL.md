---
name: test-zentorch-unit
description: >-
  Run zentorch tests (unit, op, model, miscellaneous, export, vLLM, LLM, or
  pre-trained model tests) with the required ZenDNN cache env vars. Use when the
  user asks to run or discover zentorch tests.
---

<!-- Copyright &copy; 2026 Advanced Micro Devices, Inc. All rights reserved. -->

# Run zentorch tests

When the user asks to run tests, follow this skill.

**Agent action:** Run the mapped command (adjust scope as needed):

```bash
.claude/skills/test-zentorch-unit/scripts/test.sh                     # unit tests (default)
.claude/skills/test-zentorch-unit/scripts/test.sh all
.claude/skills/test-zentorch-unit/scripts/test.sh op_tests
.claude/skills/test-zentorch-unit/scripts/test.sh model_tests
.claude/skills/test-zentorch-unit/scripts/test.sh miscellaneous_tests
.claude/skills/test-zentorch-unit/scripts/test.sh export_tests
.claude/skills/test-zentorch-unit/scripts/test.sh vllm_tests
.claude/skills/test-zentorch-unit/scripts/test.sh llm
.claude/skills/test-zentorch-unit/scripts/test.sh pre_trained
.claude/skills/test-zentorch-unit/scripts/test.sh test/unittests/op_tests/test_bmm.py
```

The script validates the requested scope before any package checks or
installation, verifies that zentorch is installed, sets the required env vars,
installs test deps, and runs the scope. Use the manual commands below only if
the script fails or a `-k` / `-p` filter is needed that the script does not
support.

See [zentorch-test-flow.md](zentorch-test-flow.md) for the test workflow.

---

## Environment

Requires an activated, non-`base` Python environment **with zentorch installed**
— the same environment used across the other zentorch skills. Confirm what is
active:

```bash
echo "${VIRTUAL_ENV:-${CONDA_DEFAULT_ENV:-none}}"
```

If zentorch is not built/installed yet, follow the `build-zentorch-setup-env`
or `build-zentorch-from-source` skill first.

---

## 1. Validate what to run

Match the user's request to the right command. If the user just says "run
tests" with no specifics, run all unit tests:

```bash
python -m unittest discover -s ./test/unittests
```

The script rejects an unknown scope or nonexistent file immediately, before
checking zentorch or running the dependency installer. A direct test file is
run with `python -m unittest <file>`; it does not use discovery.

## 2. Verify zentorch and set required environment variables

The script first verifies `import zentorch`. Then disable ZenDNN caching before
running any tests:

```bash
python -c "import zentorch"
export ZENDNNL_MATMUL_WEIGHT_CACHE=0
export ZENDNNL_ZP_COMP_CACHE=0
export ZENDNNL_ENABLE_POSTOP_CACHE=0
```

## 3. Install test dependencies

```bash
python test/install_requirements.py
```

Installs `transformers`, `expecttest`, `parameterized`, `hypothesis`, and
`deprecated`, plus the `torchvision` and `torchao` versions matching the
installed PyTorch.

### Test categories

Every category is discovered the same way (`python -m unittest discover -s <path>`):

| User says something like        | Scope                 | Path                                    |
|---------------------------------|-----------------------|-----------------------------------------|
| "run all tests"                 | `all`                 | `./test`                                |
| "run unit tests" / "unittests"  | `unittests`           | `./test/unittests`                      |
| "run op tests"                  | `op_tests`            | `./test/unittests/op_tests`             |
| "run model tests"               | `model_tests`         | `./test/unittests/model_tests`          |
| "run miscellaneous tests"       | `miscellaneous_tests` | `./test/unittests/miscellaneous_tests`  |
| "run export tests"              | `export_tests`        | `./test/unittests/export_tests`         |
| "run vllm tests"                | `vllm_tests`          | `./test/unittests/vllm_tests`           |
| "run llm tests"                 | `llm`                 | `./test/llm_tests`                      |
| "run pre-trained / pretrained"  | `pre_trained`         | `./test/pre_trained_model_tests`        |

### If the user names a specific op or feature

Find the matching test file first, then run it:

```bash
find test -name 'test_*.py' | grep -i "<keyword>"
python -m unittest <path>
```

Examples (verified paths):

| User says            | Run                                                             |
|----------------------|-----------------------------------------------------------------|
| "run bmm test"       | `python -m unittest test/unittests/op_tests/test_bmm.py`        |
| "run mm test"        | `python -m unittest test/unittests/op_tests/test_mm.py`         |
| "run linear tests"   | `python -m unittest discover -s ./test/unittests -p "test_linear*"` |
| "run woq tests"      | `python -m unittest discover -s ./test/unittests -k "woq"`      |
| "run sdpa test"      | `python -m unittest test/unittests/model_tests/test_sdpa.py`    |
| "run fused moe test" | `python -m unittest test/unittests/model_tests/test_fused_moe.py` |
| "run version test"   | `python -m unittest test/unittests/miscellaneous_tests/test_zentorch_version.py` |
| "run bert test"      | `python -m unittest test/pre_trained_model_tests/test_bert.py`  |

If multiple files match, show the matches and ask which to run, or use `-k` /
`-p` filters to run all of them.

### Generated export-test packages

The underlying export tests may write `model.pt2` and `model_z.pt2` in the
repository root. The test skill intentionally does not delete or restore files
by name because it cannot determine whether they predated the run. Check those
paths before running export tests, and clean up only artifacts known to belong
to that run.

---

## Test directory structure

```
test/
  unittests/
    op_tests/              # Individual op tests (bmm, mm, embedding, qlinear, rms_norm, ...)
    model_tests/           # Custom model fusion/graph tests (sdpa, fused_moe, pattern_matcher, ...)
    miscellaneous_tests/   # Non-op tests (version, device/isa checks)
    export_tests/          # torch.export tests
    vllm_tests/            # vLLM integration tests
  llm_tests/               # Shared LLM test utilities (llm_utils.py)
  pre_trained_model_tests/ # Real model tests (bert, cnn)
  zentorch_test_utils.py   # Shared test utilities
  install_requirements.py  # Test dependency installer
```

## Test file quick reference

### op_tests/
`test_addmm`, `test_baddbmm`, `test_bmm`, `test_dynamic_qlinear`,
`test_embedding`, `test_embedding_bag`, `test_embeg_pack_weight`,
`test_group_matmul`, `test_horizontal_embedding_bag_group`,
`test_horizontal_embedding_group`, `test_linear`, `test_matmul_direct`,
`test_matmul_tanh`, `test_mm`, `test_mm_silu`, `test_mm_silu_mul`,
`test_prepare_4d_causal_attention_mask`, `test_qlinear`, `test_qlinear_eltwise`,
`test_qlinear_mul_add`, `test_quant_embedding_bag`, `test_rms_norm`,
`test_woq_linear`, `test_woq_linear_asymmetric`

### model_tests/
Custom models exercising fusion/graph patterns: addmm/mm unary + binary fusions,
linear unary/binary, qkv fusion, sdpa, fused MoE, horizontal embedding groups,
qlinear reorder/eltwise, woq fusions, and the pattern matcher.

### miscellaneous_tests/
| File | What it tests |
|------|---------------|
| `test_zentorch_version.py` | zentorch version string |
| `test_avx512_device.py` | AVX-512 device/ISA detection |
| `test_bf16_device.py` | bf16 device support |
| `test_fp16_device.py` | fp16 device support |

### export_tests/
| File | What it tests |
|------|---------------|
| `test_exp.py` | `torch.export` support |

### vllm_tests/
| File | What it tests |
|------|---------------|
| `test_vllm_plugin.py` | vLLM plugin integration |

### pre_trained_model_tests/
| File | What it tests |
|------|---------------|
| `test_bert.py` | BERT model inference |
| `test_cnn.py` | CNN model inference |
