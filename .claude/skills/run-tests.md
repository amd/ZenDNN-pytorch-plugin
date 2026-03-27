# Skill: Run zentorch tests

When the user asks to run tests, follow this skill.

---

## 1. Install test dependencies (if not already installed)

```bash
python test/install_requirements.py
```

This installs `transformers`, `expecttest`, `parameterized`, and the correct
`torchvision` / `torchao` versions matching the installed PyTorch.

## 2. Set required environment variables

Disable ZenDNN caching before running any tests:

```bash
export ZENDNNL_MATMUL_WEIGHT_CACHE=0
export ZENDNNL_ZP_COMP_CACHE=0
```

## 3. Determine what to run

Match the user's request to the right command using the logic below.

### If the user says "run tests" with no specifics

Run all unit tests:

```bash
python -m unittest discover -s ./test/unittests
```

### If the user names a test category

| User says something like                        | Command                                                         |
|-------------------------------------------------|-----------------------------------------------------------------|
| "run unit tests" / "run unittests"              | `python -m unittest discover -s ./test/unittests`               |
| "run all tests"                                 | `python -m unittest discover -s ./test`                         |
| "run op tests"                                  | `python -m unittest discover -s ./test/unittests/op_tests`      |
| "run model tests"                               | `python -m unittest discover -s ./test/unittests/model_tests`   |
| "run miscellaneous tests"                       | `python -m unittest discover -s ./test/unittests/miscellaneous_tests` |
| "run export tests"                              | `python -m unittest discover -s ./test/unittests/export_tests`  |
| "run vllm tests"                                | `python -m unittest discover -s ./test/unittests/vllm_tests`    |
| "run llm tests"                                 | `python -m unittest discover -s ./test/llm_tests`               |
| "run pre-trained tests" / "pretrained tests"    | `python -m unittest discover -s ./test/pre_trained_model_tests` |

### If the user names a specific op or feature

Search for the matching test file first:

```bash
find test/ -name "*.py" | grep -i "<keyword>"
```

Then run it. Examples:

| User says              | Find & run                                                          |
|------------------------|---------------------------------------------------------------------|
| "run bmm test"         | `python -m unittest test/unittests/op_tests/test_bmm.py`           |
| "run embedding test"   | `python -m unittest test/unittests/op_tests/test_embedding.py`     |
| "run rope test"        | `python -m unittest test/unittests/op_tests/test_rope.py`          |
| "run woq tests"        | `python -m unittest discover -s ./test/unittests -k "woq"`         |
| "run linear tests"     | `python -m unittest discover -s ./test/unittests -p "test_linear*"` |
| "run bert test"        | `python -m unittest test/pre_trained_model_tests/test_bert.py`     |
| "run cnn test"         | `python -m unittest test/pre_trained_model_tests/test_cnn.py`      |
| "run mha test"         | `python -m unittest test/llm_tests/test_masked_mha.py`             |
| "run quantization tests" | `python -m unittest discover -s ./test/unittests -k "qlinear"`  |

If multiple files match, show the user the matches and ask which one(s) to run,
or use `-k` / `-p` filters to run all of them.

### If the user gives an exact file path

Run it directly:

```bash
python -m unittest <path>
```

---

## Test directory structure

```
test/
  unittests/
    op_tests/              # Individual op tests (bmm, mm, embedding, etc.)
    model_tests/           # Custom model fusion/graph tests
    miscellaneous_tests/   # Non-op tests (version, device checks)
    export_tests/          # torch.export tests
    vllm_tests/            # vLLM integration tests
  llm_tests/               # LLM operator tests (rope, mha)
  pre_trained_model_tests/ # Real model tests (bert, cnn)
  zentorch_test_utils.py   # Shared test utilities
  install_requirements.py  # Test dependency installer
```

## Test file quick reference

### op_tests/
| File | What it tests |
|------|---------------|
| `test_addmm.py` | addmm op |
| `test_addmm_silu.py` | addmm + silu fusion |
| `test_addmm_silu_mul.py` | addmm + silu + mul fusion |
| `test_baddbmm.py` | batched addmm |
| `test_bmm.py` | batched matmul |
| `test_dynamic_qlinear.py` | dynamic quantized linear |
| `test_embedding.py` | embedding op |
| `test_embedding_bag.py` | embedding bag op |
| `test_embeg_pack_weight.py` | embedding pack weight |
| `test_horizontal_embedding_bag_group.py` | grouped embedding bag |
| `test_horizontal_embedding_group.py` | grouped embedding |
| `test_linear.py` | linear op |
| `test_matmul_direct.py` | direct matmul |
| `test_matmul_tanh.py` | matmul + tanh fusion |
| `test_mm.py` | matrix multiply |
| `test_mm_silu.py` | mm + silu fusion |
| `test_mm_silu_mul.py` | mm + silu + mul fusion |
| `test_prepare_4d_causal_attention_mask.py` | causal attention mask |
| `test_qlinear.py` | quantized linear |
| `test_qlinear_eltwise.py` | quantized linear + elementwise |
| `test_qlinear_mul_add.py` | quantized linear + mul + add |
| `test_quant_embedding_bag.py` | quantized embedding bag |
| `test_rope.py` | rotary position embedding |
| `test_woq_linear.py` | weight-only quantized linear |

### model_tests/
Tests with custom models exercising fusion patterns: linear-unary, linear-binary,
qkv-fusion, sdpa, mini-mha, horizontal embedding groups, pattern matcher, etc.

### llm_tests/
| File | What it tests |
|------|---------------|
| `test_fused_rope_op.py` | Fused RoPE operator |
| `test_masked_mha.py` | Masked multi-head attention |

### pre_trained_model_tests/
| File | What it tests |
|------|---------------|
| `test_bert.py` | BERT model inference |
| `test_cnn.py` | CNN model inference |
