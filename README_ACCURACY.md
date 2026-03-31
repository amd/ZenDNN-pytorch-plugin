# LM Evaluation Harness - Accuracy Benchmarking

## Overview

We have migrated from the custom LM Eval implementation (`ZenDNN_tools/LM_Evaluation`) to the open-source [EleutherAI LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/v0.4.11).

**Version:** [v0.4.11](https://github.com/EleutherAI/lm-evaluation-harness/releases/tag/v0.4.11) ([PyPI](https://pypi.org/project/lm-eval/0.4.11/))

## Installation

```bash
pip install "lm-eval[vllm]==0.4.11" ray
```

## Tasks

> **Note:** For instruct/chat models, add `--apply_chat_template` to the commands below. These flags format the prompt using the model's chat template and structure few-shot examples as multi-turn conversations, which significantly improves accuracy for instruction-tuned models. Do **not** use them with base models, as they were not trained with chat-formatted inputs.

### 1. BBH CoT Few-Shot (Full)

The full [BIG-Bench Hard (BBH)](https://github.com/EleutherAI/lm-evaluation-harness/tree/v0.4.11/lm_eval/tasks/bbh) Chain-of-Thought few-shot benchmark, consisting of 27 subtasks.

- **Few-shot:** 3
- **Metric:** `exact_match` (mean, weighted by size)

For instruct/chat models:
```bash
lm_eval --model vllm \
    --model_args pretrained=<model_name>,dtype=bfloat16 \
    --tasks bbh_cot_fewshot \
    --batch_size auto \
    --trust_remote_code \
    --num_fewshot 3 \
    --gen_kwargs "max_gen_toks=2048" \
    --log_samples \
    --output_path <output_path> \
    --apply_chat_template
```

For non-instruct (base) models:
```bash
lm_eval --model vllm \
    --model_args pretrained=<model_name>,dtype=bfloat16 \
    --tasks bbh_cot_fewshot \
    --batch_size auto \
    --trust_remote_code \
    --num_fewshot 3 \
    --gen_kwargs "max_gen_toks=2048" \
    --log_samples \
    --output_path <output_path>
```

### 2. GSM8K

[Grade School Math 8K](https://github.com/EleutherAI/lm-evaluation-harness/tree/v0.4.11/lm_eval/tasks/gsm8k) — a dataset of 8.5K grade school math word problems for evaluating multi-step mathematical reasoning.

- **Few-shot:** 5
- **Metric:** `exact_match` (flexible extract)

For instruct/chat models:
```bash
lm_eval --model vllm \
    --model_args pretrained=<model_name>,dtype=bfloat16 \
    --tasks gsm8k \
    --batch_size auto \
    --trust_remote_code \
    --num_fewshot 5 \
    --gen_kwargs "max_gen_toks=2048" \
    --log_samples \
    --output_path <output_path> \
    --apply_chat_template
```

For non-instruct (base) models:
```bash
lm_eval --model vllm \
    --model_args pretrained=<model_name>,dtype=bfloat16 \
    --tasks gsm8k \
    --batch_size auto \
    --trust_remote_code \
    --num_fewshot 5 \
    --gen_kwargs "max_gen_toks=2048" \
    --log_samples \
    --output_path <output_path>
```

### 3. MMMU (Vision-Language)

[MMMU](https://github.com/EleutherAI/lm-evaluation-harness/tree/v0.4.11/lm_eval/tasks/mmmu) — Massive Multi-discipline Multimodal Understanding benchmark. Uses the `vllm-vlm` model type for multimodal (text+image) tasks.

- **Task:** `mmmu_val_tech_and_engineering`
- **Metric:** `exact_match`

For instruct/chat models:
```bash
lm_eval --model vllm-vlm \
    --model_args pretrained=<model_name>,dtype=bfloat16 \
    --tasks mmmu_val_tech_and_engineering \
    --batch_size auto \
    --trust_remote_code \
    --apply_chat_template \
    --log_samples \
    --output_path <output_path>
```

For non-instruct (base) models:
```bash
lm_eval --model vllm-vlm \
    --model_args pretrained=<model_name>,dtype=bfloat16 \
    --tasks mmmu_val_tech_and_engineering \
    --batch_size auto \
    --trust_remote_code \
    --log_samples \
    --output_path <output_path>
```

### 4. ChartQA (Vision-Language)

[ChartQA](https://github.com/EleutherAI/lm-evaluation-harness/tree/v0.4.11/lm_eval/tasks/chartqa) — Chart Question Answering benchmark for evaluating multimodal models on chart understanding. Uses the `vllm-vlm` model type.

- **Metric:** `relaxed_overall`

For instruct/chat models:
```bash
lm_eval --model vllm-vlm \
    --model_args pretrained=<model_name>,dtype=bfloat16 \
    --tasks chartqa \
    --batch_size auto \
    --trust_remote_code \
    --apply_chat_template \
    --log_samples \
    --output_path <output_path>
```

For non-instruct (base) models:
```bash
lm_eval --model vllm-vlm \
    --model_args pretrained=<model_name>,dtype=bfloat16 \
    --tasks chartqa \
    --batch_size auto \
    --trust_remote_code \
    --log_samples \
    --output_path <output_path>
```

## Model-Specific Notes

Some models require patched `lm_eval` forks or additional sampling parameters. The table below summarizes these requirements.

| Model | Requirement | Details |
|---|---|---|
| `zai-org/chatglm3-6b` | Patched `lm_eval` | `pip install "lm-eval[vllm] @ git+https://github.com/ganeshr10/lm-evaluation-harness.git@fix-eos-token-decode-fallback" ray` — see [PR #3657](https://github.com/EleutherAI/lm-evaluation-harness/pull/3657) |
| `microsoft/Phi-3.5-vision-instruct` | Patched `lm_eval` | `pip install "lm-eval[vllm] @ git+https://github.com/ganeshr10/lm-evaluation-harness.git@fix-phi3v-support" ray` — see [PR #3651](https://github.com/EleutherAI/lm-evaluation-harness/pull/3651) |
| `OpenPipe/Qwen3-14B-Instruct` | `--gen_kwargs` | `temperature=0.6,top_p=0.95,top_k=20` |
| `Qwen/Qwen3-30B-A3B` | `--gen_kwargs` | `temperature=0.6,top_p=0.95,top_k=20` |

Example (GSM8K with Qwen3-14B-Instruct):
```bash
lm_eval --model vllm \
    --model_args pretrained=OpenPipe/Qwen3-14B-Instruct,dtype=bfloat16 \
    --tasks gsm8k \
    --batch_size auto \
    --trust_remote_code \
    --num_fewshot 5 \
    --gen_kwargs "max_gen_toks=2048,temperature=0.6,top_p=0.95,top_k=20" \
    --log_samples \
    --output_path <output_path> \
    --apply_chat_template
```

## vLLM CI Alignment

The GSM8K and ChartQA tasks are inline with the accuracy benchmarks that vLLM uses in their CI:

- [GSM8K baseline script](https://github.com/vllm-project/vllm/blob/main/.buildkite/lm-eval-harness/run-lm-eval-gsm-vllm-baseline.sh)
- [ChartQA baseline script](https://github.com/vllm-project/vllm/blob/main/.buildkite/lm-eval-harness/run-lm-eval-chartqa-vllm-vlm-baseline.sh)


## References

- [LM Evaluation Harness (GitHub)](https://github.com/EleutherAI/lm-evaluation-harness/tree/v0.4.11)
- [v0.4.11 Release](https://github.com/EleutherAI/lm-evaluation-harness/tree/v0.4.11)
- [BBH CoT Few-Shot Task Config](https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.11/lm_eval/tasks/bbh/cot_fewshot/_bbh_cot_fewshot.yaml)
- [GSM8K Task Config](https://github.com/EleutherAI/lm-evaluation-harness/tree/v0.4.11/lm_eval/tasks/gsm8k)
- [MMMU Task Config](https://github.com/EleutherAI/lm-evaluation-harness/tree/v0.4.11/lm_eval/tasks/mmmu)
- [ChartQA Task Config](https://github.com/EleutherAI/lm-evaluation-harness/tree/v0.4.11/lm_eval/tasks/chartqa)
- [vLLM GSM8K CI Script](https://github.com/vllm-project/vllm/blob/main/.buildkite/lm-eval-harness/run-lm-eval-gsm-vllm-baseline.sh)
- [vLLM ChartQA CI Script](https://github.com/vllm-project/vllm/blob/main/.buildkite/lm-eval-harness/run-lm-eval-chartqa-vllm-vlm-baseline.sh)
- [BBH Paper: Challenging BIG-Bench Tasks (arXiv:2210.09261)](https://arxiv.org/abs/2210.09261)
- [GSM8K Paper: Training Verifiers to Solve Math Word Problems (arXiv:2110.14168)](https://arxiv.org/abs/2110.14168)
- [MMMU Paper: A Massive Multi-discipline Multimodal Understanding Benchmark (arXiv:2311.16502)](https://arxiv.org/abs/2311.16502)
