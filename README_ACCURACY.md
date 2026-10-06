# LM Evaluation Harness - Accuracy Benchmarking

## Overview

We have migrated from the custom LM Eval implementation (`ZenDNN_tools/LM_Evaluation`) to the open-source [EleutherAI LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/v0.4.12).

**Version:** [v0.4.12](https://github.com/EleutherAI/lm-evaluation-harness/releases/tag/v0.4.12) ([PyPI](https://pypi.org/project/lm-eval/0.4.12/))

## Installation

```bash
pip install lm-eval==0.4.12
```

Audio models run under [LMMS-Eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) instead:

```bash
pip install -e "git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git#egg=lmms_eval"
pip install "torchcodec==0.11.*" --index-url=https://download.pytorch.org/whl/cpu
pip install editdistance jiwer librosa soundfile
```

## Tasks

> **Note:** For instruct/chat models, add `--apply_chat_template` to the commands below. These flags format the prompt using the model's chat template and structure few-shot examples as multi-turn conversations, which significantly improves accuracy for instruction-tuned models. Do **not** use them with base models, as they were not trained with chat-formatted inputs.

> **Recommended:** Enable FP8 KV-cache quantization with `--kv-cache-dtype fp8_e4m3`. This stores the KV cache in `fp8_e4m3`.

### 1. GSM8K

[Grade School Math 8K](https://github.com/EleutherAI/lm-evaluation-harness/tree/v0.4.12/lm_eval/tasks/gsm8k) — a dataset of 8.5K grade school math word problems for evaluating multi-step mathematical reasoning.

- **Few-shot:** 5
- **Metric:** `exact_match` (flexible extract)

For instruct/chat models:
```bash
lm_eval --model vllm \
    --model_args pretrained=<model_name>,dtype=bfloat16,max_model_len=4096 \
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
    --model_args pretrained=<model_name>,dtype=bfloat16,max_model_len=4096 \
    --tasks gsm8k \
    --batch_size auto \
    --trust_remote_code \
    --num_fewshot 5 \
    --gen_kwargs "max_gen_toks=2048" \
    --log_samples \
    --output_path <output_path>
```

### 2. MMMU (Vision-Language)

[MMMU](https://github.com/EleutherAI/lm-evaluation-harness/tree/v0.4.12/lm_eval/tasks/mmmu) — Massive Multi-discipline Multimodal Understanding benchmark. Uses the `vllm-vlm` model type for multimodal (text+image) tasks.

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

### 3. ChartQA (Vision-Language)

[ChartQA](https://github.com/EleutherAI/lm-evaluation-harness/tree/v0.4.12/lm_eval/tasks/chartqa) — Chart Question Answering benchmark for evaluating multimodal models on chart understanding. Uses the `vllm-vlm` model type.

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

### 4. LibriSpeech (Audio)

[LibriSpeech](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/lmms_eval/tasks/librispeech) — read-speech transcription benchmark, run through LMMS-Eval with the `whisper_vllm` model type. The `librispeech` group covers the dev-clean, dev-other, test-clean and test-other splits.

- **Metric:** `wer` (lower is better)

```bash
lmms-eval --model=whisper_vllm \
    --model_args="pretrained=RedHatAI/whisper-large-v3-turbo-quantized.w8a8,dtype=bfloat16" \
    --tasks librispeech \
    --batch_size 64 \
    --output_path <output_path>
```

## Model-Specific Notes

Some models require additional model arguments. The table below summarizes these requirements.

| Model | Requirement | Details |
|---|---|---|
| `Qwen/Qwen3.5-4B`<br>`Qwen/Qwen3.5-9B`<br>`RedHatAI/Qwen3.6-35B-A3B`<br>`RedHatAI/Qwen3.5-4B-quantized.w8a8`<br>`RedHatAI/Qwen3.5-9B-quantized.w8a8` | `--model_args` + `--apply_chat_template` | `pretrained=<model_name>,dtype=bfloat16,max_model_len=4096,language_model_only=True,enable_thinking=False,mamba_ssm_cache_dtype=bfloat16` |
| `RedHatAI/Mistral-Small-3.1-24B-Instruct-2503`<br>`RedHatAI/Mistral-Small-3.1-24B-Instruct-2503-quantized.w8a8` | `--model_args` | `pretrained=<model_name>,dtype=bfloat16,language_model_only=True,tokenizer_mode=mistral,config_format=mistral`<br><br>Without `config_format=mistral` the checkpoint loads but GSM8K scores 0. |

Example (GSM8K with Qwen3-30B-A3B-Instruct-2507):
```bash
lm_eval --model vllm \
    --model_args pretrained=Qwen/Qwen3-30B-A3B-Instruct-2507,dtype=bfloat16,max_model_len=4096 \
    --tasks gsm8k \
    --batch_size auto \
    --trust_remote_code \
    --num_fewshot 5 \
    --gen_kwargs "max_gen_toks=2048" \
    --log_samples \
    --output_path <output_path> \
    --apply_chat_template
```

## vLLM CI Alignment

The GSM8K and ChartQA tasks are inline with the accuracy benchmarks that vLLM uses in their CI:

- [GSM8K baseline script](https://github.com/vllm-project/vllm/blob/main/.buildkite/lm-eval-harness/run-lm-eval-gsm-vllm-baseline.sh)
- [ChartQA baseline script](https://github.com/vllm-project/vllm/blob/main/.buildkite/lm-eval-harness/run-lm-eval-chartqa-vllm-vlm-baseline.sh)


## References

- [LM Evaluation Harness (GitHub)](https://github.com/EleutherAI/lm-evaluation-harness/tree/v0.4.12)
- [v0.4.12 Release](https://github.com/EleutherAI/lm-evaluation-harness/tree/v0.4.12)
- [GSM8K Task Config](https://github.com/EleutherAI/lm-evaluation-harness/tree/v0.4.12/lm_eval/tasks/gsm8k)
- [MMMU Task Config](https://github.com/EleutherAI/lm-evaluation-harness/tree/v0.4.12/lm_eval/tasks/mmmu)
- [ChartQA Task Config](https://github.com/EleutherAI/lm-evaluation-harness/tree/v0.4.12/lm_eval/tasks/chartqa)
- [LMMS-Eval (GitHub)](https://github.com/EvolvingLMMs-Lab/lmms-eval)
- [LibriSpeech Task Config](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/lmms_eval/tasks/librispeech)
- [vLLM GSM8K CI Script](https://github.com/vllm-project/vllm/blob/main/.buildkite/lm-eval-harness/run-lm-eval-gsm-vllm-baseline.sh)
- [vLLM ChartQA CI Script](https://github.com/vllm-project/vllm/blob/main/.buildkite/lm-eval-harness/run-lm-eval-chartqa-vllm-vlm-baseline.sh)
- [GSM8K Paper: Training Verifiers to Solve Math Word Problems (arXiv:2110.14168)](https://arxiv.org/abs/2110.14168)
- [MMMU Paper: A Massive Multi-discipline Multimodal Understanding Benchmark (arXiv:2311.16502)](https://arxiv.org/abs/2311.16502)
