Copyright &copy; 2026 Advanced Micro Devices, Inc. All rights reserved.

# Quantization Examples

vLLM inference examples for running quantized LLMs, plus a PT2E quantization workflow for DLRMv2.

## Folder Structure

```
quantization/
├── README.md
├── LLM-Compressor/
│   ├── da8w8_example.py
│   ├── da8w4_example.py
│   └── w4a16_example.py
└── torchao/
    ├── LLM/
    │   ├── da8w8_example.py
    │   └── w4a16_example.py
    └── DLRM-v2/
        └── ...
```

## LLM-Compressor

vLLM inference examples for [LLM Compressor](https://github.com/vllm-project/llm-compressor) quantized LLMs. These scripts load an AMD-published checkpoint from the Hugging Face Hub.

| File | Quantization | Model |
|------|--------------|-------|
| `LLM-Compressor/da8w8_example.py` | W8A8 (INT8 weights, dynamic INT8 activations) | [`amd/gpt-oss-20b-BF16-w8a8-llmcompressor`](https://huggingface.co/amd/gpt-oss-20b-BF16-w8a8-llmcompressor) |
| `LLM-Compressor/da8w4_example.py` | DA8W4 (INT4 weights, dynamic INT8 activations) | [`amd/Llama-3.1-8B-Instruct-w4a16-llmcompressor`](https://huggingface.co/amd/Llama-3.1-8B-Instruct-w4a16-llmcompressor) |
| `LLM-Compressor/w4a16_example.py` | W4A16 (INT4 weights, BF16 activations) | [`amd/Llama-3.1-8B-Instruct-w4a16-llmcompressor`](https://huggingface.co/amd/Llama-3.1-8B-Instruct-w4a16-llmcompressor) |

```bash
python LLM-Compressor/da8w8_example.py
python LLM-Compressor/da8w4_example.py
python LLM-Compressor/w4a16_example.py
```

`da8w4_example.py` and `w4a16_example.py` load the same INT4 checkpoint. On Zen CPU the default is DA8W4 (`VLLM_CPU_INT4_W4A8=1`). `w4a16_example.py` forces the W4A16 path by setting this env variable:

```python
os.environ["VLLM_CPU_INT4_W4A8"] = "0"
```

Other AMD-published and Red Hat AI checkpoints are listed in the [ZenDNN v6.1 Consolidated Model Support Matrix](#zendnn-v61-consolidated-model-support-matrix). Swap the `model_id` in the script to run one of those checkpoints.

## TorchAO LLM

> **Warning:** TorchAO LLM quantization support is deprecated and will be removed in the next release. Use the LLM-Compressor examples above.

vLLM inference examples for TorchAO-quantized LLMs.

| File | Quantization | Model |
|------|--------------|-------|
| `torchao/LLM/w4a16_example.py` | W4A16 asymmetric (INT4 weights, 16-bit activations) | `amd/Qwen2.5-VL-7B-Instruct-w4a16-asym-torchao-v0.17.0` |
| `torchao/LLM/da8w8_example.py` | Dynamic da8w8 (INT8 weights, dynamic INT8 activations) | `amd/gpt-oss-20b-BF16-da8w8-torchao-v0.17.0` |

```bash
python torchao/LLM/w4a16_example.py
python torchao/LLM/da8w8_example.py
```

Both TorchAO LLM scripts run with zentorch fused FFN disabled by setting the env variable:

```python
os.environ["ZENTORCH_FUSED_FFN"] = "0"
```

Each LLM script loads the pre-quantized model from the Hugging Face Hub, builds a vLLM engine, and generates text for a set of example prompts.

## TorchAO DLRM-v2

Post-training quantization (PTQ) for DLRMv2 using the PyTorch 2 Export (PT2E) framework. See [torchao/DLRM-v2/README.md](torchao/DLRM-v2/README.md) for details.

## Requirements

- PyTorch 2.14.1 (recommended) or 2.13.0
- zentorch (recommended version- 2.14.1.0)
- vLLM
- TorchAO (recommended version- 0.17.0) — TorchAO LLM and DLRM-v2 examples only
- transformers (recommended version > 5.0.0)

## Output

- Console: Generated text for each input prompt, printed as prompt/output pairs

## ZenDNN v6.1 Consolidated Model Support Matrix

Quantized checkpoints supported with zentorch in ZenDNN v6.1, run with vLLM. Each entry is the ZenDNN release that introduced support, linked to the checkpoint on the Hugging Face Hub (published by AMD, Red Hat AI, or the original model author). `-` means not supported.

| Category | Model | DA8W8 | DA8W4 | W4A16 (WOQ-Sym-PG) | W4A16 (WOQ-Asym) |
|---|---|---|---|---|---|
| MoE | [RedHatAI/Mixtral-8x7B-Instruct-v0.1](https://huggingface.co/RedHatAI/Mixtral-8x7B-Instruct-v0.1) | [v6.1](https://huggingface.co/amd/Mixtral-8x7B-Instruct-v0.1-w8a8-llmcompressor) | [v6.1](https://huggingface.co/amd/Mixtral-8x7B-Instruct-v0.1-w4a16-llmcompressor) | - | - |
| MoE | [RedHatAI/Qwen3.6-35B-A3B](https://huggingface.co/RedHatAI/Qwen3.6-35B-A3B) | [v6.1](https://huggingface.co/amd/Qwen3.6-35B-A3B-w8a8-llmcompressor) | [v6.1](https://huggingface.co/amd/Qwen3.6-35B-A3B-w4a16-llmcompressor) | - | - |
| MoE | [RedHatAI/gemma-4-26B-A4B-it](https://huggingface.co/RedHatAI/gemma-4-26B-A4B-it) | [v6.1](https://huggingface.co/amd/gemma-4-26B-A4B-it-w8a8-llmcompressor) | [v6.1](https://huggingface.co/amd/gemma-4-26B-A4B-it-w4a16-llmcompressor) | - | - |
| MoE | [unsloth/gpt-oss-20b-BF16](https://huggingface.co/unsloth/gpt-oss-20b-BF16) | [v6.0](https://huggingface.co/amd/gpt-oss-20b-BF16-w8a8-llmcompressor) | [v6.1](https://huggingface.co/amd/gpt-oss-20b-BF16-w4a16-llmcompressor) | - | - |
| MoE | [Qwen/Qwen3-30B-A3B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507) | [v6.0](https://huggingface.co/RedHatAI/Qwen3-30B-A3B-Instruct-2507-quantized.w8a8) | [v6.1](https://huggingface.co/RedHatAI/Qwen3-30B-A3B-Instruct-2507-quantized.w4a16) | - | - |
| MoE | [ibm-granite/granite-4.0-h-small](https://huggingface.co/ibm-granite/granite-4.0-h-small) | [v6.1](https://huggingface.co/amd/granite-4.0-h-small-w8a8-llmcompressor) | [v6.1](https://huggingface.co/amd/granite-4.0-h-small-w4a16-llmcompressor) | - | - |
| MoE | [ibm-granite/granite-4.0-h-tiny](https://huggingface.co/ibm-granite/granite-4.0-h-tiny) | [v6.1](https://huggingface.co/amd/granite-4.0-h-tiny-w8a8-llmcompressor) | [v6.1](https://huggingface.co/amd/granite-4.0-h-tiny-w4a16-llmcompressor) | - | - |
| Dense LLM | [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | [v6.0](https://huggingface.co/RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8) | [v6.1](https://huggingface.co/amd/Llama-3.1-8B-Instruct-w4a16-llmcompressor) | [v6.1](https://huggingface.co/amd/Llama-3.1-8B-Instruct-w4a16-llmcompressor) | [v6.0](https://huggingface.co/amd/Llama-3.1-8B-Instruct-w4a16-asym-torchao-v0.17.0) † |
| Dense LLM | [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) | [v6.0](https://huggingface.co/RedHatAI/Qwen3.5-4B-quantized.w8a8) | [v6.1](https://huggingface.co/RedHatAI/Qwen3.5-4B-quantized.w4a16) | [v6.0](https://huggingface.co/RedHatAI/Qwen3.5-4B-quantized.w4a16) | - |
| Dense LLM | [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) | [v6.0](https://huggingface.co/RedHatAI/Qwen3.5-9B-quantized.w8a8) | [v6.1](https://huggingface.co/RedHatAI/Qwen3.5-9B-quantized.w4a16) | [v6.0](https://huggingface.co/RedHatAI/Qwen3.5-9B-quantized.w4a16) | [v6.0](https://huggingface.co/amd/Qwen3.5-9B-w4a16-asym-torchao-v0.17.0) † |
| Dense LLM | [microsoft/Phi-4-reasoning-plus](https://huggingface.co/microsoft/Phi-4-reasoning-plus) | [v6.1](https://huggingface.co/amd/Phi-4-reasoning-plus-w8a8-llmcompressor) | [v6.1](https://huggingface.co/amd/Phi-4-reasoning-plus-w4a16-llmcompressor) | [v6.1](https://huggingface.co/amd/Phi-4-reasoning-plus-w4a16-llmcompressor) | [v6.0](https://huggingface.co/amd/Phi-4-reasoning-plus-w4a16-asym-torchao-v0.17.0) † |
| Dense LLM | [microsoft/Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct) | [v6.0](https://huggingface.co/RedHatAI/Phi-4-mini-instruct-quantized.w8a8) | [v6.1](https://huggingface.co/amd/Phi-4-mini-instruct-w4a16-llmcompressor) | [v6.1](https://huggingface.co/amd/Phi-4-mini-instruct-w4a16-llmcompressor) | - |
| Dense LLM | [RedHatAI/Mistral-Small-3.1-24B-Instruct-2503](https://huggingface.co/RedHatAI/Mistral-Small-3.1-24B-Instruct-2503) | [v6.1](https://huggingface.co/RedHatAI/Mistral-Small-3.1-24B-Instruct-2503-quantized.w8a8) | [v6.1](https://huggingface.co/RedHatAI/Mistral-Small-3.1-24B-Instruct-2503-quantized.w4a16) | [v6.1](https://huggingface.co/RedHatAI/Mistral-Small-3.1-24B-Instruct-2503-quantized.w4a16) | - |
| Dense LLM | [RedHatAI/gemma-4-12B-it](https://huggingface.co/RedHatAI/gemma-4-12B-it) | [v6.1](https://huggingface.co/amd/gemma-4-12B-it-w8a8-llmcompressor) | [v6.1](https://huggingface.co/amd/gemma-4-12B-it-w4a16-llmcompressor) | [v6.1](https://huggingface.co/amd/gemma-4-12B-it-w4a16-llmcompressor) | - |
| Dense LLM | [RedHatAI/Muse-Glimmer-30B](https://huggingface.co/RedHatAI/Muse-Glimmer-30B) | [v6.1](https://huggingface.co/amd/Muse-Glimmer-30B-w8a8-llmcompressor) | [v6.1](https://huggingface.co/RedHatAI/Muse-Glimmer-30B-INT4) | [v6.1](https://huggingface.co/RedHatAI/Muse-Glimmer-30B-INT4) | - |
| Dense LLM | [RedHatAI/Qwen3.8-27B](https://huggingface.co/RedHatAI/Qwen3.8-27B) | [v6.1](https://huggingface.co/amd/Qwen3.8-27B-w8a8-llmcompressor) | [v6.1](https://huggingface.co/RedHatAI/Qwen3.8-27B-INT4) | [v6.1](https://huggingface.co/RedHatAI/Qwen3.8-27B-INT4) | - |
| VLM | [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) | [v6.1](https://huggingface.co/amd/Qwen3-VL-8B-Instruct-w8a8-llmcompressor) | [v6.1](https://huggingface.co/amd/Qwen3-VL-8B-Instruct-w4a16-llmcompressor) | [v6.1](https://huggingface.co/amd/Qwen3-VL-8B-Instruct-w4a16-llmcompressor) | [v6.0](https://huggingface.co/amd/Qwen3-VL-8B-Instruct-w4a16-asym-torchao-v0.17.0) † |
| Audio | [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) | [v6.1](https://huggingface.co/RedHatAI/whisper-large-v3-quantized.w8a8) | [v6.1](https://huggingface.co/RedHatAI/whisper-large-v3-quantized.w4a16) | [v6.1](https://huggingface.co/RedHatAI/whisper-large-v3-quantized.w4a16) | - |
| Audio | [openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) | [v6.1](https://huggingface.co/RedHatAI/whisper-large-v3-turbo-quantized.w8a8) | [v6.1](https://huggingface.co/RedHatAI/whisper-large-v3-turbo-quantized.w4a16) | [v6.1](https://huggingface.co/RedHatAI/whisper-large-v3-turbo-quantized.w4a16) | - |

† TorchAO checkpoint. TorchAO support is deprecated and will be removed in the next release.

**Notes:**

1. **DA8W8** is dynamic 8-bit activation and 8-bit weight quantization. In v6.0 it was produced with TorchAO or LLM-Compressor. In v6.1, LLM-Compressor is the forward direction and TorchAO is in maintenance mode.
2. **DA8W4** is dynamic 8-bit activation and 4-bit weight quantization, a new scheme in v6.1. It is distinct from W4A16 weight-only quantization (16-bit activations).
3. **W4A16 (WOQ-Sym-PG)** is 4-bit weight-only, symmetric, per-group quantization, and includes both TorchAO and LLM-Compressor artifacts. **W4A16 (WOQ-Asym)** is asymmetric weight-only quantization.
