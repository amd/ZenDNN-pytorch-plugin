Copyright &copy; 2026 Advanced Micro Devices, Inc. All rights reserved.

# Quantization Examples

vLLM inference examples for running quantized LLMs, plus a PT2E quantization workflow for DLRMv2.

## Folder Structure

```
quantization/
├── README.md
├── LLM-Compressor/
│   ├── da8w8_example.py
│   └── w4a16_example.py
└── torchao/
    ├── LLM/
    │   ├── da8w8_example.py
    │   └── w4a16_example.py
    └── DLRM-v2/
        └── ...
```

## LLM-Compressor

vLLM inference examples for [LLM Compressor](https://github.com/vllm-project/llm-compressor) quantized LLMs.

| File | Quantization | Model |
|------|--------------|-------|
| `LLM-Compressor/da8w8_example.py` | W8A8 (INT8 weights, INT8 activations) | `RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8` |
| `LLM-Compressor/w4a16_example.py` | W4A16 (INT4 weights, FP16 activations) | `RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w4a16` |

```bash
python LLM-Compressor/da8w8_example.py
python LLM-Compressor/w4a16_example.py
```

## TorchAO LLM

vLLM inference examples for TorchAO-quantized LLMs.

| File | Quantization | Model |
|------|--------------|-------|
| `torchao/LLM/w4a16_example.py` | W4A16 asymmetric (INT4 weights, 16-bit activations) | `amd/Qwen2.5-VL-7B-Instruct-w4a16-asym-torchao-v0.17.0` |
| `torchao/LLM/da8w8_example.py` | Dynamic da8w8 (INT8 weights, dynamic INT8 activations) | `amd/gpt-oss-20b-BF16-da8w8-torchao-v0.17.0` |

```bash
python torchao/LLM/w4a16_example.py
python torchao/LLM/da8w8_example.py
```

Each LLM script loads the pre-quantized model from the Hugging Face Hub, builds a vLLM engine, and generates text for a set of example prompts.

## TorchAO DLRM-v2

Post-training quantization (PTQ) for DLRMv2 using the PyTorch 2 Export (PT2E) framework. See [torchao/DLRM-v2/README.md](torchao/DLRM-v2/README.md) for details.

## Requirements

- PyTorch 2.x (recommended version- 2.11.0)
- zentorch (recommended version- 2.11.0.1)
- vLLM
- TorchAO (recommended version- 0.17.0) — TorchAO LLM and DLRM-v2 examples only
- transformers (recommended version > 5.0.0)

## Output

- Console: Generated text for each input prompt, printed as prompt/output pairs
