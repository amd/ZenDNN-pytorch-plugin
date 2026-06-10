Copyright &copy; 2023-2036 Advanced Micro Devices, Inc. All rights reserved.

# Examples
Given below are some examples for running inference for various models with zentorch. Note that you may need to install additional packages in your environment if not already present. Assuming you have installed zentorch plugin in your environment, you can install the rest of the packages by running:
```bash
pip install -r requirements.txt
```

## BERT
### Execute the following command to run inference for bert model:
```bash
python bert_example.py
```

### Output
Last hidden states shape: torch.Size([3, 339, 1024])

## DLRM
### Execute the following command to run inference for dlrm model:
```bash
python dlrm_example.py
```
### Output
```plain
AUC Score: 0.5
```


## Resnet
### Execute the following command to run inference for resnet model:
```bash
python resnet_example.py
```
### Output
```plain
plane, carpenter's plane, woodworking plane
```

## BF16 LLM
vLLM inference example for a BF16 LLM on CPU with zentorch.

### Execute the following command:
```bash
python bf16_llm_example.py
```

### Requirements
- PyTorch 2.x (recommended version- 2.11.0)
- zentorch (recommended version- 2.11.0.1)
- vLLM
- transformers (recommended version > 5.0.0)

### Output
- Console: Generated text for each input prompt, printed as prompt/output pairs

## Quantization
Quantized LLM inference (LLM-Compressor, TorchAO) and DLRMv2 PT2E quantization examples. See [quantization/README.md](quantization/README.md) for details.
