/******************************************************************************
* Copyright (c) 2023-2025 Advanced Micro Devices, Inc.
* All rights reserved.
******************************************************************************/

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

## LLAMA: Bfloat16
### Execute the following command to run inference for llama bf16 model:
```bash
python llama_bf16_example.py
```
### Output

```plain
'Hi, How are you today? I hope you are having a great day. I'
```

## LLAMA: Weight Only Quantization
Please update the following line with the correct path to your quantized model in llama_woq_example.py:
```python
safetensor_path = "<Path to Quantized Model>"
```
### Execute the following command to run inference for llama woq model:
```bash
python llama_woq_example.py
```
### Output
```plain
'Hi, How are you today? I hope you are having a great day. I'
```

## Resnet
### Execute the following command to run inference for resnet model:
```bash
python resnet_example.py
```
### Output
```plain
airliner
```
