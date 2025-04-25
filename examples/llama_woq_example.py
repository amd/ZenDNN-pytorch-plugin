# ******************************************************************************
# * Copyright (c) 2023-2025 Advanced Micro Devices, Inc.
# * All rights reserved.
# ******************************************************************************

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import zentorch

print("\n" + "=" * 10 + " Llama WOQ Example Execution Started " + "=" * 10 + "\n")

# Load Tokenizer and WOQ Model
model_id = "meta-llama/Llama-3.1-8B"
safetensor_path = "<Path to Quantized Model>"
config = AutoConfig.from_pretrained(
    model_id,
    torchscript=True,
    return_dict=False,
    torch_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_config(
    config, trust_remote_code=True, torch_dtype=torch.bfloat16
)

# Load WOQ Model
print(f"Loading quantized model: {model_id}")
model = zentorch.load_quantized_model(model, safetensor_path)
model = model.eval()

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_id, trust_remote_code=True, padding_side="left", use_fast=False
)

# Prepare Inputs
print("Preparing inputs")
generate_kwargs = {
    "do_sample": False,
    "num_beams": 4,
    "max_new_tokens": 10,
    "min_new_tokens": 2,
}
prompt = "Hi, How are you today?"
print(f"Input prompt: {prompt}")

# Optimize Model with ZenTorch
print("Optimizing model with ZenTorch")
model = zentorch.llm.optimize(model, dtype=torch.bfloat16)

# Run Inference
print("Running inference")
with torch.inference_mode(), torch.no_grad(), torch.amp.autocast("cpu", enabled=True):
    model.forward = torch.compile(model.forward, backend="zentorch")
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    output = model.generate(input_ids, **generate_kwargs)

# Decode Output
gen_text = tokenizer.batch_decode(output, skip_special_tokens=True)
print(f'Generated text: {gen_text[0]!r}')

print("\n" + "=" * 10 + " Script Executed Successfully " + "=" * 10 + "\n")
