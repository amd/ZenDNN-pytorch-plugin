# ******************************************************************************
# * Copyright (c) 2023-2025 Advanced Micro Devices, Inc.
# * All rights reserved.
# ******************************************************************************

import torch
import zentorch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load Tokenizer and Model
model_id = "meta-llama/Llama-3.1-8B"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torchscript=True,
    return_dict=False,
    torch_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = model.eval()

# Prepare Inputs
generate_kwargs = {
    "do_sample": False,
    "temperature": 0.0,
    "num_beams": 4,
    "max_new_tokens": 10,
    "min_new_tokens": 2,
}

prompt = "Hi, How are you today?"

# Inference
model = zentorch.llm.optimize(model, dtype=torch.bfloat16)
with torch.inference_mode(), torch.no_grad(), \
     torch.amp.autocast("cpu", enabled=True), \
     zentorch.freezing_enabled():
    model.forward = torch.compile(model.forward, backend="zentorch")
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    output = model.generate(input_ids, **generate_kwargs)
    gen_text = tokenizer.batch_decode(output, skip_special_tokens=True)

# Display Output
print(gen_text)
