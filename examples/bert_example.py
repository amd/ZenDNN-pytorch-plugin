# ******************************************************************************
# * Copyright (c) 2023-2025 Advanced Micro Devices, Inc.
# * All rights reserved.
# ******************************************************************************

import torch
import zentorch
from transformers import BertTokenizer, BertModel
from datasets import load_dataset

# Load the IMDB dataset
dataset = load_dataset("imdb", split="test")
print(dataset[0]['text'])

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained(
    "bert-large-uncased", trust_remote_code=True
)

# Load the BERT model
model_id = "google-bert/bert-large-uncased"
model = BertModel.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, trust_remote_code=True
)
model = model.eval()

# Optimize model with ZenTorch
model.forward = torch.compile(model.forward, backend="zentorch")

# Inference
with torch.inference_mode(), torch.no_grad(), \
     torch.amp.autocast("cpu", enabled=True), \
     zentorch.freezing_enabled():
    # Prepare inputs
    inputs = tokenizer(
        dataset["text"][:3], return_tensors="pt", padding=True, truncation=True
    )

    # Generate outputs
    outputs = model(**inputs)

# Get last hidden states
last_hidden_states = outputs.last_hidden_state
print("Last hidden states shape:", last_hidden_states.shape)
