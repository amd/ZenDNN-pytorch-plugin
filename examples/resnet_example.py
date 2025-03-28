# ******************************************************************************
# * Copyright (c) 2023-2025 Advanced Micro Devices, Inc.
# * All rights reserved.
# ******************************************************************************

import torch
import zentorch
from transformers import AutoImageProcessor, ResNetForImageClassification
from PIL import Image
import urllib.request

print("\n" + "=" * 10 + " ResNet Example Execution Started " + "=" * 10 + "\n")

# Load Processor and Model
model_id = "microsoft/resnet-50"
print(f"Loading RESNET model: {model_id}")
processor = AutoImageProcessor.from_pretrained(model_id, use_fast=False)
model = ResNetForImageClassification.from_pretrained(
    model_id, torch_dtype=torch.bfloat16
)

# Prepare Inputs
print("Downloading and loading image")
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/EliSchwartz/"
    "imagenet-sample-images/master/n03954731_plane.JPEG",
    "airplane.jpeg"
)
image = Image.open("airplane.jpeg")

inputs = processor(image, return_tensors="pt")

# Convert inputs to BF16
inputs = {k: v.to(torch.bfloat16) for k, v in inputs.items()}

# Optimize Model with ZenTorch
print("Optimizing model with ZenTorch")
model.forward = torch.compile(model.forward, backend="zentorch")

# Run Inference
print("Running inference")
with torch.inference_mode(), torch.no_grad(), \
     torch.amp.autocast("cpu", enabled=True), \
     zentorch.freezing_enabled():
    logits = model(**inputs).logits

# Get prediction
predicted_label = logits.argmax(-1).item()
print(f"Predicted label: {model.config.id2label[predicted_label]}")

print("\n" + "=" * 10 + " Script Executed Successfully " + "=" * 10 + "\n")
