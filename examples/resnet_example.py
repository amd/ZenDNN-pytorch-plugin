# ******************************************************************************
# * Copyright (c) 2023-2025 Advanced Micro Devices, Inc.
# * All rights reserved.
# ******************************************************************************

import torch
import zentorch
from transformers import AutoImageProcessor, ResNetForImageClassification
from PIL import Image
import urllib.request

# Load Processor and Model
processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
model = ResNetForImageClassification.from_pretrained(
    "microsoft/resnet-50", torch_dtype=torch.bfloat16
)

# Prepare Inputs
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
model.forward = torch.compile(model.forward, backend="zentorch")

# Run Inference
with torch.inference_mode(), torch.no_grad(), \
     torch.amp.autocast("cpu", enabled=True), \
     zentorch.freezing_enabled():
    logits = model(**inputs).logits

predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])
