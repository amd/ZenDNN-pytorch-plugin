# ******************************************************************************
# * Copyright (c) 2023-2025 Advanced Micro Devices, Inc.
# * All rights reserved.
# ******************************************************************************

# Import dependencies
from dlrm_model import DLRMMLPerf
import torch
import numpy as np
import zentorch
import random
from sklearn.metrics import roc_auc_score

# Basic setup for reproducibility
np.random.seed(123)
random.seed(123)
torch.manual_seed(123)

# Initialize the model
DEFAULT_INT_NAMES = [f'int_{i}' for i in range(13)]
model = DLRMMLPerf(
    embedding_dim=128,
    num_embeddings_pool=[
        40000000, 39060, 17295, 7424, 20265, 3, 7122, 1543, 63, 40000000,
        3067956, 405282, 10, 2209, 11938, 155, 4, 976, 14, 40000000,
        40000000, 40000000, 590152, 12973, 108, 36
    ],
    dense_in_features=len(DEFAULT_INT_NAMES),
    dense_arch_layer_sizes=[512, 256, 128],
    over_arch_layer_sizes=[1024, 1024, 512, 256, 1],
    dcn_num_layers=3,
    dcn_low_rank_dim=512,
    use_int8=False,
    use_bf16=True
).bfloat16()

# Prepare Inputs
multi_hot = [3, 2, 1, 2, 6, 1, 1, 1, 1, 7, 3, 8, 1, 6, 9, 5, 1, 1,
             1, 12, 100, 27, 10, 3, 1, 1]
batchsize = 32768
densex = torch.randn((batchsize, 13), dtype=torch.float).to(torch.bfloat16)
index = [torch.ones((batchsize * h), dtype=torch.long) for h in multi_hot]
offset = [torch.arange(0, (batchsize + 1) * h, h, dtype=torch.long) for h in multi_hot]

# Inference with zentorch optimization
model = torch.compile(model, backend="zentorch")

with torch.inference_mode(), torch.no_grad(), \
     torch.amp.autocast("cpu", enabled=True), \
     zentorch.freezing_enabled():
    out = model(densex, index, offset)

# Simulating labels
true_labels = torch.randint(0, 2, (32768,))
predicted_probabilities = out.to(torch.float32).cpu().detach().numpy().reshape(-1)
true_labels = true_labels.cpu().detach().numpy()

# Calculate AUC
auc_score = roc_auc_score(true_labels, predicted_probabilities)
print(f"AUC Score: {auc_score}")
