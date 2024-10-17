# ******************************************************************************
# Copyright (c) 2023-2024 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

from transformers import BertTokenizer
import sys
from pathlib import Path
import torch
import random

sys.path.append(str(Path(__file__).parent.parent))
from utils import (  # noqa: 402 # noqa: F401
    TestCase,
    run_tests,
    zentorch,
    has_zentorch,
    supported_dtypes,
    reset_dynamo,
    set_seed,
)


class Test_Data:
    def __init__(self, dtype, model_name="bert-base-uncased"):
        self.dtypes = {"float32": torch.float32, "bfloat16": torch.bfloat16}
        batch_size = random.randint(1, 100)
        self.input3d = torch.randn(batch_size, 3, 224, 224).type(self.dtypes[dtype])
        input_text = "This is a sample input sentence for testing Bert Model."
        tokenizer = BertTokenizer.from_pretrained(model_name)
        input_ids = tokenizer.encode(input_text, add_special_tokens=True)
        self.input_tensor = torch.tensor(input_ids).unsqueeze(0)
