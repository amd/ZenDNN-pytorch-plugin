# ******************************************************************************
# Copyright (c) 2024 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

from torch.testing._internal.common_utils import TestCase, run_tests
import torch
import unittest
import random
import copy
from parameterized import parameterized
from torchvision import models
from transformers import BertModel, BertTokenizer

try:
    import zentorch

    HAS_ZENTORCH = True
except ImportError:
    HAS_ZENTORCH = False

supported_dtypes = [("float32")]
if zentorch._C.is_bf16_supported():
    supported_dtypes.append(("bfloat16"))
else:
    print(
        "Warning: Skipping Bfloat16 Testcases since they \
are not supported on this hardware"
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


@unittest.skipIf(not HAS_ZENTORCH, "ZENTORCH is not installed")
class Test_CNN_Models(TestCase):
    @parameterized.expand(supported_dtypes)
    def test_resnet18(self, dtype):
        data = Test_Data(dtype)
        model = models.__dict__["resnet18"](pretrained=True).eval()
        inductor_model = copy.deepcopy(model)
        torch._dynamo.reset()
        zentorch_graph = torch.compile(model, backend="zentorch", dynamic=False)
        torch._dynamo.reset()
        inductor_graph = torch.compile(inductor_model, backend="inductor")

        with torch.no_grad():
            if dtype == "bfloat16":
                with torch.cpu.amp.autocast():
                    zentorch_graph_output = zentorch_graph(data.input3d)
                    inductor_graph_output = inductor_graph(data.input3d)
            else:
                zentorch_graph_output = zentorch_graph(data.input3d)
                inductor_graph_output = inductor_graph(data.input3d)

        self.assertEqual(inductor_graph_output, zentorch_graph_output)


@unittest.skipIf(not HAS_ZENTORCH, "ZENTORCH is not installed")
class Test_Bert_Models(TestCase):
    @parameterized.expand(supported_dtypes)
    def test_bert_base(self, dtype):
        data = Test_Data(dtype, "bert-large-uncased")
        native_model = BertModel.from_pretrained("bert-large-uncased").eval()
        inductor_model = copy.deepcopy(native_model)
        torch._dynamo.reset()
        zentorch_graph = torch.compile(native_model, backend="zentorch")
        torch._dynamo.reset()
        inductor_graph = torch.compile(inductor_model, backend="inductor")

        with torch.no_grad():
            if dtype == "bfloat16":
                with torch.cpu.amp.autocast():
                    zentorch_graph_output = zentorch_graph(data.input_tensor)
                    inductor_graph_output = inductor_graph(data.input_tensor)
            else:
                zentorch_graph_output = zentorch_graph(data.input_tensor)
                inductor_graph_output = inductor_graph(data.input_tensor)

        self.assertEqual(
            zentorch_graph_output, inductor_graph_output, atol=1e-2, rtol=1e-5
        )


if __name__ == "__main__":
    run_tests()
