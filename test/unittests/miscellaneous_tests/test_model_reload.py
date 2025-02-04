# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import unittest
import torch
import sys
import json
from pathlib import Path
from torch import nn
from safetensors.torch import save_file

sys.path.append(str(Path(__file__).parent.parent))
from unittest_utils import (  # noqa: 402
    Zentorch_TestCase,
    has_zentorch,
    run_tests,
    zentorch,
    skip_test_pt_2_1,
)


class Custom_Model(nn.Module):
    def __init__(self, input_size, output_size):
        super(Custom_Model, self).__init__()
        self.layer1 = nn.Linear(input_size, output_size)

    def forward(self, a, b):
        final_result = self.layer1(a, b)
        return final_result


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
@unittest.skipIf(skip_test_pt_2_1, "Pattern matcher disabled for Torch < 2.2")
class Test_Model_Reload(Zentorch_TestCase):
    @torch.inference_mode()
    def test_wrong_model_name(self):
        model = Custom_Model(40, 30).eval()
        config = {
            "architectures": ["dummy_config"],
        }
        with open("./test/data/config.json", "w") as json_file:
            json.dump(config, json_file, indent=4)
        with self.assertRaises(ValueError) as context:
            model = zentorch.load_quantized_model(model, "./test/data/")
        self.assertTrue(
            "This quantized model with model_architecture = dummy_config "
            + "is not yet supported with zentorch's reload feature."
            in str(context.exception)
        )

    @torch.inference_mode()
    def test_without_config_file(self):
        model = Custom_Model(40, 30).eval()
        with self.assertRaises(FileNotFoundError) as context:
            model = zentorch.load_quantized_model(model, "./test/data/")
        self.assertTrue(
            "No JSON file titled 'config.json' found at this location :"
            in str(context.exception)
        )

    @torch.inference_mode()
    def test_without_safetensors_file(self):
        config = {
            "architectures": ["ChatGLMModel"],
            "quantization_config": {
                "weight": 0,
            },
        }
        with open("./test/data/config.json", "w") as json_file:
            json.dump(config, json_file, indent=4)
        model = Custom_Model(40, 30).eval()
        with self.assertRaises(FileNotFoundError) as context:
            model = zentorch.load_quantized_model(model, "./test/data/")
        self.assertTrue(
            "No file ending with .safetensors found at this location:"
            in str(context.exception)
        )

    @torch.inference_mode()
    def test_without_quantization_config(self):
        model = Custom_Model(40, 30).eval()
        weights = {
            "layer1.weight": torch.randn(10, 10),
            "layer1.bias": torch.randn(10),
        }
        save_file(weights, "./test/data/model_weights.safetensors")
        config = {
            "architectures": ["ChatGLMModel"],
        }
        with open("./test/data/config.json", "w") as json_file:
            json.dump(config, json_file, indent=4)
        with self.assertRaises(KeyError) as context:
            model = zentorch.load_quantized_model(model, "./test/data/")
        self.assertTrue(
            "quantization_config is not available" in str(context.exception)
        )

    @torch.inference_mode()
    def test_with_wrong_group_size(self):
        model = Custom_Model(40, 30).eval()
        weights = {
            "layer1.weight": torch.randn(10, 10),
            "layer1.bias": torch.randn(10),
        }
        save_file(weights, "./test/data/model_weights.safetensors")
        config = {
            "architectures": ["ChatGLMModel"],
            "quantization_config": {
                "bits": 4,
                "group_size": -3,
                "quant_method": "awq",
                "pack_method": "order",
                "zero_point": False,
            },
            "torch_dtype": "bfloat16",
        }
        with open("./test/data/config.json", "w") as json_file:
            json.dump(config, json_file, indent=4)
        with self.assertRaises(NotImplementedError) as context:
            model = zentorch.load_quantized_model(model, "./test/data/")
        self.assertTrue(
            "zentorch does not support group_size " in str(context.exception)
        )

    @torch.inference_mode()
    def test_with_wrong_config(self):
        model = Custom_Model(40, 30).eval()
        weights = {
            "layer1.weight": torch.randn(10, 10),
            "layer1.bias": torch.randn(10),
        }
        save_file(weights, "./test/data/model_weights.safetensors")
        config = {
            "architectures": ["ChatGLMModel"],
            "quantization_config": {
                "bits": 5,
                "group_size": -1,
                "quant_method": "awq",
                "pack_method": "order",
                "zero_point": False,
            },
            "torch_dtype": "bfloat16",
        }
        with open("./test/data/config.json", "w") as json_file:
            json.dump(config, json_file, indent=4)
        with self.assertRaises(NotImplementedError) as context:
            model = zentorch.load_quantized_model(model, "./test/data/")
        self.assertTrue(
            "zentorch has not yet implemented support for" in str(context.exception)
        )

    @torch.inference_mode()
    def test_with_wrong_weight_dtype_woq(self):
        model = Custom_Model(40, 30).eval()
        weights = {
            "layer1.qweight": torch.randn(10, 10),
            "layer1.bias": torch.randn(10),
        }
        save_file(weights, "./test/data/model_weights.safetensors")
        config = {
            "architectures": ["ChatGLMModel"],
            "quantization_config": {
                "bits": 4,
                "group_size": -1,
                "quant_method": "awq",
                "pack_method": "order",
                "zero_point": False,
            },
            "torch_dtype": "bfloat16",
        }

        with open("./test/data/config.json", "w") as json_file:
            json.dump(config, json_file, indent=4)
        with self.assertRaises(NotImplementedError) as context:
            model = zentorch.load_quantized_model(model, "./test/data/")
        self.assertTrue(
            "zentorch has not yet implemented support for qweights packed into "
            in str(context.exception)
        )

    @torch.inference_mode()
    def test_without_weight(self):
        model = Custom_Model(40, 30).eval()
        weights = {
            "layer1.bias": torch.randn(10, 10),
        }
        save_file(weights, "./test/data/model_weights.safetensors")
        config = {
            "architectures": ["ChatGLMModel"],
            "quantization_config": {
                "bits": 4,
                "group_size": -1,
                "quant_method": "awq",
                "pack_method": "order",
                "zero_point": False,
            },
            "torch_dtype": "bfloat16",
        }

        with open("./test/data/config.json", "w") as json_file:
            json.dump(config, json_file, indent=4)
        with self.assertRaises(ValueError) as context:
            model = zentorch.load_quantized_model(model, "./test/data/")
        self.assertTrue(
            "Encountered a non-standard weight_key" in str(context.exception)
        )

    @torch.inference_mode()
    def test_with_wrong_arg(self):
        model = Custom_Model(40, 30).eval()
        config = {
            "architectures": ["ChatGLMModel"],
            "quantization_config": {
                "bits": 4,
                "group_size": -1,
                "quant_method": "awq",
                "pack_method": "order",
                "zero_point": False,
            },
            "torch_dtype": "bfloat16",
        }
        with open("./test/data/config.json", "w") as json_file:
            json.dump(config, json_file, indent=4)
        with self.assertRaises(NotImplementedError) as context:
            model = zentorch.load_quantized_model(model, "./test/data/", "quark_safe")
        self.assertTrue(
            "zentorch has not yet implemented support for the models exported with "
            in str(context.exception)
        )


if __name__ == "__main__":
    run_tests()
