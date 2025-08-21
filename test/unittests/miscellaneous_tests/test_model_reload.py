# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import unittest
import torch
import sys
import json
import os
from pathlib import Path
from torch import nn
from safetensors.torch import save_file
import shutil

sys.path.append(str(Path(__file__).parent.parent))
from unittest_utils import (  # noqa: 402
    Zentorch_TestCase,
    has_zentorch,
    run_tests,
    zentorch,
    skip_test_pt_2_1,
)

path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


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
    def setUp(self):
        super().setUp()
        self.data_path = os.path.join(path, self._testMethodName)
        os.makedirs(self.data_path, exist_ok=True)
        self.config_file_path = os.path.join(self.data_path, "config.json")
        self.tokenizer_file_path = os.path.join(self.data_path, "tokenizer.json")
        self.safetensors_file_path = os.path.join(self.data_path, "model_weights.safetensors")

    def tearDown(self):
        shutil.rmtree(self.data_path)

    @torch.inference_mode()
    def test_without_config_file(self):
        weights = {
            "layer1.weight": torch.randn(10, 10),
            "layer1.bias": torch.randn(10),
        }
        save_file(weights, self.safetensors_file_path)
        model = Custom_Model(40, 30).eval()
        with self.assertRaises(FileNotFoundError) as context:
            model = zentorch.load_quantized_model(model, self.data_path)
        self.assertTrue(
            "No JSON file found at this location:" in str(context.exception)
        )

    @torch.inference_mode()
    def test_without_safetensors_file(self):
        config = {
            "architectures": ["ChatGLMModel"],
            "quantization_config": {
                "weight": 0,
            },
        }
        with open(self.config_file_path, "w") as json_file:
            json.dump(config, json_file, indent=4)
        model = Custom_Model(40, 30).eval()
        with self.assertRaises(FileNotFoundError) as context:
            model = zentorch.load_quantized_model(model, self.data_path)
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
        save_file(weights, self.safetensors_file_path)
        config = {
            "architectures": ["ChatGLMModel"],
        }

        with open(self.config_file_path, "w") as json_file:
            json.dump(config, json_file, indent=4)
        # HF format requires multiple config files to be present
        tokenizer = {}
        with open(self.tokenizer_file_path, "w") as json_file:
            json.dump(tokenizer, json_file, indent=4)
        with self.assertRaises(KeyError) as context:
            model = zentorch.load_quantized_model(model, self.data_path)
        self.assertTrue(
            "quantization_config is not available" in str(context.exception)
        )

    @torch.inference_mode()
    def test_without_global_quant_config(self):
        model = Custom_Model(40, 30).eval()
        weights = {
            "layer1.weight": torch.randn(10, 10),
            "layer1.bias": torch.randn(10),
        }
        save_file(weights, self.safetensors_file_path)
        config = {"architectures": ["ChatGLMModel"], "quantization_config": {}}

        with open(self.config_file_path, "w") as json_file:
            json.dump(config, json_file, indent=4)
        # HF format requires multiple config files to be present
        tokenizer = {}
        with open(self.tokenizer_file_path, "w") as json_file:
            json.dump(tokenizer, json_file, indent=4)
        with self.assertRaises(KeyError) as context:
            model = zentorch.load_quantized_model(model, self.data_path)
        self.assertTrue(
            "global_quant_config is not available." in str(context.exception)
        )

    @torch.inference_mode()
    def test_with_wrong_group_size(self):
        model = Custom_Model(40, 30).eval()
        weights = {
            "layer1.weight": torch.randn(10, 10),
            "layer1.bias": torch.randn(10),
        }
        save_file(weights, self.safetensors_file_path)
        config = {
            "architectures": ["ChatGLMModel"],
            "quantization_config": {
                "export": {"pack_method": "order"},
                "global_quant_config": {
                    "input_tensors": {},
                    "weight": {
                        "ch_axis": 0,
                        "dtype": "int8",
                        "group_size": -3,
                        "is_dynamic": False,
                        "observer_cls": "PerChannelMinMaxObserver",
                        "qscheme": "per_channel",
                        "round_method": "half_even",
                        "scale_type": "Bfloat",
                        "symmetric": True,
                    },
                },
                "layer_quant_config": {},
            },
            "torch_dtype": "float32",
        }
        with open(self.config_file_path, "w") as json_file:
            json.dump(config, json_file, indent=4)
        # HF format requires multiple config files to be present
        tokenizer = {}
        with open(self.tokenizer_file_path, "w") as json_file:
            json.dump(tokenizer, json_file, indent=4)
        with self.assertRaises(NotImplementedError) as context:
            model = zentorch.load_quantized_model(model, self.data_path)
        self.assertTrue(
            "Zentorch does not support group_size " in str(context.exception)
        )

    @torch.inference_mode()
    def test_with_wrong_config(self):
        model = Custom_Model(40, 30).eval()
        weights = {
            "layer1.weight": torch.randn(10, 10),
            "layer1.bias": torch.randn(10),
        }
        save_file(weights, self.safetensors_file_path)
        config = {
            "architectures": ["ChatGLMModel"],
            "quantization_config": {
                "export": {"pack_method": "order"},
                "global_quant_config": {
                    "input_tensors": {},
                    "weight": {
                        "ch_axis": 0,
                        "dtype": "float",
                        "group_size": None,
                        "is_dynamic": False,
                        "observer_cls": "PerChannelMinMaxObserver",
                        "qscheme": "per_channel",
                        "round_method": "half_even",
                        "scale_type": "Bfloat",
                        "symmetric": True,
                    },
                },
                "layer_quant_config": {},
            },
            "torch_dtype": "float32",
        }
        with open(self.config_file_path, "w") as json_file:
            json.dump(config, json_file, indent=4)
        # HF format requires multiple config files to be present
        tokenizer = {}
        with open(self.tokenizer_file_path, "w") as json_file:
            json.dump(tokenizer, json_file, indent=4)
        with self.assertRaises(NotImplementedError) as context:
            model = zentorch.load_quantized_model(model, self.data_path)
        self.assertTrue(
            "Zentorch has not yet implemented support for" in str(context.exception)
        )

    @torch.inference_mode()
    def test_with_wrong_weight_dtype_woq(self):
        model = Custom_Model(40, 30).eval()
        weights = {
            "layer1.weight": torch.randn(10, 10),
            "layer1.weight_scale": torch.randn(1, 10),
            "layer1.weight_zero_point": torch.randint(0, 15, (1, 10)),
            "layer1.bias": torch.randn(10),
        }
        save_file(weights, self.safetensors_file_path)
        config = {
            "architectures": ["ChatGLMModel"],
            "quantization_config": {
                "export": {"pack_method": "order"},
                "exclude": [],
                "global_quant_config": {
                    "input_tensors": {},
                    "weight": {
                        "ch_axis": 0,
                        "dtype": "int4",
                        "group_size": None,
                        "is_dynamic": False,
                        "observer_cls": "PerChannelMinMaxObserver",
                        "qscheme": "per_channel",
                        "round_method": "half_even",
                        "scale_type": "Bfloat",
                        "symmetric": True,
                    },
                },
                "layer_quant_config": {},
            },
            "torch_dtype": "bfloat16",
        }
        with open(self.config_file_path, "w") as json_file:
            json.dump(config, json_file, indent=4)
        # HF format requires multiple config files to be present
        tokenizer = {}
        with open(self.tokenizer_file_path, "w") as json_file:
            json.dump(tokenizer, json_file, indent=4)
        with self.assertRaises(NotImplementedError) as context:
            model = zentorch.load_quantized_model(model, self.data_path)
        self.assertTrue(
            "Zentorch has not yet implemented support for weights packed into "
            in str(context.exception)
        )

    @torch.inference_mode()
    def test_with_wrong_arg(self):
        model = Custom_Model(40, 30).eval()
        config = {
            "architectures": ["ChatGLMModel"],
            "quantization_config": {
                "export": {"pack_method": "order"},
                "global_quant_config": {
                    "input_tensors": {},
                    "weight": {
                        "ch_axis": 0,
                        "dtype": "int8",
                        "group_size": None,
                        "is_dynamic": False,
                        "observer_cls": "PerChannelMinMaxObserver",
                        "qscheme": "per_channel",
                        "round_method": "half_even",
                        "scale_type": "float",
                        "symmetric": True,
                    },
                },
                "layer_quant_config": {},
            },
            "torch_dtype": "float32",
        }
        with open(self.config_file_path, "w") as json_file:
            json.dump(config, json_file, indent=4)
        with self.assertRaises(NotImplementedError) as context:
            model = zentorch.load_quantized_model(model, self.data_path, "quark_safe")
        self.assertTrue(
            "Zentorch has not yet implemented support for the models exported with "
            in str(context.exception)
        )

    @torch.inference_mode()
    def test_mismatch_key_recsys(self):
        model = Custom_Model(40, 30).eval()
        weights = {
            "layer1.qweight": torch.randn(10, 10),
            "layer1.weight_scale": torch.randn(1, 10),
            "layer1.bias": torch.randn(10),
        }
        save_file(weights, self.safetensors_file_path)
        config = {
            "structure": {
                "sparse_arch": {
                    "embed_col": {
                        "embedding_bags": {
                            "0": {
                                "name": "sparse_arch.embed.embedding_bags.0",
                                "type": "QuantEmbeddingBag",
                                "weight": "sparse_arch.em.embedding_bags.0.weight",
                                "weight_quant": {
                                    "dtype": "uint4",
                                    "is_dynamic": False,
                                    "qscheme": "per_channel",
                                    "ch_axis": 0,
                                    "group_size": None,
                                    "symmetric": False,
                                    "round_method": "half_even",
                                    "scale_type": "float",
                                    "observer_cls": "PerChannelMinMaxObserver",
                                },
                            },
                            "1": {
                                "name": "sparse_arch.embed_col.embedding_bags.0",
                                "type": "QuantEmbeddingBag",
                                "weight": "sparse_arch.em.embedding_bags.0.weight",
                                "weight_quant": {
                                    "dtype": "uint4",
                                    "is_dynamic": False,
                                    "qscheme": "per_token",
                                    "ch_axis": 0,
                                    "group_size": None,
                                    "symmetric": False,
                                    "round_method": "half_even",
                                    "scale_type": "float",
                                    "observer_cls": "PerChannelMinMaxObserver",
                                },
                            },
                        },
                    },
                },
            }
        }
        with open(self.config_file_path, "w") as json_file:
            json.dump(config, json_file, indent=4)
        with self.assertRaises(ValueError) as context:
            model = zentorch.load_quantized_model(model, self.data_path, "hf_format")
        self.assertTrue(
            "embed_config_dict is NOT same as the previous " in str(context.exception)
        )

    @torch.inference_mode()
    def test_missing_key_recsys(self):
        model = Custom_Model(40, 30).eval()
        weights = {
            "layer1.qweight": torch.randn(10, 10),
            "layer1.weight_scale": torch.randn(1, 10),
            "layer1.bias": torch.randn(10),
        }
        save_file(weights, self.safetensors_file_path)
        config = {
            "structure": {
                "sparse_arch": {
                    "embed_col": {
                        "embedding_bags": {
                            "0": {
                                "name": "sparse_arch.embed_col.embedding_bags.0",
                                "type": "QuantEmbeddingBag",
                                "weight": "sparse_arch.em.embedding_bags.0.weight",
                                "weight_quant": {
                                    "dtype": "uint4",
                                    "is_dynamic": False,
                                    "qscheme": "per_channel",
                                    "ch_axis": 0,
                                    "group_size": None,
                                    "round_method": "half_even",
                                    "scale_type": "float",
                                    "observer_cls": "PerChannelMinMaxObserver",
                                },
                            },
                        },
                    },
                },
            }
        }
        with open(self.config_file_path, "w") as json_file:
            json.dump(config, json_file, indent=4)
        with self.assertRaises(ValueError) as context:
            model = zentorch.load_quantized_model(model, self.data_path, "hf_format")
        self.assertTrue("Key is missing in module_info" in str(context.exception))

    @torch.inference_mode()
    def test_wrong_module_dtype_recsys(self):
        model = Custom_Model(40, 30).eval()
        weights = {
            "layer1.qweight": torch.randn(10, 10).to(torch.int32),
            "layer1.weight_scale": torch.randn(1, 10),
            "layer1.bias": torch.randn(10),
        }
        save_file(weights, self.safetensors_file_path)
        config = {
            "structure": {
                "sparse_arch": {
                    "embed_col": {
                        "embedding_bags": {
                            "0": {
                                "name": "sparse_arch.embed_col.embedding_bags.0",
                                "type": "QuantConvolution",
                                "weight": "sparse_arch.em.embedding_bags.0.weight",
                                "weight_quant": {
                                    "dtype": "uint4",
                                    "is_dynamic": False,
                                    "qscheme": "per_channel",
                                    "ch_axis": 0,
                                    "group_size": None,
                                    "round_method": "half_even",
                                    "scale_type": "float",
                                    "observer_cls": "PerChannelMinMaxObserver",
                                },
                            },
                        },
                    },
                },
            }
        }
        with open(self.config_file_path, "w") as json_file:
            json.dump(config, json_file, indent=4)
        with self.assertRaises(NotImplementedError) as context:
            model = zentorch.load_quantized_model(model, self.data_path, "hf_format")
        self.assertTrue(
            "zentorch does not support this module type" in str(context.exception)
        )


if __name__ == "__main__":
    run_tests()
