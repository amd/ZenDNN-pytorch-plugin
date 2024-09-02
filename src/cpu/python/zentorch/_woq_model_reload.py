# ******************************************************************************
# Copyright (c) 2024 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import json
import torch
import torch.nn as nn
import os

from ._logging import get_logger
from ._quantization_utils import (
    get_op_by_name, set_op_by_name, get_module_name_str
)

# make a logger for this file
logger = get_logger(__name__)

# This set contains the strings found in the model.config.architectures[0], for
# a valid huggingface transformer model which is supported with load_woq_model api
RELOAD_SUPPORTED_MODELS = {
    "ChatGLMModel",
    "GPTJForCausalLM",
    "LlamaForCausalLM",
    "MistralForCausalLM",
    "OPTForCausalLM",
    "PhiForCausalLM",
    "Phi3ForCausalLM",
    "QWenLMHeadModel",
    "Qwen2ForCausalLM",
    # following model architectures are not yet supported
    # "BloomForCausalLM",
    # "CodeGenForCausalLM",
    # "GitForCausalLM",
    # "GPTNeoXForCausalLM",
    # "GPTBigCodeForCausalLM",
    # "MixtralForCausalLM",
    # "StableLmForCausalLM",
    # "YuanForCausalLM",
}


def build_and_replace_with_WQOLinear(
    float_module,
    weight_tensor,
    weight_scales,
    weight_zero_points=None,
    bias_tensor=None,
    group_size=-1,
    bits=4,
    use_zero_point=False,
    torch_dtype="bfloat16",
    model_architecture=None,
    module_name=None,
):
    from ._WOQLinear import ZenTorchWOQLinear

    dummy_weight_dtype = None
    # this is a hack to support ChatGLMModel through zentorch.llm.optimize
    # with this hack IPEXLinearSiluMul fusion is being disabled
    if model_architecture == "ChatGLMModel" and module_name.endswith("dense_h_to_4h"):
        dummy_weight_dtype = torch.uint8

    quant_module = ZenTorchWOQLinear(
        float_module,
        weight_tensor,
        weight_scales,
        weight_zero_points,
        bias_tensor,
        group_size,
        bits,
        torch_dtype,
        dummy_weight_dtype,
    )
    return quant_module


def get_woq_module_param_tensors(module_name, params_dict):
    weight_scales_key = module_name + ".scales"
    weight_zero_points_key = module_name + ".qzeros"
    if weight_scales_key not in params_dict.keys():
        raise KeyError("scales for module " + module_name + " are not available.")
    if weight_zero_points_key not in params_dict.keys():
        raise KeyError("qzeros for module " + module_name + " are not available.")

    # access weight_scales and weight_zero_points
    weight_scales = params_dict[weight_scales_key]
    weight_zero_points = params_dict[weight_zero_points_key]
    bias_tensor_key = module_name + ".bias"
    if bias_tensor_key in params_dict.keys():
        bias_tensor = params_dict[bias_tensor_key]
    else:
        bias_tensor = None

    return (weight_scales, weight_zero_points, bias_tensor)


def get_params_dict_from_safetensors(saved_model_path):
    logger.info("Extracting params_dict for tensors from safetensors file...")
    from safetensors import safe_open

    extension = ".safetensors"
    safetensors_files_list = []
    for root, _, files_list in os.walk(saved_model_path):
        for file_name in files_list:
            if os.path.splitext(file_name)[-1] == extension:
                file_name_path = os.path.join(root, file_name)
                safetensors_files_list.append(file_name_path)

    if len(safetensors_files_list) == 0:
        raise FileNotFoundError(
            f"No file ending with '{extension}' found at this location: "
            + saved_model_path
        )

    params_dict = {}
    for safetensors_file_path in safetensors_files_list:
        with safe_open(safetensors_file_path, framework="pt", device="cpu") as f:
            for k in f.keys():
                params_dict[k] = f.get_tensor(k)

    logger.info("Extracted params_dict successfully!!!")
    return params_dict


def get_config_information_from_config_json(config_json_path):
    logger.info("Extracting required config information from config.json file...")

    with open(config_json_path, "r") as f:
        model_config = json.load(f)

    woq_model_config = {
        "bits": model_config["quantization_config"]["bits"],
        "group_size": model_config["quantization_config"]["group_size"],
        "pack_method": model_config["quantization_config"]["pack_method"],
        "zero_point": model_config["quantization_config"]["zero_point"],
        "torch_dtype": model_config["torch_dtype"],
    }

    supported_woq_config = {
        "bits": 4,
        "group_size": -1,
        "pack_method": "order",
        "zero_point": False,
        "torch_dtype": "bfloat16",
    }

    for key in woq_model_config.keys():

        logger.info(f"Models config.json file has {key} = {woq_model_config[key]}")

        if woq_model_config[key] != supported_woq_config[key]:
            raise NotImplementedError(
                f"zentorch has not yet implemented support for {key} = "
                + f"{woq_model_config[key]}, it only supports {key} = "
                + f"{supported_woq_config[key]} for weight only quantization"
            )

    logger.info("Extracted required config information successfully!!!")
    return woq_model_config


def load_woq_model(
    model: nn.Module, saved_model_path: str, saved_model_type: str = "quark_safetensors"
) -> nn.Module:
    r"""Loads the weight only quantized model with help of original model, saved
    safetensors and config.json available from saved_model_path.

    Args:
        model (Module): original model which is used to load quantized model
        saved_model_path (str): path where safetensors and config files are available
        saved_model_type (str): model export method of quantized model

    Returns the reloaded weight only quantized model with quantized modules.
    """
    logger.info("Loading the weight only quantized model...")

    try:
        import safetensors # noqa
    except ImportError:
        raise ImportError(
            "'safetensors' package is not installed. 'safetensors' is "
            + "required for woq model loading. Please install it using "
            + "`pip install safetensors`."
        )

    model_architecture = model.config.architectures[0]
    if model_architecture not in RELOAD_SUPPORTED_MODELS:
        raise ValueError(
            "This weight only quantized model with model_architecture = "
            + f"{model_architecture} is not yet supported with zentorch's "
            + "reload feature."
        )
    if saved_model_type != "quark_safetensors":
        raise NotImplementedError(
            "zentorch has not yet implemented support for the models"
            + f" exported with '{saved_model_type}' method, it only "
            + "supports models saved/exported with 'quark_safetensors' "
            + "method."
        )

    # extract params_dict for tensors from safetensors file
    params_dict = get_params_dict_from_safetensors(saved_model_path)

    params_keys = params_dict.keys()
    weight_keys = [key for key in params_keys if key.endswith("weight")]

    # extract config information from config.json file
    woq_model_config = get_config_information_from_config_json(
        os.path.join(saved_model_path, "config.json")
    )

    for weight_key in weight_keys:
        # TODO: directly load weight_tensor or any other tensors extracted from
        # safetensors into the modules instead of creating local tensor variables
        weight_tensor = params_dict[weight_key]
        # Non-quantized weights are stored as 'weight' and
        # Quantized weights are stored as 'qweight' in safetensor file and
        # Only quantized modules will have weight_tensor.dtype == torch.int32
        if weight_key.endswith("qweight"):
            module_name = get_module_name_str(weight_key)
            if weight_tensor.dtype == torch.int32:
                weight_scales, weight_zero_points, bias_tensor = (
                    get_woq_module_param_tensors(module_name, params_dict)
                )
                # get nn.Module for corresponding module_name
                float_module = get_op_by_name(model, module_name)

                quant_module = build_and_replace_with_WQOLinear(
                    float_module,
                    weight_tensor,
                    weight_scales,
                    weight_zero_points,
                    bias_tensor,
                    woq_model_config["group_size"],
                    woq_model_config["bits"],
                    woq_model_config["zero_point"],
                    woq_model_config["torch_dtype"],
                    model_architecture,
                    module_name,
                )
                set_op_by_name(model, module_name, quant_module)
            else:
                raise NotImplementedError(
                    "zentorch has not yet implemented support for qweights packed "
                    + f"into '{weight_tensor.dtype}' tensor, it only supports "
                    + "qweight packed into 'torch.int32' tensor"
                )
        elif weight_key.endswith("weight"):
            module_name = get_module_name_str(weight_key)
            float_module = get_op_by_name(model, module_name)
            weight_device = float_module.weight.device
            float_module.weight.data = weight_tensor.to(weight_device)

            bias_tensor_key = module_name + ".bias"
            if bias_tensor_key in params_dict.keys():
                bias_tensor = params_dict[bias_tensor_key]
                bias_device = float_module.bias.device
                float_module.bias.data = bias_tensor.to(bias_device)
        else:
            raise ValueError(
                "Encountered a non-standard weight_key named " + weight_key
            )

    logger.info("The weight only quantized model is loaded successfully!!!")
    return model
