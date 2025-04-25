# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import json
import torch
import torch.nn as nn
import os
from deprecated import deprecated
import zentorch
from ._logging import get_logger
from ._quantization_utils import (
    set_op_by_name,
    get_name_and_info,
    get_torch_type_from_str,
)

# Generate a logger for this file.
logger = get_logger(__name__)

# This set contains the strings found in the model.config.architectures[0],
# for a valid Huggingface transformer model which is supported with
# load_quantized_model API.
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
    # Following model architectures are not yet supported.
    # "BloomForCausalLM",
    # "CodeGenForCausalLM",
    # "GitForCausalLM",
    # "GPTNeoXForCausalLM",
    # "GPTBigCodeForCausalLM",
    # "MixtralForCausalLM",
    # "StableLmForCausalLM",
    # "YuanForCausalLM",
}


def build_and_replace_with_woq_op(
    float_module,
    weight_tensor,
    weight_scales,
    weight_zero_points,
    bias_tensor,
    model_config,
    model_architecture=None,
    module_name=None,
    enable_weight_prepack=False,
):
    from ._WOQ_embedding_bag import ZenTorchWOQEmbeddingBag
    from ._WOQLinear import ZenTorchWOQLinear

    dummy_weight_dtype = None
    if model_architecture == "ChatGLMModel" and module_name.endswith("dense_h_to_4h"):
        dummy_weight_dtype = torch.uint8
    if isinstance(float_module, nn.EmbeddingBag):
        packed_embedding_weight = zentorch._C.zentorch_get_packed_embedding_weight(
            weight_tensor, weight_scales, weight_zero_points
        )
        quant_module = ZenTorchWOQEmbeddingBag(
            float_module,
            packed_embedding_weight,
            weight_scales,
            weight_zero_points,
            model_config["group_size"],
            model_config["weight_bits"],
            model_config["torch_dtype"],
            model_config["eb_scale_type"],
            model_config["eb_dtype"],
        )
    elif isinstance(float_module, nn.Linear):
        if enable_weight_prepack:
            logger.warning(
                "Arg - 'enable_weight_prepack' is not being utilized for "
                "WOQ Linear as reorder of weight is not yet required "
                "for WOQ Linear."
            )
        quant_module = ZenTorchWOQLinear(
            float_module,
            weight_tensor,
            weight_scales,
            weight_zero_points,
            bias_tensor,
            model_config["group_size"],
            int(model_config["weight_bits"]),
            model_config["torch_dtype"],
            dummy_weight_dtype,
        )
    else:
        raise ValueError(
            "Zentorch currently only supports nn.Linear and nn.EmbeddingBag"
            " quantized modules"
        )
    return quant_module


def param_check(param_keys, params_dict):
    for k in param_keys:
        if k not in params_dict.keys():
            raise KeyError(k, " is not available.")


def get_woq_module_param_tensors(module_name, params_dict):
    logger.info("Fetching WOQ parameters.")
    weight_tensor = module_name + ".weight"
    weight_scales_key = module_name + ".weight_scale"
    weight_zero_points_key = module_name + ".weight_zero_point"
    param_keys = [weight_tensor, weight_scales_key, weight_zero_points_key]
    param_check(param_keys, params_dict)
    # Access weight_scales and weight_zero_points.
    weight_tensor = params_dict[weight_tensor]
    weight_scales = params_dict[weight_scales_key]
    weight_zero_points = params_dict[weight_zero_points_key]
    bias_tensor_key = module_name + ".bias"
    bias_tensor = params_dict.get(bias_tensor_key, None)
    return (weight_tensor, weight_scales, weight_zero_points, bias_tensor)


def get_static_module_param_tensors(module_name, params_dict, model_dtype):
    logger.info("Fetching static quant parameters.")
    weight_tensor = module_name + ".weight"
    weight_scales_key = module_name + ".weight_scale"
    weight_zero_points_key = module_name + ".weight_zero_point"
    input_scales_key = module_name + ".input_scale"
    input_zero_points_key = module_name + ".input_zero_point"
    param_keys = [
        weight_tensor,
        weight_scales_key,
        weight_zero_points_key,
        input_scales_key,
        input_zero_points_key,
    ]
    param_check(param_keys, params_dict)
    # Access weight_scales, weight_zero_points, input_scales and input_zero_points
    weight_tensor = params_dict[weight_tensor]
    weight_scales = params_dict[weight_scales_key]
    weight_zero_points = params_dict[weight_zero_points_key]
    input_scales = params_dict[input_scales_key]
    input_zero_points = params_dict[input_zero_points_key]
    bias_tensor_key = module_name + ".bias"
    bias_tensor = params_dict.get(bias_tensor_key, None)
    if bias_tensor is not None:
        bias_tensor = bias_tensor.to(get_torch_type_from_str(model_dtype))
    return (
        weight_tensor,
        input_scales,
        input_zero_points,
        weight_scales,
        weight_zero_points,
        bias_tensor,
    )


# Iterates through each module in the model configuration,
# checking for specific types: QuantEmbeddingBag and QuantLinear,
# validates the required keys, at last combines these configurations
# into a dictionary named as model_config.
def get_recsys_config(config):
    prev_embed_config_dict = None
    prev_weight_config_dict = None
    prev_activation_config_dict = None
    for name, module_info in get_name_and_info(config["structure"]):
        if module_info["type"] == "QuantEmbeddingBag":
            embed_config_dict = {}
            required_keys = ["symmetric", "qscheme", "scale_type", "dtype"]
            for key in required_keys:
                if key not in module_info["weight_quant"]:
                    raise ValueError("Key is missing in module_info")
                embed_config_dict["eb_" + key] = module_info["weight_quant"][key]
            embed_config_dict["eb_bits"] = module_info["weight_quant"]["dtype"][-1]
            if (
                prev_embed_config_dict is not None
                and prev_embed_config_dict != embed_config_dict
            ):
                raise ValueError(
                    f"{name}: embed_config_dict is NOT same as the previous "
                    "quantized embedding layer."
                )
            prev_embed_config_dict = embed_config_dict
        elif module_info["type"] == "QuantLinear":
            weight_config_dict = {}
            activation_config_dict = {}
            required_keys = ["symmetric", "qscheme"]
            for key in required_keys:
                if (
                    key not in module_info["weight_quant"]
                    or key not in module_info["input_quant"]
                ):
                    raise ValueError("Key is missing in module_info")
                weight_config_dict["weight_" + key] = module_info["weight_quant"][key]
                activation_config_dict["activation_" + key] = module_info[
                    "input_quant"
                ][key]
            weight_config_dict["weight_bits"] = module_info["weight_quant"]["dtype"][-1]
            activation_config_dict["activation_bits"] = module_info["input_quant"][
                "dtype"
            ][-1]

            if (
                prev_weight_config_dict is not None
                and prev_weight_config_dict != weight_config_dict
            ):
                raise ValueError(
                    f"{name}: weight_config_dict is NOT same as the "
                    "previous quantized linear layer."
                )
            if (
                prev_activation_config_dict is not None
                and prev_activation_config_dict != activation_config_dict
            ):
                raise ValueError(
                    f"{name}: activation_config_dict is NOT same as the "
                    "previous quantized linear layer."
                )

            prev_weight_config_dict = weight_config_dict
            prev_activation_config_dict = activation_config_dict
        else:
            if module_info["weight_quant"]:
                raise NotImplementedError(
                    f"zentorch does not support this module type {module_info['type']}",
                )
    model_config = {**embed_config_dict, **weight_config_dict, **activation_config_dict}
    model_config["group_size"] = None
    return model_config


def get_params_dict_from_safetensors(saved_model_path):
    logger.info("Extracting params_dict for tensors from safetensors file...")
    from safetensors import safe_open
    from pathlib import Path

    safe_files = list(Path(saved_model_path).rglob("*.safetensors"))
    if len(safe_files) == 0:
        raise FileNotFoundError(
            "No file ending with .safetensors found at this location: "
            + saved_model_path
        )
    params_dict = {}
    for safetensors_file_path in safe_files:
        with safe_open(safetensors_file_path, framework="pt", device="cpu") as f:
            for k in f.keys():
                params_dict[k] = f.get_tensor(k)
    logger.info("Extracted params_dict successfully!!!")
    return params_dict


def get_llm_config(config):
    if "quantization_config" in config:
        if "global_quant_config" in config["quantization_config"]:
            global_config = config["quantization_config"]["global_quant_config"]
        else:
            raise KeyError("global_quant_config is not available.")
    else:
        raise KeyError("quantization_config is not available.")
    quant_type = "weight-only"
    model_config = {}
    supported_config = {}
    if bool(global_config["weight"]):
        model_config = {
            "weight_symmetric": global_config["weight"]["symmetric"],
            "pack_method": config["quantization_config"]["export"]["pack_method"],
            "torch_dtype": config["torch_dtype"],
            "weight_qscheme": global_config["weight"]["qscheme"],
            "weight_bits": global_config["weight"]["dtype"][-1],
            "group_size": global_config["weight"]["group_size"],
        }
        if model_config["group_size"] is None:
            model_config["group_size"] = -1
        # Weight-only quantization supported config.
        supported_config = {
            "pack_method": ("order",),
            "weight_symmetric": (True,),
            "torch_dtype": ("bfloat16",),
            "weight_qscheme": ("per_channel", "per_group"),
            "weight_bits": ("4",),
            "group_size": -1,
        }
    if bool(global_config["input_tensors"]):
        quant_type = "static"
        # Static quantization supported config.
        supported_config = {
            "pack_method": ("order",),
            "activation_symmetric": (
                True,
                False,
            ),
            "weight_symmetric": (True,),
            "torch_dtype": (
                "bfloat16",
                "float32",
            ),
            "activation_qscheme": ("per_tensor",),
            "weight_qscheme": ("per_channel", "per_tensor"),
            "activation_bits": ("8",),
            "weight_bits": ("8",),
            "group_size": -1,
        }
        static_config = {
            "activation_symmetric": global_config["input_tensors"]["symmetric"],
            "activation_qscheme": global_config["input_tensors"]["qscheme"],
            "activation_bits": global_config["input_tensors"]["dtype"][-1],
        }
        model_config.update(static_config)

    group_size = (
        -1 if model_config["group_size"] is None else model_config["group_size"]
    )
    if group_size != -1 and (group_size == 0 or group_size < supported_config["group_size"]):
        raise NotImplementedError(
            f"Zentorch does not support group_size {group_size}."
            " Supported values are group_size = ",
            supported_config["group_size"],
            "and group_size > 0 for weight-only quantization.",
        )

    for key, value in model_config.items():
        logger.info("Model's config.json file has %s = %s", key, value)
        if key == "group_size":
            continue
        if value not in supported_config[key]:
            raise NotImplementedError(
                "Zentorch has not yet implemented support for %s = %s."
                "It only supports %s = %s "
                "for %s quantization.", key, value, key, supported_config[key], quant_type
            )

    logger.info("Extracted required config information successfully!")
    return model_config


def get_model_config(model, saved_model_path):
    json_files = [
        pos_json
        for pos_json in os.listdir(saved_model_path)
        if pos_json.endswith(".json")
    ]
    logger.info("Extracting required config information from JSON file...")
    if len(json_files) == 0:
        raise FileNotFoundError(
            "No JSON file found at this location: ",
            saved_model_path,
        )
    elif len(json_files) == 1:
        with open(os.path.join(saved_model_path, json_files[0]), "r") as f:
            config = json.load(f)
        model_config = get_recsys_config(config)
        # The recsys config does not provide dtype information,
        # therefore we retrieve it from the model.
        # TODO: Handle the dtype options for AMP.
        if next(model.parameters()).dtype == torch.bfloat16:
            model_config["torch_dtype"] = "bfloat16"
        elif next(model.parameters()).dtype == torch.float:
            model_config["torch_dtype"] = "float32"
        else:
            raise ValueError("Only float or Bfloat models are supported.")
    else:
        with open(os.path.join(saved_model_path, "config.json"), "r") as f:
            config = json.load(f)
        model_config = get_llm_config(config)
    return model_config


def load_quantized_model(
    model: nn.Module,
    saved_model_path: str,
    saved_model_type: str = "hf_format",
    enable_weight_prepack: bool = False,
) -> nn.Module:
    r"""Loads the quantized model with the help of the original model, saved
    safetensors, and config.json available from saved_model_path.

    Args:
        model (Module): original model which is used to load the quantized model.
        saved_model_path (str): path where safetensors and config files are available.
        saved_model_type (str): model export method of the quantized model.
        enable_weight_prepack (bool): control to enable weight prepacking.

    Returns the reloaded quantized model with quantized modules.
    """
    if saved_model_type != "hf_format":
        raise NotImplementedError(
            "Zentorch has not yet implemented support for the models "
            "exported with %s method, it only "
            "supports models saved/exported with 'hf_format' "
            "method.", saved_model_type
        )
    if model is None:
        raise ValueError("Model should not be none for loading the quantized model.")
    logger.info("Loading the quantized model...")
    # TODO: After loading the config, check the model params dtype
    # and then pass the dtype accordingly to build_and_replace_with_Q_Linear().
    # Extract params_dict for tensors from safetensors file.
    params_dict = get_params_dict_from_safetensors(saved_model_path)
    model_config = get_model_config(model, saved_model_path)
    model_architecture = None
    if hasattr(model, "config"):
        torch._check(
            hasattr(model.config, "architectures"),
            "Model config does not have an 'architectures' attribute.",
        )
        model_architecture = model.config.architectures[0]

    if (
        model_architecture not in RELOAD_SUPPORTED_MODELS
        and model_architecture is not None
    ):
        raise ValueError(
            "This quantized model with model_architecture = "
            + f"{model_architecture} is not yet supported with zentorch."
        )

    params_keys = params_dict.keys()
    for module_name, float_module in list(
        dict(model.named_modules(remove_duplicate=False)).items()
    ):
        # TODO: Directly load weight_tensor or any other tensors extracted from
        # safetensors into the modules instead of creating local tensor variables.
        input_scale_key = module_name + ".input_scale"
        weight_scale_key = module_name + ".weight_scale"
        weight_tensor_key = module_name + ".weight"
        if input_scale_key in params_keys:
            (
                weight_tensor,
                input_scales,
                input_zero_points,
                weight_scales,
                weight_zero_points,
                bias_tensor,
            ) = get_static_module_param_tensors(
                module_name, params_dict, model_config["torch_dtype"]
            )
            from ._StaticQuantizedLinear import ZenTorchStaticQuantizedLinear

            quant_module = ZenTorchStaticQuantizedLinear(
                float_module,
                weight_tensor,
                weight_scales,
                weight_zero_points,
                model_config["weight_bits"],
                input_scales,
                input_zero_points,
                model_config["activation_bits"],
                bias_tensor,
                False,  # group_size
                model_config["torch_dtype"],
                model_config["activation_symmetric"],
                model_config["weight_symmetric"],
                enable_weight_prepack,
            )
            set_op_by_name(model, module_name, quant_module)
        elif weight_scale_key in params_keys:
            # In WOQ model's only 4-bit quantized modules will have
            # weight_tensor.dtype == torch.int32.
            weight_tensor, weight_scales, weight_zero_points, bias_tensor = (
                get_woq_module_param_tensors(module_name, params_dict)
            )
            if weight_tensor.dtype != torch.int32:
                raise NotImplementedError(
                    "Zentorch has not yet implemented support for weights packed "
                    "into %s tensor, it only supports "
                    "weight packed into 'torch.int32' tensor.", weight_tensor.dtype
                )
            quant_module = build_and_replace_with_woq_op(
                float_module,
                weight_tensor,
                weight_scales,
                weight_zero_points,
                bias_tensor,
                model_config,
                model_architecture,
                module_name,
                enable_weight_prepack,
            )
            set_op_by_name(model, module_name, quant_module)
        elif weight_tensor_key in params_keys:
            weight_tensor = params_dict[weight_tensor_key]
            weight_device = float_module.weight.device
            float_module.weight.data = weight_tensor.to(weight_device)
            bias_tensor_key = module_name + ".bias"
            if bias_tensor_key in params_dict:
                bias_tensor = params_dict[bias_tensor_key]
                bias_device = float_module.bias.device
                float_module.bias.data = bias_tensor.to(bias_device)

    logger.info("The quantized model is loaded successfully!")
    return model


@deprecated(
    "It will be removed after ZenDNN v5.0.1 release."
    "Please use load_quantized_model() API."
)
def load_woq_model(
    model: nn.Module, saved_model_path: str, saved_model_type: str = "hf_format"
) -> nn.Module:
    return load_quantized_model(model, saved_model_path)
