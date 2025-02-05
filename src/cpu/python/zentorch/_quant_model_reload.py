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
from ._quantization_utils import get_op_by_name, set_op_by_name, get_module_name_str

# Generate a logger for this file
logger = get_logger(__name__)

# This set contains the strings found in the model.config.architectures[0], for
# a valid huggingface transformer model which is supported with load_quantized_model api
RELOAD_SUPPORTED_MODELS = {
    "ChatGLMModel",
    "DLRMV2",
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


def build_and_replace_with_Q_EmbeddingBag(
    float_module,
    packed_embedding_weight,
    weight_scales,
    weight_zero_points,
    weight_bits,
    group_size,
    use_zero_point=False,
    torch_dtype="bfloat16",
    scale_dtype="float",
    quant_dtype="uint4",
):
    from ._WOQ_embedding_bag import ZenTorchWOQEmbeddingBag

    quant_module = ZenTorchWOQEmbeddingBag(
        float_module,
        packed_embedding_weight,
        weight_scales,
        weight_zero_points,
        group_size,
        weight_bits,
        torch_dtype,
        scale_dtype,
        quant_dtype,
    )

    return quant_module


def build_and_replace_with_Q_Linear(
    float_module,
    weight_tensor,
    weight_scales,
    weight_zero_points,
    bias_tensor,
    group_size,
    weight_bits,
    use_zero_point=False,
    torch_dtype="bfloat16",
    model_architecture=None,
    module_name=None,
    input_scales=None,
    input_zero_points=None,
    input_bits=None,
    input_symmetric=False,
):
    from ._WOQLinear import ZenTorchWOQLinear
    from ._StaticQuantizedLinear import ZenTorchStaticQuantizedLinear

    dummy_weight_dtype = None
    # This workaround is required for Woq ChatGLMModel with zentorch.llm.optimize.
    # It bypasses quantized layer modification, that prevents IPEXLinearSiluMul fusion.
    if model_architecture == "ChatGLMModel" and module_name.endswith("dense_h_to_4h"):
        dummy_weight_dtype = torch.uint8

    if input_scales:
        quant_module = ZenTorchStaticQuantizedLinear(
            float_module,
            weight_tensor,
            weight_scales,
            weight_zero_points,
            weight_bits,
            input_scales,
            input_zero_points,
            input_bits,
            bias_tensor,
            group_size,
            torch_dtype,
            input_symmetric=input_symmetric,
        )
    else:
        quant_module = ZenTorchWOQLinear(
            float_module,
            weight_tensor,
            weight_scales,
            weight_zero_points,
            bias_tensor,
            group_size,
            weight_bits,
            torch_dtype,
            dummy_weight_dtype,
        )
    return quant_module


def param_check(param_keys, params_dict):
    for k in param_keys:
        if k not in params_dict.keys():
            raise KeyError(k, " is not available")


def get_embedding_module_param_tensors(module_name, params_dict):
    logger.info("Fetch embedding parameters.")
    weight_scales_key = module_name + ".weight_scale"
    weight_zero_points_key = module_name + ".weight_zero_point"
    param_keys = [weight_scales_key, weight_zero_points_key]
    param_check(param_keys, params_dict)
    # access weight_scales and weight_zero_points
    weight_scales = params_dict[weight_scales_key]
    weight_zero_points = params_dict[weight_zero_points_key]
    bias_tensor_key = module_name + ".bias"
    bias_tensor = params_dict.get(bias_tensor_key, None)
    return (weight_scales, weight_zero_points, bias_tensor)


def get_woq_module_param_tensors(module_name, params_dict):
    logger.info("Fetch WOQ parameters.")
    weight_scales_key = module_name + ".scales"
    weight_zero_points_key = module_name + ".qzeros"
    param_keys = [weight_scales_key, weight_zero_points_key]
    param_check(param_keys, params_dict)
    # access weight_scales and weight_zero_points
    weight_scales = params_dict[weight_scales_key]
    weight_zero_points = params_dict[weight_zero_points_key]
    bias_tensor_key = module_name + ".bias"
    bias_tensor = params_dict.get(bias_tensor_key, None)
    return (weight_scales, weight_zero_points, bias_tensor)


def get_static_module_param_tensors(module_name, params_dict):
    logger.info("Fetch Static quant parameters.")
    weight_scales_key = module_name + ".weight_scale"
    weight_zero_points_key = module_name + ".weight_zero_point"
    input_scales_key = module_name + ".input_scale"
    input_zero_points_key = module_name + ".input_zero_point"
    param_keys = [
        weight_scales_key,
        weight_zero_points_key,
        input_scales_key,
        input_zero_points_key,
    ]
    param_check(param_keys, params_dict)
    # access weight_scales, weight_zero_points, input_scales and input_zero_points

    weight_scales = params_dict[weight_scales_key]
    weight_zero_points = params_dict[weight_zero_points_key]
    input_scales = params_dict[input_scales_key]
    input_zero_points = params_dict[input_zero_points_key]
    bias_tensor_key = module_name + ".bias"
    bias_tensor = params_dict.get(bias_tensor_key, None)
    return (
        input_scales,
        input_zero_points,
        weight_scales,
        weight_zero_points,
        bias_tensor,
    )


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


def get_model_config(config_json_path):
    logger.info("Extracting required config information from config.json file...")

    with open(config_json_path, "r") as f:
        config = json.load(f)
    if "quantization_config" in config:
        quant_config = config["quantization_config"]
    else:
        raise KeyError("quantization_config is not available")
    if quant_config["quant_method"] == "quark":
        logger.info("Static model config")
        model_config = {
            "activation_symmetric": quant_config["global_quant_config"][
                "input_tensors"
            ]["symmetric"],
            "weight_symmetric": quant_config["global_quant_config"]["weight"][
                "symmetric"
            ],
            "pack_method": config["quantization_config"]["export"]["pack_method"],
            "torch_dtype": config["torch_dtype"],
            "activation_qscheme": quant_config["global_quant_config"]["input_tensors"][
                "qscheme"
            ],
            "weight_qscheme": quant_config["global_quant_config"]["weight"]["qscheme"],
            "activation_bits": quant_config["global_quant_config"]["input_tensors"][
                "dtype"
            ][-1],
            "weight_bits": quant_config["global_quant_config"]["weight"]["dtype"][-1],
        }
        supported_config = {
            "pack_method": ("order",),
            "activation_symmetric": (
                True,
                False,
            ),
            "weight_symmetric": (True,),
            "torch_dtype": ("float32",),
            "activation_qscheme": ("per_tensor",),
            "weight_qscheme": ("per_channel", "per_tensor"),
            "activation_bits": ("8",),
            "weight_bits": ("8",),
        }
        if bool(quant_config["layer_quant_config"]):
            # Specialized case for embedding bag, will be extended for KV-cache
            model_config["layer_quant_config"] = True
            model_config["eb_symmetric"] = quant_config["layer_quant_config"]["weight"][
                "symmetric"
            ]
            model_config["eb_qscheme"] = quant_config["layer_quant_config"]["weight"][
                "qscheme"
            ]
            model_config["eb_dtype"] = quant_config["layer_quant_config"]["weight"][
                "dtype"
            ]
            model_config["eb_zero_point"] = quant_config["layer_quant_config"][
                "weight"
            ]["zero_point"]
            model_config["eb_bits"] = model_config["eb_dtype"][-1]
            model_config["eb_scale_type"] = quant_config["layer_quant_config"][
                "weight"
            ]["scale_type"]

            # Parallel entries for supported_config
            supported_config["layer_quant_config"] = (True,)
            supported_config["eb_symmetric"] = (False,)
            supported_config["eb_qscheme"] = ("per_channel",)
            supported_config["eb_dtype"] = ("uint4",)
            supported_config["eb_bits"] = ("4",)
            supported_config["eb_zero_point"] = (0,)
            supported_config["eb_scale_type"] = ("float",)
    else:
        logger.info("WOQ model config")
        model_config = {
            "weight_bits": quant_config["bits"],
            "group_size": quant_config["group_size"],
            "pack_method": quant_config["pack_method"],
            "zero_point": quant_config["zero_point"],
            "torch_dtype": config["torch_dtype"],
        }
        supported_config = {
            "weight_bits": (4,),
            "group_size": -1,
            "pack_method": ("order",),
            "zero_point": (False,),
            "torch_dtype": ("bfloat16",),
        }

        group_size = quant_config["group_size"]
        if group_size == 0 or group_size < supported_config["group_size"]:
            raise NotImplementedError(
                f"zentorch does not support group_size {group_size}."
                " Supported values are group_size = ",
                supported_config["group_size"],
                "and group_size > 0 for weight-only quantization.",
            )

    for key, value in model_config.items():
        logger.info(f"Models config.json file has {key} = {value}")
        if key == "group_size":
            continue
        if value not in supported_config[key]:
            raise NotImplementedError(
                f"zentorch has not yet implemented support for {key} = {value}."
                f"It only supports {key} = {supported_config[key]} "
                "for static and weight-only quantization."
            )

    logger.info("Extracted required config information successfully!!!")
    return model_config


def load_quantized_model(
    model: nn.Module, saved_model_path: str, saved_model_type: str = "quark_safetensors"
) -> nn.Module:
    r"""Loads the quantized model with help of original model, saved
    safetensors and config.json available from saved_model_path.

    Args:
        model (Module): original model which is used to load quantized model
        saved_model_path (str): path where safetensors and config files are available
        saved_model_type (str): model export method of quantized model

    Returns the reloaded quantized model with quantized modules.
    """
    logger.info("Loading the quantized model...")

    try:
        import safetensors  # noqa: F401
    except ImportError:
        raise ImportError(
            "'safetensors' package is not installed. 'safetensors' is "
            + "required for woq model loading. Please install it using "
            + "`pip install safetensors`."
        )
    with open(os.path.join(saved_model_path, "config.json"), "r") as f:
        models_config = json.load(f)
    if hasattr(model, "config"):
        torch._check(
            hasattr(model.config, "architectures"),
            "model config does not have a architectures attribute",
        )
        model_architecture = model.config.architectures[0]
    else:
        model_architecture = models_config["architectures"][0]
    if model_architecture not in RELOAD_SUPPORTED_MODELS:
        raise ValueError(
            "This quantized model with model_architecture = "
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
    model_config = get_model_config(os.path.join(saved_model_path, "config.json"))

    if model_architecture == "ChatGLMModel" and [
        key for key in params_keys if key.endswith("input_scale")
    ]:
        raise ValueError(
            "zentorch has not yet implemented"
            " static quantization support for chatglm model"
        )
    for weight_key in weight_keys:
        # TODO: directly load weight_tensor or any other tensors extracted from
        # safetensors into the modules instead of creating local tensor variables
        weight_tensor = params_dict[weight_key]

        if weight_key.endswith("weight"):
            module_name = get_module_name_str(weight_key)
            static_key = module_name + ".input_scale"
            if weight_key.endswith("qweight"):
                # Woq model's weights are stored as 'qweight' in safetensor file and
                # Only 4 bit quantized modules will have
                # weight_tensor.dtype == torch.int32
                if weight_tensor.dtype == torch.int32:
                    weight_scales, weight_zero_points, bias_tensor = (
                        get_woq_module_param_tensors(module_name, params_dict)
                    )
                    # get nn.Module for corresponding module_name
                    float_module = get_op_by_name(model, module_name)

                    quant_module = build_and_replace_with_Q_Linear(
                        float_module,
                        weight_tensor,
                        weight_scales,
                        weight_zero_points,
                        bias_tensor,
                        model_config["group_size"],
                        model_config["weight_bits"],
                        model_config["zero_point"],
                        model_config["torch_dtype"],
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
            elif "embedding_bags" in weight_key:
                # Ideally, we should be supporting input_tensors, output_tensors
                # and weight quant information and loading individually.
                weight_scales, weight_zero_points, bias_tensor = (
                    get_embedding_module_param_tensors(module_name, params_dict)
                )
                packed_embedding_weight = (
                    zentorch._C.zentorch_get_packed_embedding_weight(
                        weight_tensor, weight_scales, weight_zero_points
                    )
                )
                float_module = get_op_by_name(model, module_name)
                quant_module = build_and_replace_with_Q_EmbeddingBag(
                    float_module,
                    packed_embedding_weight,
                    weight_scales,
                    weight_zero_points,
                    model_config["eb_bits"],
                    None,  # group_size
                    model_config["eb_zero_point"],
                    model_config["torch_dtype"],
                    model_config["eb_scale_type"],
                    model_config["eb_dtype"],
                )
                set_op_by_name(model, module_name, quant_module)

            elif static_key in params_keys:
                (
                    input_scales,
                    input_zero_points,
                    weight_scales,
                    weight_zero_points,
                    bias_tensor,
                ) = get_static_module_param_tensors(module_name, params_dict)
                # get nn.Module for corresponding module_name
                float_module = get_op_by_name(model, module_name)
                quant_module = build_and_replace_with_Q_Linear(
                    float_module,
                    weight_tensor,
                    weight_scales,
                    weight_zero_points,
                    bias_tensor,
                    None,
                    model_config["weight_bits"],
                    False,
                    model_config["torch_dtype"],
                    model_architecture,
                    module_name,
                    input_scales,
                    input_zero_points,
                    model_config["activation_bits"],
                    model_config["activation_symmetric"],
                )
                set_op_by_name(model, module_name, quant_module)
            else:
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
    logger.info("The quantized model is loaded successfully!")
    return model


@deprecated(
    "It will be removed after ZenDNN v5.0.1 release."
    "Please use load_quantized_model() API."
)
def load_woq_model(
    model: nn.Module, saved_model_path: str, saved_model_type: str = "quark_safetensors"
) -> nn.Module:
    return load_quantized_model(model, saved_model_path)
