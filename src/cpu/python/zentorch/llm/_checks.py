# ******************************************************************************
# Copyright (c) 2024 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import torch
from torch.torch_version import TorchVersion
from .._logging import get_logger

# make a logger for this file
logger = get_logger(__name__)

# This set contains the strings found in the model.config.architectures[0], for
# a valid huggingface transformer model
SUPPORTED_MODELS = {
    "Qwen2ForCausalLM",
    "ChatGLMModel",
    "GPTJForCausalLM",
    "LlamaForCausalLM",
    "PhiForCausalLM",
    "Phi3ForCausalLM",
    "MistralForCausalLM",
    "GPTNeoXForCausalLM",
    "OPTForCausalLM",
    "BloomForCausalLM",
    "CodeGenForCausalLM",
    "GPTBigCodeForCausalLM",
    "StableLmForCausalLM",
    "GitForCausalLM",
    "MixtralForCausalLM",
    "QWenLMHeadModel",
    "YuanForCausalLM",
}


def get_installed_ipex_version():
    # Previous approach made use of freeze API from pip._internal.operations
    # This caused the script to error out in certain cases. This was due to
    # conflicts in imports of distutils used in pip and setuptools. To avoid
    # the above situation, the usage of importlib.metadata.version is done. The
    # usage of importlib.metadata is the recommended way of achieving this.
    # This will not actually import module, but will find the version from
    # metadata stored in dist-info or egg-info.

    from importlib.metadata import version, PackageNotFoundError
    try:
        return version("intel_extension_for_pytorch")
    except PackageNotFoundError:
        return None


def essential_checks(model, dtype):
    if hasattr(model, "config") and hasattr(model.config, "architectures"):
        is_well_supported_model = model.config.architectures[0] in SUPPORTED_MODELS

        if is_well_supported_model:
            installed_ipex_version = get_installed_ipex_version()
            if installed_ipex_version:
                # Zentorch will work with IPEX of atleast 2.3
                min_version = TorchVersion("2.3")
                installed_ipex_version = TorchVersion(installed_ipex_version)

                if installed_ipex_version >= min_version:
                    # All checks good...
                    if dtype != torch.bfloat16:
                        logger.warning(
                            "The supported datatype for the most optimal "
                            "performance with zentorch is bfloat16."
                        )
                        return False
                    return True
                else:
                    logger.warning(
                        "zentorch.llm.optimize requires IPEX: at least "
                        f"{min_version} but your IPEX is "
                        f"{installed_ipex_version}. Some of the ZenTorch "
                        "specific optimizations for LLMs might not be "
                        "triggered."
                    )
                    return False

            else:
                logger.warning(
                    "Intel Extension for PyTorch not installed. So, the "
                    "ZenTorch specific optimizations for LLMs might not "
                    "be triggered."
                )
                return False
        else:
            logger.warning(
                "Complete set of optimizations are currently unavailable"
                " for this model."
            )
            return False
    else:
        logger.warning(
            "Cannot detect the model transformers family by "
            "model.config.architectures. Please pass a valid HuggingFace LLM "
            "model to the zentorch.llm.optimize API.",
        )
        return False
