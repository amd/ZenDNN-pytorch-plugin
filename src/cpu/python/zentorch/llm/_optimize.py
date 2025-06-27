# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import torch
from ._checks import essential_checks
import zentorch._C
import zentorch._WOQLinear as WOQLinear
from .._logging import get_logger

# make a logger for this file
logger = get_logger(__name__)


def check_for_shared_params(model):
    param_dict = {}

    # Get the singleton instance
    manager = zentorch._C.DataPointerManager.getInstance()
    manager.clear()

    # Currently, WoQ model weights are skipped.
    # As we expect linear and embedding to have separate quantized weights
    # Static quantized models are not supported.
    for name, module in model.named_modules():
        if hasattr(module, "weight"):
            if isinstance(module.weight, WOQLinear.DummyWeight):
                continue
            data_ptr = module.weight.data_ptr()
            if data_ptr not in param_dict:
                param_dict[data_ptr] = []
            param_dict[data_ptr].append((name, module))

    # Add shared pointers to the manager
    # ZenDNN already handles reordering the same weight across linear layers.
    # It keeps a dict which has the shape and data pointer of ordered weights.
    # It checks if the weight is present in the table then it doesn't reorder
    # again. Hence we don't add weights shared across layers to the manager.
    for data_ptr, name_module_pairs in param_dict.items():
        if len(name_module_pairs) > 1:
            names = [name for name, module in name_module_pairs]
            modules = [module for name, module in name_module_pairs]
            logger.warning("Shared param found: %s, data_ptr: %s", names, hex(data_ptr))
            for module in modules:
                if not isinstance(module, torch.nn.Linear):
                    manager.addPointer(module.weight.data_ptr())
                    break


def optimize(model, dtype=torch.bfloat16):
    if essential_checks(model, dtype):
        import intel_extension_for_pytorch as ipex
        from ._model_conversion_functions import model_convert_lowering, customize_model

        ipex_t = ipex.transformers

        # For masked multihead attention, the meta registration uses dynamic shape outputs
        # To ensure the dynamic shapes do not cause a greph break
        torch._dynamo.config.capture_dynamic_output_shape_ops = True
        torch._dynamo.config.capture_scalar_outputs = True
        # Runtime over-riding of IPEX model_convert_lowering with ZenTorch
        # model_convert_lowering. So, after this line whenever IPEX
        # model_convert_lowering is called control would go to ZenTorch
        # model_convert_lowering.
        model = customize_model(model)
        ipex_t.optimize.model_convert_lowering = model_convert_lowering

        model = ipex.llm.optimize(model, optimizer=None, dtype=dtype)

    check_for_shared_params(model)

    return model
