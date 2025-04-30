# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import torch
from ._checks import essential_checks


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

    return model
