# ******************************************************************************
# Copyright (c) 2024 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import torch
from ._checks import essential_checks


def optimize(model, dtype=torch.bfloat16):
    if essential_checks(model, dtype):
        import intel_extension_for_pytorch as ipex
        from ._model_conversion_functions import model_convert_lowering, customize_model

        ipex_t = ipex.transformers
        # Runtime over-riding of IPEX model_convert_lowering with ZenTorch
        # model_convert_lowering. So, after this line whenever IPEX
        # model_convert_lowering is called control would go to ZenTorch
        # model_convert_lowering.
        model = customize_model(model)
        ipex_t.optimize.model_convert_lowering = model_convert_lowering

        model = ipex.llm.optimize(model, optimizer=None, dtype=dtype)

    return model
