# ****************************************************************************************************************************
# Modifications Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
# All rights reserved.
#
# Was sourced from
# https://github.com/intel/intel-extension-for-pytorch/blob/v2.3.0%2Bcpu/intel_extension_for_pytorch/transformers/optimize.py
# ****************************************************************************************************************************

from ._checks import essential_checks, get_installed_ipex_version
from torch.torch_version import TorchVersion
from .._logging import get_logger

# make a logger for this file
logger = get_logger(__name__)


# model_convert_lowering is inspired from IPEX 2.3.0 (commit id: d3c5244)
# Added an extra argument 'cache_weight_for_large_batch' to make the
# function compatible for IPEX 2.4.0s
def model_convert_lowering(
    _model,
    device,
    dtype,
    sample_inputs,
    deployment_mode,
    is_quantization=False,
    woq=False,
    cache_weight_for_large_batch=False,
):
    if not essential_checks(_model, dtype):
        return _model
    if cache_weight_for_large_batch:
        logger.warning(
            "cache_weight_for_large_batch is not "
            "supported in zentorch_llm_optimize"
        )
    if woq:
        logger.warning(
            "woq is not supported in "
            "zentorch_llm_optimize"
        )
    if is_quantization:
        logger.warning(
            "is_quantization is not supported in "
            "zentorch_llm_optimize"
        )
    if deployment_mode:
        logger.warning(
            "deployment_mode is not supported in "
            "zentorch_llm_optimize"
        )

    import intel_extension_for_pytorch as ipex

    if device == "cpu":
        # Basically this entire if-else can be removed.
        # If is_quatization is True, control would never go inside this if-else.
        # If is_quantization is False, control would go to the first condition,
        # that is ipex._C.is_llga_fp32_bf16_enabled(). If this condition is
        # true then, ipex.optimize is invoked, which ZenTorch would never want.
        # So, disabling this would be good. Next, in the else of the previous
        # if, various dtype specific branches are there. For fp32 and bf16, we
        # want to disable tpp and ipex.optimize for ZenTorch kernels to be
        # invoked. So, disabling them. ZenTorch currently doesn't work with
        # torch.half, so disabling that section also would be good.

        # ipex.cpu._auto_kernel_selection._disable_tpp()
        # if not is_quantization:
        #     if ipex._C.is_llga_fp32_bf16_enabled():
        #         ipex.cpu._auto_kernel_selection._disable_tpp()
        #         _model = ipex.optimize(
        #             _model.eval(),
        #             dtype=dtype,
        #             inplace=True,
        #             weights_prepack=False,
        #         )
        #     else:
        #         # Need to comment this out
        #        import torch
        #        if dtype is torch.float32:
        #             # this call also support bf32 path
        #             # Commenting the float32 path as well, as we want
        #             # control to go to zentorch even here.
        #             # _model = ipex.optimize(
        #             #     _model.eval(),
        #             #     dtype=dtype,
        #             #     inplace=True,
        #             #     auto_kernel_selection=True,
        #             # )
        #             pass
        #         elif dtype is torch.half:
        #             _model = ipex.optimize(
        #                 _model.eval(),
        #                 dtype=dtype,
        #                 inplace=True,
        #                 auto_kernel_selection=True,
        #             )
        #         elif dtype is torch.bfloat16:
        #             # This is where tpp kernels and ipex optimization is introduced,
        #             # which is not required for zentorch, for ops to be in aten
        #             # namespace. This will ease the process of replacing aten ops
        #             # to zentorch ops.
        #             # ipex.cpu._auto_kernel_selection._enable_tpp()
        #             # _model = ipex.optimize(_model.eval(), dtype=dtype, inplace=True)
        #             pass

        if not is_quantization or woq:
            import transformers

            supported_classes = [
                transformers.models.llama.modeling_llama.LlamaRMSNorm,
            ]
            if _model.config.architectures[0] in [
                # Out of the following models, zentorch well supports only
                # Phi model.
                "BaichuanForCausalLM",
                "YuanForCausalLM",
                "Phi3ForCausalLM",
                "Phi4MMForCausalLM",
            ]:
                supported_classes.append(type(_model.model.layers[0].input_layernorm))
            # Out of the following models, zentorch currently supports
            # none. Control would never go into any of these if
            # statements
            if (
                _model.config.architectures[0] == "ChatGLMModel"
                and _model.config.rmsnorm
            ):
                supported_classes.append(
                    type(_model.transformer.encoder.layers[0].input_layernorm)
                )
            if _model.config.architectures[0] == "QWenLMHeadModel":
                supported_classes.append(type(_model.transformer.h[0].ln_1))
            if _model.config.architectures[0] == "Qwen2ForCausalLM":
                supported_classes.append(
                    transformers.models.qwen2.modeling_qwen2.Qwen2RMSNorm
                )
            if hasattr(transformers.models, "mistral"):
                supported_classes.append(
                    transformers.models.mistral.modeling_mistral.MistralRMSNorm
                )
            if hasattr(transformers.models, "mixtral"):
                supported_classes.append(
                    transformers.models.mixtral.modeling_mixtral.MixtralRMSNorm
                )
            for supported_class in supported_classes:
                ipex.transformers.optimize.lowering_class_cpu(
                    _model,
                    supported_class,
                    ipex.transformers.models.cpu.fusions.mha_fusion._IPEXRMSNormCPU,
                    _model.config,
                    tpp=False,
                    woq=False,
                )

        for model_name in ["model", "transformer"]:
            if hasattr(_model, model_name) and hasattr(
                getattr(_model, model_name), "_use_sdpa"
            ):
                getattr(_model, model_name)._use_sdpa = False
            if hasattr(_model, model_name):
                cur_mod = getattr(_model, model_name)
                for submodel_name in ["encoder", "decoder"]:
                    if hasattr(cur_mod, submodel_name) and hasattr(
                        getattr(cur_mod, submodel_name), "_use_sdpa"
                    ):
                        getattr(cur_mod, submodel_name)._use_sdpa = False

        for supported_mlp_class in [
            ipex.transformers.models.reference.modules.decoder._IPEXDecoderLayerRef
        ]:
            ipex.transformers.optimize.lowering_class_cpu(
                _model,
                supported_mlp_class,
                ipex.transformers.models.cpu.modules.decoder._IPEXDecoderLayerCPU,
                _model.config,
                tpp=False,
                woq=woq,
            )

        for supported_mha_class in [
            ipex.transformers.models.reference.modules.attentions._IPEXAttentionRef
        ]:
            ipex.transformers.optimize.lowering_class_cpu(
                _model,
                supported_mha_class,
                ipex.transformers.models.cpu.modules.attentions._IPEXAttentionCPU,
                _model.config,
                tpp=False,
                woq=woq,
            )

    return _model


# Customize Model by applying modification based on the architecture
def customize_model(model):
    # Over riding forward class of ChatGLM by modifying the Rope op
    # to address the dimensionality error with compile flow.
    if model.config.architectures[0] == "ChatGLMModel":
        from intel_extension_for_pytorch.transformers.models.reference import (
            models,
        )
        from ._custom_model_forward import ChatGLMModel_forward
        models.ChatGLMModel_forward = (
            ChatGLMModel_forward
        )
        from intel_extension_for_pytorch.transformers.models.reference.modules import (
            attentions,
        )
        from ._custom_model_forward import _GLM2Attention_forward
        attentions._GLM2Attention_forward = (
            _GLM2Attention_forward
        )
    if model.config.architectures[0] == "MistralForCausalLM":
        from intel_extension_for_pytorch.transformers.models.reference import (
            models,
        )
        from ._custom_model_forward import MistralModel_forward
        models.MistralModel_forward = (
            MistralModel_forward
        )

    if model.config.architectures[0] == "MixtralForCausalLM":
        from intel_extension_for_pytorch.transformers.models.reference.modules import (
            decoder,
        )
        from intel_extension_for_pytorch.transformers.models.reference import (
            models,
        )
        from ._custom_model_forward import (
            MixtralDecoderLayer_forward, MixtralModel_forward
        )
        decoder.MixtralDecoderLayer_forward, models.MixtralModel_forward = (
            MixtralDecoderLayer_forward, MixtralModel_forward
        )
    # Over riding forward of RotaryEmbedding class for modifying
    # the longrope op to address graph breaks for v2.4 and above.
    installed_ipex_version = get_installed_ipex_version()
    installed_ipex_version = TorchVersion(installed_ipex_version)
    min_version = TorchVersion("2.4")
    if installed_ipex_version >= min_version:
        from intel_extension_for_pytorch.transformers.models.reference. \
            fusions.mha_fusion import RotaryEmbedding
        from ._custom_model_forward import RotaryEmbedding_forward
        RotaryEmbedding.forward = RotaryEmbedding_forward

        # Over riding _IPEXConcatLinearRef class to add qkv fusion support
        # for weight only quantized qkv(ZenTorchWOQLinear) modules
        from intel_extension_for_pytorch.transformers.models.reference.fusions import (
            linear_fusion,
        )
        from ._custom_models_reference_linear_fusion import _ZenTorchConcatLinearRef

        linear_fusion._IPEXConcatLinearRef = _ZenTorchConcatLinearRef

    return model
