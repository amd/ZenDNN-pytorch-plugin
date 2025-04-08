# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import zentorch
from dlrm_model import DLRMMLPerf
import torch
import os


def get_model():
    """Get the model."""

    num_embeddings_per_feature = [
        40000000,
        39060,
        17295,
        7424,
        20265,
        3,
        7122,
        1543,
        63,
        40000000,
        3067956,
        405282,
        10,
        2209,
        11938,
        155,
        4,
        976,
        14,
        40000000,
        40000000,
        40000000,
        590152,
        12973,
        108,
        36,
    ]
    embedding_dim = 128
    dcn_num_layers = 3
    dcn_low_rank_dim = 512
    dense_arch_layer_sizes = [512, 256, 128]
    over_arch_layer_sizes = [1024, 1024, 512, 256, 1]
    DEFAULT_INT_NAMES = [
        "int_0",
        "int_1",
        "int_2",
        "int_3",
        "int_4",
        "int_5",
        "int_6",
        "int_7",
        "int_8",
        "int_9",
        "int_10",
        "int_11",
        "int_12",
    ]

    return DLRMMLPerf(
        embedding_dim=embedding_dim,
        num_embeddings_pool=num_embeddings_per_feature,
        dense_in_features=len(DEFAULT_INT_NAMES),
        dense_arch_layer_sizes=dense_arch_layer_sizes,
        over_arch_layer_sizes=over_arch_layer_sizes,
        dcn_num_layers=dcn_num_layers,
        dcn_low_rank_dim=dcn_low_rank_dim,
    )


def get_compiled_model(args):
    model = get_model()
    if args.model == "quant32":
        try:
            model = zentorch.load_quantized_model(
                model, saved_model_path=args.model_path
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load quantized model from {args.model_path}. Error: {e}"
            ) from e
    elif args.model == "fp32":
        try:
            model.load_state_dict(
                torch.load(
                    os.path.join(args.model_path, "dlrm-multihot-pytorch.pt"),
                    weights_only=False,
                )
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load fp32 model from {args.model_path}. Error: {e}"
            ) from e
    elif args.model == "qdq_model":
        try:
            from quark.torch.quantization.api import load_params
        except ImportError as e:
            raise ImportError("Please install quark package to use qdq_model. ") from e
        try:
            model = load_params(
                model,
                json_path=os.path.join(args.model_path, "DLRM_INT.json"),
                safetensors_path=os.path.join(args.model_path, "DLRM_INT.safetensors"),
                compressed=True,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load qdq_model from {args.model_path}. Error: {e}"
            ) from e
    elif args.model == "quant16":
        try:
            model = model.to(dtype=torch.bfloat16)
            model = zentorch.load_quantized_model(
                model, saved_model_path=args.model_path
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load quant16 model from {args.model_path}. Error: {e}"
            ) from e
    else:
        raise ValueError(
            f"Unsupported model type: {args.model}. Supported types are: quant32, fp32, qdq_model, quant16."
        )
    print("Sharing memory", flush=True)
    model = model.cpu().share_memory()
    print("share_memory ready", flush=True)
    try:
        compiled_graph = torch.compile(model, backend="zentorch")
    except Exception as e:
        raise RuntimeError(f"Failed to compile model with zentorch. Error: {e}") from e

    return compiled_graph
