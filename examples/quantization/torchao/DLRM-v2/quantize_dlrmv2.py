# *******************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# *******************************************************************
"""
DLRMv2 PT2E Quantization Script

Quantizes DLRMv2 model using TorchAO PT2E quantization:
- Linear layers: INT8 static (X86InductorQuantizer)
- Embedding tables: Optional UINT4 (EmbeddingBagUInt4Quantizer)
"""

# Standard library
import argparse
import logging
import os
import sys

# PyTorch
import torch
import torch.nn as nn
import torch._inductor.config as inductor_config
from torch.profiler import profile, ProfilerActivity

import zentorch

# Add DLRMv2 model directory to Python path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "model")
sys.path.insert(0, MODEL_DIR)

# DLRMv2 model (imports require MODEL_DIR on path)
from dlrm_model import DLRMMLPerf  # noqa: E402

# TorchAO quantization
from torchao.quantization.pt2e.quantize_pt2e import (  # noqa: E402
    prepare_pt2e,
    convert_pt2e,
)
import torchao.quantization.pt2e.quantizer.x86_inductor_quantizer as x86_inductor_quantizer  # noqa: E402
from torchao.quantization.pt2e.quantizer.x86_inductor_quantizer import (  # noqa: E402
    X86InductorQuantizer,
    get_default_x86_inductor_quantization_config,
)
from torchao.quantization.pt2e.quantizer.composable_quantizer import (  # noqa: E402
    ComposableQuantizer,
)

# Local
from custom_quantizers import EmbeddingBagUInt4Quantizer  # noqa: E402

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

inductor_config.freezing = True

# MLPerf DLRMv2 configuration
NUM_EMBEDDINGS_PER_FEATURE = [
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
NUM_DENSE_FEATURES = 13
NUM_SPARSE_FEATURES = 26
FIXED_INTERACTIONS_PER_SAMPLE = 3


def get_dlrm_model():
    """Create DLRMv2 model with MLPerf configuration."""
    model = DLRMMLPerf(
        embedding_dim=128,
        num_embeddings_pool=NUM_EMBEDDINGS_PER_FEATURE,
        dense_in_features=NUM_DENSE_FEATURES,
        dense_arch_layer_sizes=[512, 256, 128],
        over_arch_layer_sizes=[1024, 1024, 512, 256, 1],
        dcn_num_layers=3,
        dcn_low_rank_dim=512,
    )
    model.eval()
    logger.info(
        "Created DLRMv2: %d embeddings, %d dense features",
        len(NUM_EMBEDDINGS_PER_FEATURE),
        NUM_DENSE_FEATURES,
    )
    return model


def load_calibration_data(args):
    """Load calibration dataset for quantization."""
    from model import multihot_criteo

    # the datasets we support
    SUPPORTED_DATASETS = {
        "multihot-criteo": (
            multihot_criteo.MultihotCriteo,
            multihot_criteo.pre_process_criteo_dlrm,
            multihot_criteo.DlrmPostProcess(),
            {"randomize": "total", "memory_map": True},
        ),
    }

    if args.dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Unsupported dataset: {args.dataset_name}. "
            f"Supported datasets: {list(SUPPORTED_DATASETS.keys())}"
        )
    wanted_dataset, pre_proc, _, kwargs = SUPPORTED_DATASETS[args.dataset_name]

    ds = wanted_dataset(
        num_embeddings_per_feature=NUM_EMBEDDINGS_PER_FEATURE,
        data_path=args.dataset_path,
        name=args.dataset_name,
        pre_process=pre_proc,
        count=args.count_samples,
        samples_to_aggregate_fix=args.samples_to_aggregate_fix,
        samples_to_aggregate_min=args.samples_to_aggregate_min,
        samples_to_aggregate_max=args.samples_to_aggregate_max,
        samples_to_aggregate_quantile_file=(args.samples_to_aggregate_quantile_file),
        samples_to_aggregate_trace_file=(args.samples_to_aggregate_trace_file),
        max_ind_range=args.max_ind_range,
        **kwargs,
    )
    return ds


def quantize_model_pt2e(
    model,
    max_batchsize,
    calibration_data,
    quantize_embeddings_uint4=False,
):
    """Quantize DLRM model using PT2E composable quantization.

    Strategy:
    - Linear layers: INT8 static activations + INT8 weights
      (X86InductorQuantizer)
    - Embedding tables: Optional UINT4 per-channel weight
      quantization (EmbeddingBagUInt4Quantizer)

    Args:
        model: The DLRMv2 model to quantize.
        max_batchsize: Maximum batch size for calibration.
        calibration_data: Calibration dataset.
        quantize_embeddings_uint4: If True, quantize EmbeddingBag
            layers with UINT4.

    Returns:
        Quantized model.
    """
    dsx, lsi, lso, labels = calibration_data.test_data.load_batch(
        range(0, max_batchsize)
    )

    # Export model
    logger.info("Exporting model...")
    # As the batch-size is not fixed at export time; it can change when you
    # run the exported model. As such Batch is dynamic via B.
    B = torch.export.Dim("batch")
    # Offsets for embedding-bag style inputs are usually of length
    # batch-size + 1 (one per batch item plus an extra end index).
    # So each offset tensor’s first dimension is specified as BP1 in the export call
    BP1 = B + 1
    # `lsi` is the list/tuple of indices tensors (one per embedding table) from the
    # calibration batch. Each indices tensor in `lsi` has a dynamic length on
    # dimension 0 (with minimum 0), so the exported model works with varying
    # batch sizes and varying numbers of indices per table. The name `nnz_i` suggests
    # “number of non-zeros” for the i-th table; that dimension can vary in length.
    # `nnz_dims` is a tuple of dynamic dimensions for each embedding table, which
    # is a tuple of dictionaries with one entry for each dimension.
    nnz_dims = tuple({0: torch.export.Dim(f"nnz_{i}", min=0)} for i in range(len(lsi)))
    # As batch size, indices and offsets can vary we will be
    # supporting the exported model with dynamic shapes.
    with torch.no_grad():
        exported_model = torch.export.export(
            model,
            (dsx, lsi, lso),
            strict=True,
            dynamic_shapes=(
                {0: B},  # densex: [B, 13]
                nnz_dims,  # lsi: each indices_i is [nnz_i]
                tuple({0: BP1} for _ in lso),  # offsets: [B+1]
            ),
        ).module()

    # Create quantizers
    quantizers = []

    linear_quantizer = X86InductorQuantizer()
    linear_quantizer.set_global(
        get_default_x86_inductor_quantization_config(
            is_qat=False, is_dynamic=False, reduce_range=False
        )
    )
    quantizers.append(linear_quantizer)

    if quantize_embeddings_uint4:
        quantizers.append(EmbeddingBagUInt4Quantizer())

    composable_quantizer = ComposableQuantizer(quantizers)

    # Prepare, calibrate, convert
    logger.info("Preparing model for quantization...")
    prepared_model = prepare_pt2e(exported_model, composable_quantizer)

    logger.info("Calibrating model to quantize...")
    with torch.no_grad():
        prepared_model(dsx, lsi, lso)

    logger.info("Converting to quantized model...")
    quantized_model = convert_pt2e(prepared_model)

    return quantized_model


def test_and_profile_inference(
    model,
    batch_size=100,
    model_dtype=torch.float32,
    warmup_runs=3,
):
    """Test and profile model inference.

    Warmup first to exclude compilation overhead.
    """
    densex = torch.randn(batch_size, NUM_DENSE_FEATURES, dtype=model_dtype)
    index_list, offset_list = [], []

    for feat_idx in range(NUM_SPARSE_FEATURES):
        total_interactions = batch_size * FIXED_INTERACTIONS_PER_SAMPLE
        index_list.append(
            torch.randint(
                0,
                NUM_EMBEDDINGS_PER_FEATURE[feat_idx],
                (total_interactions,),
            )
        )
        offset_list.append(
            torch.arange(
                0,
                (batch_size + 1) * FIXED_INTERACTIONS_PER_SAMPLE,
                FIXED_INTERACTIONS_PER_SAMPLE,
                dtype=torch.long,
            )
        )

    with torch.no_grad():
        # Warmup runs to trigger compilation
        for _ in range(warmup_runs):
            model(densex, tuple(index_list), tuple(offset_list))

        # Profile actual inference (compilation already done)
        with profile(
            activities=[ProfilerActivity.CPU],
            record_shapes=True,
        ) as prof:
            output = model(densex, tuple(index_list), tuple(offset_list))

    logger.info(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
    logger.info(
        "Inference: output shape=%s, range=[%.4f, %.4f]",
        output.shape,
        output.min(),
        output.max(),
    )
    return output


def compare_models(
    original_model,
    quantized_model,
    batch_size=100,
    model_dtype=torch.float32,
    num_tests=5,
):
    """Compare outputs between original and quantized models."""
    differences = []

    for _ in range(num_tests):
        densex = torch.randn(batch_size, NUM_DENSE_FEATURES, dtype=torch.float32)

        index_list = []
        offset_list = []

        for feat_idx in range(NUM_SPARSE_FEATURES):
            total_interactions = batch_size * FIXED_INTERACTIONS_PER_SAMPLE
            indices = torch.randint(
                0,
                NUM_EMBEDDINGS_PER_FEATURE[feat_idx],
                (total_interactions,),
            )
            offsets = torch.arange(
                0,
                (batch_size + 1) * FIXED_INTERACTIONS_PER_SAMPLE,
                FIXED_INTERACTIONS_PER_SAMPLE,
                dtype=torch.long,
            )
            index_list.append(indices)
            offset_list.append(offsets)

        with torch.no_grad():
            original_output = original_model(densex, index_list, offset_list)
            densex = densex.to(dtype=model_dtype)
            quantized_output = quantized_model(
                densex, tuple(index_list), tuple(offset_list)
            )

        diff = torch.abs(original_output - quantized_output)
        differences.append((diff.mean().item(), diff.max().item()))

    avg_mean = sum(d[0] for d in differences) / len(differences)
    avg_max = sum(d[1] for d in differences) / len(differences)
    logger.info(
        "Model comparison: avg_mean_diff=%.6f, avg_max_diff=%.6f",
        avg_mean,
        avg_max,
    )


def _modify_ops_in_exported_module(
    mod,
    model_dtype=torch.float32,
):
    """Replace embedding_bag ops with zentorch equivalents."""
    import operator

    gm = mod
    g = gm.graph
    changed = False

    emb_bag_op = torch.ops.aten.embedding_bag.padding_idx
    dequant_per_ch = torch.ops.quantized_decomposed.dequantize_per_channel.default
    zentorch_emb_op = torch.ops.zentorch.zentorch_quant_embedding_bag.default
    dequant_per_tensor = torch.ops.quantized_decomposed.dequantize_per_tensor.default

    for node in list(g.nodes):
        if (
            node.op == "call_function"
            and node.target == operator.getitem
            and len(node.users) == 0
        ):
            g.erase_node(node)
            changed = True
        if node.op == "call_function" and node.target == emb_bag_op:
            with g.inserting_before(node):
                if node.args[0].target == dequant_per_ch:
                    current_node = node.args[0]
                    current_node.replace_all_uses_with(current_node.args[0])
                    g.erase_node(current_node)
                    changed = True
                padding_idx = -1 if node.args[8] is None else node.args[8]
                new_args = [
                    node.args[0],
                    node.args[1],
                    node.args[2],
                    4,
                    model_dtype,
                    node.args[3],
                    node.args[4],
                    node.args[5],
                    node.args[6],
                    node.args[7],
                    padding_idx,
                ]
                new_node = g.call_function(
                    zentorch_emb_op,
                    args=tuple(new_args),
                    kwargs={},
                )
            node.replace_all_uses_with(new_node)
            g.erase_node(node)
            changed = True

    for node in list(g.nodes):
        if (
            node.op == "call_function"
            and node.target == operator.getitem
            and len(node.users) == 1
            and node.args[0].target == zentorch_emb_op
        ):
            node.replace_all_uses_with(node.args[0])
            g.erase_node(node)
            changed = True
        if (
            node.op == "call_function"
            and node.target in [dequant_per_ch, dequant_per_tensor]
            and model_dtype == torch.bfloat16
        ):
            node.kwargs = {"out_dtype": model_dtype}
    if changed:
        g.lint()
        gm.recompile()
    return gm


def _modify_buffers_and_params_in_exported_module(
    exported_module,
    model_dtype=torch.float32,
):
    """Pack uint4 embedding weights and optionally convert bias to bf16."""
    from _pack import create_pack_method

    pack_method = create_pack_method("awq", "int4")
    logger.info(
        "Packing frozen uint8 buffers (containing uint4 values) for embedding bag weights..."
    )
    for name, buffer in exported_module.named_buffers():
        if "_frozen_param" in name and buffer.dtype == torch.uint8:
            packed_buffer = pack_method.pack(
                buffer.to(torch.int32),
                reorder=False,
                transpose=False,
            )
            scale_key = "_scale_" + name[13:]
            zp_key = "_zero_point_" + name[13:]
            scale_buffer = exported_module._buffers[scale_key]
            zp_buffer = exported_module._buffers[zp_key]
            packed_buffer = zentorch._C.zentorch_get_packed_embedding_weight(
                packed_buffer, scale_buffer, zp_buffer
            )
            # directly overwrite _buffers
            exported_module._buffers[name] = packed_buffer
    logger.info(
        "Packed frozen uint8 buffers (containing uint4 values) for embedding bag weights"
    )
    if model_dtype == torch.bfloat16:
        logger.info("Converting bias parameters to BF16 in the quantized model...")
        converted_count = 0
        for name, param in exported_module.named_parameters():
            if "bias" in name and param.dtype == torch.float32:
                # modify param.data because param is nn.Parameter
                param.data = param.data.to(model_dtype)
                converted_count += 1
        logger.info(
            "Successfully converted %d bias parameters to %s",
            converted_count,
            model_dtype,
        )


def _build_dynamic_shapes(lsi, lso):
    """Build dynamic shape specs for torch.export."""
    B = torch.export.Dim("batch")
    BP1 = B + 1
    nnz_dims = tuple({0: torch.export.Dim(f"nnz_{i}", min=0)} for i in range(len(lsi)))
    shapes = (
        {0: B},
        nnz_dims,
        tuple({0: BP1} for _ in lso),
    )
    return shapes


def main():
    """Main quantization entry point."""
    parser = argparse.ArgumentParser(description="DLRMv2 PT2E Quantization")
    parser.add_argument(
        "--pretrained-weights",
        type=str,
        required=True,
        help="Path to weights ('random' for random init)",
    )
    parser.add_argument(
        "--model-save-path",
        type=str,
        default="",
        help="Path to save quantized model",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--max-batchsize",
        type=int,
        default=128000,
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="inductor",
        choices=["inductor", "zentorch"],
    )
    parser.add_argument(
        "--num-calibration-samples",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--quantize-embeddings-uint4",
        action="store_true",
        help="Enable UINT4 embedding quantization",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="multihot-criteo",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to the dataset for calibration",
    )
    parser.add_argument(
        "--max-ind-range",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--count-samples",
        type=int,
        help="dataset items to use",
    )
    parser.add_argument(
        "--samples-to-aggregate-fix",
        type=int,
        help="number of samples to be treated as one",
    )
    parser.add_argument(
        "--samples-to-aggregate-min",
        type=int,
        help=("min number of samples to be treated as one " "in random query size"),
    )
    parser.add_argument(
        "--samples-to-aggregate-max",
        type=int,
        help=("max number of samples to be treated as one " "in random query size"),
    )
    parser.add_argument(
        "--samples-to-aggregate-quantile-file",
        type=str,
        help=(
            "distribution quantile used to generate number "
            "of samples to be treated as one in random "
            "query size"
        ),
    )
    parser.add_argument(
        "--samples-to-aggregate-trace-file",
        type=str,
        default="dlrm_trace_of_aggregated_samples.txt",
    )
    parser.add_argument(
        "--bfloat16",
        action="store_true",
        help=("Set model to bfloat16 after quantization"),
    )
    parser.add_argument(
        "--no-mul-quantization",
        action="store_true",
        help=(
            "Disable quantization of multiplication operations "
            "(useful when mul op quantization degrades accuracy)"
        ),
    )

    args = parser.parse_args()

    model_dtype = torch.float32
    if args.bfloat16:
        model_dtype = torch.bfloat16

    if args.no_mul_quantization:
        x86_inductor_quantizer.default_quantizable_ops.discard(
            torch.ops.aten.mul.Tensor
        )

    logger.setLevel(logging.INFO)

    logger.info("=" * 60)
    logger.info("DLRMv2 PT2E Quantization")
    logger.info("  Linear: INT8 (X86InductorQuantizer)")
    if args.quantize_embeddings_uint4:
        emb_info = (
            "UINT4 (Limiting int range to 0-15) as uint4 is "
            "not a native dtype in PyTorch."
        )
    else:
        emb_info = "FP32"
    logger.info("  Embeddings: %s", emb_info)
    logger.info("=" * 60)

    # Load model
    model = get_dlrm_model()

    if args.pretrained_weights is None:
        raise ValueError(
            "--pretrained-weights is required; use '--pretrained-weights random' "
            "for random initialization."
        )

    if args.pretrained_weights.lower() == "random":
        logger.info("Using random weights")
        for module in model.modules():
            if isinstance(module, torch.nn.EmbeddingBag):
                nn.init.uniform_(module.weight, -0.1, 0.1)
    else:
        logger.info("Loading weights from %s", args.pretrained_weights)
        state_dict = torch.load(
            args.pretrained_weights,
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(state_dict)

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # Load calibration data
    calibration_data = load_calibration_data(args)

    # Quantize
    quantized_model = quantize_model_pt2e(
        model,
        args.max_batchsize,
        calibration_data,
        args.quantize_embeddings_uint4,
    )

    dsx, lsi, lso, labels = calibration_data.test_data.load_batch(
        range(0, args.max_batchsize)
    )
    # Re-export quantized graph module to generate ExportedProgram
    dynamic_shapes = _build_dynamic_shapes(lsi, lso)

    with torch.no_grad():
        exported_program = torch.export.export(
            quantized_model,
            (dsx, lsi, lso),
            strict=True,
            dynamic_shapes=dynamic_shapes,
        )
        # Exported program will have all params and buffers from
        # quantized_model but module from exported program will
        # have only required params and buffers; it will remove
        # unused params.
        exported_module = exported_program.module()

    # Re-create ExportedProgram with exported_module to ensure
    # unused params are removed from saved model.
    with torch.no_grad():
        exported_program_to_save = torch.export.export(
            exported_module,
            (dsx, lsi, lso),
            strict=True,
            dynamic_shapes=dynamic_shapes,
        )

        ep_module = exported_program_to_save.module()
        _modify_buffers_and_params_in_exported_module(ep_module, model_dtype)
        # Modify exported module to use zentorch ops
        ep_module = _modify_ops_in_exported_module(ep_module, model_dtype)

    dsx = dsx.to(dtype=model_dtype)

    with torch.no_grad():
        exported_program_to_save = torch.export.export(
            ep_module,
            (dsx, lsi, lso),
            strict=True,
            dynamic_shapes=dynamic_shapes,
        )

    # Re-export once again to remove unused scales / zero-points
    with torch.no_grad():
        exported_program_to_save = torch.export.export(
            exported_program_to_save.module(),
            (dsx, lsi, lso),
            strict=True,
            dynamic_shapes=dynamic_shapes,
        )

    # Save the exported, quantized model
    logger.info("Saving quantized_dlrmv2_model...")

    if model_dtype == torch.bfloat16:
        file_ext = "-bfloat16.pt2"
    else:
        file_ext = ".pt2"
    model_save_file_name = (
        "export_quantized_dlrmv2_model_with_torchao_and_pt2e" + file_ext
    )
    model_save_path = (
        model_save_file_name
        if args.model_save_path == ""
        else os.path.join(args.model_save_path, model_save_file_name)
    )
    torch.export.save(exported_program_to_save, model_save_path)
    logger.info("Saved quantized_dlrmv2_model to file %s", model_save_path)

    # Compile
    logger.info("Compiling with %s backend...", args.backend)
    with torch.no_grad():
        if args.backend == "zentorch":
            if not inductor_config.pattern_matcher:
                logger.warning("Pattern matcher was disabled, enabling it...")
                inductor_config.pattern_matcher = True
            logger.info(
                "Pattern matcher enabled: %s",
                inductor_config.pattern_matcher,
            )
            compiled_model = torch.compile(
                exported_program_to_save.module(),
                backend="zentorch",
            )
        else:
            compiled_model = torch.compile(
                exported_program_to_save.module(),
                backend="inductor",
            )

    # Test
    test_and_profile_inference(compiled_model, args.batch_size, model_dtype)

    # Compare
    if model_dtype != torch.bfloat16:
        compare_models(model, compiled_model, args.batch_size, model_dtype)
    else:
        logger.info(
            "Skipping comparison for bfloat16 model because "
            "when we modify the bias to bfloat16 inplace in "
            "the exported module, it will affect the original "
            "model's bias as well because they are sharing the "
            "same parameter and weight param will still be "
            "float32 in original model, which causes the model "
            "to throw error during comparison."
        )

    logger.info("=" * 60)
    logger.info("Quantization completed successfully!")
    logger.info("=" * 60)

    return quantized_model


if __name__ == "__main__":
    quantized_model = main()
