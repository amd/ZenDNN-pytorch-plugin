# DLRMv2 PT2E Quantization

Post-training quantization (PTQ) for DLRMv2 using PyTorch 2 Export (PT2E) quantization framework.

## Overview

| Layer Type | Quantization | Quantizer |
|------------|--------------|-----------|
| Linear | INT8 static (weights + activations) | `X86InductorQuantizer` |
| Embedding | Optional UINT4 (weights only) | `EmbeddingBagUInt4Quantizer` |

## Files

```
DLRM-v2/
├── _pack.py                 # INT4 weight packing utilities
├── custom_quantizers.py     # EmbeddingBagUInt4Quantizer
├── quantize_dlrmv2.py       # Main quantization script
├── README.md
└── model/
    ├── dataset.py           # Base Dataset class
    ├── dlrm_model.py        # DLRMMLPerf model definition
    └── multihot_criteo.py   # Multi-hot Criteo dataset loader
```

## Usage

```bash
# With pre-trained weights and zentorch backend
python quantize_dlrmv2.py \
    --pretrained-weights /path/to/dlrm-multihot-pytorch.pt \
    --dataset-path /path/to/multihot-criteo-dataset/ \
    --quantize-embeddings-uint4 \
    --backend zentorch \
    --no-mul-quantization \
    --batch-size 100  \
    --max-batchsize 128000 \
#    --model-save-path /path/to/save-model  # Optional
```

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--pretrained-weights` | (required path) | Path to weights or `random` |
| `--dataset-path` | (required path) | Path to dataset |
| `--quantize-embeddings-uint4` | False | Enable UINT4 embedding quantization |
| `--backend` | inductor | Compilation backend: `inductor` or `zentorch` |
| `--batch-size` | 100 | Batch size for testing |
| `--max-batchsize` | 128000 | Max batch size for calibration |
| `--no-mul-quantization` | False | Disable the quantization of mul op |
| `--model-save-path` | "" | Specific path to save the quantized model |


## Quantization Details

### Linear Layers (X86InductorQuantizer)
- **Weights**: INT8 per-channel symmetric
- **Activations**: UINT8 per-tensor asymmetric (static)
- **Calibration**: Required for activation range statistics

### Embedding Tables (EmbeddingBagUInt4Quantizer)
- **Weights**: UINT4 per-channel (stored as uint8 with range 0-15)
- **Targets**: `aten.embedding.default`, `aten.embedding_bag.default`
- **Note**: PyTorch doesn't have native uint4, so we use uint8 with restricted range

## Quantized Weight Packing (format required by ZenDNN int4 embedding bag ops)
As PyTorch native does not have uint4, we will pack them with the custom packing method.
- **_modify_buffers_and_params_in_exported_module**: Finds UINT4 embedding buffers, bit-packs them, fuses with scale/zero_point via ZenDNN, overwrites module buffers, and optionally casts bias to bf16.
    - **Pack method**: Packs 8 consecutive 4-bit values into one int32 (bitwise) so shape goes from [R, C] to [R, C//8].
    - **zentorch_get_packed_embedding_weight**: Takes packed int32 weight plus per-row scale and zero_point, computes per-row bias as -zero_point * scale, and returns one buffer per table where each row is the packed int32s followed by two fp16 values (scale and bias) for fused lookup and dequant.
- **Packed Weights**: EmbeddingBag weights only (when `--quantize-embeddings-uint4` is used).
- The final packed weights are in the format required for ZenDNN’s int4 embedding-bag ops.

## Why `--no-mul-quantization` flag?
- X86InductorQuantizer by default enables the mul op quantization with linear layer quantization.
- To disable this we have introduced `--no-mul-quantization` flag.
- Quantizing mul can hurt accuracy (e.g. in DLRM’s dense/over arch or interactions). `--no-mul-quantization` flag is useful when mul op quantization degrades accuracy. It keeps mul in higher precision and often improves or restores accuracy.

## Output

- Console: Profiling results, quantization stats, accuracy comparison
- **Export Quantized Model** at specified path or current path

## Programmatic Usage

```python
from custom_quantizers import EmbeddingBagUInt4Quantizer
from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e
from torchao.quantization.pt2e.quantizer.x86_inductor_quantizer import (
    X86InductorQuantizer,
    get_default_x86_inductor_quantization_config,
)
from torchao.quantization.pt2e.quantizer.composable_quantizer import ComposableQuantizer

# Setup quantizers
linear_quantizer = X86InductorQuantizer()
linear_quantizer.set_global(get_default_x86_inductor_quantization_config(
    is_qat=False, is_dynamic=False, reduce_range=False
))

quantizer = ComposableQuantizer([
    linear_quantizer,
    EmbeddingBagUInt4Quantizer(),  # Optional
])

# Export → Prepare → Calibrate → Convert
exported = torch.export.export(model, example_inputs, strict=True).module()
prepared = prepare_pt2e(exported, quantizer)

for batch in calibration_data:
    prepared(*batch)

quantized = convert_pt2e(prepared)

# Compile with backend
compiled = torch.compile(quantized, backend="inductor")
```

## Requirements

- PyTorch 2.x with PT2E support (recommended version- 2.10.0)
- TorchAO (`torchao.quantization.pt2e`) (recommended version- 0.15.0)
- DLRMv2 model (`dlrm_model.DLRMMLPerf`)
- Dataset(`multihot-criteo`)
- scikit-learn
- zentorch (recommended version- 5.2.0)

## ROC AUC Accuracy Scores

The expected ROC AUC accuracy listed in the following table.
<table>
  <tr>
   <td><strong>Model</strong></td>
   <td><strong>ROC AUC</strong></td>
  </tr>
  <tr>
   <td> fp32 DLRM-v2</td>
   <td>0.8031</td>
  </tr>
  <tr>
   <td>export quant fp32 DLRM-v2</td>
   <td>0.802717</td>
  </tr>
  <tr>
   <td>export quant bf16 DLRM-v2</td>
   <td>0.802710</td>
  </tr>
</table> 