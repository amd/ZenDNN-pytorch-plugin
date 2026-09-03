# Fused FFN Layer Support for vLLM and Zentorch

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Motivation](#motivation)
3. [Architecture Overview](#architecture-overview)
4. [Design Principles](#design-principles)
5. [Implementation Details](#implementation-details)
6. [API Reference](#api-reference)
7. [Integration Flow](#integration-flow)
8. [Testing Strategy](#testing-strategy)
9. [Performance Characteristics](#performance-characteristics)
10. [Deployment Guide](#deployment-guide)
11. [Troubleshooting](#troubleshooting)
12. [Future Enhancements](#future-enhancements)

---

## Executive Summary

This document describes the design and implementation of fused Feed-Forward Network (FFN) layer support for vLLM inference engine using AMD's Zentorch library. The solution provides **expected 2-4x speedup (untested)** on FFN layers for AMD Zen CPU platforms through kernel fusion, with **zero modifications to model files** and automatic activation for compatible models.

### Key Achievements

- ✅ **Native C++ operator** — `zentorch_fused_ffn_concat.out`, registered in the zentorch extension, delegating the fused W13 → gated activation → W2 chain to `zentorch_group_matmul_out_impl`
- ✅ **torch.compile support** — a dedicated Inductor lowering + AOTI C-shim (`aoti_torch_cpu_zentorch_fused_ffn_concat_out`), mirroring the `zentorch_fused_moe` routing (no `make_fallback`)
- ✅ **Out-variant operator design** compliant with PyTorch 2.12+ requirements
- ✅ **Generic MLP fusion** across all vLLM models with compatible structure
- ✅ **Zero model / vLLM file modifications** — enablement is a zentorch-side monkey-patch, opt-in via `ZENTORCH_FUSED_FFN=1`
- ✅ **Automatic pattern detection** for Pattern A (merged gate_up) and Pattern B (separate gate/up)
- ✅ **All gated activations the op accepts** — `silu`, `gelu`, `gelu_tanh`, `swigluoai`
- ✅ **Hypothesis-based op tests** reusing the shared group-matmul data strategy
- ✅ **Memory efficiency** with automatic weight deduplication

### Performance Impact

**Note**: Performance improvements listed below are theoretical/untested. Actual performance may vary.

| Metric | Expected Improvement |
|--------|---------------------|
| FFN Layer Speedup | 2-4x (untested) |
| End-to-End Inference | 10-20% (untested) |
| Activation Memory | Potential reduction (intermediate activations may not be written to memory, but not guaranteed) |

---

## Motivation

### Problem Statement

Feed-Forward Network (FFN) layers in transformer models:
- Account for **40-60% of total inference time**
- Consist of multiple sequential operations (gate projection, up projection, activation, down projection)
- Suffer from memory bandwidth bottlenecks due to separate kernel launches
- Result in suboptimal performance on CPU inference workloads

### Solution Overview

The fused FFN implementation combines multiple operations into a single optimized kernel:
- **Gate projection** (w1): `hidden → intermediate`
- **Up projection** (w3): `hidden → intermediate`
- **Gated activation** (SwiGLU/GeGLU): `activation(w1) * w3`
- **Down projection** (w2): `intermediate → hidden`

All fused into: `w2 @ (activation(w1 @ x) * (w3 @ x))`

### Why This Approach?

1. **Reduced Memory Traffic**: Single kernel may reduce intermediate buffer writes/reads for activations (not guaranteed)
2. **Improved Cache Locality**: Data stays in cache across operations
3. **Fewer Kernel Launches**: Eliminates PyTorch kernel launch overhead
4. **Optimized for Zen CPUs**: Leverages AMD-specific SIMD optimizations via ZenDNN

**Note on MoE Support**: A PR for fused MoE (Mixture of Experts) support using the `zentorch_group_matmul.out` operator already exists and served as an inspiration for this fused FFN design. The current implementation focuses on single-expert FFN layers.

---

## Architecture Overview

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                         vLLM Inference                       │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │  Model Loading Pipeline                             │    │
│  │  ┌──────────────────────────────────────────────┐  │    │
│  │  │  1. initialize_model()                        │  │    │
│  │  │  2. install_fused_mlp_forwards_for_model()   │  │    │
│  │  │     └─> Monkey-patch MLP classes             │  │    │
│  │  │  3. load_weights()                            │  │    │
│  │  │  4. process_mlp_weights_after_loading()      │  │    │
│  │  │     └─> dispatch_cpu_fused_mlp()             │  │    │
│  │  └──────────────────────────────────────────────┘  │    │
│  └────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │  Inference Execution                                │    │
│  │  ┌──────────────────────────────────────────────┐  │    │
│  │  │  MLP.forward(x)                               │  │    │
│  │  │    ├─> cpu_mlp_forward set?                  │  │    │
│  │  │    │   └─> YES: fused_ffn_concat (fast path) │  │    │
│  │  │    └─> NO: original forward (fallback)        │  │    │
│  │  └──────────────────────────────────────────────┘  │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ torch.ops.zentorch.zentorch_fused_ffn_concat.out
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                      Zentorch Library (C++)                   │
│  ┌────────────────────────────────────────────────────┐    │
│  │  zentorch_fused_ffn_concat.out  (C++ out-variant)   │    │
│  │  ┌──────────────────────────────────────────────┐  │    │
│  │  │  • View input/output as 2D                    │  │    │
│  │  │  • zentorch_group_matmul_out_impl (1 expert): │  │    │
│  │  │      W13 GEMM → gated act → W2 GEMM           │  │    │
│  │  │  • Result written into `output` (gemm_output) │  │    │
│  │  └──────────────────────────────────────────────┘  │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### MLP Pattern Detection

The implementation supports the common MLP pattern found in transformer models:

**Pattern: Merged Gate-Up Projection**
```python
class MLP:
    gate_up_proj: Linear  # [2*intermediate, hidden]
    down_proj: Linear      # [hidden, intermediate]
    act_fn: SiluAndMul
```
---

## Design Principles

### 1. Zero Model File Modifications

**Principle**: All integration happens through runtime monkey-patching, requiring no changes to model definition files.

**Rationale**: 
- Maintains compatibility with upstream vLLM
- Simplifies maintenance and updates
- Enables/disables fusion without code changes
- Works across all models automatically

**Implementation**: 
- Detect MLP modules by attribute inspection
- Monkey-patch class forward methods
- Store fused weights as module attributes

### 2. Generic Pattern Detection

**Principle**: Detect MLP modules by structure, not by class name or import path.

**Rationale**:
- Works across all vLLM models without per-model code
- Future-proof against model architecture changes
- Supports custom/proprietary model architectures

**Implementation**:
```python
# Pattern detection
has_pattern = (hasattr(module, 'gate_up_proj') and 
                 hasattr(module, 'down_proj'))
```

### 3. Out-Variant Operator Design

**Principle**: Use PyTorch out-variant operator semantics (output tensor passed as parameter).

**Rationale**:
- Complies with PyTorch 2.12+ deprecation of in-place custom operators
- Explicit memory management
- Better integration with torch.compile
- Clearer operator semantics

**Implementation**: the operator is registered in C++ via `STABLE_TORCH_LIBRARY`
with a void-returning, output-mutating schema (the leading `Tensor(a!) output`
is mutated in place):

```
zentorch_fused_ffn_concat.out(
    Tensor(a!) output, Tensor input, Tensor w13_weight, Tensor w2_weight,
    Tensor? w13_bias=None, Tensor? w2_bias=None, str activation="silu",
    Tensor? w13_scale=None, Tensor? w2_scale=None, *,
    str zentorch_op_name="zentorch::fused_ffn_concat") -> ()
```

### 4. Fail-Safe Filtering

**Principle**: Only apply fusion when ALL conditions are met; gracefully fall back otherwise.

**Rationale**:
- Ensures correctness on unsupported platforms
- Prevents silent performance degradation
- Enables gradual rollout

**Implementation**: opt-in `ZENTORCH_FUSED_FFN=1` gate, then a per-module
checklist (Dtype fp32/bf16, supported gated Activation, single-expert / non-MoE,
2D weights present). Any failing check leaves the native `forward` in place.

### 5. Memory Efficiency

**Principle**: Free original weights after fusion to avoid duplication.

**Rationale**:
- Prevents duplication of MLP weights in memory
- Original weights (gate_up, down) are freed via `torch.empty(0)`
- Fused weights (w13, w2) remain in the model
- **Net weight memory savings**: None - fused weights occupy similar space as originals
- **Activation memory savings**: Potential reduction if intermediate activations are not written to memory, but not guaranteed

**Implementation**: Replace original weights with `torch.empty(0)` after fusion, store fused weights as module attributes

---

## Implementation Details

### Zentorch Components

#### 1. Weight Conversion Helpers

**Location**: `src/cpu/python/zentorch/vllm/_fused_mlp_patch.py`

**Purpose**: Convert vLLM weight layouts to the fused `[w1 | w3]` format.

**Functions**:

```python
def prepare_fused_ffn_weights(
    gate_up_weight: torch.Tensor,  # [2*intermediate, hidden]
    down_weight: torch.Tensor,      # [hidden, intermediate]
    gate_up_bias: Optional[torch.Tensor] = None,
    down_bias: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Pattern: gate_up already concatenated as [w1; w3].
    Pass-through since weights are already in correct format.
    
    Returns:
        (w13_weight, w2_weight, w13_bias, w2_bias)
    """
    return gate_up_weight, down_weight, gate_up_bias, down_bias
```

#### 2. Fused FFN Operator (C++, Out-Variant)

**Location**: `src/cpu/cpp/FusedFFN.cpp` / `FusedFFN.hpp`
(kernel `zentorch::zentorch_fused_ffn_concat_out_impl`)

**Schema** (registered via `STABLE_TORCH_LIBRARY_FRAGMENT` and
`STABLE_TORCH_LIBRARY_IMPL(zentorch, CPU)`):
```
zentorch_fused_ffn_concat.out(
    Tensor(a!) output, Tensor input, Tensor w13_weight, Tensor w2_weight,
    Tensor? w13_bias=None, Tensor? w2_bias=None, str activation="silu",
    Tensor? w13_scale=None, Tensor? w2_scale=None, *,
    str zentorch_op_name="zentorch::fused_ffn_concat") -> ()
```

**Implementation Details**:
- Views 3D `[batch, seq_len, hidden]` inputs/outputs as 2D
  `[batch*seq_len, hidden]`. `output` is never copied — it aliases the caller's
  buffer (the mutated `Tensor(a!)`); `input` is made contiguous if needed.
- Wraps each operand in a length-1 list and calls `zentorch_group_matmul_out_impl`
  as a single-expert group:
  - `gemm_outputs = [output_2d]` receives the W2 down-projection result
  - `w2_weights = [w2_weight]` enables the fused W13 → gated act → W2 chain
  - `moe_output` / `topk_weights` / `row_ptrs` are empty (no MoE reduce)
- The weight dtype selects the kernel inside `GroupMatmul` (bf16/f32/fp16,
  DA8W8 with `w13_scale`/`w2_scale`, or DA8W4).
- Void return; `output` is mutated in place.

**Key Design Decision**: the result is written through `gemm_outputs=[output]`
paired with a fused `w2_weights` entry (not `moe_output`). A non-empty
`gemm_outputs` with a fused w2 makes ZenDNN write the final down-projection
directly into the caller's `output`, and leaves `input` untouched. (An earlier
prototype that routed the result elsewhere left `output` uninitialized — the
root cause of the gsm8k exact_match=0 bug, now guarded by a regression test.)

#### 3. torch.compile Lowering + AOTI Shim

Like `zentorch_fused_moe`, the op is routed through a dedicated Inductor
lowering and AOTI C-shim (instead of `make_fallback`) so `cpp_wrapper` emits a
direct C-ABI call rather than the slow `custom_op_wrapper` Python path:

- **Meta / fake** (`src/cpu/python/zentorch/_meta_registrations.py`):
  `@register_meta("zentorch_fused_ffn_concat", "out")` returns `None`
  (out-variant). The op is deliberately **not** added to `make_fallback`.
- **Lowering** (`src/cpu/python/zentorch/_lowerings.py`):
  `_ZentorchFusedFFNConcat(_ZentorchVoidShimFallbackOutBase)` registered on
  `torch.ops.zentorch.zentorch_fused_ffn_concat.out`, with
  `_zen_shim_name = "aoti_torch_cpu_zentorch_fused_ffn_concat_out"`.
- **Shim** (`src/cpu/cpp/shim_cpu_zentorch.{hpp,cpp}`):
  `aoti_torch_cpu_zentorch_fused_ffn_concat_out(...)` bridges the C-ABI handles
  to the kernel.

### vLLM Integration Components

#### 1. Operator Registration (in zentorch, not vLLM)

The operator is registered natively in the zentorch C++ extension (see the
Fused FFN Operator above); vLLM no longer registers it via
`direct_register_custom_op`. Consequences:

- The op, its meta/fake impl, and its torch.compile lowering all ship with
  zentorch and are available as
  `torch.ops.zentorch.zentorch_fused_ffn_concat.out` as soon as
  `import zentorch` runs.
- The vLLM integration is a **zentorch-side plugin** (out-of-tree monkey-patch),
  so no files under `vllm/` are modified.

#### 2. MLP Fusion Module

**Location**: `src/cpu/python/zentorch/vllm/_fused_mlp_patch.py`
(shipped with zentorch; opt-in via `ZENTORCH_FUSED_FFN=1`)

**Core Functions**:

**a. Activation Mapping**:
```python
ACTIVATION_MAPPING = {
    'SiluAndMul': 'silu',
    'GeluAndMul': 'gelu',
    'NewGELU': 'gelu_tanh',
}

def _get_activation_string(act_fn) -> Optional[str]:
    """Convert vLLM activation function to zentorch activation string."""
    if act_fn is None:
        return None
    act_name = type(act_fn).__name__
    return ACTIVATION_MAPPING.get(act_name)
```

**b. Fusion Eligibility Check**:
```python
def _should_use_fused_ffn(activation, dtype, num_experts=1) -> bool:
    """Fuse only for fp32/bf16, a supported gated act, single expert."""
    if dtype not in (torch.float32, torch.bfloat16):
        return False
    if activation not in _SUPPORTED_MOE_ACTIVATIONS:
        return False
    return True
```

> **Note on scope**: the C++ operator itself accepts more than this path uses —
> `swigluoai` in addition to `silu`/`gelu`/`gelu_tanh`, plus fp16 and int8
> (DA8W8/DA8W4 via `w13_scale`/`w2_scale`). The vLLM auto-fusion path stays on
> fp32/bf16 + `silu`/`gelu`/`gelu_tanh` because those are the only vLLM
> activation classes wired up in `ACTIVATION_MAPPING`. Whole-feature enablement
> is gated by the `ZENTORCH_FUSED_FFN=1` environment variable.

**c. Fusion Initialization**:
```python
def dispatch_cpu_fused_mlp(
    mlp_module: torch.nn.Module,
    activation_fn,
    num_experts: int = 1,
    remove_weights: bool = True
) -> None:
    """Initialize MLP fusion - sets up mlp_module.cpu_mlp_forward."""
    
    # Check if weights are already empty (already fused or freed)
    if gate_up_weight.numel() == 0 or down_weight.numel() == 0:
        logger.debug_once("CPU MLP fusion: weights already empty, skipping")
        return
    
    # Check if weights are 2D
    if gate_up_weight.dim() != 2 or down_weight.dim() != 2:
        logger.debug_once("CPU MLP fusion: weights not 2D, skipping")
        return
    
    # Extract and clone weights
    w13_weight_data = gate_up_weight.clone().detach().contiguous()
    w2_weight_data = down_weight.clone().detach().contiguous()
    ...
    
    # Prepare fused weights
    w13, w2, w13_bias, w2_bias = prepare_fused_ffn_weights_from_merged(...)
    
    # Store fused weights as module attributes
    mlp_module._fused_w13_weight = w13
    mlp_module._fused_w2_weight = w2
    mlp_module._fused_w13_bias = w13_bias
    mlp_module._fused_w2_bias = w2_bias
    mlp_module._fused_activation = activation
    
    # Create fused forward function
    def fused_forward_fn(x):
        # Allocate output tensor (same shape as input)
        output = torch.empty_like(x)
        # Call the C++ out-variant op
        torch.ops.zentorch.zentorch_fused_ffn_concat.out(
            output,
            x,
            w13_weight=mlp_module._fused_w13_weight,
            w2_weight=mlp_module._fused_w2_weight,
            w13_bias=mlp_module._fused_w13_bias,
            w2_bias=mlp_module._fused_w2_bias,
            activation=mlp_module._fused_activation
        )
        return output
    
    mlp_module.cpu_mlp_forward = fused_forward_fn
    
    # Free original weights if requested
    if remove_weights:
        mlp_module.gate_up_proj.weight = nn.Parameter(torch.empty(0), requires_grad=False)
        mlp_module.down_proj.weight = nn.Parameter(torch.empty(0), requires_grad=False)
```

**d. Monkey-Patching**:
```python
_ORIGINAL_MLP_FORWARDS: Dict[type, Callable] = {}

def install_fused_mlp_forward(mlp_class: type) -> None:
    """Monkey-patch MLP class to use fused forward when available."""
    if mlp_class in _ORIGINAL_MLP_FORWARDS:
        return  # Already patched
    
    # Save original forward
    _ORIGINAL_MLP_FORWARDS[mlp_class] = mlp_class.forward
    
    # Replace with fused version
    def fused_forward(self, x):
        if hasattr(self, 'cpu_mlp_forward') and self.cpu_mlp_forward is not None:
            return self.cpu_mlp_forward(x)  # Fused path
        return _ORIGINAL_MLP_FORWARDS[type(self)](self, x)  # Fallback
    
    mlp_class.forward = fused_forward
```

#### 3. Model Loader Integration (zentorch-side, no vLLM edits)

**Location**: `src/cpu/python/zentorch/vllm/_fused_mlp_patch.py`

Rather than editing vLLM, the plugin **wraps**
`vllm.model_executor.model_loader.base_loader.process_weights_after_loading`
through a deferred post-import hook (`patch_now_or_on_import`). The wrapper runs
fusion *before* the original (which contains the quant-processing loop), so the
required "fuse before weights are freed" ordering is preserved:

```python
def _zen_process_weights_after_loading(model, *args, **kwargs):
    try:
        install_fused_mlp_forwards_for_model(model)   # patch MLP.forward
        process_mlp_weights_after_loading(model)      # build fused weights
    except Exception:
        logger.warning("[zentorch] fused MLP setup failed; using native MLP",
                       exc_info=True)
    return orig_pwal(model, *args, **kwargs)          # then quant processing
```

The wrapper is armed only when `ZENTORCH_FUSED_FFN=1` (see
`_apply_fused_mlp_patch_impl`); otherwise the native MLP path is used unchanged.

**Why BEFORE the original `process_weights_after_loading`?**
- Even unquantized (FP32/BF16) models have `UnquantizedLinearMethod` as quant_method
- `quant_method.process_weights_after_loading()` may free/process original weights
- MLP fusion needs access to original weights before they're freed
- This was the root cause of "weights already empty" errors

**Generic MLP Detection**:
```python
def _is_mlp_module(module):
    """Structural MLP detection (Pattern A or Pattern B)."""
    has_pattern_a = hasattr(module, "gate_up_proj") and hasattr(module, "down_proj")
    has_pattern_b = (
        hasattr(module, "gate_proj")
        and hasattr(module, "up_proj")
        and hasattr(module, "down_proj")
    )
    return has_pattern_a or has_pattern_b

def install_fused_mlp_forwards_for_model(model):
    """Patch `forward` on every detected MLP class in the model."""
    patched_classes = set()
    for module in model.modules():
        if _is_mlp_module(module):
            mlp_class = type(module)
            if mlp_class not in patched_classes:
                install_fused_mlp_forward(mlp_class)
                patched_classes.add(mlp_class)

def process_mlp_weights_after_loading(model):
    """Build fused weights for every eligible MLP module."""
    for module in model.modules():
        if _is_mlp_module(module):
            act_fn = getattr(module, "act_fn", None)
            num_experts = getattr(module, "num_experts", 1)
            dispatch_cpu_fused_mlp(module, act_fn, num_experts, remove_weights=True)
```

---

## API Reference

### Zentorch API

#### `zentorch_fused_ffn_concat.out` (C++ Out-Variant Operator)

**Signature**:
```C++
torch.ops.zentorch.zentorch_fused_ffn_concat.out(
    output: torch.Tensor,       # Tensor(a!) — mutated in place
    input: torch.Tensor,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    w13_bias: Optional[torch.Tensor] = None,
    w2_bias: Optional[torch.Tensor] = None,
    activation: str = "silu",
    w13_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    *,
    zentorch_op_name: str = "zentorch::fused_ffn_concat",
) -> None
```

**Parameters**:
- `output` (Tensor): Pre-allocated output tensor, shape `[batch, seq_len, hidden]` or `[tokens, hidden]` (mutated in place)
- `input` (Tensor): Input activations, shape `[batch, seq_len, hidden]` or `[tokens, hidden]`
- `w13_weight` (Tensor): Concatenated gate|up weights, shape `[2*intermediate, hidden]`
- `w2_weight` (Tensor): Down projection weights, shape `[hidden, intermediate]`
- `w13_bias` (Tensor, optional): Concatenated gate|up bias, shape `[2*intermediate]`
- `w2_bias` (Tensor, optional): Down projection bias, shape `[hidden]`
- `activation` (str): `"silu"`, `"gelu"`, `"gelu_tanh"`, or `"swigluoai"`
- `w13_scale` (Tensor, optional): per-channel/per-group weight scale for quantized w13 (DA8W8/DA8W4)
- `w2_scale` (Tensor, optional): per-channel/per-group weight scale for quantized w2 (DA8W8/DA8W4)

**Returns**: None (output is mutated in-place)

**Supported Dtypes**: `torch.float32`, `torch.bfloat16`, `torch.float16`
(fp16 needs AVX512-FP16); quantized weights via `w13_scale`/`w2_scale` (bf16
activation). The vLLM auto-fusion path currently only drives fp32/bf16.

**Supported Activations** (forwarded to ZenDNN's `map_activation_to_gated_act`):
- `"silu"` - SwiGLU activation (most common)
- `"gelu"` / `"gelu_tanh"` - GeGLU (both route to `gelu_and_mul` / gelu_erf)
- `"swigluoai"` - SwiGLU-OAI (interleaved gate/up layout)

### vLLM API

#### `prepare_fused_ffn_weights`

Converts Pattern (merged gate_up) weights to fused format.

**Signature**:
```python
prepare_fused_ffn_weights_from_merged(
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    gate_up_bias: Optional[torch.Tensor] = None,
    down_bias: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]
```

**Returns**: `(w13_weight, w2_weight, w13_bias, w2_bias)`

#### `dispatch_cpu_fused_mlp`

Initializes MLP fusion for a given MLP module.

**Signature**:
```python
dispatch_cpu_fused_mlp(
    mlp_module: torch.nn.Module,
    activation_fn,
    num_experts: int = 1,
    remove_weights: bool = True
) -> None
```

**Parameters**:
- `mlp_module`: MLP module to fuse
- `activation_fn`: vLLM activation function instance (e.g., `SiluAndMul()`)
- `num_experts`: Number of experts (>1 for MoE, fusion skipped)
- `remove_weights`: Whether to free original weights after fusion

**Side Effects**:
- Sets `mlp_module.cpu_mlp_forward` if fusion succeeds
- Optionally frees original weights (sets to `torch.empty(0)`)

#### `install_fused_mlp_forward`

Monkey-patches an MLP class to use fused forward when available.

**Signature**:
```python
install_fused_mlp_forward(mlp_class: type) -> None
```

**Parameters**:
- `mlp_class`: MLP class to patch (e.g., `LlamaMLP`)

**Side Effects**:
- Replaces `mlp_class.forward` with wrapper that checks for fusion
- Stores original forward in `_ORIGINAL_MLP_FORWARDS`

---

## Integration Flow

### Model Loading Sequence

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. Model Initialization                                         │
│    - initialize_model(vllm_config, model_config, prefix)        │
│    - Model structure created, weights uninitialized             │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. Monkey-Patch Installation                                    │
│    - install_fused_mlp_forwards_for_model(model)                │
│    - Iterate through model.modules()                            │
│    - For each MLP class: install_fused_mlp_forward(mlp_class)   │
│    - MLP.forward now checks for cpu_mlp_forward                 │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. Weight Loading                                               │
│    - load_weights(model, model_config)                          │
│    - Weights loaded from checkpoint into module.weight          │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. MLP Fusion Processing (BEFORE quant_method!)                 │
│    - process_mlp_weights_after_loading(model)                   │
│    - Iterate through model.modules()                            │
│    - For each MLP instance: dispatch_cpu_fused_mlp()            │
│      • Check eligibility (platform, dtype, activation, etc.)    │
│      • Clone and prepare fused weights                          │
│      • Store as module attributes (_fused_w13_weight, etc.)     │
│      • Create fused_forward_fn and assign to cpu_mlp_forward    │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 5. Quant Method Processing                                      │
│    - quant_method.process_weights_after_loading(module)         │
│    - May free/process original weights                          │
│    - Fused weights already saved, unaffected                    │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 6. Model Ready for Inference                                    │
│    - Original weights may be freed                              │
│    - MLP.forward uses cpu_mlp_forward (fused path)              │
└─────────────────────────────────────────────────────────────────┘
```

### Inference Execution Flow

```
Input Tensor (x)
    │
    ↓
┌───────────────────────────────┐
│ MLP.forward(x)                │
│                                │
│ ┌───────────────────────────┐ │
│ │ Check fusion availability │ │
│ └───────────────────────────┘ │
└───────────────────────────────┘
    │
    ├─── has cpu_mlp_forward? ───┐
    │                             │
   YES                           NO
    │                             │
    ↓                             ↓
┌───────────────────────┐  ┌──────────────────────┐
│ Fused Path (Fast)     │  │ Fallback Path        │
│                       │  │                      │
│ 1. Allocate output    │  │ 1. gate_up_proj(x)   │
│    torch.empty_like() │  │ 2. act_fn(gate_up)   │
│                       │  │ 3. down_proj(act)    │
│ 2. fused_ffn_concat(  │  │                      │
│      output, x, ...)  │  │ return output        │
│                       │  │                      │
│ 3. return output      │  │                      │
└───────────────────────┘  └──────────────────────┘
    │                             │
    └─────────────┬───────────────┘
                  │
                  ↓
            Output Tensor
```

---

## Testing Strategy

### Unit Tests

**Location**: `test/unittests/op_tests/test_fused_ffn_concat_op.py`

These are zentorch operator-level tests for
`torch.ops.zentorch.zentorch_fused_ffn_concat.out`. They inherit
`GroupMatmulTestCase` and reuse its Hypothesis-driven data strategy — the
single-expert FFN uses the first expert's `[2*I, H]` / `[H, I]` gated weights —
mirroring the reference and tolerance conventions of `test_group_matmul.py`.

#### Test Categories

**1. Accuracy — 2D input** (`test_fused_ffn_accuracy_2d`)
- Iterates every activation in `_SUPPORTED_MOE_ACTIVATIONS`
  (`silu`, `gelu`, `gelu_tanh`, `swigluoai`), with and without bias, across all
  `supported_dtypes`.
- Compares against a pure-PyTorch `linear → gated act → linear` reference
  (`silu`/`gelu`/`gelu_tanh` use the split-half layout; `swigluoai` uses the
  interleaved gate/up layout).

**2. Accuracy — 3D input** (`test_fused_ffn_accuracy_3d`)
- `[B, S, H]` activation, exercising the op's internal flatten-to-2D path.

**3. Out-variant / gsm8k regression** (`test_output_buffer_populated_regression`)
- NaN-poisons `output` and asserts the fused result lands in `output` while
  `input` is left unchanged (guards the gsm8k exact_match=0 root cause).

**4. Error handling** (`test_unsupported_activation`)
- `"relu"` / `"tanh"` raise `RuntimeError` ("unsupported activation").

**Tolerances** (per dtype, mirroring `test_group_matmul.py`):
- FP32: `atol=rtol=1e-3`
- BF16 / FP16 fused chain: `atol=rtol=5e-1` (the `fused_bf16` band — the
  two-GEMM + gated-activation chain accumulates error at reduced precision).

Run with:
```bash
python -m unittest test.unittests.op_tests.test_fused_ffn_concat_op
```

### Integration Testing

**Test Procedure** (with `ZENTORCH_FUSED_FFN=1` on a Zen CPU):
1. Load a model with vLLM
2. Verify `zentorch::zentorch_fused_ffn_concat` appears in the execution trace
3. Run inference and compare outputs against the native MLP path
4. Monitor memory usage

**Expected Behavior**:
- FFN fusion enabled when `ZENTORCH_FUSED_FFN=1` and the MLP is eligible
- No warnings or errors during execution
- Numerical outputs match the unfused baseline (within tolerance)

---

## Performance Characteristics

### Expected Improvements

**IMPORTANT**: All performance improvements listed below are theoretical and untested. Actual results may vary significantly.

| Metric | Baseline | Expected with Fusion | Expected Improvement |
|--------|----------|---------------------|---------------------|
| FFN Layer Throughput | 100% | 200-400% | **2-4x (untested)** |
| End-to-End Latency | 100% | 80-90% | **10-20% faster (untested)** |
| Weight Memory | 100% | ~100% | **No net savings** - fused weights replace originals |
| Activation Memory | 100% | <100% | **Potential savings** - intermediate activations may not be written back (not guaranteed) |

### Performance Factors

**Positive Impact**:
- Larger batch sizes → Better amortization of kernel launch overhead
- Longer sequence lengths → More computation per kernel launch
- AMD Zen 3/4 CPUs → Better SIMD utilization
- BF16 precision → 2x memory bandwidth vs FP32

**Neutral/Negative Impact**:
- Very small batch sizes (batch=1, seq_len<128) → Kernel launch overhead dominant
- Non-Zen CPUs → Fusion disabled, no impact
- INT8 quantization → Fusion still helps, but quantization benefit larger

### Benchmark Methodology

**Micro-Benchmark** (FFN layer only):
```bash
cd /proj/rdi/staff/armukhop/ZenDNN_PyTorch_Plugin
python benchmarks/benchmark_fused_ffn.py \
    --batch-size 1,4,8 \
    --seq-len 128,512,2048 \
    --hidden-size 4096 \
    --intermediate-size 14336
```

**Macro-Benchmark** (End-to-end inference):
```bash
cd /proj/rdi/staff/armukhop/vllm
python benchmarks/benchmark_latency.py \
    --model meta-llama/Llama-3.1-8B \
    --batch-size 1 \
    --input-len 512 \
    --output-len 128 \
    --device cpu
```

---

## Deployment Guide

### Prerequisites

1. **Hardware**: AMD Zen4 CPU or above (the zentorch Zen CPU plugin requires Zen4+)
2. **Software**:
   - A supported CPU PyTorch build (see the repo README compatibility matrix)
   - zentorch built from this source tree (ships the C++ op + the vLLM plugin)
   - vLLM (CPU) for the end-to-end path

### Installation

**1. Build & install zentorch** (see the `build-zentorch-from-source` skill):
```bash
.claude/skills/build-zentorch-from-source/scripts/build.sh
```

**2. Install vLLM (CPU)** — any build whose torch matches zentorch's.

The fused-FFN operator, its meta impl, and its torch.compile lowering are part
of the zentorch wheel; no vLLM source changes are required. The vLLM MLP-fusion
plugin ships inside zentorch and is armed by `ZENTORCH_FUSED_FFN=1`.

### Verification

**1. Check the operator is registered**:
```python
import torch, zentorch
print(hasattr(torch.ops.zentorch, "zentorch_fused_ffn_concat"))
print(torch.ops.zentorch.zentorch_fused_ffn_concat.out)  # OpOverload
```

**2. Run the operator unit tests**:
```bash
python -m unittest test.unittests.op_tests.test_fused_ffn_concat_op
```

**3. Test End-to-End** (enable the vLLM MLP fusion):
```bash
ZENTORCH_FUSED_FFN=1 python -c "
from vllm import LLM
llm = LLM(model='meta-llama/Llama-3.1-8B', device='cpu')
print(llm.generate('Hello, world!'))
"
```

**4. Verify Fusion Active**: at debug log level, look for
```
[zentorch] CPU MLP fusion: fused_ffn_concat (Pattern A, activation=silu, dtype=torch.bfloat16)
```

### Configuration

**Environment Variables**:
- `ZENTORCH_FUSED_FFN=1` - Enable the vLLM dense-MLP → fused-FFN replacement
  (default: disabled)

**Runtime Control**:
When `ZENTORCH_FUSED_FFN=1`, an MLP module is fused only when all hold:
- Model has a compatible MLP structure (Pattern A or Pattern B)
- Dtype is FP32 or BF16
- Activation maps to SiLU, GELU, or GELU-tanh
- Not MoE with multiple experts

Any failing check leaves that module on its native `forward`.

---

## Troubleshooting

### Common Issues

#### 1. Fusion Not Activating

**Symptom**: No "[zentorch] CPU MLP fusion: fused_ffn_concat ..." log message

**Diagnosis**:
```bash
# Is the feature enabled?
echo "$ZENTORCH_FUSED_FFN"    # must be 1

# Operator availability
python -c "import torch, zentorch; print(hasattr(torch.ops.zentorch, 'zentorch_fused_ffn_concat'))"

# Enable debug logging
export VLLM_LOGGING_LEVEL=DEBUG
```

**Possible Causes**:
- `ZENTORCH_FUSED_FFN` not set to 1 → fusion disabled by default
- Not running on a Zen CPU → zentorch Zen plugin inactive
- zentorch not installed / import failed → install zentorch
- Model dtype not FP32/BF16 → fusion skipped
- Unsupported activation → check `ACTIVATION_MAPPING`

#### 2. Weights Already Empty Error

**Symptom**: "CPU MLP fusion: weights already empty, skipping"

**Root Cause**: MLP fusion called AFTER quant_method processing

**Solution**: Verify `process_mlp_weights_after_loading()` is called BEFORE quant_method loop in `process_weights_after_loading()`

#### 3. Out-Variant Deprecation Warning

**Symptom**: "UserWarning: zentorch::zentorch_fused_ffn_concat... in-place ... deprecated"

**Root Cause**: an older in-place operator signature being used

**Solution**: the current op is a native out-variant (`Tensor(a!) output`, `-> ()`);
rebuild zentorch so `torch.ops.zentorch.zentorch_fused_ffn_concat.out` is present.

#### 4. Test Failures

**Symptom**: `test_fused_ffn_concat_op` reports large mismatches at bf16/fp16.

**Root Cause**: comparing the fused (two-GEMM + gated-activation) chain at
reduced precision against an fp32 reference with a too-tight tolerance.

**Solution**: use the `fused_bf16` band (`atol=rtol=5e-1`) for bf16/fp16, as the
test already does (mirrors `test_group_matmul.py`).

### Debug Checklist

- [ ] zentorch installed and importable (`import zentorch` succeeds)
- [ ] `ZENTORCH_FUSED_FFN=1` set
- [ ] Running on an AMD Zen CPU
- [ ] Model uses FP32 or BF16
- [ ] Activation is SiLU, GELU, or GELU-tanh
- [ ] Not MoE with multiple experts
- [ ] `torch.ops.zentorch.zentorch_fused_ffn_concat` is available
- [ ] Logs show "[zentorch] CPU MLP fusion: fused_ffn_concat ..."

---

## Future Enhancements

### Planned Features

1. **INT8 Quantization Support**
   - Operator-level support already exists: pass `w13_scale`/`w2_scale` and the
     `GroupMatmul` backend runs DA8W8/DA8W4 dynamic-quant kernels
   - Remaining work: teach the vLLM auto-fusion path to detect quantized MLPs
     and supply the scales

2. **MoE (Mixture of Experts) Support**
   - The auto-fusion path skips MoE with `num_experts > 1`
   - The dedicated `zentorch_fused_moe` op already covers grouped MoE
   - This single-expert FFN op shares the same `zentorch_group_matmul` backend

3. **Additional Activations**
   - `silu`, `gelu`, `gelu_tanh`, `swigluoai` are supported today
   - Others (ReLU, Swish, Mish, ...) depend on ZenDNN kernel support; the
     vLLM auto-fusion path additionally needs an `ACTIVATION_MAPPING` entry

4. **FP16 Support**
   - Operator-level support exists (requires AVX512-FP16)
   - The vLLM auto-fusion path is still restricted to FP32/BF16 pending
     accuracy validation on AMD Zen CPUs

5. **torch.compile Integration**
   - Currently works but not optimized
   - Could benefit from custom fusion passes

### Research Directions

1. **Weight Compression**
   - Fused operator with compressed weight formats
   - GPTQ, AWQ, or custom compression

2. **Dynamic Batching**
   - Optimize for variable batch sizes
   - Adaptive kernel selection

3. **Multi-Socket NUMA**
   - NUMA-aware weight placement
   - Socket-local computation

---

## Appendix

### Commit History

**Zentorch Repository** (`zentorch_fused_ffn` branch):
- `7b8bf97d` - Add vLLM weight conversion helpers
- `6c179f51` - Fix zentorch_group_matmul out variant op call
- `378959ee` - Convert fused_ffn_concat to out-variant operator

**vLLM Repository** (`zentorch_fused_mlp` branch):
- `daed842c39` - Add generic MLP fusion for Zen CPU using fused_ffn_concat
- `e22789b3d0` - Add generic MLP fusion with custom op registration
- `4d63db9b42` - Fix test activation mapping for mock classes
- `f896e83000` - Fix weight extraction to clone tensors for fused FFN
- `6060590713` - Store fused weights as module attributes and ensure contiguity
- `4bff2a7af8` - Add early validation and skip fusion for empty/invalid weights
- `298631981d` - Move MLP fusion before quant_method processing (CRITICAL)
- `6d5c5501a5` - Convert fused_ffn_concat to out-variant operator in vLLM
- `c5df1b9d4f` - Update test mock to match out-variant fused_ffn_concat signature

### Related Documentation

- **ZenDNN Documentation**: [link]
- **vLLM Documentation**: https://docs.vllm.ai/
- **PyTorch Custom Operators**: https://pytorch.org/tutorials/advanced/custom_ops_landing_page.html
- **Transformer Architecture**: "Attention Is All You Need" (Vaswani et al., 2017)

### Contact

For questions or issues:
- **GitHub Issues**: [zentorch repo], [vLLM repo]
- **Email**: [team contact]

---

**Document Version**: 2.0  
**Last Updated**: August 31, 2026  
**Status**: Final — updated for the native C++ `zentorch_fused_ffn_concat.out`
operator (with Inductor lowering + AOTI shim) and the zentorch-side vLLM
MLP-fusion plugin (`ZENTORCH_FUSED_FFN=1`).
