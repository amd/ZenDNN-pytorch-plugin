# `fused_recurrent_gated_delta_rule_packed_decode`

| Field | Value |
|---|---|
| Python op | `torch.ops.zentorch.gdn_fused_recurrent_gated_delta_rule_packed_decode` |
| C++ symbol | `zentorch::zentorch_gdn_fused_recurrent_gated_delta_rule_packed_decode` |
| Kernel impl | `src/cpu/cpp/kernels/layers/gdn/FusedRecurrentDecode.cpp` |
| Schema + binding | `src/cpu/cpp/GDN_ops.cpp` |
| Triton source | `vllm/model_executor/layers/fla/ops/fused_recurrent.py` |
| Tests + oracle | `test/unittests/op_tests/layers/gdn/test_fused_recurrent_gated_delta_rule_packed_decode.py` |
| Profiler span | `zentorch::gdn::fused_recurrent_gated_delta_rule_packed_decode` |
| Backends | `CPU`, `Meta` |
| ISA | **AVX-512** (per-source compile flags in `CMakeLists.txt`) |

## What this op does

The **decode hot path** for the GDN attention block. For each batch
entry (one decode token per row), reads the previous recurrent state
from the SSM cache slot identified by `ssm_state_indices[b]`, runs one
step of the gated delta-rule recurrence, writes the output `(B, 1, HV,
V)` and the updated state back to the cache. The op is a pure
side-effect entry point that mutates `out` and `initial_state` in
place and returns `()`.

Unlike `fused_sigmoid_gating_delta_rule_update` (which handles
spec-decode with `seqlen > 1`), this op handles strictly `seqlen = 1`
plain decode batches. The packing of `q`/`k`/`v` into a single
`mixed_qkv` tensor (the model's pre-conv `qkv_proj` output, unpacked
inline) is what gives this op its "packed" name.

This op is **AVX-512 vectorised**: the inner V*K state-row passes are
done via `__m512` intrinsics with per-pass fusion (the original 5
separate row passes are collapsed into 2 — `load_scale_dot` and
`update_state_and_compute_out` — so each state row is brought into L1
once per call).

## How it's wired into Qwen3.5 / Qwen3-Next GDN

In `gdn_linear_attn.py` (plain decode-only path, after the conv):

```python
fused_recurrent_gated_delta_rule_packed_decode(
    mixed_qkv=mixed_qkv_decode_non_spec,   # (B, qkv_dim)
    a=a_decode_non_spec,                    # (B, HV)
    b=b_decode_non_spec,                    # (B, HV)
    A_log=self.A_log,
    dt_bias=self.dt_bias,
    scale=self.scale,
    initial_state=ssm_state,                # (num_cache_lines, HV, V, K)
    out=core_attn_out_non_spec_decode,      # (B, 1, HV, V) — written in-place
    ssm_state_indices=non_spec_state_indices,
    use_qk_l2norm_in_kernel=True,
)
```

`forward_cpu_zen` substitutes the cpp op directly.

## Schema

```
zentorch::gdn_fused_recurrent_gated_delta_rule_packed_decode(
    Tensor mixed_qkv, Tensor a, Tensor b, Tensor A_log, Tensor dt_bias,
    float scale, Tensor(a!) initial_state, Tensor(b!) out,
    Tensor ssm_state_indices, bool use_qk_l2norm_in_kernel,
    *, str zentorch_op_name='zentorch::gdn_fused_recurrent_gated_delta_rule_packed_decode'
) -> ()
```

Returns `()` (void). Both `initial_state` and `out` are mutated in
place — `(a!)` and `(b!)` declare distinct alias sets so PyTorch's
dispatcher tracks them separately.

## Per-tensor layout contract

| Tensor | Shape | Dtype | Notes |
|---|---|---|---|
| `mixed_qkv` | `(B, qkv_dim)` with `qkv_dim = 2*H*K + HV*V` | fp32 or bf16 | inner dim unit-stride |
| `a`, `b` | `(B, HV)` | same as `mixed_qkv` | inner dim unit-stride |
| `A_log`, `dt_bias` | `(HV,)` | floating | contiguous |
| `initial_state` | `(num_cache_lines, HV, V, K)` | fp32 or bf16 | **inner `(V, K)` slab must be contiguous**; outer strides are read from the tensor (may be non-default, matching vLLM's hybrid KV cache layout) |
| `out` | `(B, 1, HV, V)` | same as `mixed_qkv` | inner dim unit-stride |
| `ssm_state_indices` | `(B,)` | int32 | contiguous |

`H = q_dim / K`, `HV = initial_state.size(1)`, `r = HV / H` (GQA ratio).
The op infers `H` from `qkv_dim - HV*V` and `K`.

## Hybrid precision contract

All compute runs in fp32. Inputs upcast at the load site, outputs cast
at the store site:

- `mixed_qkv` / `a` / `b` / `out` share `model_dtype` ∈ {fp32, bf16}.
- `initial_state` is independently `state_dtype` ∈ {fp32, bf16} —
  often differs from `model_dtype` (e.g. bf16 model with fp32 state).
- `A_log` and `dt_bias` are independently floating; the cpp upcasts
  them to fp32 once at op entry (cheap, `(HV,)`-sized cast).

Dispatch handles 4 `(model_t, state_t)` combinations via the
`dispatch_dtypes` switch.

## Math (per batch entry `b`, per V-head `hv`)

```python
slot = ssm_state_indices[b]
if slot <= 0:                            # NULL_BLOCK_ID
    out[b, 0, hv, :] = 0
    continue

i_h = hv // r
q = mixed_qkv[b, i_h*K : (i_h+1)*K]                    # (K,)
k = mixed_qkv[b, H*K + i_h*K : H*K + (i_h+1)*K]
v = mixed_qkv[b, 2*H*K + hv*V : 2*H*K + (hv+1)*V]      # (V,)

if use_qk_l2norm_in_kernel:
    q = q * rsqrt(sum(q^2) + 1e-6)
    k = k * rsqrt(sum(k^2) + 1e-6)
q *= scale

# Gating (per (b, hv) scalar).
sp    = softplus(a[b, hv] + dt_bias[hv], threshold=20)
g     = -exp(A_log[hv]) * sp
beta  = sigmoid(b[b, hv])
exp_g = exp(g)

# Recurrence: state is (V, K) per (slot, hv).
state = initial_state[slot, hv]                          # (V, K)
state = state * exp_g
v_corr = (v - state @ k) * beta                          # (V,)
state = state + outer(v_corr, k)                         # (V, K)
out[b, 0, hv, :] = state @ q
initial_state[slot, hv] = state
```

## Parallelization

- **Outer parallel axis:** `(b, hv)` pairs flattened to a single
  iteration space of length `B * HV`. `at::parallel_for(grain=1)` lets
  ATen pick the partition; for typical decode shapes (`B * HV` ~ 16–
  256), this schedules ~one pair per thread.
- **Inside a pair:** the V*K state passes are AVX-512-vectorised via
  `load_scale_dot` and `update_state_and_compute_out`. Per-thread
  workspace (`state_ws`, `q_ws`, `k_ws`, `v_ws`, `v_corr_ws`) is
  allocated once per partition and reused across pairs.

## Production-narrowing rationale

| Variant | Production caller? | Cpp op support |
|---|---|---|
| `seqlen > 1` (spec-decode) | covered by `fused_sigmoid_gating_delta_rule_update` | not supported (op assumes `B = num_decode_seqs`, `seqlen=1` packed in `(B, 1, HV, V)`) |
| fp16 anything | none | rejected at runtime |
| Higher-rank `out` | none | rejected |

## Tolerances

One-step recurrence; `max(out_tol, state_tol)` slack covers the
hybrid-precision compute. Project defaults apply at production shapes.

## Meta function

Registered in `_meta_registrations.py` as
`meta_gdn_fused_recurrent_gated_delta_rule_packed_decode`. Returns
`None` — this is a pure side-effect op that mutates `initial_state`
and `out` in place. The `(a!)` / `(b!)` schema annotations carry the
mutation information through the dispatcher; `make_fallback` honours
them.
