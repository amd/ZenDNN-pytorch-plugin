# `fused_sigmoid_gating_delta_rule_update`

| Field | Value |
|---|---|
| Python op | `torch.ops.zentorch.gdn_fused_sigmoid_gating_delta_rule_update` |
| C++ symbol | `zentorch::zentorch_gdn_fused_sigmoid_gating_delta_rule_update` |
| Kernel impl | `src/cpu/cpp/kernels/layers/gdn/FusedSigmoidGatingDeltaRuleUpdate.cpp` |
| Schema + binding | `src/cpu/cpp/GDN_ops.cpp` |
| Triton source | `vllm/model_executor/layers/fla/ops/fused_sigmoid_gating.py` |
| Tests + oracle | `test/unittests/op_tests/layers/gdn/test_fused_sigmoid_gating_delta_rule_update.py` |
| Profiler span | `zentorch::gdn::fused_sigmoid_gating_delta_rule_update` |
| Backends | `CPU`, `Meta` |

## What this op does

The **decode and spec-decode** fused op. For each varlen sequence in
the batch (typically `seqlen=1` for plain decode, `seqlen=draft_len`
for spec decode), it computes the gating tensors `g` and `beta` from
`(a, b, A_log, dt_bias)` on the fly and walks the per-token recurrent
delta-rule update against the cache state. Per-token state updates are
written back to slot `ssm_state_indices[n, t]` (or `[n]` in 1-D mode).

The op covers two related code paths:
- **Plain decode**: `ssm_state_indices` is 1-D `(N,)`; one slot per
  sequence.
- **Spec decode**: `ssm_state_indices` is 2-D `(N, max_query_len)`;
  one slot per `(sequence, draft-token-position)`. The op uses
  `num_accepted_tokens[n] - 1` to find the initial state for sequence
  `n` (the state after the last accepted token from the previous step).

## How it's wired into Qwen3.5 / Qwen3-Next GDN

In `gdn_linear_attn.py` (decode and spec paths, after the conv +
`fused_post_conv_prep` chain):

```python
o_decode = fused_sigmoid_gating_delta_rule_update(
    A_log=self.A_log, a=a_decode, b=b_decode, dt_bias=self.dt_bias,
    q=q_decode, k=k_decode, v=v_decode,
    beta_temp=1.0, threshold=20.0, scale=self.scale,
    initial_state=ssm_state,
    cu_seqlens=cu_seqlens_decode,
    ssm_state_indices=ssm_state_indices,
    num_accepted_tokens=num_accepted_tokens,
    use_qk_l2norm_in_kernel=True,
)
```

`forward_cpu_zen` substitutes the cpp op directly.

## Schema

```
zentorch::gdn_fused_sigmoid_gating_delta_rule_update(
    Tensor A_log, Tensor a, Tensor b, Tensor dt_bias,
    Tensor q, Tensor k, Tensor v,
    float beta_temp, float threshold, float scale,
    Tensor(a!) initial_state,
    Tensor cu_seqlens, Tensor ssm_state_indices,
    Tensor? num_accepted_tokens,
    bool use_qk_l2norm_in_kernel,
    *, str zentorch_op_name='zentorch::gdn_fused_sigmoid_gating_delta_rule_update'
) -> Tensor
```

`Tensor(a!)` on `initial_state` declares it as mutated in-place.
Returns the attention output `o` of shape `(B, T, HV, V)` in `q.dtype`.

## Per-tensor layout contract

| Tensor | Shape | Dtype | Mutation |
|---|---|---|---|
| `A_log`, `dt_bias` | `(HV,)` | floating | none |
| `a`, `b` | `(B=1, T, HV)` | same as `q` | none |
| `q`, `k` | `(B=1, T, H, K)` | fp32 or bf16 | none |
| `v` | `(B=1, T, HV, V)` | same as `q` | none |
| `initial_state` | `(num_cache_lines, HV, V, K)` | fp32 (typically) | **in-place writes to per-token slots** |
| `cu_seqlens` | `(N+1,)` | int32 | none |
| `ssm_state_indices` | `(N,)` or `(N, max_query_len)` | int32 | none |
| `num_accepted_tokens` | `(N,)` or `None` | int32 | none |
| return `o` | `(B, T, HV, V)` | `q.dtype` | fresh contiguous |

`HV = v.size(2)` is the V-head count; `H = q.size(2)` is the K-head
count; `r = HV / H` is the GQA ratio. `q`/`k` are GQA-expanded
internally via `repeat_interleave`.

## Math (per token)

```python
# Null-slot contract: a sequence whose initial-state index is
# <= null_block_id (0) is treated as "no state to load" -- the op
# zero-fills o[0, bos:eos] for that sequence and skips the recurrence
# (mirrors gdn_fused_recurrent_gated_delta_rule_packed_decode). The
# null-slot path is dormant in production today; all schedules pass
# allocated slots >= 1.

# Gating.
sp        = softplus(beta_temp * (a[t] + dt_bias), threshold=threshold) / beta_temp
g[t]      = -exp(A_log) * sp                                          # (HV,)
beta_t    = sigmoid(b[t])                                              # (HV,)

# Optional L2-norm + scale.
if use_qk_l2norm_in_kernel:
    q[t] = q[t] * rsqrt(sum(q[t]^2, dim=-1, keepdim=True) + 1e-6)
    k[t] = k[t] * rsqrt(sum(k[t]^2, dim=-1, keepdim=True) + 1e-6)
q[t] *= scale

# GQA expansion: (H, K) -> (HV, K) via repeat_interleave.
q_e = q[t].repeat_interleave(r, dim=0)
k_e = k[t].repeat_interleave(r, dim=0)

# Recurrence (state shape (HV, V, K)).
h = h * exp(g[t]).reshape(HV, 1, 1)
kv_mem = (h * k_e.unsqueeze(-2)).sum(-1)                # (HV, V)
delta  = (v[t] - kv_mem) * beta_t.unsqueeze(-1)         # (HV, V)
h      = h + delta.unsqueeze(-1) * k_e.unsqueeze(-2)    # (HV, V, K)
o[t]   = (h * q_e.unsqueeze(-2)).sum(-1)                # (HV, V)

# Per-token state writeback.
final_idx = ssm_state_indices[n, t]   # 2-D mode (or ssm_state_indices[n] in 1-D)
if final_idx > 0:                     # null_block_id = 0
    initial_state[final_idx] = h
```

All recurrent math runs in fp32; the output is cast to `q.dtype` and
the state writeback is cast to `initial_state.dtype` (typically fp32).

## Production-narrowing rationale

| Variant | Production caller? | Cpp op support |
|---|---|---|
| KDA-style (`is_kda=True`) per-K decay | none | not supported |
| `inplace_final_state=False` | none | not supported (always in-place) |
| Higher rank `q`/`k`/`v` | none | not supported |

## Tolerances

The per-token recurrence is a single-step update (no compounding
across tokens beyond the loop length, which is `T_n <= max_query_len`,
typically 1–4 in decode). Project default tolerances with
`max(out_tol, state_tol)` slack apply.

## Meta function

Registered in `_meta_registrations.py` as
`meta_gdn_fused_sigmoid_gating_delta_rule_update`. Returns
`q.new_empty(v.size())` — shape `(B, T, HV, V)` taken from `v.size()`,
options (device/dtype/layout) inherited from `q` to mirror the cpp's
`at::empty({B, T, HV, V_dim}, q.options())` precisely. The
ZENTORCH_CHECK enforces `q.dtype == k.dtype == v.dtype` so the result
agrees with the older `v.new_empty(...)` form for valid inputs, but
deriving from `q` keeps FakeTensorMode in sync with the cpp's source-
of-truth under torch.compile. `initial_state` mutation is declared via
`Tensor(a!)`.
