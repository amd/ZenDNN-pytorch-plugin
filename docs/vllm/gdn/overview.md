# Overview: how Qwen3.5 / Qwen3-Next reaches the GDN cpp ops

This document is the call-graph reference for the project. It traces, top-down,
what runs on every forward pass of `Qwen3_5ForConditionalGeneration` on CPU,
showing exactly where `torch.ops.zentorch.gdn_*` is invoked.

> For the per-kernel math/contract, see `kernels/<name>.md`.

## A. Engine-level (once per benchmark)

```
vllm bench throughput
  → run_vllm → LLM(...)
  → LLMEngine.from_engine_args
  → EngineCoreClient.make_client → SyncMPClient
  → EngineCoreProc.__init__
  → MultiprocExecutor → WorkerProc
  → CPUWorker.load_model
  → CPUModelRunner.load_model
  → vllm.model_executor.model_loader.get_model(...)
  → CPUModelRunner.warming_up_model
    → CPUWorker.determine_available_memory
    → gpu_model_runner.profile_run → _dummy_run
       └─ Qwen3_5ForConditionalGeneration.forward       ← goes into Section B
```

## B. Per-token-batch model forward

```
Qwen3_5ForConditionalGeneration.forward
└─ Qwen3_5ForCausalLM.forward → self.model(...)
   └─ Qwen3_5Model.forward (subclass of Qwen3NextModel.forward)
      ├─ embed_input_ids → VocabParallelEmbedding
      ├─ for layer in self.layers:                       ← Section C
      │     layer(positions, hidden_states, residual)
      └─ self.norm(...) → Qwen3NextRMSNorm
   → compute_logits → LogitsProcessor → ParallelLMHead
```

## C. Per `Qwen3_5DecoderLayer.forward` (= `Qwen3NextDecoderLayer.forward`)

```
input_layernorm                       → Qwen3NextRMSNorm (fused add+RMS)
↓
either                                ── linear_attention layer ──
  linear_attn (GatedDeltaNetAttention) → see Section D
or                                    ── full_attention layer ──
  self_attn (Qwen3NextAttention)
    ├─ qkv_proj (QKVParallelLinear)
    ├─ optional attn_output_gate split
    ├─ q_norm / k_norm (Qwen3NextRMSNorm)
    ├─ rotary_emb
    ├─ attn (TORCH_SDPA on CPU)
    └─ o_proj (RowParallelLinear)
↓
post_attention_layernorm              → Qwen3NextRMSNorm
↓
mlp                                   → Qwen3NextMLP (dense)  OR  Qwen3NextSparseMoeBlock (MoE)
```

Only the `linear_attention` branch reaches our cpp ops.

## D. `GatedDeltaNetAttention.forward` on CPU

On CPU, `forward(...)` dispatches to `forward_cpu(...)`, which
zentorch overrides with `forward_cpu_zen` at vLLM startup (see the
invocation flow in `README.md`). `forward_cpu_zen` runs the entire
GDN block by orchestrating `torch.ops.zentorch.gdn_*` ops:

```
Part 1 — Input projection
├─ in_proj_qkvz   (the LoRA variant `in_proj_qkv + in_proj_z` is
│                  explicitly rejected by `forward_cpu_zen` with
│                  `NotImplementedError` — LoRA is unsupported on CPU)
├─ in_proj_ba
└─ fix_query_key_value_ordering / split + reshape

Part 2 — Core attention (the cpp-op-heavy zone)
├─ Decode-only fast path
│   ├─ torch.ops.zentorch.gdn_causal_conv1d_update
│   └─ torch.ops.zentorch.gdn_fused_recurrent_gated_delta_rule_packed_decode
│      (decode hot kernel; AVX-512 vectorised)
└─ General mixed (prefill + decode in same batch) path
    ├─ torch.ops.zentorch.gdn_causal_conv1d_update              (decode rows)
    ├─ torch.ops.zentorch.gdn_causal_conv1d_fn                  (prefill rows)
    ├─ torch.ops.zentorch.gdn_fused_post_conv_prep              (prefill rows)
    ├─ torch.ops.zentorch.gdn_fused_sigmoid_gating_delta_rule_update  (decode rows)
    └─ torch.ops.zentorch.gdn_chunk_gated_delta_rule_fwd        (prefill rows)
        └─ fused: composes chunk_local_cumsum + l2norm_fwd
                + chunk_scaled_dot_kkt_fwd + solve_tril
                + recompute_w_u_fwd + chunk_gated_delta_rule_fwd_h
                + chunk_fwd_o
                (these ops are also exposed individually for testing)

Part 3 — Output projection
├─ torch.ops.zentorch.gdn_rms_norm_gated
└─ out_proj (RowParallelLinear)
```

## E. GDN cpp op inventory

Each row corresponds to one `torch.ops.zentorch.gdn_*` op. All 14
schemas + CPU impls are registered from the single consolidated
bindings file `src/cpu/cpp/GDN_ops.cpp`; the actual kernel math lives
in `src/cpu/cpp/kernels/layers/gdn/<KernelImpl>.cpp`. Each op also has
a meta registration in `_meta_registrations.py` and an inlined oracle
+ unittest under `test/unittests/op_tests/layers/gdn/`.

| Op name | Kernel impl | Per-kernel doc |
| --- | --- | --- |
| `gdn_chunk_local_cumsum` | `kernels/layers/gdn/ChunkLocalCumsum.cpp` | [`kernels/chunk_local_cumsum.md`](kernels/chunk_local_cumsum.md) |
| `gdn_l2norm_fwd` | `kernels/layers/gdn/L2NormFwd.cpp` | `kernels/l2norm_fwd.md` |
| `gdn_chunk_scaled_dot_kkt_fwd` | `kernels/layers/gdn/ChunkScaledDotKktFwd.cpp` | `kernels/chunk_scaled_dot_kkt_fwd.md` |
| `gdn_solve_tril` | `kernels/layers/gdn/SolveTril.cpp` | `kernels/solve_tril.md` |
| `gdn_recompute_w_u_fwd` | `kernels/layers/gdn/RecomputeWUFwd.cpp` | `kernels/recompute_w_u_fwd.md` |
| `gdn_chunk_gated_delta_rule_fwd_h` | `kernels/layers/gdn/ChunkGatedDeltaRuleFwdH.cpp` | `kernels/chunk_gated_delta_rule_fwd_h.md` |
| `gdn_chunk_fwd_o` | `kernels/layers/gdn/ChunkFwdO.cpp` | `kernels/chunk_fwd_o.md` |
| `gdn_chunk_gated_delta_rule_fwd` | `kernels/layers/gdn/ChunkGatedDeltaRuleFwd.cpp` | `kernels/chunk_gated_delta_rule_fwd.md` |
| `gdn_rms_norm_gated` | `kernels/layers/gdn/RmsNormGated.cpp` | `kernels/rms_norm_gated.md` |
| `gdn_causal_conv1d_fn` | `kernels/layers/gdn/CausalConv1dFn.cpp` | `kernels/causal_conv1d_fn.md` |
| `gdn_causal_conv1d_update` | `kernels/layers/gdn/CausalConv1dUpdate.cpp` | `kernels/causal_conv1d_update.md` |
| `gdn_fused_post_conv_prep` | `kernels/layers/gdn/FusedPostConvPrep.cpp` | `kernels/fused_post_conv_prep.md` |
| `gdn_fused_sigmoid_gating_delta_rule_update` | `kernels/layers/gdn/FusedSigmoidGatingDeltaRuleUpdate.cpp` | `kernels/fused_sigmoid_gating_delta_rule_update.md` |
| `gdn_fused_recurrent_gated_delta_rule_packed_decode` | `kernels/layers/gdn/FusedRecurrentDecode.cpp` | `kernels/fused_recurrent_gated_delta_rule_packed_decode.md` |

`gdn_chunk_gated_delta_rule_fwd` is a single fused cpp op that internally
performs the work of `chunk_local_cumsum`, `l2norm_fwd`,
`chunk_scaled_dot_kkt_fwd`, `solve_tril`, `recompute_w_u_fwd`,
`chunk_gated_delta_rule_fwd_h`, and `chunk_fwd_o`. The component ops
exist as standalone bindings so each can be unit-tested in isolation;
only the fused op is invoked from `forward_cpu_zen`.
