/******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#include "EnvReader.hpp"
#include "GroupMatmul.hpp"
#include "Utils.hpp"
#include <ATen/Parallel.h>
#include <ATen/record_function.h>
#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <torch/library.h>
#include <unordered_map>
#include <utility>
#include <vector>

namespace zentorch {

namespace {

// ---------------------------------------------------------------------------
// FusedMoE scratchpad — opt-in growing allocator
//
// Singleton holding a single 64-byte-aligned buffer that grows monotonically
// (never shrinks) as larger per-call working sets are observed. Each
// `zentorch_fused_moe` call's per-expert [M_e, H] grouped-input buffers are
// placed contiguously inside this block and exposed as zero-copy
// `at::from_blob` tensors with a no-op deleter. After the call returns the
// at::from_blob handles are destroyed but the underlying memory persists for
// the next call to reuse — avoiding the per-tensor allocator round-trip the
// default `at::empty` path goes through.
//
// Modeled on vLLM's `cpu_utils::ScratchPadManager` (csrc/cpu/utils.cpp).
//
// Enabled by default. Set `ZENTORCH_USE_SCRATCHPAD=0` to disable and fall
// back to the `at::empty`-per-expert allocation path.
//
// Thread-safety: not safe for concurrent `zentorch_fused_moe` calls from
// different threads. Matches the assumption of vLLM's CPU MoE path
// (single inference stream per process); intra-call work parallelizes via
// `at::parallel_for` / OMP, not via overlapping op invocations.
//
// Lifetime safety: the at::from_blob tensors are local to a single
// `zentorch_fused_moe` call and destroyed before the next call begins, so
// a `reserve()` that grows (and frees) the underlying buffer can never
// dangle a still-live tensor. Tensors must NOT escape the op into Python /
// graph storage.
// ---------------------------------------------------------------------------

class FusedMoEScratchpad {
public:
  static constexpr size_t kAlignment = 64;
  static constexpr size_t kAllocUnit = 4 * 1024;            // 4 KB grow unit
  static constexpr size_t kInitialBytes = kAllocUnit * 128; // 512 KB seed

  static FusedMoEScratchpad &get() {
    static FusedMoEScratchpad sp;
    return sp;
  }

  std::byte *data() noexcept { return static_cast<std::byte *>(ptr_); }

  void reserve(size_t bytes) {
    bytes = round_up(bytes);
    if (bytes <= size_) {
      return;
    }
    void *new_ptr = std::aligned_alloc(kAlignment, bytes);
    TORCH_CHECK(new_ptr != nullptr, "FusedMoEScratchpad: aligned_alloc(", bytes,
                ") failed");
    if (ptr_ != nullptr) {
      std::free(ptr_);
    }
    ptr_ = new_ptr;
    size_ = bytes;
  }

private:
  FusedMoEScratchpad() : size_(0), ptr_(nullptr) { reserve(kInitialBytes); }
  ~FusedMoEScratchpad() {
    if (ptr_ != nullptr) {
      std::free(ptr_);
    }
  }
  FusedMoEScratchpad(const FusedMoEScratchpad &) = delete;
  FusedMoEScratchpad &operator=(const FusedMoEScratchpad &) = delete;

  static size_t round_up(size_t s) {
    return ((s + kAllocUnit - 1) / kAllocUnit) * kAllocUnit;
  }

  size_t size_;
  void *ptr_;
};

inline size_t round_up_to(size_t bytes, size_t alignment) {
  return ((bytes + alignment - 1) / alignment) * alignment;
}

// ---------------------------------------------------------------------------
// ExpertSliceCache — per-tensor cache of dim-0 expert views
//
// Caches the [N, K] / [K] view tensors produced by `select(0, e)` so that we
// only pay the ATen dispatcher cost once per (tensor, expert) pair across the
// life of the process — not once per `zentorch_fused_moe` invocation.
//
// Why we need a multi-entry cache:
//   MoE models stack many MoE layers (e.g. Qwen3-30B-A3B has 48), and every
//   decode step calls `zentorch_fused_moe` once per layer. A single-slot
//   cache (last tensor seen) thrashes between layers and rebuilds every call.
//   Keying by `TensorImpl*` lets each layer's weight/bias/scale tensors hold
//   their own slice list, so the hit rate after step 1 is effectively 100%.
//
// Why we avoid `unbind(0)`:
//   `unbind(0)` is implemented as a loop of `select(0, i)` internally, so it
//   pays the same per-expert dispatcher cost AND adds an outer `aten::unbind`
//   wrapper on top. Measured 8.7s of `select` becoming 5.9s of `select` +
//   5.7s of `unbind` — a net regression. Calling `select` directly inside
//   the cache build keeps the one-time fill cheaper.
//
// Lifetime / safety:
//   - The cache stores `at::Tensor` (strong) handles, so the underlying
//     `TensorImpl` is kept alive for the process. For inference workloads
//     where model weights live until shutdown this is benign (~1 MB total
//     for 48 layers × 128 experts × {w13, w2, biases, scales}).
//   - `TensorImpl*` can in principle be reused if a tensor is freed and a
//     new one allocated at the same address. Holding a strong ref prevents
//     that for the cached tensors themselves, eliminating the aliasing risk.
//   - Not thread-safe across concurrent `zentorch_fused_moe` calls. Matches
//     existing assumptions of this op (single inference stream per process).
// ---------------------------------------------------------------------------
class ExpertSliceCache {
public:
  // Returns the per-expert views for `tensor`. On first encounter, builds
  // them with one `select(0, e)` per expert; subsequent calls are an
  // unordered_map lookup.
  const std::vector<at::Tensor> &get(const at::Tensor &tensor) {
    const auto *impl = tensor.unsafeGetTensorImpl();
    auto it = entries_.find(impl);
    if (it != entries_.end()) {
      return it->second;
    }
    const int64_t E = tensor.size(0);
    std::vector<at::Tensor> slices;
    slices.reserve(E);
    for (int64_t e = 0; e < E; ++e) {
      slices.emplace_back(tensor.select(0, e));
    }
    return entries_.emplace(impl, std::move(slices)).first->second;
  }

private:
  std::unordered_map<const c10::TensorImpl *, std::vector<at::Tensor>> entries_;
};

} // namespace

// ---------------------------------------------------------------------------
// Phase 1 — Token-Expert Grouping
//
// For each routed (token t, slot k) we need expert e = topk_id[t][k] to
// receive a copy of input[t] in its per-expert input buffer. We do this in
// two passes:
//
//   Pass 1 (single-threaded; trivially cheap O(T*K) bookkeeping):
//     - Walk the T*K (t, k) pairs in flat order.
//     - On the first encounter of expert e, append e to `active_expert_ids`.
//       To resolve an already-seen expert back to its active slot a we
//       linear-scan `active_expert_ids`; E_a is small in practice (tens for
//       typical MoE configs) and the scan stays in L1.
//     - For each (t, k), record `topk_to_expert_row[i] = (active_idx, pos)`
//       where `pos = tokens_per_active[a]++` is the deterministic write
//       position in active slot a's eventual input tensor (no atomics needed).
//     - Also append the source token id `t` to
//       `source_tokens_per_active[a]` so Pass-2 can iterate per-expert
//       without re-scanning `topk_to_expert_row`.
//
//   Pass 2 (per-expert parallel; the actual data movement):
//     - Allocate one [M_e, H] tensor per active expert, indexed by
//       active_idx (so `grouped_inputs.size() == E_a`, ready to hand to
//       `group_matmul` directly).
//     - `at::parallel_for` over the E_a active experts (`grain_size=1`):
//       each worker owns one destination buffer and walks its own
//       `source_tokens_per_active[a]` list to memcpy the source rows in.
//       This exposes ~E_a tasks (instead of `total_pairs / grain_size`)
//       and confines each thread's writes to a single destination buffer,
//       avoiding the scattered-write and false-sharing pattern of a
//       per-pair scheme.
//
// All Pass-1 auxiliary state is sized to E_a (not E) — `active_expert_ids`
// is the single source of truth for the active set, no reverse map needed.
//
// ----- Worked example: T=3, H=4, E=5 experts, K=2 top-k routing -----
//
// Inputs:
//   input  [T, H] = [[ t0_h0, t0_h1, t0_h2, t0_h3 ],   # token 0
//                    [ t1_h0, t1_h1, t1_h2, t1_h3 ],   # token 1
//                    [ t2_h0, t2_h1, t2_h2, t2_h3 ]]   # token 2
//
//   topk_id [T, K] = [[ 3, 0 ],     # token 0 -> experts {3, 0}
//                     [ 3, 1 ],     # token 1 -> experts {3, 1}
//                     [ 0, 1 ]]     # token 2 -> experts {0, 1}
//
// After Pass 1 (single sweep over T*K=6 pairs, first-encounter ordering):
//   active_expert_ids        = [ 3, 0, 1 ]              # E_a = 3
//   source_tokens_per_active = [ [0, 1], [0, 2], [1, 2] ]
//                                # a=0 (expert 3): tokens 0, 1
//                                # a=1 (expert 0): tokens 0, 2
//                                # a=2 (expert 1): tokens 1, 2
//
//   topk_to_expert_row [T*K = 6 entries] = (active_idx, row_in_expert):
//     i=0 (t=0,k=0) -> (a=0, row=0)   # expert 3
//     i=1 (t=0,k=1) -> (a=1, row=0)   # expert 0
//     i=2 (t=1,k=0) -> (a=0, row=1)   # expert 3
//     i=3 (t=1,k=1) -> (a=2, row=0)   # expert 1
//     i=4 (t=2,k=0) -> (a=1, row=1)   # expert 0
//     i=5 (t=2,k=1) -> (a=2, row=1)   # expert 1
//
// After Pass 2 (per-expert parallel memcpy):
//   grouped_inputs[0]  (active_idx 0 = expert 3, M=2) = [ input[0], input[1] ]
//   grouped_inputs[1]  (active_idx 1 = expert 0, M=2) = [ input[0], input[2] ]
//   grouped_inputs[2]  (active_idx 2 = expert 1, M=2) = [ input[1], input[2] ]
// ---------------------------------------------------------------------------

struct TokenExpertMapping {
  // Size E_a. grouped_inputs[a] is the [M_e, H] input tensor for the
  // a-th active expert (active expert id = active_expert_ids[a]).
  std::vector<at::Tensor> grouped_inputs;
  // Size E_a. active_expert_ids[a] = the original expert id (in [0, E))
  // for the a-th active expert. Order is first-encounter in `topk_id`.
  std::vector<int32_t> active_expert_ids;
  // T*K entries; index (t*K + k) holds (active_idx, row_in_expert).
  // Kept token-major because Phase 5's `row_ptrs` setup walks pairs in this
  // order to build the MoE weighted-reduce postop input.
  std::vector<std::pair<int32_t, int32_t>> topk_to_expert_row;
  // Size E_a. source_tokens_per_active[a] lists the source `input` row ids
  // (t values) that fill grouped_inputs[a], in the deterministic in-expert
  // order assigned by Pass-1. Pass-2 consumes this directly so each worker
  // thread walks one contiguous list and writes one destination buffer.
  std::vector<std::vector<int32_t>> source_tokens_per_active;
};

static TokenExpertMapping
build_token_expert_mapping(const at::Tensor &input, const at::Tensor &topk_id) {

  const int64_t T = input.size(0);
  const int64_t H = input.size(1);
  const int64_t K = topk_id.size(1);
  const int64_t total_pairs = T * K;
  const int64_t row_bytes = H * input.element_size();

  // topk_id is contiguous int32 [T, K]; flat indexing is i = t*K + k.
  const int32_t *topk_id_serialized = topk_id.const_data_ptr<int32_t>();

  // ----- Pass 1: register active experts + assign per-active positions ----
  // `active_expert_ids` is the only structure that tracks the active set;
  // resolving an already-seen expert back to its active slot is a small
  // linear scan (E_a is in the tens for typical MoE configs, stays in L1).
  // Per-active source-token lists are built in lockstep so Pass-2 can walk
  // one contiguous list per expert without a second sweep.
  TokenExpertMapping mapping;
  mapping.topk_to_expert_row.resize(total_pairs);
  std::vector<int32_t> tokens_per_active; // grows alongside active_expert_ids
  {
    RECORD_FUNCTION("zentorch::fused_moe::pass1_active_set_build",
                    c10::ArrayRef<c10::IValue>({}));
    for (int64_t i = 0; i < total_pairs; ++i) {
      const int32_t e = topk_id_serialized[i];
      const int32_t t = static_cast<int32_t>(i / K);
      auto it = std::find(mapping.active_expert_ids.begin(),
                          mapping.active_expert_ids.end(), e);
      int32_t a;
      if (it == mapping.active_expert_ids.end()) {
        a = static_cast<int32_t>(mapping.active_expert_ids.size());
        mapping.active_expert_ids.emplace_back(e);
        tokens_per_active.emplace_back(0);
        mapping.source_tokens_per_active.emplace_back();
      } else {
        a = static_cast<int32_t>(it - mapping.active_expert_ids.begin());
      }
      const int32_t pos = tokens_per_active[a]++;
      mapping.topk_to_expert_row[i] = {a, pos};
      mapping.source_tokens_per_active[a].emplace_back(t);
    }
  } // RECORD_FUNCTION pass1_active_set_build
  const int64_t E_a = static_cast<int64_t>(mapping.active_expert_ids.size());

  // ----- Allocate per-active-expert [M_e, H] tensors ----------------------
  // Default path (`ZENTORCH_USE_SCRATCHPAD=1`, the default): pack every
  // per-expert [M_e, H] region contiguously into a single 64-byte-aligned
  // block reused (and grown when needed) across calls, then expose each
  // region as a zero-copy `at::from_blob` tensor. Each region's byte length
  // is rounded up to a 64-byte multiple so the next region's base also
  // starts on a 64-byte boundary; the within-region row stride stays at the
  // natural H * elem_size — identical to what `at::empty` would give us, so
  // ZenDNN and the Pass-2 memcpy both see the same layout under either path.
  //
  // Fallback path (`ZENTORCH_USE_SCRATCHPAD=0`): one `at::empty` per active
  // expert. PyTorch's caching allocator amortizes well for stable per-call
  // working-set sizes but still pays a per-tensor metadata round-trip on
  // every call.
  const int int_env_value =
      EnvReader::getEnvVariableAsInt("ZENTORCH_USE_SCRATCHPAD");
  const bool use_scratchpad = static_cast<bool>(int_env_value);

  mapping.grouped_inputs.resize(E_a);
  {
    RECORD_FUNCTION("zentorch::fused_moe::scratchpad_allocation",
                    c10::ArrayRef<c10::IValue>({}));
    if (use_scratchpad) {
      const size_t row_bytes_sz = static_cast<size_t>(row_bytes);
      std::vector<size_t> region_offsets(E_a);
      size_t total_bytes = 0;
      for (int64_t a = 0; a < E_a; ++a) {
        region_offsets[a] = total_bytes;
        total_bytes += round_up_to(static_cast<size_t>(tokens_per_active[a]) *
                                       row_bytes_sz,
                                   FusedMoEScratchpad::kAlignment);
      }
      auto &sp = FusedMoEScratchpad::get();
      sp.reserve(total_bytes);
      std::byte *base = sp.data();
      for (int64_t a = 0; a < E_a; ++a) {
        mapping.grouped_inputs[a] = at::from_blob(
            base + region_offsets[a], {tokens_per_active[a], H},
            /*deleter=*/[](void *) {}, input.options());
      }
    } else {
      for (int64_t a = 0; a < E_a; ++a) {
        mapping.grouped_inputs[a] = at::detail::empty_strided_cpu(
            {tokens_per_active[a], H}, {H, 1}, input.options());
      }
    }
  } // RECORD_FUNCTION scratchpad_allocation

  // ----- Pass 2: per-expert parallel memcpy --------------------------------
  // Each worker thread owns exactly one active expert and streams its rows
  // sequentially into that expert's [M_e, H] destination buffer. Compared
  // with parallelizing over the T*K (t, k) pairs in token-major order, this
  // gives us:
  //   - One destination buffer per thread (no cache-line / store-buffer
  //     contention between threads writing into different experts).
  //   - Sequential writes within each destination (auto-vectorizable, ideal
  //     for streaming stores).
  //   - More tasks than the per-pair scheme could expose: per-pair was
  //     capped at `total_pairs / grain_size` (= 4 for T*K=256, grain_size=64),
  //     here we expose `E_a` tasks (~32-64 for typical MoE configs) — enough
  //     to actually use the available cores.
  //
  // grain_size=1 lets the runtime hand one expert per task. Same idiom used
  // by `kernels/zen_Sdpa.cpp` and the per-row loop in `QuantEmbedBag.cpp`.
  //
  // `std::byte` (not `char`) makes the byte-pointer arithmetic intent
  // explicit and avoids the implementation-defined signedness of plain
  // `char`. Like `char`/`unsigned char`, it is exempt from strict aliasing
  // so it can legally point into any tensor's storage.
  const auto *src_base =
      reinterpret_cast<const std::byte *>(input.const_data_ptr());

  std::vector<std::byte *> dst_base(E_a);
  {
    RECORD_FUNCTION("zentorch::fused_moe::pass2_dst_base_setup",
                    c10::ArrayRef<c10::IValue>({}));
    for (int64_t a = 0; a < E_a; ++a) {
      dst_base[a] =
          reinterpret_cast<std::byte *>(mapping.grouped_inputs[a].data_ptr());
    }
  } // RECORD_FUNCTION pass2_dst_base_setup

  {
    RECORD_FUNCTION("zentorch::fused_moe::pass2_parallel_memcpy",
                    c10::ArrayRef<c10::IValue>({}));
    at::parallel_for(
        0, E_a, /*grain_size=*/1, [&](int64_t a_begin, int64_t a_end) {
          for (int64_t a = a_begin; a < a_end; ++a) {
            std::byte *dst = dst_base[a];
            const auto &src_tokens = mapping.source_tokens_per_active[a];
            const int64_t M_e = static_cast<int64_t>(src_tokens.size());
            for (int64_t p = 0; p < M_e; ++p) {
              std::memcpy(dst + p * row_bytes,
                          src_base +
                              static_cast<int64_t>(src_tokens[p]) * row_bytes,
                          row_bytes);
            }
          }
        });
  } // RECORD_FUNCTION pass2_parallel_memcpy

  return mapping;
}

// ---------------------------------------------------------------------------
// zentorch_fused_moe — Fused Mixture-of-Experts operator (out variant)
//
// Schema mirrors vLLM's `cpu_fused_moe` signature so that vLLM's CPUFusedMOE
// dispatch can swap the op name without rewriting call sites:
//
//   torch.ops.zentorch.zentorch_fused_moe(
//       output, input, w13, w2, w13_bias, w2_bias,
//       topk_weights, topk_id, skip_weighted, act,
//       w13_scales=None, w2_scales=None,
//       *, zentorch_op_name="zentorch::zentorch_fused_moe")
//
// `output` is mutated in place (Tensor(a!)). Returns ().
//
// ----------------------------- Input contract ------------------------------
// All shape/dtype/bias validation is performed once, at patch-install time
// in `src/cpu/python/zentorch/vllm/__init__.py` (the only producer of calls
// into this op). The C++ op trusts its inputs and assumes:
//
//   input          : 2D [T, H], f32 or bf16, contiguous
//   output         : 2D [T, H], same dtype as input, ZERO-INITIALIZED
//                    (Phase 5 accumulates into it)
//   w13            : 3D [E, 2*I, H], same dtype as input
//   w2             : 3D [E, H, I],   same dtype as input
//   w13_bias       : None or [E, 2*I] (same dtype as input)
//   w2_bias        : None or [E, H]   (same dtype as input)
//   topk_weights   : 2D [T, K], f32, contiguous
//   topk_id        : 2D [T, K], int32, contiguous, values in [0, E)
//   skip_weighted  : bool; if true, requires K == 1
//   act            : one of {"silu", "gelu", "gelu_tanh", "swigluoai"}
// ---------------------------------------------------------------------------

void zentorch_fused_moe(
    at::Tensor &output, const at::Tensor &input, const at::Tensor &w13,
    const at::Tensor &w2, const c10::optional<at::Tensor> &w13_bias,
    const c10::optional<at::Tensor> &w2_bias, const at::Tensor &topk_weights,
    const at::Tensor &topk_id, bool skip_weighted, std::string_view act,
    const c10::optional<at::Tensor> &w13_scales,
    const c10::optional<at::Tensor> &w2_scales, std::string zentorch_op_name) {

  const int64_t T = input.size(0);
  const int64_t K = topk_id.size(1);
  const int64_t total_pairs = T * K;
  const int64_t row_bytes = input.size(1) * input.element_size();

  // ---------------------- Phase 1: token-expert grouping ---------------------
  TokenExpertMapping mapping;
  {
    RECORD_FUNCTION("zentorch::fused_moe::token_expert_grouping",
                    c10::ArrayRef<c10::IValue>({}));
    mapping = build_token_expert_mapping(input, topk_id);
  }
  const int64_t E_a = static_cast<int64_t>(mapping.active_expert_ids.size());

  // ---------------------- Temporary Guard ------------------------------------
  // zentorch_group_matmul_out_impl requires E_a > 1. E_a == 1 is structurally
  // guaranteed when K == 1 and T == 1 (apply_router_weight_on_input=True
  // path). Rather than letting the call reach group_matmul_direct and crash
  // on its inputs.size() > 1 assertion, fail here with a clear, actionable
  // message.
  TORCH_CHECK(
      E_a > 1, "zentorch_fused_moe: only ", E_a,
      " expert(s) received tokens. "
      "zentorch_group_matmul_out_impl requires at least 2 active experts. "
      "This typically occurs with K=1 routing "
      "(apply_router_weight_on_input=True) "
      "at single-token decode (T=1). Use the standard vLLM cpu_fused_moe "
      "path "
      "for this configuration, or unset ZENTORCH_FUSED_MOE.");

  // ---------------------- Phase 2: build weight & bias slices ---------------
  // Weight contract with zentorch_group_matmul_out_impl: w13 / w2 lists are
  // sized E (all experts), with the E_a active experts placed FIRST in
  // active_idx order so they line up with `mapping.grouped_inputs[a]`, then the
  // inactive experts appended in their original [0, E) order. Example:
  // active_expert_ids = [3, 0, 1] over E=6 experts -> weight order [3, 0, 1, 2,
  // 4, 5]. GroupMatmul ties current GEMM work to inputs.size(), so only the
  // first E_a entries participate in this dispatch's matmul computation; the
  // trailing inactive weights are still passed through the prepack-extras path
  // to ZenDNN for prepack cache warming.
  //
  // Bias lists stay sized E_a (active only) - biases are only consumed for
  // experts that receive tokens, so there's no reason to materialize slices
  // for the inactive set.
  const int64_t E = w13.size(0);
  std::vector<at::Tensor> w13_slices(E);
  std::vector<c10::optional<at::Tensor>> w2_weight_slices(E);
  std::vector<c10::optional<at::Tensor>> w13_bias_slices(E_a);
  std::vector<c10::optional<at::Tensor>> w2_bias_slices(E_a);
  std::vector<c10::optional<at::Tensor>> w13_scale_slices(E_a);
  std::vector<c10::optional<at::Tensor>> w2_scale_slices(E_a);

  const bool has_w13_bias = w13_bias.has_value() && w13_bias->defined();
  const bool has_w2_bias = w2_bias.has_value() && w2_bias->defined();
  const bool has_w13_scales = w13_scales.has_value() && w13_scales->defined();
  const bool has_w2_scales = w2_scales.has_value() && w2_scales->defined();

  // Per-tensor caches of dim-0 expert views. Each cache is keyed by
  // TensorImpl*, so the 48 MoE layers in models like Qwen3-30B-A3B each get
  // their own entry and do not evict each other. After the first decode
  // step, every lookup below is an unordered_map hit (no `select` calls).
  // Caches are function-local statics: zero churn while the process runs,
  // released at process shutdown along with the model weights.
  static ExpertSliceCache w13_cache;
  static ExpertSliceCache w2_cache;
  static ExpertSliceCache w13_bias_cache;
  static ExpertSliceCache w2_bias_cache;
  static ExpertSliceCache w13_scales_cache;
  static ExpertSliceCache w2_scales_cache;

  const auto &w13_all_slices = w13_cache.get(w13);
  const auto &w2_all_slices = w2_cache.get(w2);
  const std::vector<at::Tensor> *w13_bias_all_slices =
      has_w13_bias ? &w13_bias_cache.get(*w13_bias) : nullptr;
  const std::vector<at::Tensor> *w2_bias_all_slices =
      has_w2_bias ? &w2_bias_cache.get(*w2_bias) : nullptr;
  const std::vector<at::Tensor> *w13_scales_all_slices =
      has_w13_scales ? &w13_scales_cache.get(*w13_scales) : nullptr;
  const std::vector<at::Tensor> *w2_scales_all_slices =
      has_w2_scales ? &w2_scales_cache.get(*w2_scales) : nullptr;

  // Pass 2a: active experts in active_idx order (positions [0, E_a)).
  std::vector<bool> is_active(E, false);
  for (int64_t a = 0; a < E_a; ++a) {
    const int64_t e = mapping.active_expert_ids[a];
    is_active[e] = true;
    w13_slices[a] = w13_all_slices[e];
    w2_weight_slices[a] = w2_all_slices[e];
    w13_bias_slices[a] =
        has_w13_bias ? c10::optional<at::Tensor>((*w13_bias_all_slices)[e])
                     : c10::nullopt;
    w2_bias_slices[a] =
        has_w2_bias ? c10::optional<at::Tensor>((*w2_bias_all_slices)[e])
                    : c10::nullopt;
    w13_scale_slices[a] =
        has_w13_scales ? c10::optional<at::Tensor>((*w13_scales_all_slices)[e])
                       : c10::nullopt;
    w2_scale_slices[a] =
        has_w2_scales ? c10::optional<at::Tensor>((*w2_scales_all_slices)[e])
                      : c10::nullopt;
  }

  // Pass 2b: inactive experts in original order (positions [E_a, E)).
  // Weight-only — no bias slices needed since these experts are not consumed.
  int64_t fill_idx = E_a;
  for (int64_t e = 0; e < E; ++e) {
    if (is_active[e]) {
      continue;
    }
    w13_slices[fill_idx] = w13_all_slices[e];
    w2_weight_slices[fill_idx] = w2_all_slices[e];
    ++fill_idx;
  }

  // ---------------------- Phase 5 setup: row_ptrs for weighted reduce -------
  // ZenDNN's MoE postop reads each (t, k) result via a raw row
  // address, then accumulates `topk_weights[t, k] * row` into `output[t]`. We
  // reuse the per-expert `grouped_inputs` buffers as the W2 output
  // destinations: by the time W2 needs to write, W13 has already consumed its
  // inputs from these buffers within the fused chain, and both shapes are
  // [M_e, H]. row_ptrs therefore point directly into
  // `mapping.grouped_inputs`.
  at::Tensor row_ptrs =
      at::detail::empty_strided_cpu({total_pairs}, {1}, at::kLong);
  int64_t *row_ptrs_data = row_ptrs.data_ptr<int64_t>();
  for (int64_t i = 0; i < total_pairs; ++i) {
    const auto [a, pos] = mapping.topk_to_expert_row[i];
    row_ptrs_data[i] = reinterpret_cast<int64_t>(
        static_cast<std::byte *>(mapping.grouped_inputs[a].data_ptr()) +
        pos * row_bytes);
  }

  // When `skip_weighted` is set, vLLM has already pre-applied router weights
  // to the input. Pass an all-ones weight vector so the postop accumulates
  // raw expert outputs. (Schema requires K == 1 in this case.)
  const at::Tensor effective_topk_weights =
      skip_weighted ? at::ones_like(topk_weights) : topk_weights;

  const bool two_pass =
      static_cast<bool>(EnvReader::getEnvVariableAsInt("ZENTORCH_TWO_PASS"));

  if (two_pass) {
    // W13 is [E, 2*I, H]; w13_slices[a] is [2*I, H]. After gated activation
    // the intermediate is [M_e, I]; W2 is [E, H, I] so W2's output is [M_e,
    // H].
    const int64_t N = w13.size(1); // 2*I (W13 row dim)
    const int64_t I = N / 2;       // post-gated-activation hidden dim

    // ----- Call 1: W13 + gated activation only
    // -------------------------------- gemm_outputs are [M_e, N] per active
    // expert; the kernel writes the gated-activation result into the first I
    // columns.
    std::vector<at::Tensor> w13_gemm_outs(E_a);
    for (int64_t a = 0; a < E_a; ++a) {
      const int64_t M_e = mapping.grouped_inputs[a].size(0);
      w13_gemm_outs[a] =
          at::detail::empty_strided_cpu({M_e, N}, {N, 1}, input.options());
    }

    const std::vector<c10::optional<at::Tensor>> empty_optional_vec_E{};
    const std::vector<c10::optional<at::Tensor>> empty_optional_vec_Ea{};

    {
      RECORD_FUNCTION("zentorch::fused_moe::two_pass::w13_activation",
                      c10::ArrayRef<c10::IValue>({}));
      zentorch_group_matmul_out_impl(
          /*gemm_outputs=*/w13_gemm_outs,
          /*inputs=*/mapping.grouped_inputs,
          /*w13_weights=*/w13_slices,
          /*w2_weights=*/empty_optional_vec_E,
          /*moe_output=*/c10::nullopt,
          /*topk_weights=*/c10::nullopt,
          /*row_ptrs=*/c10::nullopt,
          /*activation=*/act,
          /*w13_bias=*/w13_bias_slices,
          /*w2_bias=*/empty_optional_vec_Ea,
          /*w13_scales=*/w13_scale_slices,
          /*w2_scales=*/empty_optional_vec_Ea,
          /*zentorch_op_name=*/zentorch_op_name);
    }

    // ----- Call 2: W2 only, with MoE weighted reduce
    // -------------------------- Inputs are the gated-activation outputs [M_e,
    // I] (first I cols of Call 1's output, made contiguous so the kernel sees a
    // tight stride). gemm_outputs reuse mapping.grouped_inputs so the pre-built
    // row_ptrs continue to point at the correct W2 destination rows.
    std::vector<at::Tensor> activation_outputs(E_a);
    for (int64_t a = 0; a < E_a; ++a) {
      activation_outputs[a] = w13_gemm_outs[a].narrow(1, 0, I).contiguous();
    }

    // For Call 2, W2 acts as the only matmul, so we hand it in as
    // `w13_weights` (vector<at::Tensor>). Preserve the active-prefix +
    // inactive-tail layout so ZenDNN's prepack warmer still sees all E
    // experts.
    std::vector<at::Tensor> w2_as_w13(E);
    for (int64_t e = 0; e < E; ++e) {
      w2_as_w13[e] = w2_weight_slices[e].value();
    }

    {
      RECORD_FUNCTION("zentorch::fused_moe::two_pass::w2_reduce",
                      c10::ArrayRef<c10::IValue>({}));
      zentorch_group_matmul_out_impl(
          /*gemm_outputs=*/mapping.grouped_inputs,
          /*inputs=*/activation_outputs,
          /*w13_weights=*/w2_as_w13,
          /*w2_weights=*/empty_optional_vec_E,
          /*moe_output=*/output,
          /*topk_weights=*/effective_topk_weights,
          /*row_ptrs=*/row_ptrs,
          /*activation=*/"none",
          /*w13_bias=*/w2_bias_slices,
          /*w2_bias=*/empty_optional_vec_Ea,
          /*w13_scales=*/w2_scale_slices,
          /*w2_scales=*/empty_optional_vec_Ea,
          /*zentorch_op_name=*/zentorch_op_name);
    }

    return;
  }

  // ---------------------- Single-call fused execution -----------------------
  // gemm_outputs is empty -> ZenDNN allocates W13 outputs internally.
  // We pass `mapping.grouped_inputs` as `w2_outputs`: the W2 down-projection
  // writes back into the same [M_e, H] buffers we used as W13 inputs, saving
  // an allocation per active expert. The full chain
  // (W13 -> gated_act -> W2 -> weighted_reduce -> output) runs inside one
  // `group_matmul_direct` call.
  // This path handles both bf16/f32 weights and int8 weights (with scales).
  zentorch_group_matmul_out_impl(
      /*gemm_outputs=*/{},
      /*inputs=*/mapping.grouped_inputs,
      /*w13_weights=*/w13_slices,
      /*w2_weights=*/w2_weight_slices,
      /*moe_output=*/output,
      /*topk_weights=*/effective_topk_weights,
      /*row_ptrs=*/row_ptrs,
      /*activation=*/act,
      /*w13_bias=*/w13_bias_slices,
      /*w2_bias=*/w2_bias_slices,
      /*w13_scales=*/w13_scale_slices,
      /*w2_scales=*/w2_scale_slices,
      /*zentorch_op_name=*/zentorch_op_name);
}

// ---------------------------------------------------------------------------
// Op registration
// ---------------------------------------------------------------------------

TORCH_LIBRARY_FRAGMENT(zentorch, m) {
  m.def("zentorch_fused_moe(Tensor(a!) output, Tensor input, "
        "Tensor w13, Tensor w2, "
        "Tensor? w13_bias, Tensor? w2_bias, "
        "Tensor topk_weights, Tensor topk_id, "
        "bool skip_weighted, str act, "
        "Tensor? w13_scales=None, Tensor? w2_scales=None, "
        "*, str zentorch_op_name='zentorch::zentorch_fused_moe') -> ()");
}

TORCH_LIBRARY_IMPL(zentorch, CPU, m) {
  m.impl("zentorch_fused_moe", zentorch::zentorch_fused_moe);
}

} // namespace zentorch