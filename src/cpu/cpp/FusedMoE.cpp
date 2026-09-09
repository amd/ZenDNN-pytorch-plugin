/******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#include "FusedMoE.hpp"
#include "EnvReader.hpp"
#include "GroupMatmul.hpp"
#include "Memory.hpp"
#include "Utils.hpp"
#include "lowoha_operators/reorder/lowoha_reorder.hpp"
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
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
// `zentorch_fused_moe` call's per-expert [M_e, H] grouped-input buffers
// (and [M_e, 1] per-token scale buffers when unique-token quant is on)
// are placed contiguously inside this block and exposed as zero-copy
// `torch::stable::from_blob` tensors, which are non-owning by contract. After
// the call returns the from_blob handles are destroyed but the underlying
// memory persists for the next call to reuse — avoiding the per-tensor
// allocator round-trip the default `new_empty` path goes through.
//
// Modeled on vLLM's `cpu_utils::ScratchPadManager` (csrc/cpu/utils.cpp).
//
// Enabled by default. Set `ZENTORCH_USE_SCRATCHPAD=0` to disable and fall
// back to the `new_empty`-per-expert allocation path.
//
// Thread-safety: not safe for concurrent `zentorch_fused_moe` calls from
// different threads. Matches the assumption of vLLM's CPU MoE path
// (single inference stream per process); intra-call work parallelizes via
// `torch::stable::parallel_for` / OMP, not via overlapping op invocations.
//
// Lifetime safety: the from_blob tensors are local to a single
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
    ZENTORCH_CHECK(new_ptr != nullptr, "FusedMoEScratchpad: aligned_alloc(",
                   bytes, ") failed");
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
//   Keying per tensor lets each layer's weight/bias/scale tensors hold their
//   own slice list, so the hit rate after step 1 is effectively 100%.
//
// Why we avoid `unbind(0)`:
//   `unbind(0)` is implemented as a loop of `select(0, i)` internally, so it
//   pays the same per-expert dispatcher cost AND adds an outer `aten::unbind`
//   wrapper on top. Measured 8.7s of `select` becoming 5.9s of `select` +
//   5.7s of `unbind` — a net regression. Calling `select` directly inside
//   the cache build keeps the one-time fill cheaper.
//
// Keying a stable tensor:
//   `get()` is a new handle each unbox, so key on `data_ptr()` instead. That
//   stays within the stable ABI. Collisions can't happen here: each weight
//   role has its own cache and expert weights don't alias.
//
// Lifetime / safety:
//   - Each entry holds a strong ref to the *base* tensor (`Entry::base`), not
//     just the view slices. This pins the keyed storage so its address can't
//     be freed and recycled into a different tensor (which would return stale
//     slices).
//   - For long-lived model weights this is benign. Callers that pass a fresh
//     weight tensor every call grow the map by one (process-lifetime) entry.
//   - Not thread-safe across concurrent `zentorch_fused_moe` calls. Matches
//     existing assumptions of this op (single inference stream per process).
// ---------------------------------------------------------------------------
class ExpertSliceCache {
public:
  // Returns the per-expert views for `tensor`. On first encounter, builds
  // them with one `select(0, e)` per expert; subsequent calls are an
  // unordered_map lookup. A hit is always the same tensor: `Entry::base`
  // pins the keyed storage so its address cannot be recycled to a
  // different tensor while the entry lives, so no identity re-check is
  // needed.
  const std::vector<torch::stable::Tensor> &
  get(const torch::stable::Tensor &tensor) {
    const void *key = tensor.data_ptr();
    auto it = entries_.find(key);
    if (it != entries_.end()) {
      return it->second.slices;
    }
    auto emplaced = entries_.emplace(key, build_entry(tensor));
    return emplaced.first->second.slices;
  }

  // Drop all cached entries (and the strong tensor refs they hold).
  void clear() { entries_.clear(); }

private:
  struct Entry {
    // Strong ref to the keyed tensor: pins its storage so the map key cannot
    // be recycled to a different tensor while this entry exists.
    torch::stable::Tensor base;
    std::vector<torch::stable::Tensor> slices;
  };

  static Entry build_entry(const torch::stable::Tensor &tensor) {
    const int64_t E = tensor.size(0);
    std::vector<torch::stable::Tensor> slices;
    slices.reserve(E);
    for (int64_t e = 0; e < E; ++e) {
      slices.emplace_back(torch::stable::select(tensor, 0, e));
    }
    return Entry{tensor, std::move(slices)};
  }

  std::unordered_map<const void *, Entry> entries_;
};

// Singleton owning every FusedMoE per-expert view cache, so they can be
// flushed together via zentorch_flush_moe_weight_cache. Single inference
// stream per process (this op's existing assumption), so no locking.
struct MoEWeightCaches {
  ExpertSliceCache w13, w2, w13_bias, w2_bias, w13_scales, w2_scales;

  static MoEWeightCaches &instance() {
    static MoEWeightCaches inst;
    return inst;
  }

  void flush() {
    w13.clear();
    w2.clear();
    w13_bias.clear();
    w2_bias.clear();
    w13_scales.clear();
    w2_scales.clear();
  }
};

// Free-function entry point for the flush op (defined below).
void flush_moe_weight_cache_impl() { MoEWeightCaches::instance().flush(); }

// ---------------------------------------------------------------------------
// Phase 1 — Token-Expert Grouping
//
// For each routed (token t, slot k) we need expert e = topk_id[t][k] to
// receive a copy of input[t] in its per-expert input buffer. We do this in
// two passes:
//
//   Pass 1 (single-threaded O(T*K) bookkeeping; stays serial so first-seen
//   active_idx assignment is deterministic):
//     - Walk the T*K (t, k) pairs in flat order.
//     - On the first encounter of expert e, append e to `active_expert_ids`.
//       Already-seen experts resolve via `expert_to_active[e]` (size E,
//       filled with -1), not a linear scan of `active_expert_ids`.
//     - For each (t, k), record `topk_to_expert_row[i] = (active_idx, pos)`
//       where `pos = tokens_per_active[a]++` is the deterministic write
//       position in active slot a's eventual input tensor (no atomics needed).
//     - Also append the source token id `t` to
//       `source_tokens_per_active[a]` so Pass-2 can iterate per-expert
//       without re-scanning `topk_to_expert_row`. Each list is reserved to
//       T on first encounter (M_e <= T when a token hits an expert once).
//
//   Pass 2 (per-expert parallel memcpy of unique source rows):
//     - Allocate one [M_e, H] tensor per active expert, indexed by
//       active_idx (so `grouped_inputs.size() == E_a`, ready to hand to
//       `group_matmul` directly). Allocation dtype is always the
//       activation dtype (bf16). Unique-token quant packs int8 payload
//       into that bf16-sized storage; grouping never quantizes.
//       DA8W8/DA8W4 also allocate per-token scales.
//     - `torch::stable::parallel_for` over the E_a active experts
//       (`grain_size=1`): each worker owns one destination buffer and
//       walks its own `source_tokens_per_active[a]` list to memcpy the
//       source rows in.
//       This exposes ~E_a tasks (instead of `total_pairs / grain_size`)
//       and confines each thread's writes to a single destination buffer,
//       avoiding the scattered-write and false-sharing pattern of a
//       per-pair scheme.
//
// Pass-1 lists stay sized to E_a. The only size-E structure is the
// `expert_to_active` reverse map used during the T*K walk and discarded
// when mapping returns.
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
// After Pass 2 (per-expert parallel memcpy of `src`; bf16 shown):
//   grouped_inputs[0]  (active_idx 0 = expert 3, M=2) = [ src[0], src[1] ]
//   grouped_inputs[1]  (active_idx 1 = expert 0, M=2) = [ src[0], src[2] ]
//   grouped_inputs[2]  (active_idx 2 = expert 1, M=2) = [ src[1], src[2] ]
// DA8W8/DA8W4: grouped_inputs is still a bf16 [M_e, H] allocation; Pass-2
// packs unique-token int8 rows into the leading M_e*H bytes (plus a
// scale broadcast into grouped_src_scales).
// ---------------------------------------------------------------------------

struct TokenExpertMapping {
  // Size E_a. grouped_inputs[a] is the [M_e, H] buffer for the a-th
  // active expert (active expert id = active_expert_ids[a]). Dtype is
  // the activation dtype (bf16). Unique-token quant packs int8 rows
  // into this storage; W2 may reuse it as a bf16 dest.
  std::vector<torch::stable::Tensor> grouped_inputs;
  // Size E_a. active_expert_ids[a] = the original expert id (in [0, E))
  // for the a-th active expert. Order is first-encounter in `topk_id`.
  std::vector<int32_t> active_expert_ids;
  // T*K entries; index (t*K + k) holds (active_idx, row_in_expert).
  // Kept token-major because Phase 5's `row_ptrs` setup walks pairs in this
  // order to build the MoE weighted-reduce postop input.
  std::vector<std::pair<int32_t, int32_t>> topk_to_expert_row;
  // Size E_a. source_tokens_per_active[a] lists the unique-token row ids
  // (t values) that fill each expert's grouped buffer, in the deterministic
  // in-expert order assigned by Pass-1. Pass-2 consumes this directly so
  // each worker thread walks one contiguous list and writes one dest buffer.
  std::vector<std::vector<int32_t>> source_tokens_per_active;
  // DA8W8/DA8W4: [M_e, 1] per-token scales, filled by grouping memcpy of
  // already-quantized unique-token scales. Empty for bf16.
  std::vector<torch::stable::Tensor> grouped_src_scales;
};

static void quantize_unique_tokens_s8(const torch::stable::Tensor &input,
                                      torch::stable::Tensor &input_quant,
                                      torch::stable::Tensor &token_src_scales) {
  const int64_t T = input.size(0);
  const int64_t H = input.size(1);

  ZENTORCH_CHECK(input.scalar_type() == c10::kBFloat16,
                 "zentorch_fused_moe: unique-token quant requires bf16 input");
  ZENTORCH_CHECK(input.is_contiguous(),
                 "zentorch_fused_moe: unique-token quant requires contiguous "
                 "bf16 input");
  ZENTORCH_CHECK(T > 0 && H > 0,
                 "zentorch_fused_moe: unique-token quant requires T>0 and H>0");
  ZENTORCH_CHECK(is_avx512_supported(),
                 "zentorch_fused_moe: unique-token "
                 "dynamic_per_token_quant_bf16_s8_native requires AVX-512F/"
                 "BW/VL");

  input_quant = torch::stable::new_empty(input, {T, H}, c10::kChar);
  // Native kernel writes one f32 scale per row. GroupMatmul converts to
  // wei_scale dtype when they differ (DLP requires src_scale.dt ==
  // wei_scale.dt).
  token_src_scales = torch::stable::new_empty(input, {T, 1}, c10::kFloat);

  zendnnl::lowoha::reorder::dynamic_per_token_quant_bf16_s8_native(
      reinterpret_cast<const uint16_t *>(input.const_data_ptr()),
      static_cast<int8_t *>(input_quant.data_ptr()),
      static_cast<float *>(token_src_scales.data_ptr()), T, H);
}

static TokenExpertMapping
build_token_expert_mapping(const torch::stable::Tensor &input,
                           const torch::stable::Tensor &topk_id, int64_t E,
                           const torch::stable::Tensor *src_scales) {

  const int64_t T = input.size(0);
  const int64_t H = input.size(1);
  const int64_t K = topk_id.size(1);
  const int64_t total_pairs = T * K;
  const int64_t row_bytes = H * static_cast<int64_t>(input.element_size());
  const auto dtype = input.scalar_type();

  // topk_id is contiguous int32 [T, K]; flat indexing is i = t*K + k.
  const int32_t *topk_id_serialized = topk_id.const_data_ptr<int32_t>();

  // ----- Pass 1: register active experts + assign per-active positions ----
  // First-seen order of `active_expert_ids` is unchanged. `expert_to_active[e]`
  // is O(1) (size E, -1 = unseen) so Qwen-scale E_a does not turn each pair
  // into a scan of the growing active list. Source-token lists are reserved
  // to T on first encounter to avoid realloc in the hot loop.
  TokenExpertMapping mapping;
  mapping.topk_to_expert_row.resize(total_pairs);
  const size_t e_a_bound =
      static_cast<size_t>(E < total_pairs ? E : total_pairs);
  mapping.active_expert_ids.reserve(e_a_bound);
  mapping.source_tokens_per_active.reserve(e_a_bound);
  std::vector<int32_t> tokens_per_active;
  tokens_per_active.reserve(e_a_bound);
  std::vector<int32_t> expert_to_active(static_cast<size_t>(E), -1);
  {
    ZENTORCH_RECORD_SCOPE("zentorch::fused_moe::pass1_active_set_build");
    for (int64_t i = 0; i < total_pairs; ++i) {
      const int32_t e = topk_id_serialized[i];
      ZENTORCH_CHECK(static_cast<uint32_t>(e) < static_cast<uint32_t>(E),
                     "zentorch_fused_moe: topk_id value ", e,
                     " is outside [0, ", E, ")");
      const int32_t t = static_cast<int32_t>(i / K);
      int32_t a = expert_to_active[static_cast<size_t>(e)];
      if (a < 0) {
        a = static_cast<int32_t>(mapping.active_expert_ids.size());
        expert_to_active[static_cast<size_t>(e)] = a;
        mapping.active_expert_ids.emplace_back(e);
        tokens_per_active.emplace_back(0);
        mapping.source_tokens_per_active.emplace_back();
        mapping.source_tokens_per_active.back().reserve(static_cast<size_t>(T));
      }
      const int32_t pos = tokens_per_active[a]++;
      mapping.topk_to_expert_row[i] = {a, pos};
      mapping.source_tokens_per_active[a].emplace_back(t);
    }
  } // RECORD_FUNCTION pass1_active_set_build
  const int64_t E_a = static_cast<int64_t>(mapping.active_expert_ids.size());

  // ----- Allocate per-active-expert [M_e, H] tensors ----------------------
  // Default path (`ZENTORCH_USE_SCRATCHPAD=1`, the default): pack every
  // per-expert [M_e, H] region (and [M_e, 1] scale region when
  // `src_scales` is set) contiguously into a single 64-byte-aligned
  // block reused (and grown when needed) across calls, then expose each
  // region as a zero-copy `torch::stable::from_blob` tensor. Each region's
  // byte length is rounded up to a 64-byte multiple so the next region's base
  // also starts on a 64-byte boundary; the within-region row stride stays at
  // the natural H * elem_size — identical to what a fresh allocation would
  // give us, so ZenDNN and the Pass-2 memcpy both see the same layout under
  // either path. Input and scale regions for one expert are packed
  // back-to-back so the Pass-2 worker that owns that expert writes a
  // contiguous scratchpad span.
  //
  // Fallback path (`ZENTORCH_USE_SCRATCHPAD=0`): one `new_empty` per active
  // expert. PyTorch's caching allocator amortizes well for stable per-call
  // working-set sizes but still pays a per-tensor metadata round-trip (plus,
  // on the stable ABI, a dispatcher hop) on every call.
  const int int_env_value =
      EnvReader::getEnvVariableAsInt("ZENTORCH_USE_SCRATCHPAD");
  const bool use_scratchpad = static_cast<bool>(int_env_value);
  const bool has_src_scales = src_scales != nullptr;

  mapping.grouped_inputs.resize(E_a);
  if (has_src_scales) {
    mapping.grouped_src_scales.resize(E_a);
  }
  {
    ZENTORCH_RECORD_SCOPE("zentorch::fused_moe::scratchpad_allocation");
    if (use_scratchpad) {
      const size_t row_bytes_sz = static_cast<size_t>(row_bytes);
      const size_t scale_bytes_per_token =
          has_src_scales ? static_cast<size_t>(src_scales->element_size()) : 0;
      std::vector<size_t> region_offsets(E_a);
      std::vector<size_t> scale_offsets(has_src_scales ? E_a : 0);
      size_t total_bytes = 0;
      for (int64_t a = 0; a < E_a; ++a) {
        const size_t M_e = static_cast<size_t>(tokens_per_active[a]);
        region_offsets[a] = total_bytes;
        total_bytes +=
            round_up_to(M_e * row_bytes_sz, FusedMoEScratchpad::kAlignment);
        if (has_src_scales) {
          scale_offsets[a] = total_bytes;
          total_bytes += round_up_to(M_e * scale_bytes_per_token,
                                     FusedMoEScratchpad::kAlignment);
        }
      }
      auto &sp = FusedMoEScratchpad::get();
      sp.reserve(total_bytes);
      std::byte *base = sp.data();
      // Hoisted: each accessor is a shim call. Grouped token regions are
      // always the activation dtype (bf16); unique-token int8 is packed
      // into that storage by Pass-2.
      const auto device = input.device();
      for (int64_t a = 0; a < E_a; ++a) {
        mapping.grouped_inputs[a] = torch::stable::from_blob(
            base + region_offsets[a],
            {static_cast<int64_t>(tokens_per_active[a]), H}, {H, 1}, device,
            dtype);
      }
      for (int64_t a = 0; a < E_a; ++a) {
        ZENTORCH_CHECK(mapping.grouped_inputs[a].is_contiguous(),
                       "zentorch_fused_moe: grouped_inputs[", a,
                       "] must be a contiguous bf16 buffer");
      }
      if (has_src_scales) {
        const auto scale_device = src_scales->device();
        const auto scale_dtype = src_scales->scalar_type();
        for (int64_t a = 0; a < E_a; ++a) {
          mapping.grouped_src_scales[a] = torch::stable::from_blob(
              base + scale_offsets[a],
              {static_cast<int64_t>(tokens_per_active[a]), 1}, {1, 1},
              scale_device, scale_dtype);
        }
      }
    } else {
      for (int64_t a = 0; a < E_a; ++a) {
        mapping.grouped_inputs[a] = torch::stable::new_empty(
            input, {static_cast<int64_t>(tokens_per_active[a]), H}, dtype);
      }
      for (int64_t a = 0; a < E_a; ++a) {
        ZENTORCH_CHECK(mapping.grouped_inputs[a].is_contiguous(),
                       "zentorch_fused_moe: grouped_inputs[", a,
                       "] must be a contiguous bf16 buffer");
      }
      if (has_src_scales) {
        for (int64_t a = 0; a < E_a; ++a) {
          mapping.grouped_src_scales[a] = torch::stable::new_empty(
              *src_scales, {static_cast<int64_t>(tokens_per_active[a]), 1});
        }
      }
    }
  } // RECORD_FUNCTION scratchpad_allocation

  return mapping;
}

// Pass 2: memcpy unique source rows into per-expert buffers.
// Grouping does not quantize.
//   * Float path: copy bf16 rows into contiguous bf16 grouped_inputs.
//   * Unique-token path: `src` is a real int8 [T, H] buffer; dest is a
//     contiguous bf16 [M_e, H] buffer. Copy H int8 bytes per row into
//     the leading M_e*H bytes of that bf16 storage.
// When `src_scales` is set, also broadcast the matching [T, 1]
// unique-token scales into per-expert [M_e, 1] buffers.
static void scatter_tokens_to_experts(TokenExpertMapping &mapping,
                                      const torch::stable::Tensor &src,
                                      const torch::stable::Tensor *src_scales) {
  const int64_t E_a = static_cast<int64_t>(mapping.active_expert_ids.size());
  const int64_t H = src.size(1);
  const bool has_src_scales = src_scales != nullptr;
  const bool src_is_s8 = src.scalar_type() == c10::kChar;
  const int64_t src_row_bytes = H * static_cast<int64_t>(src.element_size());
  // Dest stride follows src row width (H for packed int8, 2H for bf16).
  // Do not key this on has_src_scales: a bf16 src with scales would copy
  // 2H-byte rows at stride H and overlap writes.
  const int64_t dst_row_bytes = src_is_s8 ? H : src_row_bytes;
  ZENTORCH_CHECK(src_is_s8 == has_src_scales,
                 "zentorch_fused_moe: int8 grouping and unique-token src "
                 "scales must be used together");
  ZENTORCH_CHECK(!has_src_scales || mapping.grouped_src_scales.size() ==
                                        static_cast<size_t>(E_a),
                 "zentorch_fused_moe: grouped_src_scales not allocated");
  ZENTORCH_CHECK(!src_is_s8 ||
                     mapping.grouped_inputs[0].scalar_type() == c10::kBFloat16,
                 "zentorch_fused_moe: unique-token grouping dest must be "
                 "contiguous bf16");

  const auto *src_base =
      reinterpret_cast<const std::byte *>(src.const_data_ptr());
  const int64_t scale_bytes =
      has_src_scales ? static_cast<int64_t>(src_scales->element_size()) : 0;
  const auto *scale_src =
      has_src_scales
          ? reinterpret_cast<const std::byte *>(src_scales->const_data_ptr())
          : nullptr;

  std::vector<std::byte *> dst_base(E_a);
  std::vector<std::byte *> scale_dst_base(has_src_scales ? E_a : 0);
  {
    ZENTORCH_RECORD_SCOPE("zentorch::fused_moe::pass2_dst_base_setup");
    for (int64_t a = 0; a < E_a; ++a) {
      dst_base[a] =
          reinterpret_cast<std::byte *>(mapping.grouped_inputs[a].data_ptr());
      if (has_src_scales) {
        scale_dst_base[a] = reinterpret_cast<std::byte *>(
            mapping.grouped_src_scales[a].data_ptr());
      }
    }
  } // RECORD_FUNCTION pass2_dst_base_setup

  {
    ZENTORCH_RECORD_SCOPE("zentorch::fused_moe::pass2_parallel_memcpy");
    torch::stable::parallel_for(
        0, E_a, /*grain_size=*/1, [&](int64_t a_begin, int64_t a_end) {
          for (int64_t a = a_begin; a < a_end; ++a) {
            std::byte *dst = dst_base[a];
            const auto &src_tokens = mapping.source_tokens_per_active[a];
            const int64_t M_e = static_cast<int64_t>(src_tokens.size());
            for (int64_t p = 0; p < M_e; ++p) {
              std::memcpy(dst + p * dst_row_bytes,
                          src_base + static_cast<int64_t>(src_tokens[p]) *
                                         src_row_bytes,
                          src_row_bytes);
              if (has_src_scales) {
                std::memcpy(scale_dst_base[a] + p * scale_bytes,
                            scale_src + static_cast<int64_t>(src_tokens[p]) *
                                            scale_bytes,
                            scale_bytes);
              }
            }
          }
        });
  } // RECORD_FUNCTION pass2_parallel_memcpy
}

} // namespace

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
// This single op serves three weight regimes, dispatched by weight dtype
// inside `zentorch_group_matmul_out_impl` (see GroupMatmul.cpp):
//   * bf16 / f32 weights            — plain grouped GEMM (no scales).
//   * DA8W8 weights + scales        — unique-token s8 activation quant, then
//     int8 grouping. W2 still dynamically quantizes the post-activation rows.
//   * packed-s4 weights (DA8W4) + per-group scales — bf16 grouping;
//     ZenDNN quantizes each expert row (`dynamic_quant=true`). Unique-
//     token is DA8W8-only (fused_moe op1_internal rejects src=s8 +
//     wei=s4 + dst=bf16). Packed weights come in either container,
//     `[E, N, K/8]` int32 or `[E, N, K/2]` int8 (full-width int8 is
//     DA8W8, not DA8W4).
//     See docs/zentorch_fused_moe.md §"DA8W4 weights".
//
// ----------------------------- Input contract ------------------------------
// Shape/dtype/bias validation is performed once by the producing Python layer.
// The C++ op trusts its inputs and assumes:
//
//   input          : 2D [T, H], contiguous. f32, bf16, or fp16 (bf16 ONLY
//                    for the quantized regimes: DA8W8 and DA8W4).
//   output         : 2D [T, H], same dtype as input, UNINITIALIZED
//                    (Phase 5's reduce writes every element)
//   w13            : 3D [E, 2*I, H] (input dtype or int8) or, for DA8W4,
//                    [E, 2*I, H/8] int32 / [E, 2*I, H/2] int8
//   w2             : 3D [E, H, I]   (input dtype or int8) or, for DA8W4,
//                    [E, H, I/8]   int32 / [E, H, I/2]   int8
//   w13_bias       : None or [E, 2*I] (same dtype as input)
//   w2_bias        : None or [E, H]   (same dtype as input)
//   w13_scales     : None (float), per-channel (DA8W8), or per-group (DA8W4)
//   w2_scales      : as w13_scales for the down projection
//   topk_weights   : 2D [T, K], f32, contiguous
//   topk_id        : 2D [T, K], int32, contiguous, values in [0, E)
//   skip_weighted  : bool; if true, requires K == 1
//   act            : one of {"silu", "gelu", "gelu_tanh", "swigluoai"}
// ---------------------------------------------------------------------------

void zentorch_fused_moe(torch::stable::Tensor &output,
                        const torch::stable::Tensor &input,
                        const torch::stable::Tensor &w13,
                        const torch::stable::Tensor &w2,
                        const std::optional<torch::stable::Tensor> &w13_bias,
                        const std::optional<torch::stable::Tensor> &w2_bias,
                        const torch::stable::Tensor &topk_weights,
                        const torch::stable::Tensor &topk_id,
                        bool skip_weighted, std::string_view act,
                        const std::optional<torch::stable::Tensor> &w13_scales,
                        const std::optional<torch::stable::Tensor> &w2_scales,
                        std::string zentorch_op_name) {

  const int64_t T = input.size(0);
  const int64_t K = topk_id.size(1);
  const int64_t E = w13.size(0);
  const int64_t total_pairs = T * K;
  const int64_t row_bytes =
      input.size(1) * static_cast<int64_t>(input.element_size());

  // Opt-in split of W13+act and W2+reduce. Off by default. Unique-token
  // pre-quant is the fused-path optimization only — never combined with
  // this split (two-pass already group-quantizes each GEMM inside ZenDNN).
  const bool two_pass =
      static_cast<bool>(EnvReader::getEnvVariableAsInt("ZENTORCH_TWO_PASS"));
  // Unique-token vs grouped quant on the fused call. Default ON. Set
  // ZENTORCH_MOE_PREQUANT=0 to A/B the same fused path with bf16 grouping.
  const bool moe_prequant = static_cast<bool>(
      EnvReader::getEnvVariableAsInt("ZENTORCH_MOE_PREQUANT"));

  // Unique-token s8 quant is fused DA8W8 only. ZenDNN fused_moe
  // op1_internal allows mixed src/dst (s8 src, bf16 dst) only when
  // wei=s8. Packed s4 (DA8W4, wei=s4) keeps bf16 grouping and
  // dynamic_quant=true. Off when two_pass is set or
  // ZENTORCH_MOE_PREQUANT=0.
  // Full-width int8: w13 is [E, N, H]. Packed s4 is [E, N, H/2] int8
  // or [E, N, H/8] int32.
  const bool da8w8_weights =
      w13.scalar_type() == c10::kChar && w13.size(2) == input.size(1);
  const bool use_prequant = !two_pass && moe_prequant &&
                            w13_scales.has_value() && w13_scales->defined() &&
                            da8w8_weights &&
                            input.scalar_type() == c10::kBFloat16;

  torch::stable::Tensor input_quant;
  torch::stable::Tensor token_src_scales;
  if (use_prequant) {
    ZENTORCH_RECORD_SCOPE("zentorch::fused_moe::unique_token_dynamic_quant");
    quantize_unique_tokens_s8(input, input_quant, token_src_scales);
  }

  // ---------------------- Phase 1: token-expert grouping ---------------------
  // Always allocate grouped_inputs as bf16 [M_e, H] from `input`. Unique-
  // token int8 is packed into that storage by scatter; it must not drive
  // the allocation dtype.
  TokenExpertMapping mapping;
  {
    ZENTORCH_RECORD_SCOPE("zentorch::fused_moe::token_expert_grouping");
    mapping = build_token_expert_mapping(
        input, topk_id, E, use_prequant ? &token_src_scales : nullptr);
    if (use_prequant) {
      scatter_tokens_to_experts(mapping, input_quant, &token_src_scales);
    } else {
      scatter_tokens_to_experts(mapping, input, nullptr);
    }
  }
  const int64_t E_a = static_cast<int64_t>(mapping.active_expert_ids.size());
  std::vector<std::optional<torch::stable::Tensor>> src_scale_slices;
  if (use_prequant) {
    src_scale_slices.resize(E_a);
    for (int64_t a = 0; a < E_a; ++a) {
      src_scale_slices[a] = mapping.grouped_src_scales[a];
    }
  }

  // ---------------------- Temporary Guard ------------------------------------
  // zentorch_group_matmul_out_impl requires E_a > 1. E_a == 1 is structurally
  // guaranteed when K == 1 and T == 1 (apply_router_weight_on_input=True
  // path). Rather than letting the call reach group_matmul_direct and crash
  // on its inputs.size() > 1 assertion, fail here with a clear, actionable
  // message.
  ZENTORCH_CHECK(
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
  std::vector<torch::stable::Tensor> w13_slices(E);
  std::vector<std::optional<torch::stable::Tensor>> w2_weight_slices(E);
  std::vector<std::optional<torch::stable::Tensor>> w13_bias_slices(E_a);
  std::vector<std::optional<torch::stable::Tensor>> w2_bias_slices(E_a);
  std::vector<std::optional<torch::stable::Tensor>> w13_scale_slices(E_a);
  std::vector<std::optional<torch::stable::Tensor>> w2_scale_slices(E_a);

  const bool has_w13_bias = w13_bias.has_value() && w13_bias->defined();
  const bool has_w2_bias = w2_bias.has_value() && w2_bias->defined();
  const bool has_w13_scales = w13_scales.has_value() && w13_scales->defined();
  const bool has_w2_scales = w2_scales.has_value() && w2_scales->defined();

  // Per-tensor caches of dim-0 expert views. Each cache is keyed by storage
  // address, so the 48 MoE layers in models like Qwen3-30B-A3B each get
  // their own entry and do not evict each other. After the first decode
  // step, every lookup below is an unordered_map hit (no `select` calls).
  // Caches live in a process-lifetime singleton: zero churn while the process
  // runs, released at shutdown along with the model weights. Tests can drop
  // all of them between cases via zentorch_flush_moe_weight_cache.
  MoEWeightCaches &caches = MoEWeightCaches::instance();

  const auto &w13_all_slices = caches.w13.get(w13);
  const auto &w2_all_slices = caches.w2.get(w2);
  const std::vector<torch::stable::Tensor> *w13_bias_all_slices =
      has_w13_bias ? &caches.w13_bias.get(*w13_bias) : nullptr;
  const std::vector<torch::stable::Tensor> *w2_bias_all_slices =
      has_w2_bias ? &caches.w2_bias.get(*w2_bias) : nullptr;
  const std::vector<torch::stable::Tensor> *w13_scales_all_slices =
      has_w13_scales ? &caches.w13_scales.get(*w13_scales) : nullptr;
  const std::vector<torch::stable::Tensor> *w2_scales_all_slices =
      has_w2_scales ? &caches.w2_scales.get(*w2_scales) : nullptr;

  // Pass 2a: active experts in active_idx order (positions [0, E_a)).
  // uint8_t, not vector<bool>: vector<bool> is not a real container.
  std::vector<uint8_t> is_active(static_cast<size_t>(E), 0);
  for (int64_t a = 0; a < E_a; ++a) {
    const int64_t e = mapping.active_expert_ids[a];
    is_active[static_cast<size_t>(e)] = 1;
    w13_slices[a] = w13_all_slices[e];
    w2_weight_slices[a] = w2_all_slices[e];
    w13_bias_slices[a] =
        has_w13_bias
            ? std::optional<torch::stable::Tensor>((*w13_bias_all_slices)[e])
            : std::nullopt;
    w2_bias_slices[a] =
        has_w2_bias
            ? std::optional<torch::stable::Tensor>((*w2_bias_all_slices)[e])
            : std::nullopt;
    w13_scale_slices[a] =
        has_w13_scales
            ? std::optional<torch::stable::Tensor>((*w13_scales_all_slices)[e])
            : std::nullopt;
    w2_scale_slices[a] =
        has_w2_scales
            ? std::optional<torch::stable::Tensor>((*w2_scales_all_slices)[e])
            : std::nullopt;
  }

  // Pass 2b: inactive experts in original order (positions [E_a, E)).
  // Weight-only — no bias slices needed since these experts are not consumed.
  int64_t fill_idx = E_a;
  for (int64_t e = 0; e < E; ++e) {
    if (is_active[static_cast<size_t>(e)]) {
      continue;
    }
    w13_slices[fill_idx] = w13_all_slices[e];
    w2_weight_slices[fill_idx] = w2_all_slices[e];
    ++fill_idx;
  }

  // ---------------------- Phase 5 setup: row_ptrs for weighted reduce -------
  // ZenDNN's MoE postop reads each (t, k) result via a raw row
  // address, then accumulates `topk_weights[t, k] * row` into `output[t]`.
  // W2 destinations are bf16 [M_e, H]. grouped_inputs is always allocated
  // as that dtype, so W2 reuses it (unique-token int8 occupies only the
  // leading M_e*H bytes and is consumed by W13 before W2 writes).
  const std::vector<torch::stable::Tensor> *w2_row_bufs =
      &mapping.grouped_inputs;
  torch::stable::Tensor row_ptrs;
  {
    ZENTORCH_RECORD_SCOPE("zentorch::fused_moe::gather_row_ptrs");
    row_ptrs = torch::stable::new_empty(input, {total_pairs},
                                        torch::headeronly::ScalarType::Long);
    std::vector<std::byte *> dest_bases(static_cast<size_t>(E_a));
    for (int64_t a = 0; a < E_a; ++a) {
      dest_bases[a] = static_cast<std::byte *>((*w2_row_bufs)[a].data_ptr());
    }
    int64_t *row_ptrs_data = row_ptrs.mutable_data_ptr<int64_t>();
    for (int64_t i = 0; i < total_pairs; ++i) {
      const int32_t a = mapping.topk_to_expert_row[i].first;
      const int32_t pos = mapping.topk_to_expert_row[i].second;
      row_ptrs_data[i] =
          reinterpret_cast<int64_t>(dest_bases[a] + pos * row_bytes);
    }
  }

  // GroupMatmul's s8 path keys off Tensor dtype == int8. Unique-token
  // payload is packed int8 in bf16 storage; wrap the leading M_e*H bytes
  // as a non-owning int8 view for W13.
  std::vector<torch::stable::Tensor> grouped_s8_views;
  if (use_prequant) {
    ZENTORCH_RECORD_SCOPE("zentorch::fused_moe::s8_views");
    grouped_s8_views.resize(E_a);
    for (int64_t a = 0; a < E_a; ++a) {
      const auto &buf = mapping.grouped_inputs[a];
      grouped_s8_views[a] =
          torch::stable::from_blob(buf.data_ptr(), {buf.size(0), buf.size(1)},
                                   {buf.size(1), 1}, buf.device(), c10::kChar);
    }
  }

  // When `skip_weighted` is set, vLLM has already pre-applied router weights
  // to the input. Pass an all-ones weight vector so the postop accumulates
  // raw expert outputs. (Schema requires K == 1 in this case.)
  const torch::stable::Tensor effective_topk_weights =
      skip_weighted
          ? torch::stable::fill_(torch::stable::empty_like(topk_weights), 1.0)
          : topk_weights;

  // Unique-token pre-quant is fused-path only (`use_prequant` is false
  // whenever two_pass is set or ZENTORCH_MOE_PREQUANT=0). ZENTORCH_TWO_PASS
  // splits W13 and W2 into two group_matmul_direct calls with bf16
  // grouped inputs.
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
    std::vector<torch::stable::Tensor> w13_gemm_outs(E_a);
    for (int64_t a = 0; a < E_a; ++a) {
      const int64_t M_e = mapping.grouped_inputs[a].size(0);
      w13_gemm_outs[a] = torch::stable::new_empty(input, {M_e, N});
    }

    const std::vector<std::optional<torch::stable::Tensor>>
        empty_optional_vec_E{};
    const std::vector<std::optional<torch::stable::Tensor>>
        empty_optional_vec_Ea{};

    {
      ZENTORCH_RECORD_SCOPE("zentorch::fused_moe::two_pass::w13_activation");
      zentorch_group_matmul_out_impl(
          /*gemm_outputs=*/w13_gemm_outs,
          /*inputs=*/mapping.grouped_inputs,
          /*w13_weights=*/w13_slices,
          /*w2_weights=*/empty_optional_vec_E,
          /*moe_output=*/std::nullopt,
          /*topk_weights=*/std::nullopt,
          /*row_ptrs=*/std::nullopt,
          /*activation=*/act,
          /*w13_bias=*/w13_bias_slices,
          /*w2_bias=*/empty_optional_vec_Ea,
          /*w13_scales=*/w13_scale_slices,
          /*w2_scales=*/empty_optional_vec_Ea,
          /*src_scales=*/{},
          /*zentorch_op_name=*/zentorch_op_name);
    }

    // ----- Call 2: W2 only, with MoE weighted reduce
    // -------------------------- Inputs are the gated-activation outputs [M_e,
    // I] (first I cols of Call 1's output, made contiguous so the kernel sees a
    // tight stride). gemm_outputs reuse grouped_inputs (bf16 W13 src) so
    // the pre-built row_ptrs still target the W2 destination rows.
    std::vector<torch::stable::Tensor> activation_outputs(E_a);
    for (int64_t a = 0; a < E_a; ++a) {
      activation_outputs[a] = torch::stable::contiguous(
          torch::stable::narrow(w13_gemm_outs[a], 1, 0, I));
    }

    // For Call 2, W2 acts as the only matmul, so we hand it in as the
    // non-optional `w13_weights` list. Preserve the active-prefix +
    // inactive-tail layout so ZenDNN's prepack warmer still sees all E
    // experts.
    std::vector<torch::stable::Tensor> w2_as_w13(E);
    for (int64_t e = 0; e < E; ++e) {
      w2_as_w13[e] = w2_weight_slices[e].value();
    }

    {
      ZENTORCH_RECORD_SCOPE("zentorch::fused_moe::two_pass::w2_reduce");
      zentorch_group_matmul_out_impl(
          /*gemm_outputs=*/*w2_row_bufs,
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
          /*src_scales=*/{},
          /*zentorch_op_name=*/zentorch_op_name);
    }

    return;
  }

  // ---------------------- Single-call fused execution -----------------------
  // W13 -> gated_act -> W2 -> weighted_reduce in one `group_matmul_direct`.
  //   * float: gemm_outputs empty -> ZenDNN allocates W13; W2 reuses
  //     grouped_inputs (matched src/dst precision).
  //   * prequant: W13 src is the int8 view of bf16 grouped_inputs;
  //     gemm_outputs are the same bf16 buffers (W2 dest / row_ptrs).
  zentorch_group_matmul_out_impl(
      /*gemm_outputs=*/use_prequant ? *w2_row_bufs
                                    : std::vector<torch::stable::Tensor>{},
      /*inputs=*/use_prequant ? grouped_s8_views : mapping.grouped_inputs,
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
      /*src_scales=*/src_scale_slices,
      /*zentorch_op_name=*/zentorch_op_name);
}

// ---------------------------------------------------------------------------
// Op registration
// ---------------------------------------------------------------------------

// Drops all FusedMoE per-expert view caches (ExpertSliceCache). Primarily a
// test hook so each case starts with no cross-call view-cache state.
void zentorch_flush_moe_weight_cache() { flush_moe_weight_cache_impl(); }

STABLE_TORCH_LIBRARY_FRAGMENT(zentorch, m) {
  // `output` is the leading schema arg (mirroring vLLM's cpu_fused_moe), not a
  // trailing `.out` kwarg, so it already lines up positionally with the
  // kernel's first parameter for TORCH_BOX.
  m.def("zentorch_fused_moe(Tensor(a!) output, Tensor input, "
        "Tensor w13, Tensor w2, "
        "Tensor? w13_bias, Tensor? w2_bias, "
        "Tensor topk_weights, Tensor topk_id, "
        "bool skip_weighted, str act, "
        "Tensor? w13_scales=None, Tensor? w2_scales=None, "
        "*, str zentorch_op_name='zentorch::zentorch_fused_moe') -> ()");

  m.def("zentorch_flush_moe_weight_cache() -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(zentorch, CPU, m) {
  m.impl("zentorch_fused_moe", TORCH_BOX(&zentorch::zentorch_fused_moe));
}

// The flush hook is keyed on CompositeExplicitAutograd, not CPU: its schema
// takes no tensors, so the dispatcher computes an empty key set and looks for a
// backend-agnostic kernel. A CPU-only registration raises "no tensor arguments
// to this function ... no fallback function is registered" on call.
STABLE_TORCH_LIBRARY_IMPL(zentorch, CompositeExplicitAutograd, m) {
  m.impl("zentorch_flush_moe_weight_cache",
         TORCH_BOX(&zentorch::zentorch_flush_moe_weight_cache));
}

} // namespace zentorch
