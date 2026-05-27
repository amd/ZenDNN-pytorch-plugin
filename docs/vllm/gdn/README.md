# Gated Delta Net (GDN) on CPU

Hand-written C++ kernels and a vLLM `forward_cpu` override that let
GatedDeltaNet models (Qwen3-Next, Qwen3.5, Qwen3-Next-A22B, …) run
end-to-end on CPU through vLLM with parity-level accuracy and a
measurable throughput improvement over upstream's reference CPU path.

## Documentation layout

| Path | Contents |
|---|---|
| [`overview.md`](overview.md) | Top-down call graph from `vllm bench throughput` to the GDN cpp ops |
| [`kernels/<name>.md`](kernels/) | Per-kernel math, dtype contract, parallelisation plan |

## Source layout (runtime)

| Path | Contents |
|---|---|
| `src/cpu/cpp/GDN_ops.cpp` | Single bindings file: forward decls + one `TORCH_LIBRARY_FRAGMENT` with all 14 `m.def` schemas + one `TORCH_LIBRARY_IMPL` with all 14 `m.impl` entries |
| `src/cpu/cpp/kernels/layers/gdn/<OpName>.cpp` | 14 per-op kernel implementations (the C++ math); compiled into `libCPUkernels.a` with AVX-512 flags via `cmake/modules/FindCPUkernels.cmake` |
| `src/cpu/python/zentorch/_meta_registrations.py` | `@register_meta` and `make_fallback` entries for every GDN op (Dynamo / FakeTensorMode / Inductor lowering) |
| `src/cpu/python/zentorch/vllm/layers/gdn/forward.py` | `forward_cpu_zen` — the override that stitches `torch.ops.zentorch.gdn_*` calls together to implement `GatedDeltaNetAttention.forward_cpu` |
| `src/cpu/python/zentorch/vllm/layers/gdn/patch.py` | Installs the `forward_cpu` override on `GatedDeltaNetAttention` at vLLM startup (deferred via a `sys.meta_path` import hook) |
| `src/cpu/python/zentorch/vllm/__init__.py` | Registers `GatedDeltaNetPatch` with the vLLM plugin manager |

## Source layout (tests, out of the wheel)

| Path | Contents |
|---|---|
| `test/unittests/op_tests/layers/gdn/test_<name>.py` | Per-op unittest files; each exercises one `torch.ops.zentorch.gdn_<name>` op against an independent oracle defined at module scope in the same file |
| `test/unittests/op_tests/layers/gdn/helpers/shapes.py` | `Qwen35_4B_GDN` constants, `GDNShape` dataclass, `common_seq_lens()` |
| `test/unittests/op_tests/layers/gdn/helpers/varlen.py` | `prepare_chunk_indices` / `prepare_chunk_offsets` + `NULL_BLOCK_ID` / `PAD_SLOT_ID` sentinels |
| `test/zentorch_test_utils.py::default_tolerance(*dtypes)` | Project-wide per-dtype atol/rtol table (shared with every op test, not GDN-specific) |

## Invocation flow

```
vLLM startup (in every process: driver / EngineCore / Worker)
└── plugin loader fires `vllm.general_plugins` entry points
    └── zentorch.vllm:register
        └── manager.apply_all()
            └── GatedDeltaNetPatch.apply()
                └── zentorch.vllm.layers.gdn.patch.apply_deferred()
                    │  (defers until gdn_linear_attn module is imported,
                    │   then installs the class-method override:)
                    └── zentorch.vllm.layers.gdn.patch.apply()
                        └── GatedDeltaNetAttention.forward_cpu  ←  forward_cpu_zen
                              (the entire CPU heavy-lifting pipeline lives in
                               forward.py and dispatches to torch.ops.zentorch.gdn_*)
```

At inference time, vLLM calls `GatedDeltaNetAttention.forward(...)` which
on CPU now routes through `forward_cpu_zen`, which orchestrates the C++
ops registered by `GDN_ops.cpp`.

## Running the tests

After building and installing zentorch (see top-level `README.md` for the
build instructions), from the repo root:

```bash
# Full GDN suite
python -m unittest discover -s test/unittests/op_tests/layers/gdn -t . -v

# A single op's tests
python -m unittest test.unittests.op_tests.layers.gdn.test_chunk_local_cumsum -v

# A single test method
python -m unittest \
    test.unittests.op_tests.layers.gdn.test_l2norm_fwd.Test_GDN_L2NormFwd.test_unit_norm_property \
    -v
```

The `-t .` (top-level dir) in the `discover` form is required so the
absolute imports (`from test.unittests.op_tests.layers.gdn...`) resolve.

Every per-op test compares the C++ op's output against an independent
oracle defined inline at the top of its test file, using
project-default tolerances from
`test/zentorch_test_utils.py::default_tolerance(*dtypes)`.
