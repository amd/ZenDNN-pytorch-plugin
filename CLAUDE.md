# CLAUDE.md — zentorch (ZenDNN PyTorch Plugin)

## Working preferences

**Autonomy**: Work independently and execute commands without asking for permission each time. Only ask the user when:
- There's an actual problem or error that blocks progress
- A critical decision needs user input (e.g., choosing between multiple valid approaches)
- An action is destructive and irreversible (e.g., force push, deleting branches)

For routine operations (building, installing, running tests, reading files, etc.), proceed directly without requesting permission.

**Script-first workflows**: When a task matches a skill in `.claude/skills/`, read
that skill file first, then run the mapped `scripts/*.sh` script before any manual
steps. Only fall back to manual commands if the script fails or the user asks
for a manual path.

| User intent | Skill file | Run first |
|-------------|------------|-----------|
| Prepare Python env / install PyTorch | `create-env.md` | `scripts/install_pytorch.sh` |
| Fresh setup and build | `setup-env.md` | `scripts/setup.sh` |
| Rebuild from source | `build-from-source.md` | `scripts/build.sh` |
| Verify install | any build skill | `scripts/verify.sh` |
| Run tests | `run-tests.md` | `scripts/test.sh [scope]` |
| Lint code | `lint.md` | `scripts/lint.sh python`, `cpp`, or `shell` |

Scripts assume an activated Python environment (see
[README.md §2.2.2.1](README.md#22221-create-conda-environment-for-the-build)).

**Environment setup**: Do not prescribe conda commands in skills or scripts.
Refer to README.md for authoritative environment creation, PyTorch version
pinning, and build instructions.

## Project overview

zentorch is a PyTorch C++ extension that accelerates inference on AMD EPYC CPUs.
It registers a custom `torch.compile` backend called `zentorch` that applies
ZenDNN graph optimizations (pattern fusion, op replacement, embedding/matmul
kernels) on the ATen IR produced by AOTAutograd.

## Repository layout

```
setup.py                  # Wheel packaging + CppExtension build entry point
CMakeLists.txt            # Top-level cmake; builds libzentorch.so
cmake/modules/            # ZenDNN fetch/build, dependency wiring
src/cpu/cpp/              # C++ operator bindings and integration code
src/cpu/python/zentorch/  # Python package (backend, llm, vllm plugin)
test/                     # All tests (unittests, llm_tests, pre_trained_model_tests)
scripts/                  # Env setup helpers + dev workflow scripts (setup/build/verify/test/lint) + common.sh
.claude/skills/           # Step-by-step agent skill guides
benchmark/                # Benchmark configs (BERT, DLRM-v2, etc.)
third_party/              # Auto-populated at build time (ZenDNN)
```

## Repos

| Audience   | Repo                                                         |
|------------|--------------------------------------------------------------|
| Developer  | `git@github.com:AMD-Zenai/ZenDNN_PyTorch_Plugin.git` (internal) |
| End user   | `https://github.com/amd/ZenDNN-pytorch-plugin.git` (public)    |

## Branches

- **main** — latest development (supports PyTorch 2.13.0, 2.12.1, 2.12.0, and 2.11.0)
- **r5.2** — stable release (supports PyTorch 2.10.0 and 2.9.1)
- **master** (public repo) — weekly development releases

See README.md for the authoritative PyTorch/Python compatibility matrix.

## Build system

The build is a two-phase process:

1. **CMake phase** (`setup.py` triggers cmake via `CustomBuildExtension`):
   - Fetches or copies ZenDNN into `third_party/ZenDNN`
   - Builds ZenDNN as a static archive (`libzendnnl_archive.a`) with all deps
     (oneDNN, libxsmm, fbgemm, aoclutils, aocl-dlp)
   - Compiles `libzentorch.so` linking against ZenDNN and PyTorch
2. **setuptools phase** (`python setup.py bdist_wheel`):
   - Builds `_C` CppExtension (Bindings.cpp) linking `libzentorch.so`
   - Packages everything into a wheel under `dist/`

### Key environment variables

| Variable                      | Purpose                                              |
|-------------------------------|------------------------------------------------------|
| `ZENTORCH_USE_LOCAL_ZENDNN=1` | Use ZenDNN from `../ZenDNN` instead of git fetch     |
| `ZENDNNL_MANYLINUX_BUILD=1`  | Required for RHEL/Fedora/AlmaLinux/CentOS builds     |
| `DEBUG=1`                     | Debug build (disables -O2, sets cmake Debug)         |
| `ZENTORCH_VLLM_PLUGIN_BUILD` | Set to `0` to skip building vLLM plugin (default: 1) |

## Testing

Before running tests, disable ZenDNN caching:
```
export ZENDNNL_MATMUL_WEIGHT_CACHE=0
export ZENDNNL_ZP_COMP_CACHE=0
```

Install test deps: `python test/install_requirements.py`

| Scope             | Command                                                  |
|-------------------|----------------------------------------------------------|
| All tests         | `python -m unittest discover -s ./test`                  |
| Unit tests only   | `python -m unittest discover -s ./test/unittests`        |
| LLM tests         | `python -m unittest discover -s ./test/llm_tests`        |
| Pre-trained tests | `python -m unittest discover -s ./test/pre_trained_model_tests` |
| Single file       | `python -m unittest test/unittests/op_tests/test_bmm.py` |
| By name pattern   | `python -m unittest discover -s ./test/unittests -k "woq"` |
| By file pattern   | `python -m unittest discover -s ./test/unittests -p "test_mm*"` |

Or use `scripts/test.sh [scope]`.

## Coding conventions

- C++20 standard, compiled with `-Wall -Werror` (PyTorch 2.13's c10 headers require C++20)
- Python package lives under `src/cpu/python/zentorch/`
- Ops are registered via `TORCH_LIBRARY` / `TORCH_LIBRARY_IMPL` macros in `Bindings.cpp`
- Linting: `.flake8` config in repo root; `linter/py_cpp_linter.sh` for CI checks

## Scripts

User-facing workflow scripts under `scripts/` (shared helpers in `scripts/common.sh`):

| Command | Purpose |
|---------|---------|
| `scripts/install_pytorch.sh [--force]` | Install/validate pinned PyTorch CPU |
| `scripts/setup.sh` | Full fresh setup: deps, build, install, verify |
| `scripts/build.sh` | Rebuild zentorch from source |
| `scripts/verify.sh` | Print zentorch version and build config |
| `scripts/test.sh [scope]` | Run tests with required env vars |
| `scripts/lint.sh python` | flake8 Python lint |
| `scripts/lint.sh cpp` | clang-format C++ check |
| `scripts/lint.sh shell` | shellcheck on `.sh` files |

## Skills

See `.claude/skills/` for step-by-step guides:
- `create-env.md` — Prepare a Python environment with pinned PyTorch CPU.
  Refers to README for environment creation. User chooses their own env name.
- `setup-env.md` — Full fresh setup: validate PyTorch, install deps, build and
  install zentorch. Validates/reinstalls PyTorch even when reusing an env.
- `build-from-source.md` — Build/rebuild zentorch from source.
  - Auto-detects developer vs end-user by inspecting `git remote get-url origin`:
    - `AMD-Zenai` in origin → **developer** path (local ZenDNN + `ZENTORCH_USE_LOCAL_ZENDNN=1`)
    - `amd/ZenDNN-pytorch-plugin` in origin → **end-user** path (cmake auto-fetches ZenDNN)
  - Verify step prints both `__version__` and `__config__`.
- `run-tests.md` — Test dependency setup, env vars, and commands for running
  unit tests, LLM tests, pre-trained model tests, or filtered subsets
- `lint.md` — flake8 (Python), clang-format (C++), and shellcheck (shell scripts)
