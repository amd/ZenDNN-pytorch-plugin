# CLAUDE.md — zentorch (ZenDNN PyTorch Plugin)

## Working preferences

**Autonomy**: Work independently and execute commands without asking for permission each time. Only ask the user when:
- There's an actual problem or error that blocks progress
- A critical decision needs user input (e.g., choosing between multiple valid approaches)
- An action is destructive and irreversible (e.g., force push, deleting branches)

For routine operations (building, installing, running tests, reading files, etc.), proceed directly without requesting permission.

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
scripts/                  # Environment setup helpers
benchmark/                # Benchmark configs (BERT, DLRM-v2, etc.)
third_party/              # Auto-populated at build time (ZenDNN)
```

## Repos

| Audience   | Repo                                                         |
|------------|--------------------------------------------------------------|
| Developer  | `git@github.com:AMD-Zenai/ZenDNN_PyTorch_Plugin.git` (internal) |
| End user   | `https://github.com/amd/ZenDNN-pytorch-plugin.git` (public)    |

## Branches

- **main** — latest development (supports PyTorch 2.11.0 and 2.10.0)
- **r5.2** — stable release (supports PyTorch 2.10.0 and 2.9.1)
- **master** (public repo) — weekly development releases

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

## Coding conventions

- C++17 standard, compiled with `-Wall -Werror`
- Python package lives under `src/cpu/python/zentorch/`
- Ops are registered via `TORCH_LIBRARY` / `TORCH_LIBRARY_IMPL` macros in `Bindings.cpp`
- Linting: `.flake8` config in repo root; `linter/py_cpp_linter.sh` for CI checks

## Skills

See `.claude/skills/` for step-by-step guides:
- `create-env.md` — Create a conda environment for zentorch development. Default
  environment name is `agent_env`. Installs PyTorch, torchvision, and torchaudio.
  Use this when creating a new conda environment.
- `build-from-source.md` — Build/rebuild zentorch from source.
  - First checks if `agent_env` exists and asks user to continue with existing or create new
  - If continuing with existing env: uninstalls zentorch, pulls latest code (git pull)
  - For developers: also pulls latest ZenDNN code
  - Auto-detects developer vs end-user by inspecting `git remote get-url origin`:
    - `AMD-Zenai` in origin → **developer** path (local ZenDNN + `ZENTORCH_USE_LOCAL_ZENDNN=1`)
    - `amd/ZenDNN-pytorch-plugin` in origin → **end-user** path (cmake auto-fetches ZenDNN)
- `setup-env.md` — Full fresh setup: create conda env, install PyTorch (CPU),
  install deps, build and install zentorch. Start here for first-time setup.
- `run-tests.md` — Test dependency setup, env vars, and commands for running
  unit tests, LLM tests, pre-trained model tests, or filtered subsets
