# AGENTS.md — zentorch (ZenDNN PyTorch Plugin)

This is the primary agent guide for the repository. It tracks the available
skills under `.claude/skills/` and the core project reference.

## Working preferences

**Autonomy**: Work independently and execute commands without asking for
permission each time. Only ask the user when:
- There's an actual problem or error that blocks progress
- A critical decision needs user input (e.g., choosing between multiple valid approaches)
- An action is destructive and irreversible (e.g., force push, deleting branches)

For routine operations (building, installing, running tests, reading files,
etc.), proceed directly without requesting permission.

**Skill-first workflows**: When a task matches a skill under `.claude/skills/`,
read that skill's `SKILL.md` first, then run its bundled `scripts/*.sh` before
any manual steps. Only fall back to manual commands if the script fails or the
user asks for a manual path.

**Environment convention (shared by all skills)**: Use a single activated,
non-`base` Python environment for the whole workflow (create-env → build → test
→ lint). The user chooses the environment name; skills never assume or create a
fixed one (see [README.md section 2.2.2.1](README.md)). Every skill checks the
active environment the same way and reuses it.

## Skills

All skills live under `.claude/skills/<name>/SKILL.md`, each with an entry script
under its own `scripts/` that sources the shared `scripts/common.sh` helper (one
source of truth):

| Skill | Purpose | Run first |
|-------|---------|-----------|
| [`create-env`](.claude/skills/create-env/SKILL.md) | Prepare a Python env; install pinned PyTorch CPU for the branch | `.claude/skills/create-env/scripts/install_pytorch.sh` |
| [`setup-env`](.claude/skills/setup-env/SKILL.md) | Fresh end-to-end setup: validate PyTorch, install deps, build, install, verify | `.claude/skills/setup-env/scripts/setup.sh` |
| [`build-zentorch-from-source`](.claude/skills/build-zentorch-from-source/SKILL.md) | Build/rebuild zentorch from source and verify | `.claude/skills/build-zentorch-from-source/scripts/build.sh` |
| [`run-tests`](.claude/skills/run-tests/SKILL.md) | Run tests (unit, op, model, misc, export, vLLM, LLM, pre-trained) | `.claude/skills/run-tests/scripts/test.sh [scope]` |
| [`lint`](.claude/skills/lint/SKILL.md) | Lint Python (flake8), C++ (git clang-format), shell (shellcheck) | `.claude/skills/lint/scripts/lint.sh <python\|cpp\|shell>` |

## Project overview

zentorch is a PyTorch C++ extension that accelerates inference on AMD EPYC CPUs.
It registers a custom `torch.compile` backend called `zentorch` that applies
ZenDNN graph optimizations (pattern fusion, op replacement, embedding/matmul
kernels) on the ATen IR produced by AOTAutograd.

Public repo: `https://github.com/amd/ZenDNN-pytorch-plugin.git`

## Repository layout

```
setup.py                  # Wheel packaging + CppExtension build entry point
CMakeLists.txt            # Top-level cmake; builds libzentorch.so
cmake/modules/            # ZenDNN fetch/build, dependency wiring
src/cpu/cpp/              # C++ operator bindings and integration code
src/cpu/python/zentorch/  # Python package (backend, llm, vllm plugin)
test/                     # All tests (unittests, llm_tests, pre_trained_model_tests)
.claude/skills/           # Agent skills (each: SKILL.md + an entry script under scripts/)
scripts/                  # Shared skill helper (common.sh) + benchmark env-setup scripts
benchmark/                # Benchmark configs (BERT, DLRM-v2, etc.)
third_party/              # Auto-populated at build time (ZenDNN)
```

## Branches

- **main**/**master** — latest development (supports PyTorch 2.13.0 (recommended), 2.12.1, 2.12.0, and 2.11.0)
- **r5.2** — stable release (supports PyTorch 2.10.0 and 2.9.1)

See [README.md](README.md) for the authoritative PyTorch/Python compatibility matrix.

## Build system

Two-phase build:

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
| `ZENDNNL_MANYLINUX_BUILD=1`   | Required for RHEL/Fedora/AlmaLinux/CentOS builds     |
| `DEBUG=1`                     | Debug build (disables -O2, sets cmake Debug)         |
| `ZENTORCH_VLLM_PLUGIN_BUILD`  | Set to `0` to skip building vLLM plugin (default: 1) |

## Testing

Before running tests, disable ZenDNN caching:

```bash
export ZENDNNL_MATMUL_WEIGHT_CACHE=0
export ZENDNNL_ZP_COMP_CACHE=0
```

Install test deps: `python test/install_requirements.py`

Prefer the `run-tests` skill (`.claude/skills/run-tests/scripts/test.sh [scope]`).
Direct commands:

| Scope             | Command                                                  |
|-------------------|----------------------------------------------------------|
| All tests         | `python -m unittest discover -s ./test`                  |
| Unit tests only   | `python -m unittest discover -s ./test/unittests`        |
| Pre-trained tests | `python -m unittest discover -s ./test/pre_trained_model_tests` |
| Single file       | `python -m unittest test/unittests/op_tests/test_bmm.py` |
| By name pattern   | `python -m unittest discover -s ./test/unittests -k "woq"` |
| By file pattern   | `python -m unittest discover -s ./test/unittests -p "test_mm*"` |

## Coding conventions

- C++20 standard, compiled with `-Wall -Werror`
- Python package lives under `src/cpu/python/zentorch/`
- Ops are registered via `TORCH_LIBRARY` / `TORCH_LIBRARY_IMPL` macros in `Bindings.cpp`
- Linting: `.flake8` config in repo root; `linter/py_cpp_linter.sh` for CI checks
