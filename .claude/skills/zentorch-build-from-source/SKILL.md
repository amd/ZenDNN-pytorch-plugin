---
name: zentorch-build-from-source
description: >-
  Build or rebuild the zentorch wheel from source (cmake auto-fetches ZenDNN),
  install it, and verify the version and build config. Use when the user asks to
  build, rebuild, or compile zentorch from source.
---

<!-- Copyright &copy; 2026 Advanced Micro Devices, Inc. All rights reserved. -->

# Build zentorch from source

When the user asks to build or rebuild zentorch from source, follow this skill.

**Agent action:** Run this first (foreground, 600000ms timeout):

```bash
.claude/skills/zentorch-build-from-source/scripts/build.sh
```

If the script succeeds, stop — do not run the manual steps below. The script
prints the version and config string on completion.

**Authoritative reference:** [README.md section 2.2 (From Source)](../../../README.md)

See [zentorch-build-flow.mmd](zentorch-build-flow.mmd) for the raw Mermaid
build workflow.

---

## Environment

All zentorch skills share one environment convention:

- Use a single activated, non-`base` Python environment for the whole workflow
  (environment setup → build → test → lint). Do not switch environments between skills.
- You choose the environment name; skills never assume or create a fixed one.
  See [README.md section 2.2.2.1](../../../README.md) to create one.
- Confirm what is active before running anything:

```bash
echo "${VIRTUAL_ENV:-${CONDA_DEFAULT_ENV:-none}}"
```

If this prints `none` or `base`, activate a dedicated environment first (follow
the `setup-env` skill). The bundled scripts enforce this automatically and exit
if no non-`base` environment is active.

---

## Quick path (preferred)

```bash
.claude/skills/zentorch-build-from-source/scripts/build.sh
```

Run in the foreground with a long timeout (600000ms). ZenDNN is fetched
automatically by cmake — no local ZenDNN checkout needed. The script builds the
current checkout; run `git pull --ff-only` first if you want the latest code.
Before compiling, it:

- rejects missing and `base` environments;
- installs the repository requirements in the active environment;
- preserves an installed, branch-supported CPU-only PyTorch version (including
  supported alternates), or installs the recommended CPU version otherwise;
- builds into an isolated temporary directory and installs only that wheel, so
  a stale wheel already in `dist/` cannot be selected or overwritten.

---

## Manual fallback

Use these steps only if the script fails or the user requests a manual build.

### 1. Pull latest code

```bash
git pull --ff-only
```

### 2. Install build dependencies

```bash
python -m pip install -r requirements.txt
```

This installs the environment-local CMake and Ninja versions required by the
build instead of relying on potentially missing or outdated system tools.

### 3. Ensure supported CPU-only PyTorch

```bash
.claude/skills/setup-env/scripts/install_pytorch.sh
torch_version="$(python -c "import torch; print(torch.__version__.split('+')[0])")"
```

This preserves a supported CPU alternate. A supported CUDA or ROCm build is
reinstalled as CPU at the same base version; only a missing or unsupported
version falls back to the branch-recommended CPU version.

### 4. Uninstall existing zentorch

```bash
python -m pip uninstall zentorch -y
```

### 5. Build in an isolated directory

```bash
wheel_build_dir="$(mktemp -d)"
python setup.py bdist_wheel --dist-dir "${wheel_build_dir}"
wheel="$(find "${wheel_build_dir}" -maxdepth 1 -name '*.whl' -print -quit)"
test -n "${wheel}"
```

> For RHEL/Fedora/AlmaLinux/CentOS, set first: `export ZENDNNL_MANYLINUX_BUILD=1`

**IMPORTANT:** Run in the foreground (NOT in background) with a long timeout (600000ms).

### 6. Install the wheel

```bash
python -m pip install "${wheel}"
rm -rf "${wheel_build_dir}"
```

### 6a. Restore the selected PyTorch CPU build

The wheel install may pull a CUDA build of torch. Reinstall the CPU build you
validated before building:

```bash
python -m pip install "torch==${torch_version}" --index-url https://download.pytorch.org/whl/cpu --force-reinstall
```

### 7. Verify

```bash
python -c 'import zentorch; print(zentorch.__version__); print(*zentorch.__config__.split("\n"), sep="\n")'
```

### Build cleanup

```bash
.claude/skills/zentorch-build-from-source/scripts/clean.sh
```

This removes only the repository's generated `build/`, `dist/`, and
`src/cpu/python/zentorch.egg-info/` directories. Use this skill-owned cleanup
because `setupext_janitor` 1.1.2 calls a removed setuptools `remove_tree`
argument in current environments, causing `python setup.py clean --all` to
fail.

---

## Troubleshooting

### GLIBCXX version error

See README.md section 2.1 notes. Typical fix:

```bash
export LD_PRELOAD=<path_to_env>/lib/libstdc++.so.6:$LD_PRELOAD
```

### Debug build

```bash
export DEBUG=1
python setup.py bdist_wheel
```

### Skip vLLM plugin

```bash
export ZENTORCH_VLLM_PLUGIN_BUILD=0
python setup.py bdist_wheel
```

### Post-install runtime tuning

`scripts/zentorch_env_setup.sh` is intentionally not sourced by the build
skill. It selects model- and precision-specific runtime settings and may
install jemalloc or LLVM OpenMP packages, so applying it during build would
mutate the environment beyond installation. Source it explicitly before a
workload when those runtime settings are wanted.
