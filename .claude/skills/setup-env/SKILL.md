---
name: setup-env
description: >-
  Prepare the Python environment (install the pinned PyTorch CPU build) and/or
  do a fresh end-to-end zentorch setup: validate PyTorch, install dependencies,
  build, install, and verify zentorch. Use when the user asks to create or
  prepare an environment, install PyTorch for zentorch, get started, set up from
  scratch, or do a full fresh setup and build.
---

# Set up the environment and build zentorch

When the user asks to prepare a Python environment, install PyTorch for
zentorch, get started, set up from scratch, or do a fresh setup and build,
follow this skill.

**Agent action:** Run this first (foreground, 600000ms timeout):

```bash
.claude/skills/setup-env/scripts/setup.sh
```

If the script succeeds, stop — do not run the manual steps below. Use the manual
steps only if the script fails or the user requests a step-by-step setup.

**Authoritative reference:** [README.md section 2 (Installation)](../../../README.md)
for environment creation, PyTorch versions, and build steps.

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

If this prints `none` or `base`, create and activate a dedicated environment
first (see [README.md section 2.2.2.1](../../../README.md)), then install
PyTorch as in the steps below. The bundled scripts enforce this automatically.

---

## Quick path (preferred)

With an activated environment:

```bash
.claude/skills/setup-env/scripts/setup.sh
```

Run in the foreground with a long timeout (600000ms). The script validates
PyTorch, installs dependencies, builds, installs, and verifies zentorch
(version + config string). Validation accepts only branch-supported CPU builds
while preserving supported alternates. The wheel is built in an isolated
directory so a stale `dist/` wheel cannot be installed.

---

## Prepare the environment only (no build)

If the user only wants to prepare the environment — install the pinned PyTorch
CPU build without building zentorch — run:

```bash
.claude/skills/setup-env/scripts/install_pytorch.sh          # validate/install
.claude/skills/setup-env/scripts/install_pytorch.sh --force  # reinstall unconditionally
```

Then verify PyTorch:

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}')"
```

See the PyTorch version matrix in Step 1 below.

---

## Manual fallback

### Step 1: Validate or reinstall PyTorch (CPU)

**Always run this step**, including when reusing an existing environment — an
environment previously used on a different branch may have an incompatible
PyTorch version or a CUDA/ROCm build. Supported alternate CPU versions are
preserved.

```bash
.claude/skills/setup-env/scripts/install_pytorch.sh          # validate/install
.claude/skills/setup-env/scripts/install_pytorch.sh --force  # reinstall unconditionally
```

| Branch          | Primary PyTorch (recommended) | Alternates             |
|-----------------|-------------------------------|------------------------|
| `main`/`master` | 2.13.0                        | 2.12.1, 2.12.0, 2.11.0 |
| `r5.2`          | 2.10.0                        | 2.9.1                  |

> See [README.md](../../../README.md) for the authoritative PyTorch/Python
> compatibility matrix. Use Python 3.10 by default.

### Step 2: Uninstall existing zentorch

```bash
python -m pip uninstall zentorch -y
```

### Step 3: Install build dependencies

```bash
python -m pip install -r requirements.txt
```

### Step 4: Build zentorch

ZenDNN is fetched automatically by cmake — no local ZenDNN checkout needed.

```bash
wheel_build_dir="$(mktemp -d)"
python setup.py bdist_wheel --dist-dir "${wheel_build_dir}"
wheel="$(find "${wheel_build_dir}" -maxdepth 1 -name '*.whl' -print -quit)"
test -n "${wheel}"
```

> For RHEL/Fedora/AlmaLinux/CentOS, also set: `export ZENDNNL_MANYLINUX_BUILD=1`

**IMPORTANT:** Run the build in the foreground (NOT in background) with a long
timeout (600000ms).

### Step 5: Install the wheel

```bash
python -m pip install "${wheel}"
rm -rf "${wheel_build_dir}"
```

The wheel install may switch PyTorch to a CUDA build. Reinstall the pinned CPU
version (same version from Step 1):

```bash
python -m pip install torch==<version> --index-url https://download.pytorch.org/whl/cpu --force-reinstall
```

### Step 6: Verify

```bash
python -c 'import zentorch; print(zentorch.__version__); print(*zentorch.__config__.split("\n"), sep="\n")'
```

---

## What's next

- To run tests: follow the `run-tests` skill.
- To rebuild after code changes: follow the `build-zentorch-from-source` skill.
- To clean generated build outputs:
  `.claude/skills/build-zentorch-from-source/scripts/clean.sh`
