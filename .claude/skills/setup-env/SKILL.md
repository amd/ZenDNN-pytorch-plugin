---
name: setup-env
description: >-
  Do a fresh end-to-end zentorch setup: validate PyTorch, install dependencies,
  build, install, and verify zentorch. Use when the user asks to set up from
  scratch, get started, or do a full fresh setup and build.
---

# Fresh setup and build of zentorch

When the user asks to set up the environment, get started, or do a fresh setup
and build, follow this skill.

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
  (create-env → build → test → lint). Do not switch environments between skills.
- You choose the environment name; skills never assume or create a fixed one.
  See [README.md section 2.2.2.1](../../../README.md) to create one.
- Confirm what is active before running anything:

```bash
echo "${VIRTUAL_ENV:-${CONDA_DEFAULT_ENV:-none}}"
```

If this prints `none` or `base`, activate a dedicated environment first (follow
the `create-env` skill). The bundled scripts enforce this automatically.

---

## Quick path (preferred)

With an activated environment:

```bash
.claude/skills/setup-env/scripts/setup.sh
```

Run in the foreground with a long timeout (600000ms). The script validates
PyTorch, installs dependencies, builds, installs, and verifies zentorch
(version + config string).

---

## Manual fallback

### Step 1: Validate or reinstall PyTorch (CPU)

**Always run this step**, including when reusing an existing environment — an
environment previously used on a different branch may have an incompatible
PyTorch version.

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
pip uninstall zentorch -y
```

### Step 3: Install build dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Build zentorch

ZenDNN is fetched automatically by cmake — no local ZenDNN checkout needed.

```bash
python setup.py bdist_wheel
```

> For RHEL/Fedora/AlmaLinux/CentOS, also set: `export ZENDNNL_MANYLINUX_BUILD=1`

**IMPORTANT:** Run the build in the foreground (NOT in background) with a long
timeout (600000ms).

### Step 5: Install the wheel

```bash
pip install dist/zentorch-*.whl
```

The wheel install may switch PyTorch to a CUDA build. Reinstall the pinned CPU
version (same version from Step 1):

```bash
pip install torch==<version> --index-url https://download.pytorch.org/whl/cpu --force-reinstall --no-deps
```

### Step 6: Verify

```bash
python -c 'import zentorch; print(zentorch.__version__); print(*zentorch.__config__.split("\n"), sep="\n")'
```

---

## What's next

- To run tests: follow the `run-tests` skill.
- To rebuild after code changes: follow the `build-zentorch-from-source` skill.
- To clean the build: `python setup.py clean --all`
