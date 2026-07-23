---
name: build-zentorch-from-source
description: >-
  Build or rebuild the zentorch wheel from source (cmake auto-fetches ZenDNN),
  install it, and verify the version and build config. Use when the user asks to
  build, rebuild, or compile zentorch from source.
---

# Build zentorch from source

When the user asks to build or rebuild zentorch from source, follow this skill.

**Agent action:** Run this first (foreground, 600000ms timeout):

```bash
.claude/skills/build-zentorch-from-source/scripts/build.sh
```

If the script succeeds, stop — do not run the manual steps below. The script
prints the version and config string on completion.

**Authoritative reference:** [README.md section 2.2 (From Source)](../../../README.md)

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
.claude/skills/build-zentorch-from-source/scripts/build.sh
```

Run in the foreground with a long timeout (600000ms). ZenDNN is fetched
automatically by cmake — no local ZenDNN checkout needed. The script builds the
current checkout; run `git pull --ff-only` first if you want the latest code.

---

## Manual fallback

Use these steps only if the script fails or the user requests a manual build.

### 1. Uninstall existing zentorch

```bash
pip uninstall zentorch -y
```

### 2. Pull latest code

```bash
git pull --ff-only
```

### 3. Build

```bash
python setup.py bdist_wheel
```

> For RHEL/Fedora/AlmaLinux/CentOS, set first: `export ZENDNNL_MANYLINUX_BUILD=1`

**IMPORTANT:** Run in the foreground (NOT in background) with a long timeout (600000ms).

### 4. Install the wheel

```bash
pip install dist/zentorch-*.whl
```

### 4a. Reinstall pinned PyTorch CPU (if needed)

The wheel install may pull a CUDA build of torch. Reinstall the CPU build you
were using (see the version matrix in the `setup-env` skill):

```bash
pip install torch==<version> --index-url https://download.pytorch.org/whl/cpu --force-reinstall --no-deps
```

### 5. Verify

```bash
python -c 'import zentorch; print(zentorch.__version__); print(*zentorch.__config__.split("\n"), sep="\n")'
```

### Build cleanup

```bash
python setup.py clean --all
```

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
