---
name: create-env
description: >-
  Prepare a Python environment for zentorch by installing the pinned PyTorch CPU
  build for the current branch. Use when the user asks to create or prepare an
  environment, or install the correct PyTorch for zentorch.
---

# Prepare a Python environment for zentorch

When the user asks to create or prepare a Python environment for zentorch,
follow this skill.

**Agent action:** After confirming an active environment (see Environment
below), run:

```bash
.claude/skills/create-env/scripts/install_pytorch.sh
```

Use the manual steps below only if the script fails.

**Authoritative reference:** [README.md section 2.2.2.1](../../../README.md)
for environment creation and Python version guidance.

---

## Environment

All zentorch skills share one environment convention:

- Use a single activated, non-`base` Python environment for the whole workflow
  (create-env → build → test → lint). Do not switch environments between skills.
- You choose the environment name; skills never assume or create a fixed one.
  See [README.md section 2.2.2.1](../../../README.md) to create one (venv,
  virtualenv, conda, or another tool of your choice).
- Confirm what is active before running anything:

```bash
echo "${VIRTUAL_ENV:-${CONDA_DEFAULT_ENV:-none}}"
```

If this prints `none`, create and activate a dedicated environment first. Do not
use `base`. The bundled scripts enforce this automatically.

---

## Install pinned PyTorch (CPU)

Install the pinned CPU build for the current branch. **Do not install the latest
PyTorch.**

Preferred — run the script (validates or installs the pinned version):

```bash
.claude/skills/create-env/scripts/install_pytorch.sh
```

Manual equivalent — pick the version from the matrix for your branch:

```bash
git branch --show-current
pip install torch==<version> --index-url https://download.pytorch.org/whl/cpu
```

| Branch          | Primary PyTorch (recommended) | Alternates             |
|-----------------|-------------------------------|------------------------|
| `main`/`master` | 2.13.0                        | 2.12.1, 2.12.0, 2.11.0 |
| `r5.2`          | 2.10.0                        | 2.9.1                  |

> See [README.md](../../../README.md) for the authoritative PyTorch/Python
> compatibility matrix. Use Python 3.10 by default; for a different Python
> version, choose one supported by your branch's PyTorch release per the
> [PyTorch Release Compatibility Matrix](https://github.com/pytorch/pytorch/blob/main/RELEASE.md#release-compatibility-matrix).

---

## Verify PyTorch

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}')"
```

---

## Next steps

- **Full fresh setup:** follow the `setup-env` skill.
- **Build only:** follow the `build-zentorch-from-source` skill.
- **Run tests:** follow the `run-tests` skill.
