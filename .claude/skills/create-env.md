# Skill: Prepare Python environment for zentorch

When the user asks to create or prepare a Python environment for zentorch
development, follow this skill.

**Agent action:** After confirming an active environment (Step 1), run:

```bash
.claude/scripts/agent.sh install-pytorch
```

Use manual steps below only if the script fails.

**Authoritative reference:** [README.md §2.2.2.1](../../README.md#22221-create-conda-environment-for-the-build)
for environment creation and Python version guidance.

---

## Step 1: Confirm an active Python environment

Do **not** assume a fixed environment name. Ask the user which environment they
want to use, or confirm the currently active one:

```bash
echo "${VIRTUAL_ENV:-${CONDA_DEFAULT_ENV:-none}}"
```

If no environment is active, direct the user to README.md §2.2.2.1 to create
and activate a dedicated Python environment (venv, virtualenv, or other tool of
their choice). Do not use the base environment.

---

## Step 2: Install pinned PyTorch (CPU)

Detect the expected PyTorch version from the current branch and repo role, then
install the pinned CPU build. **Do not install the latest PyTorch.**

Preferred — run the script:

```bash
.claude/scripts/agent.sh install-pytorch
```

Manual equivalent:

```bash
git branch --show-current
git remote get-url origin
```

| Role       | Origin contains               | Branch  | Primary PyTorch | Alternate   |
|------------|-------------------------------|---------|-----------------|-------------|
| Developer  | `AMD-Zenai`                   | `main`  | 2.11.0          | 2.10.0      |
| Developer  | `AMD-Zenai`                   | `r5.2`  | 2.10.0          | 2.9.1       |
| End user   | `amd/ZenDNN-pytorch-plugin`   | `main`/`master` | 2.11.0  | 2.10.0      |
| End user   | `amd/ZenDNN-pytorch-plugin`   | `r5.2`  | 2.10.0          | 2.9.1       |

```bash
pip install torch==<version> --index-url https://download.pytorch.org/whl/cpu
```

> Use Python 3.10 by default (see README). If you need a different Python
> version, choose one supported by the PyTorch version for your branch and
> verify against the [PyTorch Release Compatibility Matrix](https://github.com/pytorch/pytorch/blob/main/RELEASE.md#release-compatibility-matrix).

---

## Step 3: Verify PyTorch

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}')"
```

---

## Step 4: Install build dependencies (optional)

If planning to build zentorch immediately:

```bash
pip install -r requirements.txt
```

---

## Step 5: Next steps

- **Full setup:** follow `setup-env.md` or run `.claude/scripts/agent.sh setup`
- **Build only:** follow `build-from-source.md` or run `.claude/scripts/agent.sh build`
- **Run tests:** follow `run-tests.md` or run `.claude/scripts/agent.sh test`
