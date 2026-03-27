# Skill: Set up environment and build zentorch

When the user asks to set up the environment, get started, or do a fresh setup
and build, follow this skill. It creates a conda environment, installs PyTorch
(CPU), and builds zentorch from source end-to-end.

---

## Step 0: Auto-detect developer vs end user

Run this command from the repo root:

```bash
git remote get-url origin
```

| Origin URL contains        | Role      |
|----------------------------|-----------|
| `AMD-Zenai`               | Developer |
| `amd/ZenDNN-pytorch-plugin` | End user  |

If the remote doesn't match either pattern, ask the user which role applies.

---

## Step 1: Check for existing agent_env

Check if the `agent_env` conda environment exists:

```bash
conda env list | grep -w agent_env
```

**If `agent_env` exists**, ask the user:
- Continue with the existing `agent_env` environment?
- Or delete and recreate it fresh?

If the user chooses to continue with existing `agent_env`:
1. Uninstall any existing zentorch:
   ```bash
   conda run -n agent_env pip uninstall zentorch -y
   ```
2. Skip to Step 3 (Install build dependencies).

If the user chooses to recreate, delete the old one first:
```bash
conda env remove -n agent_env -y
```

**If `agent_env` does not exist**, create it:

```bash
conda create -n agent_env python=3.10 -y
```

> Python 3.10 – 3.13 are supported. No experimental versions (3.13T/3.14/3.14T).

---

## Step 2: Install PyTorch (CPU)

Detect the current branch to pick the right PyTorch version:

```bash
git branch --show-current
```

**Developer (internal repo):**

| Branch  | Command                                                                                        |
|---------|------------------------------------------------------------------------------------------------|
| `main`  | `conda run -n agent_env pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu` |
| `r5.2`  | `conda run -n agent_env pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu` |

**End user (public repo):**

| Branch   | Command                                                                                        |
|----------|------------------------------------------------------------------------------------------------|
| `master` | `conda run -n agent_env pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu` |
| `r5.2`   | `conda run -n agent_env pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu` |

---

## Step 3: Install build dependencies

```bash
conda run -n agent_env pip install -r requirements.txt
```

---

## Step 4: Developer only — ensure local ZenDNN

Skip this step for end users.

Check whether `../ZenDNN` exists relative to the repo root:

```bash
ls ../ZenDNN
```

If it does NOT exist, clone it:

```bash
git clone https://github.com/amd/ZenDNN.git ../ZenDNN
```

Expected layout:

```
<parent_dir>/
  ZenDNN/                      # ZenDNN checked out here
  ZenDNN_PyTorch_Plugin/       # this repo
```

---

## Step 5: Build zentorch

**Developer:**

```bash
conda run -n agent_env bash -c "export ZENTORCH_USE_LOCAL_ZENDNN=1 && python setup.py bdist_wheel"
```

**End user:**

```bash
conda run -n agent_env python setup.py bdist_wheel
```

> For RHEL/Fedora/AlmaLinux/CentOS, also set: `export ZENDNNL_MANYLINUX_BUILD=1`

**IMPORTANT**: Run the build command in foreground (NOT in background) with a long timeout (600000ms).

---

## Step 6: Install the wheel

```bash
conda run -n agent_env pip install dist/zentorch-*.whl
```

### Step 6a: Reinstall PyTorch CPU (if needed)

The wheel installation may switch PyTorch from CPU to CUDA version. Reinstall CPU version:

```bash
conda run -n agent_env pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --force-reinstall --no-deps
```

---

## Step 7: Verify

```bash
conda run -n agent_env python -c "import zentorch; print(zentorch.__version__)"
```

If this prints the version without errors, setup is complete.

---

## What's next

- To run tests, follow the `run-tests.md` skill.
- To rebuild after code changes, follow the `build-from-source.md` skill
  (the conda env and PyTorch are already in place).
- To clean the build: `conda run -n agent_env python setup.py clean --all`
