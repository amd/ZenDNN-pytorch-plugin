# Skill: Create conda environment for zentorch

When the user asks to create a conda environment for zentorch development, follow this skill.

---

## Step 1: Check if environment already exists

Ask the user for the desired environment name (default: `agent_env`).

Check if the environment already exists:

```bash
conda env list | grep -w <env_name>
```

If the environment exists, ask the user:
- Continue with the existing environment?
- Or provide a new name for a fresh environment?

If the user chooses to continue with existing, skip to Step 4 (Verify).

---

## Step 2: Create the conda environment

Create a new conda environment with Python 3.10 (recommended for PyTorch 2.10/2.11):

```bash
conda create -n <env_name> python=3.10 -y
```

---

## Step 3: Install PyTorch (CPU version)

Activate the environment and install the latest PyTorch CPU:

```bash
conda run -n <env_name> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

This installs the latest stable PyTorch CPU version (currently supports 2.10.0 and 2.11.0).

---

## Step 4: Verify installation

Verify PyTorch is installed correctly:

```bash
conda run -n <env_name> python -c "import torch; print(f'PyTorch {torch.__version__}')"
```

---

## Step 5: Install build dependencies (optional)

If planning to build zentorch immediately, install build dependencies:

```bash
conda run -n <env_name> pip install cmake ninja setuptools wheel
```

---

## Step 6: Next steps

The environment is ready. To use it:

**Activate the environment:**
```bash
conda activate <env_name>
```

**Build zentorch:**
Follow the `build-from-source.md` skill.

**Run tests:**
Follow the `run-tests.md` skill.

---

## Environment deletion (if needed)

To delete an environment:

```bash
conda env remove -n <env_name>
```
