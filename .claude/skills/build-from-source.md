# Skill: Build zentorch from source

When the user asks to rebuild or build zentorch from source, follow this skill.

---

## Step 0: Check for existing agent_env

Check if the `agent_env` conda environment exists:

```bash
conda env list | grep -w agent_env
```

If `agent_env` exists, ask the user:
- Continue with the existing `agent_env` environment?
- Or create a new environment? (if so, follow `create-env.md` skill first)

If the user chooses to continue with existing `agent_env`, proceed to Step 1.
If no `agent_env` exists, follow the `create-env.md` skill first to create it.

---

## Step 1: Auto-detect developer vs end user

Run this command from the repo root:

```bash
git remote get-url origin
```

| Origin URL contains        | Role      | Build path to follow        |
|----------------------------|-----------|-----------------------------|
| `AMD-Zenai`               | Developer | **Developer build** (below) |
| `amd/ZenDNN-pytorch-plugin` | End user  | **End-user build** (below)  |

If the remote doesn't match either pattern, ask the user which path to follow.

---

## Developer build (internal repo + local ZenDNN)

Developers use a local ZenDNN checkout as a sibling directory. The cmake build
copies it into `third_party/` instead of fetching from GitHub.

### 1. Uninstall existing zentorch (if continuing with existing env)

If using an existing `agent_env`, uninstall zentorch first:

```bash
conda run -n agent_env pip uninstall zentorch -y
```

### 2. Pull latest code

Update both ZenDNN_PyTorch_Plugin and ZenDNN to latest:

```bash
git pull
```

Check whether `../ZenDNN` exists:

```bash
ls ../ZenDNN
```

If it does NOT exist, clone it:

```bash
git clone https://github.com/amd/ZenDNN.git ../ZenDNN
```

If it exists, pull latest:

```bash
cd ../ZenDNN && git pull && cd -
```

### 3. Build with local ZenDNN

```bash
conda run -n agent_env bash -c "export ZENTORCH_USE_LOCAL_ZENDNN=1 && python setup.py bdist_wheel"
```

> For RHEL/Fedora/AlmaLinux/CentOS, also set: `export ZENDNNL_MANYLINUX_BUILD=1`

**IMPORTANT**: Run this in foreground (NOT in background) with a long timeout (600000ms).

### 4. Install the wheel

```bash
conda run -n agent_env pip install dist/zentorch-*.whl
```

### 4a. Reinstall PyTorch CPU (if needed)

The wheel installation may switch PyTorch from CPU to CUDA version. Reinstall CPU version:

```bash
conda run -n agent_env pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --force-reinstall --no-deps
```

### 5. Verify

```bash
conda run -n agent_env python -c "import zentorch; print(zentorch.__version__)"
```

### Build cleanup

```bash
conda run -n agent_env python setup.py clean --all
```

---

## End-user build (public repo)

ZenDNN is fetched automatically by cmake — no local ZenDNN checkout needed.

### 1. Uninstall existing zentorch (if continuing with existing env)

If using an existing `agent_env`, uninstall zentorch first:

```bash
conda run -n agent_env pip uninstall zentorch -y
```

### 2. Pull latest code

```bash
git pull
```

### 3. Build

```bash
conda run -n agent_env python setup.py bdist_wheel
```

> For RHEL/Fedora/AlmaLinux/CentOS, set first: `export ZENDNNL_MANYLINUX_BUILD=1`

**IMPORTANT**: Run this in foreground (NOT in background) with a long timeout (600000ms).

### 4. Install the wheel

```bash
conda run -n agent_env pip install dist/zentorch-*.whl
```

### 4a. Reinstall PyTorch CPU (if needed)

The wheel installation may switch PyTorch from CPU to CUDA version. Reinstall CPU version:

```bash
conda run -n agent_env pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --force-reinstall --no-deps
```

### 5. Verify

```bash
conda run -n agent_env python -c "import zentorch; print(zentorch.__version__)"
```

---

## Troubleshooting

### GLIBCXX version error

If you see `ImportError: libstdc++.so.6: version 'GLIBCXX_x.y.zz' not found`:

```bash
export LD_PRELOAD=$(conda info --base)/envs/agent_env/lib/libstdc++.so.6:$LD_PRELOAD
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
