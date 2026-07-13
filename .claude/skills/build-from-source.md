# Skill: Build zentorch from source

When the user asks to rebuild or build zentorch from source, follow this skill.

**Agent action:** Run this first (foreground, 600000ms timeout):

```bash
scripts/build.sh
```

If the script succeeds, stop — do not run manual steps below. The script prints
version and config string on completion.

**Authoritative reference:** [README.md §2.2 From Source](../../README.md#22-from-source)

---

## Quick path (preferred)

With an activated Python environment:

```bash
scripts/build.sh
```

Run in foreground with a long timeout (600000ms).

---

## Manual fallback

## Step 0: Confirm active Python environment

Ask the user which environment to use. Do **not** lock to a specific name.

```bash
echo "${VIRTUAL_ENV:-${CONDA_DEFAULT_ENV:-none}}"
```

If nothing is active, follow `create-env.md` first (refer to README.md §2.2.2.1
for environment creation).

---

## Step 1: Auto-detect developer vs end user

```bash
git remote get-url origin
```

| Origin URL contains           | Role      | Build path        |
|-------------------------------|-----------|-------------------|
| `AMD-Zenai`                   | Developer | Local ZenDNN      |
| `amd/ZenDNN-pytorch-plugin`   | End user  | Auto-fetch ZenDNN |

If the remote doesn't match either pattern, ask the user which path to follow.

---

## Developer build (internal repo + local ZenDNN)

### 1. Uninstall existing zentorch

```bash
pip uninstall zentorch -y
```

### 2. Pull latest code

```bash
git pull
ls ../ZenDNN || git clone https://github.com/amd/ZenDNN.git ../ZenDNN
cd ../ZenDNN && git pull && cd -
```

### 3. Build with local ZenDNN

```bash
export ZENTORCH_USE_LOCAL_ZENDNN=1
python setup.py bdist_wheel
```

> For RHEL/Fedora/AlmaLinux/CentOS, also set: `export ZENDNNL_MANYLINUX_BUILD=1`

**IMPORTANT**: Run in foreground (NOT in background) with a long timeout (600000ms).

### 4. Install the wheel

```bash
pip install dist/zentorch-*.whl
```

### 4a. Reinstall pinned PyTorch CPU (if needed)

```bash
pip install torch==<pinned_version> --index-url https://download.pytorch.org/whl/cpu --force-reinstall --no-deps
```

Use the version from `scripts/install_pytorch.sh` / current branch table
in `setup-env.md`.

### 5. Verify

```bash
scripts/verify.sh
```

Prints both version and build config string:

```bash
python -c 'import zentorch; print(zentorch.__version__); print(*zentorch.__config__.split("\n"), sep="\n")'
```

### Build cleanup

```bash
python setup.py clean --all
```

---

## End-user build (public repo)

ZenDNN is fetched automatically by cmake — no local ZenDNN checkout needed.

### 1. Uninstall existing zentorch

```bash
pip uninstall zentorch -y
```

### 2. Pull latest code

```bash
git pull
```

### 3. Build

```bash
python setup.py bdist_wheel
```

> For RHEL/Fedora/AlmaLinux/CentOS, set first: `export ZENDNNL_MANYLINUX_BUILD=1`

**IMPORTANT**: Run in foreground (NOT in background) with a long timeout (600000ms).

### 4. Install the wheel

```bash
pip install dist/zentorch-*.whl
```

### 4a. Reinstall pinned PyTorch CPU (if needed)

```bash
pip install torch==<pinned_version> --index-url https://download.pytorch.org/whl/cpu --force-reinstall --no-deps
```

### 5. Verify

```bash
scripts/verify.sh
```

---

## Troubleshooting

### GLIBCXX version error

See README.md §2.1 notes. Typical fix:

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
