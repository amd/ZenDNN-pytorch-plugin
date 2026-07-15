# Skill: Set up environment and build zentorch

When the user asks to set up the environment, get started, or do a fresh setup
and build, follow this skill.

**Agent action:** Run this first (foreground, 600000ms timeout):

```bash
scripts/setup.sh
```

If the script succeeds, stop — do not run manual steps below. Use manual steps
only if the script fails or the user requests a step-by-step manual setup.

**Authoritative reference:** [README.md §2 Installation](../../README.md#2-installation)
for environment creation, PyTorch versions, and build steps.

---

## Quick path (preferred)

If the user already has an activated Python environment:

```bash
scripts/setup.sh
```

Run in foreground with a long timeout (600000ms). The script validates PyTorch,
installs dependencies, builds, and verifies zentorch (version + config string).

---

## Manual fallback

## Step 1: Confirm active Python environment

Ask the user which environment to use. Do **not** assume a fixed environment
name. Check what is currently active:

```bash
echo "${VIRTUAL_ENV:-${CONDA_DEFAULT_ENV:-none}}"
```

If nothing is active, direct the user to README.md §2.2.2.1 to create and
activate a dedicated environment before continuing.

---

## Step 2: Validate or reinstall PyTorch (CPU)

**Always run this step**, including when reusing an existing environment. An
environment previously used on a different branch may have an incompatible
PyTorch version.

Preferred:

```bash
scripts/install_pytorch.sh
```

This installs or validates the pinned PyTorch version for the current branch.
Use `--force` to reinstall unconditionally:

```bash
scripts/install_pytorch.sh --force
```

| Branch          | Primary PyTorch | Alternate              |
|-----------------|-----------------|------------------------|
| `main`/`master` | 2.13.0          | 2.12.1, 2.12.0, 2.11.0 |
| `r5.2`          | 2.10.0          | 2.9.1                  |

> Use Python 3.10 by default (see README). Choose a Python version supported by
> your branch's PyTorch release per the [PyTorch Release Compatibility Matrix](https://github.com/pytorch/pytorch/blob/main/RELEASE.md#release-compatibility-matrix).

---

## Step 3: Uninstall existing zentorch

```bash
pip uninstall zentorch -y
```

---

## Step 4: Install build dependencies

```bash
pip install -r requirements.txt
```

---

## Step 5: Build zentorch

ZenDNN is fetched automatically by cmake — no local ZenDNN checkout needed.

```bash
python setup.py bdist_wheel
```

> For RHEL/Fedora/AlmaLinux/CentOS, also set: `export ZENDNNL_MANYLINUX_BUILD=1`

**IMPORTANT**: Run the build in foreground (NOT in background) with a long
timeout (600000ms).

---

## Step 6: Install the wheel

```bash
pip install dist/zentorch-*.whl
```

### Step 6a: Reinstall pinned PyTorch CPU (if needed)

The wheel may switch PyTorch to a CUDA build. Reinstall the pinned CPU version:

```bash
pip install torch==<pinned_version> --index-url https://download.pytorch.org/whl/cpu --force-reinstall --no-deps
```

Use the same `<pinned_version>` from Step 2.

---

## Step 7: Verify

```bash
scripts/verify.sh
```

Or manually:

```bash
python -c 'import zentorch; print(zentorch.__version__); print(*zentorch.__config__.split("\n"), sep="\n")'
```

---

## What's next

- To run tests: `scripts/test.sh` or follow `run-tests.md`
- To rebuild after code changes: `scripts/build.sh` or
  follow `build-from-source.md`
- To clean the build: `python setup.py clean --all`
