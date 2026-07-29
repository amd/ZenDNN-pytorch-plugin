---
name: build-vllm-zentorch
description: >-
  Install vLLM plus the zentorch plugin against the ZenDNN backend. vLLM is
  obtained either from a pre-built CPU wheel (pip) or built from source — always
  ask the user which. If the user provides vLLM PR(s), force a source build and
  cherry-pick them. Always confirm the vLLM version with the user and validate it
  against zentorch's supported range (read fresh from the repo, never hardcoded).
  The ZenDNN backend is compiled by zentorch's own cmake from a local ZenDNN
  checkout at third_party/ZenDNN — always build against that local checkout
  (ZENTORCH_USE_LOCAL_ZENDNN=1) unless the user explicitly asks for a
  FetchContent/tagged build.
---

<!-- Copyright &copy; 2026 Advanced Micro Devices, Inc. All rights reserved. -->

# Build vLLM + zentorch (ZenDNN backend)

Backend is **zendnn** (vanilla `vllm-project/vllm`). vLLM comes from **pip
(pre-built CPU wheel)** or **source** — always ask. A vLLM PR forces source +
cherry-pick.

## Sections
- [Flow](#flow)
- [Inputs to confirm (ask the user)](#inputs-to-confirm-ask-the-user)
- [ALWAYS: confirm + validate the vLLM version](#always-confirm--validate-the-vllm-version)
- [vLLM acquisition (pip vs source vs source+cherry-pick)](#vllm-acquisition-pip-vs-source-vs-sourcecherry-pick)
- [Environment prep (native)](#environment-prep-native)
- [vLLM install — branches on acquisition](#vllm-install--branches-on-acquisition)
- [Backend: build against the local ZenDNN at `third_party/ZenDNN`](#backend-build-against-the-local-zendnn-at-third_partyzendnn)
- [Install zentorch](#install-zentorch)
- [Outputs](#outputs)
- [Runtime environment (document for the user)](#runtime-environment-document-for-the-user)
- [Smoke test](#smoke-test)
- [Benchmark validation (confirm the build works)](#benchmark-validation-confirm-the-build-works)
- [Docker](#docker)
- [Gotchas](#gotchas)

## Flow

Stage 0 → 7. vLLM is installed **before** the backend and zentorch, so the
plugin registration binds against an already-present vLLM.

See [vllm-zentorch-build-flow.mmd](vllm-zentorch-build-flow.mmd) for the raw
Mermaid build workflow.

## Environment

All zentorch skills share one environment convention:

- Use a single activated, non-`base` Python environment for the whole workflow.
  Do not switch environments between skills.
- You choose the environment name; skills never assume or create a fixed one.
- Confirm what is active before running anything:

```bash
echo "${VIRTUAL_ENV:-${CONDA_DEFAULT_ENV:-none}}"
```

Unlike the other build skills, this one pins the Python version to whatever the
vLLM requirements table calls for (see [Environment prep](#environment-prep-native)),
because vLLM's CPU wheels are built for a specific interpreter.

## Inputs to confirm (ask the user)
1. **vLLM acquisition**: `pip` (pre-built CPU wheel) or `source`. **Always ask —
   no default.** (Overridden to `source` automatically if a PR is given.)
2. **vLLM version**: always confirm (see next section). If the user is unsure,
   auto-suggest the latest supported version.
3. **vLLM PR(s)** (optional): if provided → force `source` + cherry-pick; ask the
   **base version/tag** to cherry-pick onto (must be in supported range).
4. **environment**: `native` (venv/conda) or `docker`.
5. **git refs**: zentorch (default `main`), ZenDNN (default `main`).

## ALWAYS: confirm + validate the vLLM version
Confirm the version with the user **before** any `pip install vllm` or source
checkout. The supported range is **read fresh from this repo each build — never
hardcoded** (it drifts with the zentorch version):
```bash
grep -E 'VLLM_(MIN|MAX)_VERSION' src/cpu/python/zentorch/vllm/_core.py
grep -nE '0\.[0-9]+\.[0-9]+' src/cpu/python/zentorch/vllm/_core.py   # _VERSION_MAP entries
```
Cross-check the out-of-tree runtime range and the Python/PyTorch/TorchAO
requirements in `src/cpu/python/zentorch/vllm/README.md` — read the `| vLLM |`,
`| Python |`, `| PyTorch |` and `| TorchAO |` rows of the requirements table:
```bash
grep -E '^\| (vLLM|Python|PyTorch|TorchAO) \|' src/cpu/python/zentorch/vllm/README.md
```
No version numbers are reproduced in this skill on purpose: `_core.py` and that
table are the only sources of truth, and any copy here would silently rot.

**Auto-suggest the latest supported version:** from the grep output, pick the
highest `_VERSION_MAP` entry that is `≤ VLLM_MAX_VERSION` and within the
out-of-tree runtime range; offer it as the default when the user is unsure. The
user may override with any in-range version.

If the chosen version is outside `[VLLM_MIN_VERSION, VLLM_MAX_VERSION]` or not in
`_VERSION_MAP`, **stop and warn** the user before building. Never `pip install
vllm` before the version is confirmed AND validated.

## vLLM acquisition (pip vs source vs source+cherry-pick)
Decision tree:
1. **User provided vLLM PR(s)?**
   - **Yes** → force **source + cherry-pick** (pip cannot carry PR patches).
   - **No** → **ask** pip vs source (no default).
2. Confirm + validate the version (section above) on whichever path.
3. Run the matching install branch under
   [vLLM install](#vllm-install--branches-on-acquisition).

## Environment prep (native)
Create the environment at the Python version from the README requirements table
(grep it as shown above — do not assume a version):
```bash
conda create -n <env-name> python=<grepped-python-version> -y && conda activate <env-name>
sudo apt-get update -y
sudo apt-get install -y gcc-12 g++-12 libnuma-dev python3-dev
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 10 \
     --slave /usr/bin/g++ g++ /usr/bin/g++-12
```

**No sudo / no gcc-12 (e.g. Ubuntu 24.04 ships gcc-13, LSF batch hosts have no
root):** skip apt entirely.
- **Compiler:** any modern gcc works — do NOT hardcode gcc-12. Export the
  available one explicitly so CMake picks it (and so a stale cache can't force a
  missing gcc-12): `export CC=$(which gcc) CXX=$(which g++)`.
- **libnuma headers** (`numa.h`) without apt: install via conda-forge and expose
  the headers/libs to the build:
  ```bash
  conda install -y -c conda-forge libnuma numactl
  export CPATH="$CONDA_PREFIX/include:$CPATH"
  export LIBRARY_PATH="$CONDA_PREFIX/lib:$LIBRARY_PATH"
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
  ```

## vLLM install — branches on acquisition

**vLLM is installed first, before the ZenDNN backend and zentorch below** —
zentorch's wheel build packages the `zentorch.vllm` plugin and registers the
`vllm.platform_plugins` entry point, so installing zentorch last makes its plugin
registration bind against an already-present vLLM.


### pip — pre-built CPU wheel
The default PyPI `vllm` wheel is built with CUDA, so `pip install vllm==<version>`
pulls a GPU wheel that crashes on a CPU host (needs `libcudart.so.*`, lacks the
CPU custom ops like `torch.ops._C.init_cpu_memory_env`). CPU wheels live in a
separate registry — vLLM's own wheel index at `https://wheels.vllm.ai/<version>/cpu`.

Preferred (`uv`) — install straight from the CPU wheel registry:
```bash
uv pip install vllm --extra-index-url https://wheels.vllm.ai/<version>/cpu \
    --index-strategy first-index --torch-backend cpu
```
Plain `pip` cannot use that index reliably (and `--torch-backend cpu` is
`uv`-only), so with pip install the versioned `+cpu` wheel from the GitHub release
instead. `--extra-index-url .../whl/cpu` here only routes **torch** to CPU; it does
not change which vLLM wheel is chosen:
```bash
pip install --upgrade pip
VLLM_VERSION=<confirmed-version>
# read the x86_64 +cpu wheel filename (the manylinux tag drifts — don't hardcode):
gh release view v${VLLM_VERSION} --repo vllm-project/vllm \
   --json assets -q '.assets[].name' | grep -E '\+cpu-.*x86_64\.whl'
WHEEL=<name-from-above>
pip install "https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/${WHEEL}" \
    --extra-index-url https://download.pytorch.org/whl/cpu
# verify it is the +cpu build:
python -c "import vllm; assert '+cpu' in vllm.__version__, vllm.__version__; print(vllm.__version__)"
```
- If `gh` is unavailable, pick the `+cpu ... x86_64.whl` (or `aarch64` on ARM) from
  `https://github.com/vllm-project/vllm/releases/tag/v<version>`.
- **If you ever re-install vLLM after zentorch**, do it as a final step and
  re-verify `import zentorch` — reinstalling vLLM leaves the zentorch wheel intact
  but torch may have moved.

### source — vanilla vLLM (no PR)
```bash
git clone https://github.com/vllm-project/vllm.git && cd vllm
git checkout <confirmed-version>
pip install --upgrade pip
# vLLM > 0.19.0:
pip install -v -r requirements/build/cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu
# vLLM < 0.19.0 instead: requirements/cpu-build.txt
pip install -v -r requirements/cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu
VLLM_TARGET_DEVICE=cpu python setup.py install
cd ..
```

### source + cherry-pick — PR(s) provided
Ask the user for the **base version/tag** (confirm + validate it as above), then
clone at that base and cherry-pick the PR commit(s) **in the order the user
listed them**. **On any conflict: stop and surface it to the user — do NOT
auto-resolve.**
```bash
git clone https://github.com/vllm-project/vllm.git && cd vllm
git checkout <base-version>                    # confirmed + validated base tag
# for each PR N, in the user's stated order:
git fetch origin pull/<N>/head:pr<N>
git cherry-pick pr<N>                          # STOP on conflict — ask the user
# then build from source (same as the source branch):
pip install --upgrade pip
pip install -v -r requirements/build/cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu
pip install -v -r requirements/cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu
VLLM_TARGET_DEVICE=cpu python setup.py install
cd ..
```

## Backend: build against the local ZenDNN at `third_party/ZenDNN`

The ZenDNN backend build steps live in ZenDNN's own skill — run it and follow it
as the single source of truth when the checkout provides one:
- `third_party/ZenDNN/.claude/skills/build-zendnn`

**Policy: always build against a local ZenDNN checkout** by exporting
`ZENTORCH_USE_LOCAL_ZENDNN=1` for the wheel build (see
[Install zentorch](#install-zentorch)), **unless the user explicitly asks for a
FetchContent/tagged build.** Two reasons:

- The default path (`ZENTORCH_USE_LOCAL_ZENDNN=0`, which is what `setup.py` sets
  when the variable is unset) clones a **pinned tag** declared in
  `cmake/modules/zendnnl.cmake`. Read that pin fresh — never assume a tag from
  memory or from this document — and confirm it still exists on the remote before
  relying on it:
  ```bash
  grep -nE 'GIT_REPOSITORY|GIT_TAG' cmake/modules/zendnnl.cmake
  git ls-remote --tags <grepped-repository> | grep <grepped-tag>
  ```
  If `git ls-remote` returns nothing the pin is stale, FetchContent fails with
  `fatal: invalid reference`, and the build dies.
- Worse, that FetchContent uses `SOURCE_DIR = third_party/ZenDNN`, so a failed
  checkout **wipes whatever is at that path** (leaving only `.git`) and re-points
  the remote at the public repo. Recover with:
  `git -C third_party/ZenDNN checkout -f main && \
  git -C third_party/ZenDNN reset --hard origin/<zendnn-ref>`.

### Two ways to supply the local ZenDNN

**Option A — cmake copies the sibling checkout (no setup).** When
`ZENTORCH_USE_LOCAL_ZENDNN=1` and `third_party/ZenDNN` does **not** exist,
`zendnnl.cmake` copies `../ZenDNN` (the sibling of this repo) into
`third_party/ZenDNN` for you. Nothing to configure — just make sure the sibling
checkout is at the ref you want.

**Option B — symlink to the sibling checkout (preferred).** A symlink is
preferred over the copy for two reasons: edits in the sibling ZenDNN are picked
up live rather than frozen at copy time, and the tree is not duplicated on disk.
It is also what keeps a ZenDNN checkout's own untracked files (such as its
`.claude/skills`) reachable, which no `git submodule` checkout would fetch.

Replace any submodule or directory at `third_party/ZenDNN` with a symlink, once
per clone of the plugin:
```bash
git submodule deinit -f third_party/ZenDNN 2>/dev/null || true
rm -rf third_party/ZenDNN .git/modules/third_party/ZenDNN
git rm --cached -f third_party/ZenDNN 2>/dev/null || true
git config --remove-section submodule.'third_party/ZenDNN' 2>/dev/null || true
ln -s ../../ZenDNN third_party/ZenDNN     # target resolves to the sibling ZenDNN working copy
```

Then, per build, check out the ref and verify the target resolves:
```bash
git -C third_party/ZenDNN checkout <zendnn-ref>   # default: main
git -C third_party/ZenDNN log --oneline -1        # note the hash — verified later
ls third_party/ZenDNN/CMakeLists.txt              # target resolves
```
Do NOT run `git submodule update --init third_party/ZenDNN` afterward — it would
re-clone the public ZenDNN over the symlink.

## Install zentorch
Runs after vLLM is installed. The native wheel build is **not re-derived here** —
it is the `build-zentorch-from-source` skill's job. Run that skill's script with
the `--for-vllm` flag from this repository root:

```bash
export ZENTORCH_USE_LOCAL_ZENDNN=1
export CC=$(which gcc) CXX=$(which g++)
.claude/skills/build-zentorch-from-source/scripts/build.sh --for-vllm
```

Run it in the foreground with a long timeout (600000ms).

`--for-vllm` exists precisely for this skill. Without it the script pins and then
restores the branch-recommended CPU PyTorch and installs the wheel with full
dependency resolution — all of which would replace the exact torch vLLM chose.
With it the script instead:

- leaves the installed torch alone (no `ensure_pytorch_cpu`, no restore);
- installs the wheel with `--no-deps` so pip cannot pull a second torch;
- records `torch.__version__` before the build and **fails** if it changed by the
  end.

`ZENTORCH_USE_LOCAL_ZENDNN` is exported by this skill, not by the script — the
script is deliberately agnostic about where ZenDNN comes from.

Because the script installs `requirements.txt` into the active environment before
building, zentorch's pure-python runtime dependencies (including `deprecated`,
without which `import zentorch` raises `ModuleNotFoundError`) are already present
despite the `--no-deps` wheel install. Only a hand-rolled `pip install --no-deps`
outside the script needs `pip install deprecated` afterward.

**Build against the exact torch vLLM installed — do not let pip swap it.** The
vLLM install above (pip `+cpu` wheel or source) already pinned a specific CPU
torch; zentorch must compile and install against *that* torch, not a fresh one.
Capture it right after the vLLM step:
```bash
TORCH_VLLM=$(python -c "import torch; print(torch.__version__)"); echo "$TORCH_VLLM"
# cross-check it is within zentorch's supported PyTorch range (grep the README
# requirements table — never hardcode); if out of range, STOP (version mismatch).
grep -E '^\| PyTorch \|' src/cpu/python/zentorch/vllm/README.md
```
The script's own assertion covers the build itself, but confirm parity again
after any later pip activity:
```bash
python -c "import torch; print(torch.__version__)"   # MUST equal $TORCH_VLLM
```
If it differs, pip pulled a different torch — reinstall vLLM's torch
(`$TORCH_VLLM`) and re-run `build.sh --for-vllm` against it.

## Outputs
Concrete artifacts produced by the steps above:
- **zentorch wheel**: built into a temporary directory by `build.sh` and
  preserved into `<this repo>/dist/zentorch-*.whl` (filename is version-pinned).
- **vLLM**: either a **pre-built CPU wheel** installed via pip, or built **from
  source** from the cloned `vllm-project/vllm` checkout (source is required when
  a PR is cherry-picked).
- **torch**: the CPU torch pulled in by the vLLM install, left untouched by the
  `--for-vllm` build.

Expected installed state after all steps:
- `pip show zentorch` reports the version matching `dist/zentorch-*.whl`.
- `python -c "from vllm import LLM"` imports vLLM (pip or from-source).
- `python -c "import torch, zentorch"` succeeds (`deprecated` present).

## Runtime environment (document for the user)
```bash
export VLLM_CPU_KVCACHE_SPACE=90        # tune to KV-cache utilization
export VLLM_CPU_OMP_THREADS_BIND=0-95   # 0-95 Genoa, 0-127 Turin
export TORCHINDUCTOR_FREEZING=1
export VLLM_USE_AOT_COMPILE=0           # required when FREEZING=1 — see note below
export HF_TOKEN=<token>
# defaults (already on): ZENTORCH_LINEAR=1, ZENDNNL_MATMUL_WEIGHT_CACHE=1, ZENDNNL_MATMUL_ALGO=1
```

**`VLLM_USE_AOT_COMPILE=0` is required whenever `TORCHINDUCTOR_FREEZING=1`.**
Inductor freezing and AOT compile are two features that don't go hand in hand — so
with freezing on, AOT compile must be off. Leaving AOT compile on (the default)
alongside freezing makes the torch.compile path fail at model warmup:
```
torch/_functorch/_aot_autograd/autograd_cache.py: unwrap_output_code
AssertionError: expected OutputCode, got <class 'function'>
```
This is a consequence of the freezing + AOT-compile combination (not a zentorch
issue, and not a bug per se — reproduced independent of the model on Qwen3 0.6B
and 4B). Setting `VLLM_USE_AOT_COMPILE=0` keeps the inductor compile (and its
optimization, including freezing) while skipping the incompatible AOT path. It is
preferred over `--enforce-eager`, which disables compilation entirely.

## Smoke test
```bash
python -c "from vllm import LLM"
python -c 'import torch, zentorch; print(*zentorch.__config__.split("\n"), sep="\n")'
```
Expect (with monkey-patch warnings) a line like:
`INFO ... Platform plugin zentorch is activated`.

## Benchmark validation (confirm the build works)
The smoke test only proves the modules import. Run an end-to-end
`vllm bench throughput` on a small model to confirm the engine actually
initializes and runs inference with the zentorch backend. **This is the success
criterion for the build.**

Set the runtime env (see [Runtime environment](#runtime-environment-document-for-the-user)) —
note `VLLM_USE_AOT_COMPILE=0` is required — then run a tiny public model
(Qwen3-0.6B needs no HF token):
```bash
export VLLM_CPU_KVCACHE_SPACE=40
export VLLM_CPU_OMP_THREADS_BIND=0-127     # 0-95 Genoa, 0-127 Turin
export TORCHINDUCTOR_FREEZING=1
export VLLM_USE_AOT_COMPILE=0              # avoids the AOTAutograd OutputCode crash
export HF_HOME=<writable-cache-dir>        # model download cache

vllm bench throughput \
  --model Qwen/Qwen3-0.6B \
  --input-len 128 --output-len 128 --num-prompts 32 \
  --dtype bfloat16 \
  2>&1 | tee bench.log
```

**Pass criteria — all three must hold:**
1. Log contains `Platform plugin zentorch is activated` (confirms the zentorch
   plugin drove the run, not stock vLLM CPU).
2. A final `Throughput: <N> requests/s, <N> total tokens/s, <N> output tokens/s`
   line is printed (engine ran inference to completion).
3. Exit code 0.

Quick assertion (fails loudly if either signal is missing):
```bash
grep -q "Platform plugin zentorch is activated" bench.log \
  && grep -qE "Throughput: .* tokens/s" bench.log \
  && echo "BUILD VALIDATED" || echo "BUILD VALIDATION FAILED — inspect bench.log"
```

**Failure signatures and causes** (all seen during bring-up):
- `ImportError: libcudart.so.*` or `_C ... has no attribute init_cpu_memory_env`
  → the **CUDA vLLM wheel** was installed, not the `+cpu` wheel. Reinstall per
  [pip — pre-built CPU wheel](#pip--pre-built-cpu-wheel), then re-verify
  `import zentorch`.
- `expected OutputCode, got <class 'function'>` → AOT compile was left on while
  `TORCHINDUCTOR_FREEZING=1` (the two don't go together); set
  `VLLM_USE_AOT_COMPILE=0` (see
  [Runtime environment](#runtime-environment-document-for-the-user)).
- No `Platform plugin zentorch is activated` line → zentorch not installed into
  the active env, or its vLLM out-of-tree plugin was omitted
  (`ZENTORCH_VLLM_PLUGIN_BUILD=0`). Note that when the OOT plugin is omitted,
  zentorch can still activate via its **in-tree** vLLM plugin instead — in that
  case the run is driven by zentorch but this exact OOT log line won't appear, so
  confirm which plugin path is active before assuming a broken build; otherwise
  rebuild/reinstall the zentorch wheel with the OOT plugin enabled.

Verified working (Qwen3-4B, compile path, `VLLM_USE_AOT_COMPILE=0`):
`Throughput: 1.85 requests/s, 2131.95 total tokens/s, 236.88 output tokens/s`.

Fuller offline/online benchmarking (model list, `vllm serve` +
`vllm bench serve`) is not yet encoded here.

## Docker
Dockerfiles are **vendored in this skill** at `docker/<Ubuntu|RHEL>/Dockerfile`
so the skill is self-contained — they depend only on public inputs. Ask the
distro, then build the matching one. Each Dockerfile builds vLLM (CPU) from
source + the zentorch wheel against a local ZenDNN checkout.

Build args:
- `VLLM_VERSION` — **required, no default.** Confirm and validate it against the
  supported range first; the build fails fast if it is unset, so the version can
  never drift out of a stale default.
- `ZENTORCH_REPO` / `ZENDNN_REPO` — default to the public upstreams; override to
  point at a different fork if needed.
- `ZENTORCH_REF` / `ZENDNN_REF` — default `main`.
- `GIT_TOKEN` — build-time only, for private repos. **Never hardcode a PAT in the
  Dockerfile**; pass it in (prefer BuildKit `--secret` over `--build-arg`).

```bash
cd .claude/skills/build-vllm-zentorch/docker/Ubuntu   # or RHEL
docker build -t zentorch-cpu \
  --build-arg VLLM_VERSION=<confirmed-version> \
  --build-arg ZENTORCH_REF=main --build-arg ZENDNN_REF=main \
  -f Dockerfile .
```
Run with the HF token at runtime, never baked in: `docker run -e HF_TOKEN=... ...`.
**Keep these Dockerfiles in sync** with the native build gotchas (CPU wheel /
`+cpu`, `rm -rf build`, explicit `CC/CXX`, `deprecated` dep).

## Gotchas
- **The default backend path depends on a pinned tag that can go stale:**
  `ZENTORCH_USE_LOCAL_ZENDNN=0` (what `setup.py` assumes when the variable is
  unset) makes cmake clone the tag pinned in `cmake/modules/zendnnl.cmake`. Grep
  the pin and confirm it exists with `git ls-remote` before trusting it; export
  `=1` to build against a local checkout instead.
- **A failed FetchContent clobbers `third_party/ZenDNN`** (SOURCE_DIR is that
  path): it leaves only `.git` and re-points the remote at the public repo, so the
  *next* build fails with "does not contain CMakeLists.txt". This is why a
  **symlink** to the sibling `../ZenDNN` is preferred over a live submodule.
- **A local ZenDNN newer than the pinned tag breaks the cmake integration.**
  `cmake/modules/ZenDnnlFwkIntegrate.cmake` presets only a few `ZENDNNL_*`
  variables and then hands off to ZenDNN's own `fwk/ZenDnnlFwkIntegrate.cmake`,
  so the two move as a pair. Building against a ZenDNN checkout ahead of the
  pinned tag surfaces as cmake configure errors that name ZenDNN internals, for
  example `include could not find requested file: ZenDnnlFwkMacros` or
  `At least one of ZENDNNL_LIB_BUILD_ARCHIVE or ZENDNNL_LIB_BUILD_SHARED must be
  ON`. Neither is a compiler problem — check out the pinned tag (or the newest
  published one at or below it) in the local ZenDNN and rebuild.
- **zentorch must build against vLLM's torch.** vLLM pins the CPU torch; use
  `build.sh --for-vllm` so pip never swaps it. The flag also asserts the torch
  version is unchanged once the build finishes.
- **`third_party/ZenDNN` is a symlink, not a submodule.** A ZenDNN checkout's own
  untracked files are not fetched by `git submodule update`, which would also
  replace the symlink with the public repo. Never re-init the submodule over it.
- **Never hardcode the vLLM range or Python/PyTorch/TorchAO versions** — grep
  `_core.py` and the README requirements table fresh each build.
- **Confirm the vLLM version with the user before `pip install vllm`** — and
  validate it against the supported range first.
- **A PR forces source + cherry-pick** — pip wheels cannot carry PR patches.
  Cherry-pick in the user's stated order; stop on conflict.
- Install wheels by glob; version-pinned filenames drift.
- **`pip install vllm==<v>` installs the CUDA wheel, not CPU.** Fails on CPU
  hosts (`libcudart.so.*`, missing `init_cpu_memory_env`). Install the versioned
  `+cpu` wheel from the GitHub release. `--torch-backend cpu` is `uv`-only.
- **With `TORCHINDUCTOR_FREEZING=1`, set `VLLM_USE_AOT_COMPILE=0`** — freezing and
  AOT compile don't go hand in hand, so leaving AOT compile on crashes at model
  warmup with `expected OutputCode, got <class 'function'>`. Preferred over
  `--enforce-eager`, which disables compilation entirely.
- **Stale `build/` cache pins the old compiler.** Clear generated outputs with
  `.claude/skills/build-zentorch-from-source/scripts/clean.sh` before a rebuild,
  and export `CC/CXX` — do not rely on gcc-12 existing.
- **No sudo?** Get `numa.h` from conda-forge (`libnuma numactl`) and export
  `CPATH`/`LIBRARY_PATH`/`LD_LIBRARY_PATH` instead of apt.
