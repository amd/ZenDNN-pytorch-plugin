# zentorch build-from-source flow

This flow mirrors
[`scripts/build.sh`](scripts/build.sh) and the source-build procedure in
[README.md section 2.2](../../../README.md).

The build script installs repository requirements, validates a supported
CPU-only PyTorch build, and compiles into an isolated wheel directory:

```bash
python -m pip install -r requirements.txt
.claude/skills/setup-env/scripts/install_pytorch.sh
python setup.py bdist_wheel
```

Supported alternate CPU versions are preserved. A supported CUDA or ROCm build
is reinstalled as CPU at the same base version. Missing and unsupported
versions use the branch-recommended CPU version. Only the single wheel in the
isolated directory can be installed. Older wheels in `dist/` cannot be selected
or overwritten.

After installation, the script restores the selected CPU PyTorch version and
verifies the zentorch version and build configuration. On
RHEL/Fedora-family systems, set `ZENDNNL_MANYLINUX_BUILD=1` when required.
Generated outputs can be removed independently of Python packaging internals:

```bash
.claude/skills/build-zentorch-from-source/scripts/clean.sh
```

```mermaid
flowchart TD
    start(["START: Build zentorch<br/>from source"])
    checkout["1. Current zentorch<br/>git checkout"]
    env{"Dedicated env<br/>active?"}
    envFail["STOP: Activate a<br/>non-base environment"]
    requirements["2. Install repository<br/>requirements"]
    version{"Supported base<br/>version?"}
    cpu{"CPU-only<br/>build?"}
    sameCpu["3. Reinstall same<br/>base version as CPU"]
    recommended["3. Install recommended<br/>CPU PyTorch"]
    selected["4. Record selected<br/>CPU version"]
    uninstall["5. Uninstall existing<br/>zentorch"]
    isolate["6. Create isolated<br/>wheel directory"]
    build["7. Build wheel there"]
    rhel["RHEL / Fedora:<br/>enable manylinux build"]
    wheel{"Exactly one new<br/>wheel found?"}
    buildFail["STOP: Build failed<br/>Fix error and rebuild"]
    install["8. Install current<br/>wheel"]
    restore["9. Restore selected<br/>CPU PyTorch"]
    verify["10. Import and verify<br/>version and config"]
    imports{"Verification<br/>successful?"}
    importFail["STOP: Import failed<br/>Check GLIBCXX<br/>or LD_PRELOAD"]
    done(["END: Build installed<br/>Continue to tests"])
    cleanup["Optional: Remove<br/>generated outputs"]

    start --> checkout --> env
    env -- No --> envFail
    env -- Yes --> requirements --> version
    version -- Yes --> cpu
    version -- No --> recommended --> selected
    cpu -- Yes --> selected
    cpu -- No --> sameCpu --> selected
    selected --> uninstall --> isolate --> build --> wheel
    rhel -. platform setting .-> build
    wheel -- No --> buildFail
    wheel -- Yes --> install --> restore --> verify --> imports
    imports -- No --> importFail
    imports -- Yes --> done
    done -. clean script .-> cleanup
```
