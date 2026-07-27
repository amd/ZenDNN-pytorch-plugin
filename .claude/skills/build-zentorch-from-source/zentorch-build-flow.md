# zentorch build-from-source flow

This flow mirrors
[`scripts/build.sh`](scripts/build.sh) and the source-build procedure in
[README.md section 2.2](../../../README.md).

The build script installs the repository requirements before compiling so the
active environment supplies CMake and Ninja:

```bash
python -m pip install -r requirements.txt
python setup.py bdist_wheel
```

After building, it installs the newest wheel, restores the selected CPU-only
PyTorch version, and verifies the zentorch version and build configuration. On
RHEL/Fedora-family systems, set `ZENDNNL_MANYLINUX_BUILD=1` when required.

```mermaid
flowchart TD
    start(["START: Build zentorch<br/>from source"])
    checkout["1. Current zentorch<br/>git checkout"]
    env["2. Active dedicated<br/>Python environment"]
    requirements["3. Install repository<br/>requirements"]
    uninstall["4. Uninstall existing<br/>zentorch"]
    torch["5. Select PyTorch<br/>CPU version"]
    build["6. Build wheel"]
    rhel["RHEL / Fedora:<br/>enable manylinux build"]
    wheel{"Wheel found<br/>in dist?"}
    buildFail["STOP: Build failed<br/>Fix error and rebuild"]
    install["7. Install wheel<br/>Restore PyTorch CPU"]
    verify["8. Import and verify<br/>version and config"]
    imports{"Verification<br/>successful?"}
    importFail["STOP: Import failed<br/>Check GLIBCXX<br/>or LD_PRELOAD"]
    done(["END: Build installed<br/>Continue to tests"])

    start --> checkout --> env --> requirements --> uninstall --> torch --> build --> wheel
    rhel -. platform setting .-> build
    wheel -- No --> buildFail
    wheel -- Yes --> install --> verify --> imports
    imports -- No --> importFail
    imports -- Yes --> done
```
