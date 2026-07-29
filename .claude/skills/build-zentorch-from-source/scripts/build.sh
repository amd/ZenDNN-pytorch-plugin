#!/usr/bin/env bash

# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

# Rebuild zentorch from source: build the wheel, install, verify.
# Builds the current checkout; update the repo separately if you want the latest.
#
# Usage: build.sh [--for-vllm]
#
#   --for-vllm  Build into an environment where vLLM already pinned PyTorch.
#               Leaves the installed torch untouched (no version pinning, no
#               restore), installs the wheel with --no-deps so pip cannot pull a
#               second torch, and fails if the torch version changed anyway.
#               Used by the build-vllm-zentorch skill.
#
# Requires an activated Python environment (not base); see README section 2.2.2.1.
#
# NOTE: the build step can take several minutes; run it in the foreground.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null)" \
    || { echo "ERROR: run this from a zentorch git checkout." >&2; exit 1; }
# shellcheck source=scripts/common.sh
source "${REPO_ROOT}/scripts/common.sh"

for_vllm=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --for-vllm) for_vllm=1 ;;
        -h|--help)
            echo "Usage: build.sh [--for-vllm]"
            exit 0
            ;;
        *) die "Unknown argument: $1 (usage: build.sh [--for-vllm])" ;;
    esac
    shift
done

require_repo_root
require_active_env

echo "Building zentorch (branch: $(current_branch))"

echo "Installing build requirements into the active environment..."
python -m pip install -r requirements.txt

# With --for-vllm the installed torch is whatever vLLM chose, and zentorch has
# to compile and install against exactly that. Skipping the pin/restore is not
# enough on its own: the wheel install must also skip dependency resolution, or
# pip pulls a second torch over vLLM's. The three go together, which is why one
# flag controls all of them.
if (( for_vllm )); then
    torch_version="$(installed_pytorch_version)"
    [[ -n "${torch_version}" ]] \
        || die "--for-vllm requires PyTorch to be installed already (install vLLM first)."
    echo "Preserving the installed PyTorch ${torch_version} for vLLM."
else
    ensure_pytorch_cpu
    torch_version="$(installed_pytorch_version)"
fi

python -m pip uninstall -y zentorch 2>/dev/null || true

# Build into an isolated directory so pre-existing dist/ wheels cannot be
# selected. Preserve an existing dist/<wheel> rather than overwriting it.
wheel_build_dir="$(mktemp -d)"
trap 'rm -rf -- "${wheel_build_dir}"' EXIT
python setup.py bdist_wheel --dist-dir "${wheel_build_dir}"
wheel="$(wheel_from_build_directory "${wheel_build_dir}")"
preserve_wheel_artifact "${wheel}"

if (( for_vllm )); then
    python -m pip install --no-deps "${wheel}"
    installed_torch="$(installed_pytorch_version)"
    [[ "${installed_torch}" == "${torch_version}" ]] \
        || die "PyTorch changed from ${torch_version} to ${installed_torch:-none} during the build. Reinstall vLLM's PyTorch and re-run."
    echo "PyTorch ${torch_version} is unchanged."
else
    python -m pip install "${wheel}"
    install_pytorch_cpu "${torch_version}"
fi

echo "Build complete."
verify_zentorch
