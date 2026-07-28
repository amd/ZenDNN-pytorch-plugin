#!/usr/bin/env bash

# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

# Rebuild zentorch from source: build the wheel, install, verify.
# Builds the current checkout; update the repo separately if you want the latest.
#
# Usage: build.sh
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

require_repo_root
require_active_env

echo "Building zentorch (branch: $(current_branch))"

echo "Installing build requirements into the active environment..."
python -m pip install -r requirements.txt

ensure_pytorch_cpu
torch_version="$(installed_pytorch_version)"

python -m pip uninstall -y zentorch 2>/dev/null || true

# Build into an isolated directory so pre-existing dist/ wheels cannot be
# selected. Preserve an existing dist/<wheel> rather than overwriting it.
wheel_build_dir="$(mktemp -d)"
trap 'rm -rf -- "${wheel_build_dir}"' EXIT
python setup.py bdist_wheel --dist-dir "${wheel_build_dir}"
wheel="$(wheel_from_build_directory "${wheel_build_dir}")"
preserve_wheel_artifact "${wheel}"
python -m pip install "${wheel}"
install_pytorch_cpu "${torch_version}"

echo "Build complete."
verify_zentorch
