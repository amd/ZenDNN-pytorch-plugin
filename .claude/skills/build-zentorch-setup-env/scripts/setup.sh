#!/usr/bin/env bash

# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

# Fresh setup: validate PyTorch, install deps, build and install zentorch, verify.
#
# Usage: setup.sh
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

echo "Setting up zentorch (branch: $(current_branch))"

python -m pip uninstall -y zentorch 2>/dev/null || true
ensure_pytorch_cpu
python -m pip install -r requirements.txt

# Preserve the validated, supported CPU version across wheel installation.
torch_version="$(installed_pytorch_version)"
wheel_build_dir="$(mktemp -d)"
trap 'rm -rf -- "${wheel_build_dir}"' EXIT
python setup.py bdist_wheel --dist-dir "${wheel_build_dir}"
wheel="$(wheel_from_build_directory "${wheel_build_dir}")"
preserve_wheel_artifact "${wheel}"
python -m pip install "${wheel}"
install_pytorch_cpu "${torch_version}"

echo "Setup complete."
verify_zentorch
