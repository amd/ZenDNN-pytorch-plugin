#!/usr/bin/env bash

# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

# Remove generated zentorch build, wheel, and package-metadata directories.
#
# Usage: clean.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null)" \
    || { echo "ERROR: run this from a zentorch git checkout." >&2; exit 1; }
# shellcheck source=scripts/common.sh
source "${REPO_ROOT}/scripts/common.sh"

require_repo_root

generated_paths=(
    "${REPO_ROOT}/build"
    "${REPO_ROOT}/dist"
    "${REPO_ROOT}/src/cpu/python/zentorch.egg-info"
)

for path in "${generated_paths[@]}"; do
    if [[ -e "${path}" ]]; then
        rm -rf -- "${path}"
        echo "Removed ${path#"${REPO_ROOT}/"}"
    fi
done

echo "Build cleanup complete."
