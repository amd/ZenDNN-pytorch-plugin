#!/usr/bin/env bash

# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

# Install or validate the pinned PyTorch CPU build for the current branch.
#
# Usage: install_pytorch.sh [--force]
#   --force   Uninstall any existing torch/torchvision/torchaudio and reinstall.
#
# Requires an activated Python environment (not base); see README section 2.2.2.1.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null)" \
    || { echo "ERROR: run this from a zentorch git checkout." >&2; exit 1; }
# shellcheck source=scripts/common.sh
source "${REPO_ROOT}/scripts/common.sh"

require_repo_root
require_active_env

case "${1:-}" in
    --force)
        python -m pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
        install_pytorch_cpu "$(detect_pytorch_version)"
        ;;
    "")
        ensure_pytorch_cpu
        ;;
    *)
        die "Usage: install_pytorch.sh [--force]"
        ;;
esac

python -c "import torch; print(f'PyTorch {torch.__version__}')"
