#!/usr/bin/env bash
# Install or validate the pinned PyTorch CPU build for the current branch.
#
# Usage: scripts/install_pytorch.sh [--force]
#   --force   Uninstall any existing torch/torchvision/torchaudio and reinstall.
#
# Requires an activated Python environment (not base); see README section 2.2.2.1.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source-path=SCRIPTDIR source=common.sh
source "${SCRIPT_DIR}/common.sh"

require_repo_root
require_active_env

case "${1:-}" in
    --force)
        pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
        install_pytorch_cpu "$(detect_pytorch_version)"
        ;;
    "")
        ensure_pytorch_cpu
        ;;
    *)
        die "Usage: scripts/install_pytorch.sh [--force]"
        ;;
esac

python -c "import torch; print(f'PyTorch {torch.__version__}')"
