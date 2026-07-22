#!/usr/bin/env bash
# Fresh setup: validate PyTorch, install deps, build and install zentorch, verify.
#
# Usage: setup.sh
#
# Requires an activated Python environment (not base); see README section 2.2.2.1.
#
# NOTE: the build step can take several minutes; run it in the foreground.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source-path=SCRIPTDIR source=common.sh
source "${SCRIPT_DIR}/common.sh"

require_repo_root
require_active_env

echo "Setting up zentorch (branch: $(current_branch))"

pip uninstall -y zentorch 2>/dev/null || true
ensure_pytorch_cpu
pip install -r requirements.txt

# Preserve the currently-installed (supported) torch version so the wheel
# install does not silently switch it; fall back to the pinned version.
torch_version="$(installed_pytorch_version)"
python setup.py bdist_wheel
wheel="$(latest_wheel)"
[[ -n "${wheel}" ]] || die "No zentorch wheel found in dist/."
pip install "${wheel}"
pip install "torch==${torch_version:-$(detect_pytorch_version)}" \
    --index-url https://download.pytorch.org/whl/cpu --force-reinstall --no-deps

echo "Setup complete."
verify_zentorch
