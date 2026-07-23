#!/usr/bin/env bash
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

pip uninstall -y zentorch 2>/dev/null || true

# Preserve the currently-installed (supported) torch version; fall back to the
# pinned version if torch is not installed yet.
torch_version="$(installed_pytorch_version)"
python setup.py bdist_wheel
wheel="$(latest_wheel)"
[[ -n "${wheel}" ]] || die "No zentorch wheel found in dist/."
pip install "${wheel}"
pip install "torch==${torch_version:-$(detect_pytorch_version)}" \
    --index-url https://download.pytorch.org/whl/cpu --force-reinstall --no-deps

echo "Build complete."
verify_zentorch
