#!/usr/bin/env bash
# Rebuild zentorch from source: pull latest, build the wheel, install, verify.
#
# Usage: build.sh
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

echo "Building zentorch (branch: $(current_branch))"

pip uninstall -y zentorch 2>/dev/null || true
git pull --ff-only

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
