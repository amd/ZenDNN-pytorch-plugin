#!/usr/bin/env bash
# Rebuild zentorch from source: pull latest, build the wheel, install, verify.
#
# Usage: scripts/build.sh
#
# Requires an activated Python environment (not base); see README section 2.2.2.1.
# Developer vs end-user is auto-detected from the git remote; override with
# ZENTORCH_ROLE=developer or ZENTORCH_ROLE=end-user for non-standard remotes.
#
# NOTE: the build step can take several minutes; run it in the foreground.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source-path=SCRIPTDIR source=common.sh
source "${SCRIPT_DIR}/common.sh"

require_repo_root
require_active_env

role="$(detect_role)"
[[ "${role}" != "unknown" ]] || die "Could not detect role from git remote. Set ZENTORCH_ROLE=developer or end-user."
echo "Detected role: ${role} (branch: $(current_branch))"

pip uninstall -y zentorch 2>/dev/null || true
git pull --ff-only

if [[ "${role}" == "developer" ]]; then
    [[ -d ../ZenDNN ]] || git clone https://github.com/amd/ZenDNN.git ../ZenDNN
    (cd ../ZenDNN && git pull --ff-only)
    export ZENTORCH_USE_LOCAL_ZENDNN=1
fi

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
