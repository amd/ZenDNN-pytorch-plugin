#!/usr/bin/env bash
# Print the installed zentorch version and build configuration.
#
# Usage: scripts/verify.sh
#
# Requires an activated Python environment with zentorch installed.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source-path=SCRIPTDIR source=common.sh
source "${SCRIPT_DIR}/common.sh"

require_repo_root
require_active_env
verify_zentorch
