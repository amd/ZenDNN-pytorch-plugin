#!/usr/bin/env bash
# Run zentorch tests with the required environment variables.
#
# Usage: scripts/test.sh [scope]
#   scope: all | unittests (default) | op_tests | model_tests | llm |
#          pre_trained | <path/to/test_file.py>
#
# Requires an activated Python environment with zentorch installed.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source-path=SCRIPTDIR source=common.sh
source "${SCRIPT_DIR}/common.sh"

require_repo_root
require_active_env

export ZENDNNL_MATMUL_WEIGHT_CACHE=0
export ZENDNNL_ZP_COMP_CACHE=0

python test/install_requirements.py

scope="${1:-unittests}"
declare -a cmd
case "${scope}" in
    all) cmd=(python -m unittest discover -s ./test) ;;
    unittests) cmd=(python -m unittest discover -s ./test/unittests) ;;
    op_tests) cmd=(python -m unittest discover -s ./test/unittests/op_tests) ;;
    model_tests) cmd=(python -m unittest discover -s ./test/unittests/model_tests) ;;
    llm) cmd=(python -m unittest discover -s ./test/llm_tests) ;;
    pre_trained) cmd=(python -m unittest discover -s ./test/pre_trained_model_tests) ;;
    *)
        if [[ -f "${scope}" ]]; then
            cmd=(python -m unittest "${scope}")
        else
            die "Unknown scope '${scope}'. Use: all, unittests, op_tests, model_tests, llm, pre_trained, or a test file path."
        fi
        ;;
esac

echo "Running: ${cmd[*]}"
"${cmd[@]}"
