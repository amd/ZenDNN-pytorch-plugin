#!/usr/bin/env bash
# Run zentorch tests with the required environment variables.
#
# Usage: test.sh [scope]
#   scope: all | unittests (default) | op_tests | model_tests |
#          miscellaneous_tests | export_tests | vllm_tests | llm |
#          pre_trained | <path/to/test_file.py>
#
# Requires an activated Python environment with zentorch installed.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null)" \
    || { echo "ERROR: run this from a zentorch git checkout." >&2; exit 1; }
# shellcheck source=scripts/common.sh
source "${REPO_ROOT}/scripts/common.sh"

require_repo_root
require_active_env

export ZENDNNL_MATMUL_WEIGHT_CACHE=0
export ZENDNNL_ZP_COMP_CACHE=0
export ZENDNNL_ENABLE_POSTOP_CACHE=0

python test/install_requirements.py

scope="${1:-unittests}"
declare -a cmd
case "${scope}" in
    all) cmd=(python -m unittest discover -s ./test) ;;
    unittests) cmd=(python -m unittest discover -s ./test/unittests) ;;
    op_tests) cmd=(python -m unittest discover -s ./test/unittests/op_tests) ;;
    model_tests) cmd=(python -m unittest discover -s ./test/unittests/model_tests) ;;
    miscellaneous_tests) cmd=(python -m unittest discover -s ./test/unittests/miscellaneous_tests) ;;
    export_tests) cmd=(python -m unittest discover -s ./test/unittests/export_tests) ;;
    vllm_tests) cmd=(python -m unittest discover -s ./test/unittests/vllm_tests) ;;
    llm) cmd=(python -m unittest discover -s ./test/llm_tests) ;;
    pre_trained) cmd=(python -m unittest discover -s ./test/pre_trained_model_tests) ;;
    *)
        if [[ -f "${scope}" ]]; then
            cmd=(python -m unittest "${scope}")
        else
            die "Unknown scope '${scope}'. Use: all, unittests, op_tests, model_tests, miscellaneous_tests, export_tests, vllm_tests, llm, pre_trained, or a test file path."
        fi
        ;;
esac

echo "Running: ${cmd[*]}"
"${cmd[@]}"
