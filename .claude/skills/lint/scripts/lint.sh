#!/usr/bin/env bash
# Lint zentorch source.
#
# Usage: lint.sh <python|cpp|shell>
#   python : flake8 (requires an activated Python environment)
#   cpp    : git clang-format check (requires git-clang-format on PATH)
#   shell  : shellcheck over tracked .sh files (requires shellcheck)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null)" \
    || { echo "ERROR: run this from a zentorch git checkout." >&2; exit 1; }
# shellcheck source=scripts/common.sh
source "${REPO_ROOT}/scripts/common.sh"

require_repo_root

target="${1:-}"
[[ -n "${target}" ]] || die "Usage: lint.sh <python|cpp|shell>"

case "${target}" in
    python)
        require_active_env
        python -m pip install -q -r linter/requirements.txt
        python -m flake8
        echo "Python lint passed."
        ;;
    cpp)
        # The repo C++ lint (linter/py_cpp_linter.sh) uses git clang-format,
        # which the clang-format pip package does not provide. Require it and
        # point the user at an actionable install instead of a no-op fallback.
        command -v git >/dev/null 2>&1 \
            || die "git is required for C++ lint."
        if ! git clang-format -h >/dev/null 2>&1; then
            die "git clang-format is required for C++ lint (install with 'conda install -c conda-forge clang-tools' or 'sudo apt-get install clang-format')."
        fi
        git clang-format --commit "$(git rev-list HEAD | tail -n 1)" --diff
        echo "C++ lint passed."
        ;;
    shell)
        command -v shellcheck >/dev/null 2>&1 \
            || die "shellcheck is required for shell lint (install with 'conda install -c conda-forge shellcheck' or 'sudo apt-get install shellcheck')."
        shellcheck --version >/dev/null 2>&1 \
            || die "shellcheck is present but could not run."
        # Lint only tracked .sh files for deterministic results. This excludes
        # untracked/generated scripts and the git-ignored third_party/build/dist.
        mapfile -t scripts < <(git -C "${REPO_ROOT}" ls-files -- '*.sh')
        [[ ${#scripts[@]} -gt 0 ]] || { echo "No shell scripts found."; exit 0; }
        shellcheck "${scripts[@]}"
        echo "Shell lint passed."
        ;;
    *) die "Unknown lint target '${target}'. Use: python, cpp, or shell." ;;
esac
