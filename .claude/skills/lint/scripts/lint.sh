#!/usr/bin/env bash
# Lint zentorch source.
#
# Usage: lint.sh <python|cpp|shell>
#   python : flake8 (requires an activated Python environment)
#   cpp    : git clang-format check (requires git-clang-format on PATH)
#   shell  : shellcheck over tracked .sh files (requires shellcheck)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source-path=SCRIPTDIR source=common.sh
source "${SCRIPT_DIR}/common.sh"

require_repo_root

target="${1:-}"
[[ -n "${target}" ]] || die "Usage: lint.sh <python|cpp|shell>"

case "${target}" in
    python)
        require_active_env
        pip install -q -r linter/requirements.txt
        flake8
        echo "Python lint passed."
        ;;
    cpp)
        # The repo C++ lint (linter/py_cpp_linter.sh) uses git clang-format,
        # which the clang-format pip package does not provide. Require it and
        # point the user at an actionable install instead of a no-op fallback.
        if ! git clang-format -h >/dev/null 2>&1; then
            die "git clang-format is required for C++ lint (install LLVM/clang tools that provide git-clang-format)."
        fi
        git clang-format --commit "$(git rev-list HEAD | tail -n 1)" --diff
        echo "C++ lint passed."
        ;;
    shell)
        command -v shellcheck >/dev/null 2>&1 || die "shellcheck is not installed."
        mapfile -t scripts < <(
            find . \
                -path ./third_party -prune -o \
                -path ./build -prune -o \
                -path ./dist -prune -o \
                -path ./.git -prune -o \
                -type f -name '*.sh' -print
        )
        [[ ${#scripts[@]} -gt 0 ]] || { echo "No shell scripts found."; exit 0; }
        shellcheck "${scripts[@]}"
        echo "Shell lint passed."
        ;;
    *) die "Unknown lint target '${target}'. Use: python, cpp, or shell." ;;
esac
