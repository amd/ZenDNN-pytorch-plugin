#!/usr/bin/env bash
# Shared helpers for the zentorch skill workflow scripts.
#
# Single source of truth. The per-skill entry scripts under
# .claude/skills/<name>/scripts/ locate the repo root via git and source this
# file, so there is exactly one copy to maintain.
#
# Meant to be sourced, not executed directly.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

die() {
    echo "ERROR: $*" >&2
    exit 1
}

# Resolve the repository root from git so these scripts work regardless of
# where they are bundled (repo-root scripts/ or .claude/skills/<name>/scripts/)
# and regardless of the current working directory.
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null)" \
    || die "Not inside a git repository. Run from a zentorch checkout."

require_repo_root() {
    cd "${REPO_ROOT}"
    [[ -f setup.py ]] || die "Could not find the zentorch repository root (setup.py missing)."
}

current_branch() {
    git -C "${REPO_ROOT}" branch --show-current 2>/dev/null || echo "unknown"
}

# Primary (recommended) PyTorch CPU version pinned for the current branch.
# See README.md for the authoritative PyTorch/Python compatibility matrix.
detect_pytorch_version() {
    local branch
    branch="$(current_branch)"
    case "${branch}" in
        r5.2) echo "2.10.0" ;;
        *) echo "2.13.0" ;;
    esac
}

# Additional supported PyTorch versions (space-separated) for the current branch.
detect_pytorch_alternates() {
    local branch
    branch="$(current_branch)"
    case "${branch}" in
        r5.2) echo "2.9.1" ;;
        *) echo "2.12.1 2.12.0 2.11.0" ;;
    esac
}

# Name of the active Python environment (empty if none).
active_env_name() {
    if [[ -n "${VIRTUAL_ENV:-}" ]]; then
        basename "${VIRTUAL_ENV}"
    elif [[ -n "${CONDA_DEFAULT_ENV:-}" ]]; then
        echo "${CONDA_DEFAULT_ENV}"
    else
        echo ""
    fi
}

# All skills share this convention: require an activated, non-base environment.
# The user picks the environment name (see README.md section 2.2.2.1); skills do
# not prescribe one.
require_active_env() {
    local env_name
    env_name="$(active_env_name)"
    if [[ -z "${env_name}" ]]; then
        die "No active Python environment detected. Create and activate one first (see README.md section 2.2.2.1)."
    fi
    if [[ "${env_name}" == "base" ]]; then
        die "Do not use the base environment. Create a dedicated Python environment (see README.md section 2.2.2.1)."
    fi
    echo "Using active environment: ${env_name}"
}

installed_pytorch_version() {
    python -c "import torch; print(torch.__version__.split('+')[0])" 2>/dev/null || true
}

pytorch_version_supported() {
    local installed expected alternates alt
    installed="$1"
    expected="$(detect_pytorch_version)"
    alternates="$(detect_pytorch_alternates)"

    [[ "${installed}" == "${expected}" ]] && return 0
    for alt in ${alternates}; do
        [[ "${installed}" == "${alt}" ]] && return 0
    done
    return 1
}

install_pytorch_cpu() {
    local version="${1:-$(detect_pytorch_version)}"
    echo "Installing PyTorch CPU ${version}..."
    pip install "torch==${version}" --index-url https://download.pytorch.org/whl/cpu
}

ensure_pytorch_cpu() {
    local installed expected
    expected="$(detect_pytorch_version)"
    installed="$(installed_pytorch_version)"

    if [[ -z "${installed}" ]]; then
        install_pytorch_cpu "${expected}"
        return
    fi

    if pytorch_version_supported "${installed}"; then
        echo "PyTorch ${installed} is compatible with branch $(current_branch) (expected ${expected} or $(detect_pytorch_alternates))."
        return
    fi

    echo "PyTorch ${installed} is incompatible with branch $(current_branch). Reinstalling ${expected}..."
    pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
    install_pytorch_cpu "${expected}"
}

verify_zentorch() {
    python -c 'import zentorch; print("zentorch", zentorch.__version__); print(*zentorch.__config__.split("\n"), sep="\n")'
}

# Print the most recently built zentorch wheel in dist/ (empty if none).
latest_wheel() {
    find "${REPO_ROOT}/dist" -maxdepth 1 -name 'zentorch-*.whl' -printf '%T@ %p\n' 2>/dev/null \
        | sort -rn | head -n 1 | cut -d' ' -f2-
}
