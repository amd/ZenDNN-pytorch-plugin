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

# Best-effort current branch name. `git branch --show-current` is empty on a
# detached HEAD (common in CI), so fall back to GITHUB_REF_NAME and finally
# "unknown" rather than silently assuming a branch for version pinning.
current_branch() {
    local branch
    branch="$(git -C "${REPO_ROOT}" branch --show-current 2>/dev/null || true)"
    [[ -z "${branch}" ]] && branch="${GITHUB_REF_NAME:-}"
    if [[ -z "${branch}" || "${branch}" == "HEAD" ]]; then
        echo "unknown"
    else
        echo "${branch}"
    fi
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

pytorch_is_cpu_build() {
    python -c \
        "import sys, torch; sys.exit(0 if torch.version.cuda is None and getattr(torch.version, 'hip', None) is None else 1)" \
        2>/dev/null
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
    python -m pip install "torch==${version}" \
        --index-url https://download.pytorch.org/whl/cpu \
        --force-reinstall --no-deps

    [[ "$(installed_pytorch_version)" == "${version}" ]] \
        || die "Installed PyTorch version does not match ${version}."
    pytorch_is_cpu_build || die "Installed PyTorch ${version} is not CPU-only."
}

ensure_pytorch_cpu() {
    local installed expected
    expected="$(detect_pytorch_version)"
    installed="$(installed_pytorch_version)"

    if [[ -z "${installed}" ]]; then
        install_pytorch_cpu "${expected}"
        return
    fi

    if pytorch_version_supported "${installed}" && pytorch_is_cpu_build; then
        echo "PyTorch ${installed} CPU is compatible with branch $(current_branch) (expected ${expected} or $(detect_pytorch_alternates))."
        return
    fi

    if pytorch_version_supported "${installed}"; then
        echo "PyTorch ${installed} is not CPU-only. Reinstalling ${expected} CPU..."
    else
        echo "PyTorch ${installed} is incompatible with branch $(current_branch). Reinstalling ${expected} CPU..."
    fi
    python -m pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
    install_pytorch_cpu "${expected}"
}

verify_zentorch() {
    python -c 'import zentorch; print("zentorch", zentorch.__version__); print(*zentorch.__config__.split("\n"), sep="\n")'
}

# Print the single wheel produced in an isolated build directory.
wheel_from_build_directory() {
    local build_directory="$1"
    local -a wheels=()
    [[ -d "${build_directory}" ]] \
        || die "Wheel build directory does not exist: ${build_directory}"
    mapfile -d '' wheels < <(
        find "${build_directory}" -maxdepth 1 -type f -name '*.whl' -print0
    )
    [[ "${#wheels[@]}" -eq 1 ]] \
        || die "Expected one wheel from the current build, found ${#wheels[@]}."
    printf '%s\n' "${wheels[0]}"
}

# Keep a conventional dist/<wheel> artifact when that path is unused. If an
# artifact with the same name predates the build, preserve it; installation
# still uses the isolated, current-build wheel.
preserve_wheel_artifact() {
    local wheel="$1"
    local destination
    mkdir -p "${REPO_ROOT}/dist"
    destination="${REPO_ROOT}/dist/$(basename "${wheel}")"
    if [[ -e "${destination}" ]]; then
        echo "Preserving pre-existing wheel: ${destination}"
    else
        cp -p -- "${wheel}" "${destination}"
        echo "Saved current-build wheel: ${destination}"
    fi
}
