#!/usr/bin/env bash
# Agent workflow entry point for zentorch development tasks.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

usage() {
    cat <<'EOF'
Usage: .claude/scripts/agent.sh <command> [args]

Commands:
  install-pytorch [--force]   Install or validate pinned PyTorch CPU
  setup                       Full fresh setup: deps, build, install, verify
  build                       Rebuild zentorch from source
  verify                      Print zentorch version and build config
  test [scope]                Run tests (default: unittests)
  lint <python|cpp|shell>     Run linters

Test scopes: all, unittests, op_tests, model_tests, llm, pre_trained, or a test file path
EOF
}

cmd_install_pytorch() {
    require_active_env
    if [[ "${1:-}" == "--force" ]]; then
        pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
        install_pytorch_cpu "$(detect_pytorch_version)"
    else
        ensure_pytorch_cpu
    fi
    python -c "import torch; print(f'PyTorch {torch.__version__}')"
}

cmd_setup() {
    require_active_env
    local role
    role="$(detect_role)"
    [[ "${role}" != "unknown" ]] || die "Could not detect developer vs end-user from git remote."

    echo "Detected role: ${role} (branch: $(current_branch))"
    pip uninstall -y zentorch 2>/dev/null || true
    cmd_install_pytorch

    pip install -r requirements.txt

    if [[ "${role}" == "developer" ]]; then
        [[ -d ../ZenDNN ]] || git clone https://github.com/amd/ZenDNN.git ../ZenDNN
        export ZENTORCH_USE_LOCAL_ZENDNN=1
    fi

    python setup.py bdist_wheel
    pip install dist/zentorch-*.whl
    pip install "torch==$(detect_pytorch_version)" \
        --index-url https://download.pytorch.org/whl/cpu --force-reinstall --no-deps

    echo "Setup complete."
    verify_zentorch
}

cmd_build() {
    require_active_env
    local role
    role="$(detect_role)"
    [[ "${role}" != "unknown" ]] || die "Could not detect developer vs end-user from git remote."

    echo "Detected role: ${role} (branch: $(current_branch))"
    pip uninstall -y zentorch 2>/dev/null || true
    git pull

    if [[ "${role}" == "developer" ]]; then
        [[ -d ../ZenDNN ]] || git clone https://github.com/amd/ZenDNN.git ../ZenDNN
        (cd ../ZenDNN && git pull)
        export ZENTORCH_USE_LOCAL_ZENDNN=1
    fi

    python setup.py bdist_wheel
    pip install dist/zentorch-*.whl
    pip install "torch==$(detect_pytorch_version)" \
        --index-url https://download.pytorch.org/whl/cpu --force-reinstall --no-deps

    echo "Build complete."
    verify_zentorch
}

cmd_verify() {
    require_active_env
    verify_zentorch
}

cmd_test() {
    require_active_env
    export ZENDNNL_MATMUL_WEIGHT_CACHE=0
    export ZENDNNL_ZP_COMP_CACHE=0

    python test/install_requirements.py

    local scope="${1:-unittests}"
    local -a cmd
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
}

cmd_lint() {
    local target="${1:-}"
    [[ -n "${target}" ]] || die "Usage: agent.sh lint <python|cpp|shell>"

    case "${target}" in
        python)
            require_active_env
            pip install -q -r linter/requirements.txt
            flake8
            echo "Python lint passed."
            ;;
        cpp)
            if ! command -v git-clang-format >/dev/null 2>&1 && ! git clang-format -h >/dev/null 2>&1; then
                pip install -q clang-format
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
            [[ ${#scripts[@]} -gt 0 ]] || { echo "No shell scripts found."; return 0; }
            shellcheck "${scripts[@]}"
            echo "Shell lint passed."
            ;;
        *) die "Unknown lint target '${target}'. Use: python, cpp, or shell." ;;
    esac
}

require_repo_root

case "${1:-}" in
    install-pytorch) shift; cmd_install_pytorch "$@" ;;
    setup) cmd_setup ;;
    build) cmd_build ;;
    verify) cmd_verify ;;
    test) shift; cmd_test "${1:-}" ;;
    lint) shift; cmd_lint "${1:-}" ;;
    -h|--help|help|"") usage ;;
    *) die "Unknown command '${1}'. Run: agent.sh --help" ;;
esac
