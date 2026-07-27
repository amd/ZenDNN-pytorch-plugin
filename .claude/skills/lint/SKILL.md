---
name: lint
description: >-
  Lint zentorch Python (flake8), C++ (git clang-format), or shell (shellcheck)
  code. Use when the user asks to lint, format-check, or run style checks on
  zentorch code.
---

# Lint zentorch code

When the user asks to lint Python, C++, or shell code, follow this skill.

**Agent action:** Run the command matching the requested language:

```bash
.claude/skills/lint/scripts/lint.sh python
.claude/skills/lint/scripts/lint.sh cpp
.claude/skills/lint/scripts/lint.sh shell
```

Use the manual steps below only if the script fails.

---

## Environment

Python linting requires an activated, non-`base` Python environment — the same
one used across the other zentorch skills. Confirm what is active:

```bash
echo "${VIRTUAL_ENV:-${CONDA_DEFAULT_ENV:-none}}"
```

C++ and shell linting do not require a Python environment.

---

## Python

```bash
pip install -r linter/requirements.txt
flake8
```

To auto-format files flagged by flake8 with black:

```bash
flake8 --format='%(path)s' | sort -u | xargs -r black --verbose
```

- Config: `.flake8` in the repo root
- Excludes: `.git`, `build`, `dist`, `third_party`
- Also used by CI via `linter/py_cpp_linter.sh`

---

## C++

Requires `git-clang-format` (from LLVM/clang tools) on `PATH`. This is the same
tool the repo's CI linter (`linter/py_cpp_linter.sh`) uses; the `clang-format`
pip package does not provide it.

```bash
git clang-format --commit "$(git rev-list HEAD | tail -n 1)" --diff
```

To apply the suggested formatting:

```bash
git clang-format -f
```

- C++20 standard, compiled with `-Wall -Werror`
- Operator bindings live in `src/cpu/cpp/` and `Bindings.cpp`

---

## Shell

Requires `shellcheck` on the system (e.g. `sudo apt-get install shellcheck`).

```bash
find . \
  -path ./third_party -prune -o \
  -path ./build -prune -o \
  -path ./dist -prune -o \
  -path ./.git -prune -o \
  -type f -name '*.sh' -print | xargs shellcheck
```

Covers shell scripts under `scripts/`, `linter/`, and `.claude/`.
