# Skill: Lint zentorch code

When the user asks to lint Python, C++, or shell scripts, follow this skill.

**Agent action:** Run the command matching the requested language:

```bash
scripts/lint.sh python
scripts/lint.sh cpp
scripts/lint.sh shell
```

Use manual steps below only if the script fails.

---

## Quick path (preferred)

```bash
scripts/lint.sh python
scripts/lint.sh cpp
scripts/lint.sh shell
```

---

## Python

Requires an activated Python environment.

```bash
pip install -r linter/requirements.txt
flake8
```

To auto-format files flagged by flake8 with black:

```bash
flake8 --quiet | xargs black --verbose
```

- Config: `.flake8` in repo root
- Excludes: `.git`, `build`, `dist`, `third_party`
- Also used by CI via `linter/py_cpp_linter.sh`

---

## C++

```bash
# Requires git-clang-format (LLVM/clang tools) available on PATH.
git clang-format --commit $(git rev-list HEAD | tail -n 1) --diff
```

To apply suggested formatting:

```bash
git clang-format -f
```

- C++20 standard, compiled with `-Wall -Werror` (PyTorch 2.13's c10 headers require C++20)
- Operator bindings live in `src/cpu/cpp/` and `Bindings.cpp`

---

## Shell

Requires `shellcheck` installed on the system (e.g. `sudo apt-get install shellcheck`).

```bash
find . \
  -path ./third_party -prune -o \
  -path ./build -prune -o \
  -path ./dist -prune -o \
  -path ./.git -prune -o \
  -type f -name '*.sh' -print | xargs shellcheck
```

Covers shell scripts under `scripts/`, `linter/`, and `.claude/`.
