# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

"""Fail the build when a portable library imports an unstable ATen symbol.

libzentorch_stable.so exists so its ops survive a PyTorch minor upgrade that
libzentorch.so cannot. An `at::` or `c10::` symbol in it silently voids that:
the reference resolves at load time out of the libtorch_cpu.so this library
already links, so the build succeeds and the wheel ships, and the failure only
appears on the torch version the library was supposed to tolerate.

Nothing else catches this. A shared object may leave symbols undefined, so the
linker has no objection, and the source list in cmake/modules/StableAbiLibs.cmake
is a glob - a newly added .cpp is compiled in without anyone opting it in.

Run by the zentorch_stable POST_BUILD step; also usable by hand:

    python scripts/check-torch-abi.py <path to libzentorch_stable.so>
"""

from __future__ import annotations

import argparse
import sys

try:
    from torch_abi_audit import inspect_extension
except ImportError:
    sys.exit(
        "error: torch-abi-audit is not installed, so the stable-ABI audit "
        "cannot run.\n"
        "       It ships in requirements.txt: pip install -r requirements.txt"
    )

# Symbols tolerated in the portable library, matched by prefix.
#
# This is a debt list, not a policy. Every entry means the library is not truly
# portable: the symbol resolves out of libtorch_cpu.so on the build's torch and
# may not on the one it is meant to survive. Shrink it, do not extend it.
#
# c10::MessageLogger is what LOG() expands to. The ~90 LOG(INFO)/LOG(WARNING)
# call sites across the shared op sources all reach it, so removing this entry
# means giving those sources a logger that is not ATen-backed.
ALLOWED_UNSTABLE_PREFIXES: tuple[str, ...] = ("c10::MessageLogger::",)

# Shown when the audit fails, because the fix is never obvious from a mangled
# symbol name alone.
_GUIDANCE = """
Each symbol above is an ATen entry point with no stable-ABI equivalent. Usually
one of:

  - TORCH_CHECK or LOG() reaching c10 because the source did not pick up the
    header-only substitutions in Utils.hpp (they are behind
    ZENTORCH_STABLE_ABI_LIB, so the file must include Utils.hpp).
  - an at:: call with a torch::stable equivalent, e.g. at::parallel_for ->
    torch::stable::parallel_for.
  - a source that is not portable yet and should be named in the exclusion list
    in cmake/modules/StableAbiLibs.cmake, with a reason.

Find the source with:
  nm -C --undefined-only <object or library> | grep -E '\\b(at|c10)::'
"""


def audit(paths: list[str]) -> int:
    failures = 0

    for path in paths:
        report = inspect_extension(path)

        if report.error is not None:
            print(f"error: could not inspect {path}: {report.error}", file=sys.stderr)
            failures += 1
            continue

        if not report.torch.uses_torch:
            # Not an error in itself, but it means the audit proved nothing, so
            # say so rather than report a pass.
            print(f"warning: {path} does not reference torch at all", file=sys.stderr)
            continue

        symbols = sorted(report.torch.unstable_symbols)
        allowed = [s for s in symbols if s.startswith(ALLOWED_UNSTABLE_PREFIXES)]
        unexpected = [s for s in symbols if s not in allowed]

        # An allowlist entry that no longer matches anything has been fixed, so
        # say so: otherwise the list only ever grows.
        stale = [
            prefix
            for prefix in ALLOWED_UNSTABLE_PREFIXES
            if not any(s.startswith(prefix) for s in symbols)
        ]

        if not unexpected:
            note = f", {len(allowed)} allowlisted" if allowed else ", no ATen imports"
            print(
                f"stable-ABI audit passed: {path} "
                f"({report.torch.stable_shim_count} shim calls{note})"
            )
            if stale:
                print(
                    "stable-ABI audit FAILED: these ALLOWED_UNSTABLE_PREFIXES no "
                    "longer match anything and should be deleted from "
                    f"{__file__}: {', '.join(stale)}",
                    file=sys.stderr,
                )
                failures += 1
            continue

        print(
            f"stable-ABI audit FAILED: {path} imports {len(unexpected)} "
            f"unstable torch symbol(s) outside the allowlist:",
            file=sys.stderr,
        )
        for symbol in unexpected:
            print(f"    {symbol}", file=sys.stderr)
        print(_GUIDANCE, file=sys.stderr)
        failures += 1

    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "library",
        nargs="+",
        help="path to a built shared library that must be stable-ABI clean",
    )
    return audit(parser.parse_args().library)


if __name__ == "__main__":
    sys.exit(main())
