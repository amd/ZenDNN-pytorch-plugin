#!/usr/bin/env python3

# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

"""Read the perf-eval workload inventory and answer two questions about it.

usage:
  workload_info.py --list [--inventory PATH]
  workload_info.py --max-model-len chat,rag [--inventory PATH]

The inventory is the vendored clone's
`.../inventory/group_vars/all/test-workloads.yml`, so the workload table is
never duplicated here: whatever GuideLLM is told to send is what we size the
server for.

`--max-model-len` is the reason this exists. The harness starts every vLLM
instance with one context window, but a workload's traffic is described by
isl/osl (or isl_max/osl_max when it is a variable-length profile). Requests
that overflow that window are rejected by the server, so the window has to be
sized from the workload rather than left at the harness default. So:

    need     = isl_max|isl + osl_max|osl + 128   (tokenization headroom)
    declared = --max-model-len from the workload's vllm_args, if it sets one
    result   = max(need, declared), and across several workloads, the max

Several workloads share one warm stack, so the largest window wins.
"""

import argparse
import sys

try:
    import yaml
except ImportError:  # pragma: no cover - environment problem, not logic
    print("ERROR: PyYAML not installed (pip install pyyaml)", file=sys.stderr)
    sys.exit(2)

HEADROOM = 128


def load_configs(path):
    try:
        with open(path, encoding="utf-8") as fh:
            doc = yaml.safe_load(fh)
    except FileNotFoundError:
        print(
            f"ERROR: workload inventory not found: {path}\n"
            "       run scripts/setup-harness.sh first (it clones vllm-cpu-perf-eval)",
            file=sys.stderr,
        )
        sys.exit(1)
    configs = (doc or {}).get("test_configs")
    if not configs:
        print(f"ERROR: no 'test_configs' mapping in {path}", file=sys.stderr)
        sys.exit(1)
    return configs


def declared_len(cfg):
    """--max-model-len from the workload's vllm_args, or 0 if it sets none."""
    for arg in cfg.get("vllm_args") or []:
        if isinstance(arg, str) and arg.startswith("--max-model-len"):
            # both "--max-model-len=4096" and "--max-model-len 4096" appear
            value = arg.split("=", 1)[1] if "=" in arg else arg.split()[-1]
            try:
                return int(value)
            except ValueError:
                return 0
    return 0


def needed_len(cfg):
    isl = int(cfg.get("isl_max") or cfg.get("isl") or 0)
    osl = int(cfg.get("osl_max") or cfg.get("osl") or 0)
    return isl + osl + HEADROOM


def resolve(configs, names):
    missing = [n for n in names if n not in configs]
    if missing:
        print(
            f"ERROR: unknown workload(s): {', '.join(missing)}\n"
            f"       known: {', '.join(sorted(configs))}",
            file=sys.stderr,
        )
        sys.exit(1)
    return max(max(needed_len(configs[n]), declared_len(configs[n])) for n in names)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inventory", required=True, help="path to test-workloads.yml")
    ap.add_argument("--list", action="store_true", help="print the workload table")
    ap.add_argument("--max-model-len", metavar="W1,W2", help="comma-separated workloads")
    args = ap.parse_args()

    configs = load_configs(args.inventory)

    if args.list:
        print(f"{'workload':<22} {'isl':>6} {'osl':>6} {'declared':>9} {'max-model-len':>14}")
        for name in sorted(configs):
            cfg = configs[name]
            isl = cfg.get("isl_max") or cfg.get("isl") or 0
            osl = cfg.get("osl_max") or cfg.get("osl") or 0
            dec = declared_len(cfg)
            print(
                f"{name:<22} {isl:>6} {osl:>6} {dec or '-':>9} "
                f"{max(needed_len(cfg), dec):>14}"
            )
        return

    if args.max_model_len:
        names = [w.strip() for w in args.max_model_len.split(",") if w.strip()]
        if not names:
            print("ERROR: --max-model-len given an empty workload list", file=sys.stderr)
            sys.exit(2)
        print(resolve(configs, names))
        return

    ap.error("nothing to do: pass --list or --max-model-len")


if __name__ == "__main__":
    main()
