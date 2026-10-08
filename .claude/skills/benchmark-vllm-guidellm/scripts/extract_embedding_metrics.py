#!/usr/bin/env python3

# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

"""Extract key metrics from an embedding benchmark results tree and print CSV.

The embedding path of ``run_sweep.sh`` drives upstream's
``embedding-benchmark.yml``, which drives ``vllm bench serve``. Unlike guidellm
(which prints ASCII tables that ``extract_guidellm_metrics.py`` scrapes), vllm
bench writes one self-describing JSON object per measurement point:

    results/embedding/<model>/<test_name>-<timestamp>/
        test-metadata.json
        baseline/sweep-inf.json  sweep-25pct.json  sweep-50pct.json  sweep-75pct.json
        latency/concurrent-16.json  ... concurrent-196.json

Upstream ships ``scripts/conversion/convert_embedding_results.py``, which reads
the same files but has no notion of instance count or backend variant -- the two
axes a multi-instance sweep exists to measure. This script recovers both from the
run directory name (``EPYC<N>-<variant>-<tag>``, produced by run_sweep.sh) so the
rows from several sweeps can be concatenated and plotted against each other.

Usage:
    python3 extract_embedding_metrics.py results/embedding
    python3 extract_embedding_metrics.py results/embedding/*/EPYC8-* -o sweep.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys

OUTPUT_FIELDS = [
    "model",
    "backend",
    "instances",
    "cores_per_instance",
    "run",
    "scenario",
    "point",
    "request_rate",
    "max_concurrency",
    "num_prompts",
    "completed",
    "duration",
    "request_throughput",
    "total_token_throughput",
    "mean_e2el_ms",
    "median_e2el_ms",
    "p95_e2el_ms",
    "p99_e2el_ms",
]


def parse_path_meta(path: str):
    """Derive (model, backend, instances) from an embedding result path.

    Expected layout: .../results/embedding/<model>/<run-dir>/... where <run-dir>
    contains either 'native' or 'zentorch' plus an 'EPYC<N>' token giving the
    instance count (e.g. 'EPYC8-zentorch-granite-en-r2-20260804-051500' -> 8).

    Mirrors extract_guidellm_metrics.parse_path_meta, except the tree segment is
    'embedding' rather than 'llm'.
    """
    parts = os.path.normpath(os.path.abspath(path)).split(os.sep)

    model = ""
    if "embedding" in parts:
        idx = parts.index("embedding")
        if idx + 1 < len(parts):
            model = parts[idx + 1]

    backend = ""
    for seg in parts:
        low = seg.lower()
        if "zentorch" in low:
            backend = "zentorch"
            break
        if "native" in low:
            backend = "native"
            break

    instances = ""
    for seg in parts:
        m = re.search(r"EPYC(\d+)", seg, re.IGNORECASE)
        if m:
            instances = m.group(1)
            break

    return model, backend, instances


def parse_point(filename: str):
    """Map a result filename to a (scenario, point) pair.

    sweep-inf.json     -> ('baseline', 'inf')
    sweep-25pct.json   -> ('baseline', '25pct')
    concurrent-64.json -> ('latency', '64')
    """
    stem = os.path.basename(filename)[: -len(".json")]
    if stem.startswith("sweep-"):
        return "baseline", stem[len("sweep-") :]
    if stem.startswith("concurrent-"):
        return "latency", stem[len("concurrent-") :]
    return "unknown", stem


def _sort_key(scenario: str, point: str):
    """Order rows baseline-before-latency, then by load level."""
    if scenario == "baseline":
        # 'inf' is the reference max, so print it first.
        return (0, -1.0 if point == "inf" else float(re.sub(r"pct$", "", point)))
    if scenario == "latency":
        return (1, float(point) if point.isdigit() else 0.0)
    return (2, 0.0)


def find_run_dirs(roots: list[str]) -> list[str]:
    """Return the run directories under the given paths.

    A run directory is one holding a 'baseline' or 'latency' subdirectory. Paths
    that are already run directories are accepted as-is, so both
    'results/embedding' and a single 'results/embedding/<model>/<run>' work.
    """
    found: list[str] = []
    for root in roots:
        if not os.path.isdir(root):
            print(f"warning: not a directory: {root}", file=sys.stderr)
            continue
        if any(os.path.isdir(os.path.join(root, d)) for d in ("baseline", "latency")):
            found.append(root)
            continue
        for dirpath, dirnames, _ in os.walk(root):
            if any(d in dirnames for d in ("baseline", "latency")):
                found.append(dirpath)
                # A run directory never nests another one.
                dirnames[:] = []
    return sorted(set(found))


def read_metadata(run_dir: str) -> dict:
    path = os.path.join(run_dir, "test-metadata.json")
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError) as exc:
        print(f"warning: cannot read {path}: {exc}", file=sys.stderr)
        return {}


def collect_run(run_dir: str) -> list[dict]:
    model, backend, instances = parse_path_meta(run_dir)
    meta = read_metadata(run_dir)
    # The playbook records the per-instance core count run_sweep.sh passed in.
    cores = meta.get("requested_cores", "")
    # Prefer the model as vLLM reported it over the sanitized directory name.
    model = meta.get("model") or model.replace("__", "/")

    rows: list[dict] = []
    for scenario_dir in ("baseline", "latency"):
        d = os.path.join(run_dir, scenario_dir)
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            if not name.endswith(".json"):
                continue
            path = os.path.join(d, name)
            try:
                with open(path, encoding="utf-8") as fh:
                    result = json.load(fh)
            except (OSError, json.JSONDecodeError) as exc:
                print(f"warning: cannot read {path}: {exc}", file=sys.stderr)
                continue

            scenario, point = parse_point(name)
            rows.append(
                {
                    "model": model,
                    "backend": backend,
                    "instances": instances,
                    "cores_per_instance": cores if cores is not None else "",
                    "run": os.path.basename(run_dir),
                    "scenario": scenario,
                    "point": point,
                    "request_rate": result.get("request_rate", ""),
                    "max_concurrency": result.get("max_concurrency", ""),
                    "num_prompts": result.get("num_prompts", ""),
                    "completed": result.get("completed", ""),
                    "duration": result.get("duration", ""),
                    "request_throughput": result.get("request_throughput", ""),
                    "total_token_throughput": result.get("total_token_throughput", ""),
                    "mean_e2el_ms": result.get("mean_e2el_ms", ""),
                    "median_e2el_ms": result.get("median_e2el_ms", ""),
                    # p95 only exists when the run passed
                    # --metric-percentiles 95,99 (see vllm-cpu-perf-eval.patch).
                    "p95_e2el_ms": result.get("p95_e2el_ms", ""),
                    "p99_e2el_ms": result.get("p99_e2el_ms", ""),
                }
            )

    rows.sort(key=lambda r: _sort_key(r["scenario"], r["point"]))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Extract embedding benchmark metrics as CSV."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="results/embedding, a single model dir, or individual run dirs",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Write CSV here instead of stdout.",
    )
    args = parser.parse_args(argv)

    run_dirs = find_run_dirs(args.paths)
    if not run_dirs:
        print(
            "error: no run directories found (expected a 'baseline' or "
            "'latency' subdirectory somewhere under the given paths)",
            file=sys.stderr,
        )
        return 1

    rows: list[dict] = []
    for run_dir in run_dirs:
        rows.extend(collect_run(run_dir))

    if not rows:
        print("error: no result JSON files found", file=sys.stderr)
        return 2

    # Group the sweep so instance counts read in order within each scenario.
    rows.sort(
        key=lambda r: (
            r["model"],
            r["backend"],
            int(r["instances"]) if str(r["instances"]).isdigit() else 0,
            _sort_key(r["scenario"], r["point"]),
        )
    )

    if args.output:
        with open(args.output, "w", newline="", encoding="utf-8") as out:
            _write_csv(out, rows)
        print(f"wrote {len(rows)} rows to {args.output}", file=sys.stderr)
    else:
        _write_csv(sys.stdout, rows)
    return 0


def _write_csv(out, rows) -> None:
    writer = csv.DictWriter(out, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
