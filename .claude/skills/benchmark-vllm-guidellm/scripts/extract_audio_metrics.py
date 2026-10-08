#!/usr/bin/env python3

# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

"""Extract key metrics from an audio benchmark results tree and print CSV.

The audio path of ``run_sweep.sh`` drives upstream's ``audio-benchmark.yml`` in
external mode against the nginx-fronted stack, which drives GuideLLM's audio
backend against ``/v1/audio/transcriptions``. GuideLLM writes one self-describing
JSON object per scenario, with one entry per stage:

    results/audio-models/<model>/<scenario>-<test_name>-<timestamp>/
        test-metadata.json
        <stage>/benchmarks.json   (e.g. sequential/, concurrent-2/, ...)
        <stage>/stage-metadata.json

Upstream ships ``scripts/ansible/audio_enterprise_report.py`` (and the shared
``dashboard-examples/vllm_dashboard/audio_enterprise.py``), which read the same
files but have no notion of instance count or backend variant -- the two axes a
multi-instance sweep exists to measure. This script recovers both from the run
directory name (``EPYC<N>-<variant>-<tag>``, threaded into ``test_run_id`` by the
audio-benchmark.yml patch) so the rows from several sweeps can be concatenated
and plotted against each other. It is stdlib-only (no numpy/pandas) so it runs
under the ``/usr/bin/python3`` the benchmark nodes pin ansible to; it mirrors
``extract_embedding_metrics.parse_path_meta``.

Usage:
    python3 extract_audio_metrics.py results/audio-models
    python3 extract_audio_metrics.py results/audio-models/*/EPYC8-* -o sweep.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys

OUTPUT_FIELDS = [
    "model",
    "backend",
    "instances",
    "cores",
    "run",
    "scenario",
    "stage",
    "concurrency",
    "requests_per_second",
    "total_requests",
    "successful",
    "errored",
    "duration",
    "e2e_mean_ms",
    "e2e_p50_ms",
    "e2e_p95_ms",
    "e2e_p99_ms",
    "total_audio_seconds",
    "audio_throughput",
    "rtf_mean",
    "rtf_p95",
]


def parse_path_meta(path: str):
    """Derive (model, backend, instances) from an audio result path.

    Expected layout: .../results/audio-models/<model>/<run-dir>/... where
    <run-dir> contains either 'native' or 'zentorch' plus an 'EPYC<N>' token
    giving the instance count (e.g.
    'transcription-throughput-EPYC8-zentorch-whisper-small-20260804-051500' -> 8).

    Mirrors extract_embedding_metrics.parse_path_meta, except the tree segment is
    'audio-models' rather than 'embedding'.
    """
    parts = os.path.normpath(os.path.abspath(path)).split(os.sep)

    model = ""
    if "audio-models" in parts:
        idx = parts.index("audio-models")
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


def _percentile(values: list, pct: float) -> float:
    """Linear-interpolated percentile, matching numpy.percentile's default."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (pct / 100.0) * (len(ordered) - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return float(ordered[low])
    return float(ordered[low] + (ordered[high] - ordered[low]) * (rank - low))


def calculate_audio_aggregates(successful_requests: list, duration: float) -> dict:
    """Audio-specific aggregates from a request list (stdlib port of the
    dashboard's audio_enterprise.calculate_audio_aggregates)."""
    zeros = {
        "total_audio_seconds": 0.0,
        "audio_throughput": 0.0,
        "rtf_mean": 0.0,
        "rtf_p95": 0.0,
    }
    if not successful_requests:
        return zeros

    audio_seconds_list: list = []
    rtf_list: list = []
    for req in successful_requests:
        inp = req.get("input_metrics", {}) or {}
        audio_sec = inp.get("audio_seconds", 0)
        latency = req.get("request_latency")
        if audio_sec and audio_sec > 0:
            audio_seconds_list.append(audio_sec)
            if latency is not None and latency > 0:
                rtf_list.append(latency / audio_sec)

    total_audio = sum(audio_seconds_list)
    return {
        "total_audio_seconds": total_audio,
        "audio_throughput": (total_audio / duration if duration and duration > 0 else 0.0),
        "rtf_mean": (sum(rtf_list) / len(rtf_list) if rtf_list else 0.0),
        "rtf_p95": _percentile(rtf_list, 95),
    }


def find_run_dirs(roots: list) -> list:
    """Return the run directories under the given paths.

    A run directory is one holding a 'test-metadata.json' with at least one
    'benchmarks.json' somewhere below it. Paths that are already run directories
    are accepted as-is, so both 'results/audio-models' and a single
    'results/audio-models/<model>/<run>' work.
    """
    found: list = []
    for root in roots:
        if not os.path.isdir(root):
            print(f"warning: not a directory: {root}", file=sys.stderr)
            continue
        if os.path.isfile(os.path.join(root, "test-metadata.json")):
            found.append(root)
            continue
        for dirpath, _dirnames, filenames in os.walk(root):
            if "test-metadata.json" in filenames:
                found.append(dirpath)
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


def _ms(value) -> str:
    """Latency values in benchmarks.json are seconds; report milliseconds."""
    if value is None or value == "":
        return ""
    try:
        return round(float(value) * 1000.0, 2)
    except (TypeError, ValueError):
        return ""


def collect_run(run_dir: str) -> list:
    model, backend, instances = parse_path_meta(run_dir)
    meta = read_metadata(run_dir)
    cores = meta.get("core_count", "")
    scenario = meta.get("scenario_name", meta.get("scenario", ""))
    # Prefer the model vLLM reported over the sanitized directory name.
    model = meta.get("model") or model.replace("__", "/")

    rows: list = []
    for dirpath, _dirnames, filenames in os.walk(run_dir):
        if "benchmarks.json" not in filenames:
            continue
        path = os.path.join(dirpath, "benchmarks.json")
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"warning: cannot read {path}: {exc}", file=sys.stderr)
            continue

        stage = os.path.basename(dirpath)
        for bench in data.get("benchmarks", []):
            metrics = bench.get("metrics", {})
            config = bench.get("config", {})
            requests = bench.get("requests", {})
            duration = bench.get("duration", 0) or 0

            concurrency = config.get("strategy", {}).get("max_concurrency", "")
            rps = (
                metrics.get("requests_per_second", {})
                .get("successful", {})
                .get("mean", "")
            )
            totals = metrics.get("request_totals", {})
            latency = metrics.get("request_latency", {}).get("successful", {})
            pcts = latency.get("percentiles", {})

            agg = calculate_audio_aggregates(requests.get("successful", []), duration)

            rows.append(
                {
                    "model": model,
                    "backend": backend,
                    "instances": instances,
                    "cores": cores if cores is not None else "",
                    "run": os.path.basename(run_dir),
                    "scenario": scenario,
                    "stage": stage,
                    "concurrency": concurrency,
                    "requests_per_second": rps,
                    "total_requests": totals.get("total", ""),
                    "successful": totals.get("successful", ""),
                    "errored": totals.get("errored", ""),
                    "duration": round(duration, 2) if duration else duration,
                    "e2e_mean_ms": _ms(latency.get("mean")),
                    "e2e_p50_ms": _ms(pcts.get("p50")),
                    "e2e_p95_ms": _ms(pcts.get("p95")),
                    "e2e_p99_ms": _ms(pcts.get("p99")),
                    "total_audio_seconds": round(agg["total_audio_seconds"], 2),
                    "audio_throughput": round(agg["audio_throughput"], 3),
                    "rtf_mean": round(agg["rtf_mean"], 4),
                    "rtf_p95": round(agg["rtf_p95"], 4),
                }
            )

    rows.sort(key=lambda r: r["stage"])
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Extract audio benchmark metrics as CSV."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="results/audio-models, a single model dir, or individual run dirs",
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
            "error: no run directories found (expected a 'test-metadata.json' "
            "somewhere under the given paths)",
            file=sys.stderr,
        )
        return 1

    rows: list = []
    for run_dir in run_dirs:
        rows.extend(collect_run(run_dir))

    if not rows:
        print("error: no benchmarks.json result files found", file=sys.stderr)
        return 2

    # Group the sweep so instance counts read in order within each scenario.
    rows.sort(
        key=lambda r: (
            r["model"],
            r["backend"],
            int(r["instances"]) if str(r["instances"]).isdigit() else 0,
            r["scenario"],
            r["stage"],
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
