#!/usr/bin/env python3

# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

"""Merge the per-rate GuideLLM artifacts of one run back into the single-run layout.

When ``GUIDELLM_MAX_SECONDS`` is given as a ``<rate>:<seconds>`` map,
``run_sweep.sh`` runs one GuideLLM invocation per rate so each concurrency level
gets its own benchmarking window. Each invocation writes ``benchmarks-<rate>.json``,
``benchmarks-<rate>.csv`` and ``guidellm-<rate>.log`` into the shared run
directory. Every downstream consumer -- the vllm-cpu-perf-eval dashboard
(``rglob("benchmarks.json")``), ``extract_benchmark_csv.py`` and the playbook's
``extract_benchmark_timings.py`` -- looks for the literal unsuffixed names, so
this script rebuilds them from the parts.

The merged ``benchmarks.json`` is byte-for-byte equivalent in shape to what a
single multi-rate invocation produces: one ``benchmarks`` entry per rate, in the
order the rates ran. The per-rate files are left in place as the source of truth.

Usage:
    python merge_guidellm_rates.py RUN_DIR [RUN_DIR ...]
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import re
import sys

BENCH_JSON_RE = re.compile(r"^benchmarks-(?P<rate>.+)\.json$")
BENCH_CSV_RE = re.compile(r"^benchmarks-(?P<rate>.+)\.csv$")
LOG_RE = re.compile(r"^guidellm-(?P<rate>.+)\.log$")


def _rate_sort_key(rate: str):
    """Sort numerically when the rate is a number, lexically otherwise."""
    try:
        return (0, float(rate), "")
    except ValueError:
        return (1, 0.0, rate)


def find_parts(directory: str, pattern: re.Pattern) -> list[tuple[str, str]]:
    """Return [(rate, path)] for files in *directory* matching *pattern*."""
    parts = []
    for name in os.listdir(directory):
        m = pattern.match(name)
        if m:
            parts.append((m.group("rate"), os.path.join(directory, name)))
    parts.sort(key=lambda p: _rate_sort_key(p[0]))
    return parts


def _override_from_strategy(benchmark: dict) -> dict:
    """Recover the per-benchmark profile override from a benchmark's strategy.

    A single-rate invocation carries no ``--override``, so its
    ``config.benchmarks`` entry is empty. The merged file needs one entry per
    benchmark to stay parallel with the ``benchmarks`` list, so reconstruct it
    from the strategy the run actually used.
    """
    strategy = benchmark.get("config", {}).get("strategy", {})
    if strategy.get("streams") is not None:
        return {"profile.streams": strategy["streams"]}
    if strategy.get("rate") is not None:
        return {"profile.rate": strategy["rate"]}
    return {}


def _max_duration(data: dict) -> str:
    """The max_duration constraint of one per-rate file, as a string."""
    for constraint in data.get("config", {}).get("spec", {}).get("constraints", []):
        if isinstance(constraint, dict) and constraint.get("kind") == "max_duration":
            return str(constraint.get("seconds", ""))
    return ""


def merge_json(parts: list[tuple[str, str]], out_path: str) -> int:
    """Concatenate the ``benchmarks`` arrays of *parts* into one file."""
    merged = None
    benchmarks: list = []
    overrides: list = []
    windows: list[str] = []
    rates: list[str] = []

    for rate, path in parts:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)

        if merged is None:
            merged = copy.deepcopy(data)

        per_file = data.get("benchmarks", [])
        per_file_overrides = data.get("config", {}).get("benchmarks") or []
        for i, benchmark in enumerate(per_file):
            benchmarks.append(benchmark)
            override = per_file_overrides[i] if i < len(per_file_overrides) else None
            if not isinstance(override, dict) or not override:
                override = _override_from_strategy(benchmark)
            overrides.append(override)

        rates.append(rate)
        windows.append(_max_duration(data))

    if merged is None:
        return 0

    merged["benchmarks"] = benchmarks
    config = merged.setdefault("config", {})
    config["benchmarks"] = overrides

    # config.spec holds a single set of constraints, so the merged file can only
    # carry the first rate's window there. Record the real per-rate windows as
    # labels rather than losing them.
    labels = config.setdefault("metadata", {}).setdefault("labels", {})
    labels["merged_rates"] = ",".join(rates)
    labels["merged_max_seconds"] = ",".join(windows)

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(merged, fh)

    return len(benchmarks)


def merge_csv(parts: list[tuple[str, str]], out_path: str) -> int:
    """Concatenate CSVs that share a header row.

    GuideLLM writes one header row followed by one row per benchmark. If the
    headers disagree the files are not the shape this assumes, so skip rather
    than produce a silently malformed table.
    """
    header = None
    rows: list[list[str]] = []

    for _rate, path in parts:
        with open(path, newline="", encoding="utf-8") as fh:
            file_rows = list(csv.reader(fh))
        if not file_rows:
            continue
        if header is None:
            header = file_rows[0]
        elif file_rows[0] != header:
            print(
                f"  skipping CSV merge: {os.path.basename(path)} has a different header",
                file=sys.stderr,
            )
            return 0
        rows.extend(file_rows[1:])

    if header is None:
        return 0

    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)

    return len(rows)


def _is_rule(line: str, char: str) -> bool:
    """Whether *line* is a table rule drawn with *char* (e.g. |----|----|)."""
    stripped = line.strip()
    return bool(stripped) and set(stripped) <= set("|+ " + char)


def _find_tables(lines: list[str]) -> list[dict]:
    """Locate the console tables in a GuideLLM log.

    Each table is a run of pipe-prefixed lines: a '=' border, three header
    rows, a '-' rule, the data rows, and a closing '=' border. The title is the
    nearest preceding text line ("i Request Latency Statistics (...)"), which is
    what identifies the same table across per-rate logs.
    """
    tables = []
    i = 0
    seen: dict[str, int] = {}

    while i < len(lines):
        if not lines[i].lstrip().startswith("|"):
            i += 1
            continue

        start = i
        while i < len(lines) and lines[i].lstrip().startswith("|"):
            i += 1
        end = i  # one past the last pipe line

        block = lines[start:end]
        rule = next(
            (n for n in range(len(block) - 1, -1, -1) if _is_rule(block[n], "-")),
            None,
        )
        border = next(
            (
                n
                for n in range(len(block) - 1, -1, -1)
                if _is_rule(block[n], "=") and rule is not None and n > rule
            ),
            None,
        )
        if rule is None or border is None:
            continue

        title = ""
        for n in range(start - 1, -1, -1):
            if lines[n].strip():
                title = lines[n].strip()
                break
        # A title can repeat within one log, so key on its occurrence too.
        key = f"{title}#{seen.get(title, 0)}"
        seen[title] = seen.get(title, 0) + 1

        tables.append(
            {
                "key": key,
                "rule": start + rule,
                "data_start": start + rule + 1,
                "data_end": start + border,
            }
        )

    return tables


def _cell_widths(rule_line: str) -> list[int]:
    """Column widths of a table, read off its |----|----| rule."""
    return [len(c) for c in rule_line.rstrip("\n").split("|")[1:-1]]


def _render_row(row: str, widths: list[int]) -> str:
    """Re-pad a data row to *widths* so appended rows line up with the table."""
    cells = [c.strip() for c in row.rstrip("\n").split("|")[1:-1]]
    if len(cells) != len(widths):
        return row.rstrip("\n") + "\n"
    out = []
    for i, cell in enumerate(cells):
        out.append(" " + cell.ljust(max(widths[i] - 2, len(cell))) + " ")
    return "|" + "|".join(out) + "|\n"


def merge_logs(parts: list[tuple[str, str]], out_path: str) -> tuple[int, list[str]]:
    """Rebuild one console log holding every rate's rows.

    The consumers of guidellm.log -- the perf dashboard's parser and
    extract_guidellm_metrics.py -- read the first occurrence of each summary
    table and stop at its closing border, so simply concatenating the per-rate
    logs would hand them the first rate only. Instead, take the first log that
    produced tables and splice the other rates' data rows into each of its
    tables, which is the shape a single multi-rate run writes.

    Returns (rates merged, rates whose log held no table).
    """
    parsed = []
    empty: list[str] = []
    for rate, path in parts:
        with open(path, encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
        tables = _find_tables(lines)
        if not tables:
            empty.append(rate)
            continue
        parsed.append((rate, lines, {t["key"]: t for t in tables}))

    if not parsed:
        return 0, empty

    # The first log with tables supplies the prose: the run config, the model
    # name and the CPU binding the dashboard also reads out of this file. Those
    # are identical across the per-rate runs.
    _base_rate, base_lines, base_tables = parsed[0]

    extra: dict[str, list[str]] = {}
    for _rate, lines, tables in parsed[1:]:
        for key, table in tables.items():
            extra.setdefault(key, []).extend(lines[table["data_start"]:table["data_end"]])

    # Rewrite back to front so the earlier insertion points stay valid.
    out_lines = list(base_lines)
    for table in sorted(base_tables.values(), key=lambda t: t["data_end"], reverse=True):
        rows = extra.get(table["key"])
        if not rows:
            continue
        widths = _cell_widths(base_lines[table["rule"]])
        out_lines[table["data_end"]:table["data_end"]] = [
            _render_row(row, widths) for row in rows
        ]

    with open(out_path, "w", encoding="utf-8") as out:
        out.writelines(out_lines)

    return len(parsed), empty


def merge_run_dir(directory: str) -> bool:
    json_parts = find_parts(directory, BENCH_JSON_RE)
    if not json_parts:
        print(f"No benchmarks-<rate>.json found in {directory}", file=sys.stderr)
        return False

    print(f"Merging {len(json_parts)} rate(s) in {directory}")

    count = merge_json(json_parts, os.path.join(directory, "benchmarks.json"))
    print(f"  benchmarks.json : {count} benchmark(s) from {[r for r, _ in json_parts]}")

    csv_parts = find_parts(directory, BENCH_CSV_RE)
    if csv_parts:
        rows = merge_csv(csv_parts, os.path.join(directory, "benchmarks.csv"))
        if rows:
            print(f"  benchmarks.csv  : {rows} row(s)")

    log_parts = find_parts(directory, LOG_RE)
    if log_parts:
        merged, empty = merge_logs(log_parts, os.path.join(directory, "guidellm.log"))
        print(f"  guidellm.log    : {merged} log(s)")
        if empty:
            # A rate whose run died before printing its summary tables. Its own
            # guidellm-<rate>.log still holds the failure.
            print(
                f"  WARNING: no summary table in the log of rate(s) {','.join(empty)}; "
                "they are missing from guidellm.log",
                file=sys.stderr,
            )

    return True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Merge per-rate GuideLLM outputs into benchmarks.json / .csv / guidellm.log."
    )
    parser.add_argument(
        "run_dir",
        nargs="+",
        help="Run directory holding benchmarks-<rate>.json (usually .../external-endpoint)",
    )
    args = parser.parse_args(argv)

    ok = True
    for directory in args.run_dir:
        if not os.path.isdir(directory):
            print(f"error: not a directory: {directory}", file=sys.stderr)
            ok = False
            continue
        ok = merge_run_dir(directory) and ok

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
