#!/usr/bin/env python3

# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

"""Extract key metrics from a guidellm console log and print them as CSV.

guidellm prints two relevant tables to the console (which people usually
redirect into a ``guidellm.log`` file):

  * "Request Latency Statistics"  -> Request Latency (Sec), TTFT (ms), TPOT (ms) ... each with Mdn + p95
  * "Server Throughput Statistics" -> Output Tokens/Sec, Total Tokens/Sec ... (Mean)

This script pulls the following per-benchmark values out of that log:

    req_latency_sec_p95   total_tokens_per_sec   output_tokens_per_sec   ttft_ms_p95   tpot_ms_p95

Usage:
    python extract_guidellm_metrics.py /path/to/guidellm.log
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

# Titles guidellm uses for the two tables we care about (matched as substrings).
LATENCY_TITLE = "Request Latency Statistics"
THROUGHPUT_TITLE = "Server Throughput Statistics"

OUTPUT_FIELDS = [
    "req_latency_sec_p95",
    "total_tokens_per_sec",
    "output_tokens_per_sec",
    "ttft_ms_p95",
    "tpot_ms_p95",
]


def _clean(line: str) -> str:
    return ANSI_RE.sub("", line).rstrip("\n")


def parse_path_meta(logfile: str):
    """Derive (model, backend, instances) from a guidellm result log path.

    Expected layout: .../results/llm/<model>/<run-dir>/.../guidellm.log where
    <run-dir> contains either 'native' or 'zentorch' plus an 'EPYC<N>' token
    giving the instance count (e.g.
    'chat-20260706-181257-EPYC7-native-llama31-8b' -> 7 instances).
    """
    parts = os.path.normpath(os.path.abspath(logfile)).split(os.sep)

    model = ""
    if "llm" in parts:
        idx = parts.index("llm")
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


def _is_divider(inner_cells: list[str]) -> str | None:
    """Return '=' or '-' if the row is a divider line, else None."""
    joined = "".join(inner_cells).replace(" ", "")
    if not joined:
        return None
    chars = set(joined)
    if chars <= {"="}:
        return "="
    if chars <= {"-"}:
        return "-"
    return None


def _split_row(line: str) -> list[str] | None:
    """Split a '|a|b|c|' row into stripped inner cells, or None if not a table row."""
    if "|" not in line:
        return None
    parts = line.split("|")
    # Drop the leading/trailing edge pieces produced by the surrounding '|'.
    inner = parts[1:-1]
    if not inner:
        return None
    return [c.strip() for c in inner]


def _collect_table_lines(lines: list[str], title_idx: int) -> list[list[str]]:
    """Collect the contiguous block of table rows following a title line."""
    rows: list[list[str]] = []
    i = title_idx + 1
    started = False
    while i < len(lines):
        cells = _split_row(lines[i])
        if cells is None:
            if started:
                break
            i += 1
            continue
        started = True
        rows.append(cells)
        i += 1
    return rows


def _forward_fill(cells: list[str]) -> list[str]:
    out: list[str] = []
    last = ""
    for c in cells:
        if c:
            last = c
        out.append(last)
    return out


def _parse_table(rows: list[list[str]]):
    """Parse a table block into (groups, units, value_rows).

    groups: per-column group name (forward filled top header row)
    units:  per-column units/stat label (last header row, e.g. 'p95', 'Mean')
    value_rows: list of per-column value lists (one per benchmark)
    """
    # Normalise column count across all rows.
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]

    # Find divider positions.
    dash_idx = None
    header_end_eq = None
    for idx, r in enumerate(rows):
        d = _is_divider(r)
        if d == "-" and dash_idx is None:
            dash_idx = idx
        elif d == "=" and dash_idx is not None and header_end_eq is None:
            header_end_eq = idx

    if dash_idx is None:
        return None

    header_rows = [r for r in rows[:dash_idx] if _is_divider(r) is None]
    value_rows = [
        r for r in rows[dash_idx + 1 :] if _is_divider(r) is None
    ]

    if not header_rows:
        return None

    # guidellm merges identical adjacent header cells (leaving blanks), so
    # forward-fill each header row to recover the per-column group/stat labels.
    groups = _forward_fill(header_rows[0])
    units = _forward_fill(header_rows[-1])  # last header row = stat label (Mdn/p95/Mean)
    return groups, units, value_rows


def _find_col(groups, units, group_name, unit_name):
    gname = group_name.lower()
    uname = unit_name.lower()
    for idx in range(len(groups)):
        u = units[idx] if idx < len(units) else ""
        if groups[idx].lower() == gname and u.lower() == uname:
            return idx
    return None


def _parse_all_tables(lines: list[str], title: str):
    """Parse every occurrence of a titled table and stack their value rows.

    A single guidellm run prints each table once, with one row per rate. A
    per-rate sweep instead concatenates one log per rate, so the same title
    appears several times with one row each; both collapse to the same result.
    """
    groups = units = None
    value_rows: list[list[str]] = []

    for idx, line in enumerate(lines):
        if title not in line:
            continue
        parsed = _parse_table(_collect_table_lines(lines, idx))
        if parsed is None:
            continue
        table_groups, table_units, rows = parsed
        if groups is None:
            groups, units = table_groups, table_units
        value_rows.extend(rows)

    if groups is None:
        return None
    return groups, units, value_rows


def parse_log(text: str):
    lines = [_clean(raw) for raw in text.splitlines()]

    latency = _parse_all_tables(lines, LATENCY_TITLE)
    throughput = _parse_all_tables(lines, THROUGHPUT_TITLE)

    if latency is None and throughput is None:
        raise ValueError(
            "Could not find the guidellm 'Request Latency Statistics' or "
            "'Server Throughput Statistics' tables in the log."
        )

    records: list[dict[str, str]] = []

    lat_groups = lat_units = lat_vals = None
    if latency is not None:
        lat_groups, lat_units, lat_vals = latency
    thr_groups = thr_units = thr_vals = None
    if throughput is not None:
        thr_groups, thr_units, thr_vals = throughput

    # Locate the columns of interest.
    c_reqlat = (
        _find_col(lat_groups, lat_units, "Request Latency", "p95")
        if latency
        else None
    )
    c_ttft = (
        _find_col(lat_groups, lat_units, "TTFT", "p95") if latency else None
    )
    c_tpot = (
        _find_col(lat_groups, lat_units, "TPOT", "p95") if latency else None
    )
    c_out = (
        _find_col(thr_groups, thr_units, "Output Tokens", "Mean")
        if throughput
        else None
    )
    c_tot = (
        _find_col(thr_groups, thr_units, "Total Tokens", "Mean")
        if throughput
        else None
    )

    n_rows = max(
        len(lat_vals) if lat_vals else 0,
        len(thr_vals) if thr_vals else 0,
    )

    def get(row, col):
        if row is None or col is None or col >= len(row):
            return ""
        return row[col]

    for i in range(n_rows):
        lrow = lat_vals[i] if lat_vals and i < len(lat_vals) else None
        trow = thr_vals[i] if thr_vals and i < len(thr_vals) else None
        benchmark = ""
        if lrow:
            benchmark = lrow[0]
        elif trow:
            benchmark = trow[0]

        records.append(
            {
                "benchmark": benchmark,
                "req_latency_sec_p95": get(lrow, c_reqlat),
                "total_tokens_per_sec": get(trow, c_tot),
                "output_tokens_per_sec": get(trow, c_out),
                "ttft_ms_p95": get(lrow, c_ttft),
                "tpot_ms_p95": get(lrow, c_tpot),
            }
        )

    return records


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Extract latency/throughput metrics from a guidellm log as CSV."
    )
    parser.add_argument("logfile", help="Path to the guidellm.log file")
    parser.add_argument(
        "--no-benchmark-col",
        action="store_true",
        help="Omit the leading 'benchmark' (strategy) column.",
    )
    args = parser.parse_args(argv)

    try:
        with open(args.logfile, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    except OSError as exc:
        print(f"error: cannot read {args.logfile}: {exc}", file=sys.stderr)
        return 1

    try:
        records = parse_log(text)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if not records:
        print("error: no benchmark rows found in log", file=sys.stderr)
        return 3

    model, backend, instances = parse_path_meta(args.logfile)
    for rec in records:
        rec["model"] = model
        rec["backend"] = backend
        rec["instances"] = instances

    fields = ["model", "backend", "instances"]
    if not args.no_benchmark_col:
        fields.append("benchmark")
    fields.extend(OUTPUT_FIELDS)

    writer = csv.DictWriter(sys.stdout, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for rec in records:
        writer.writerow(rec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
