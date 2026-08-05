#!/usr/bin/env python3

# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

"""Parse scores from a guidellm.log file (the ASCII summary tables).

usage: parse_guidellm_log.py <guidellm.log>

Pulls per-strategy rows from:
  - Server Throughput Statistics (concurrency, req/s, in tok/s, out tok/s, total tok/s)
  - Request Latency Statistics  (latency sec, TTFT ms, ITL ms, TPOT ms; medians)

Strategies appear in benchmark order (concurrency 32 then 64).
"""

import re
import sys

STRATEGY_ROW = re.compile(r"\|\s*(concurrent|synchronous|throughput|constant|poisson)")

HEADER = (
    f"{'conc':>5} {'req/s':>7} {'in_tok/s':>9} {'out_tok/s':>10} "
    f"{'tot_tok/s':>10} {'lat_s':>7} {'TTFT_ms':>9} {'ITL_ms':>7} {'TPOT_ms':>8}"
)


def clean(line):
    """Split an ASCII table row "| a | b | c |" -> ['a','b','c']."""
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def find_section(lines, title):
    for i, line in enumerate(lines):
        if title in line:
            return i
    return None


def data_rows(lines, start):
    """Yield cleaned data rows of the table beginning after `start`, i.e. rows
    that start with '| concurrent' (the strategy name)."""
    rows = []
    for line in lines[start:]:
        stripped = line.strip()
        if stripped.startswith("|") and STRATEGY_ROW.match(stripped):
            rows.append(clean(line))
        elif rows and stripped.startswith(("ℹ", "✔")):
            break
    return rows


def cell(row, index):
    return row[index] if len(row) > index else "?"


def main(path):
    with open(path, encoding="utf-8", errors="replace") as handle:
        lines = handle.read().splitlines()

    # Columns: Strategy | Conc Mdn | Conc Mean | Req/s Mean | In tok/s | Out tok/s | Total tok/s
    thr_start = find_section(lines, "Server Throughput Statistics")
    thr = data_rows(lines, thr_start) if thr_start is not None else []

    # Columns: Strategy | Lat Mdn | Lat p95 | TTFT Mdn | TTFT p95 | ITL Mdn | ITL p95 | TPOT Mdn | TPOT p95
    lat_start = find_section(lines, "Request Latency Statistics")
    lat = data_rows(lines, lat_start) if lat_start is not None else []

    print(HEADER)
    for i in range(max(len(thr), len(lat))):
        t = thr[i] if i < len(thr) else []
        m = lat[i] if i < len(lat) else []
        print(
            f"{cell(t, 1):>5} {cell(t, 3):>7} {cell(t, 4):>9} {cell(t, 5):>10} "
            f"{cell(t, 6):>10} {cell(m, 1):>7} {cell(m, 3):>9} {cell(m, 5):>7} "
            f"{cell(m, 7):>8}"
        )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: parse_guidellm_log.py <guidellm.log>", file=sys.stderr)
        sys.exit(2)
    main(sys.argv[1])
