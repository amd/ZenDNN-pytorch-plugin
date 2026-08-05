#!/usr/bin/env python3

# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

"""Extract headline guidellm metrics from a benchmarks.json.

usage: extract_perf.py <benchmarks.json>

Prints one line per benchmark (rate): rate, completed, req/s, out_tok/s,
TTFT_ms (median), ITL_ms (median), TPOT_ms (median).
"""

import json
import sys

HEADER = (
    f"{'rate':>6} {'completed':>10} {'req/s':>8} {'out_tok/s':>10} "
    f"{'TTFT_ms':>9} {'ITL_ms':>8} {'TPOT_ms':>8}"
)


def stat(metric, field="median"):
    """metric is a dict like {'successful': {'median':..}, ...} or {'median':..}."""
    if metric is None:
        return None
    if isinstance(metric, dict):
        # guidellm nests under 'successful' / 'total' sometimes
        for key in ("successful", "total", "all"):
            if isinstance(metric.get(key), dict) and field in metric[key]:
                return metric[key][field]
        if field in metric:
            return metric[field]
    return None


def fmt(value, precision=2):
    return f"{value:.{precision}f}" if isinstance(value, (int, float)) else "n/a"


def requested_rate(cfg):
    """The rate a benchmark was driven at, as guidellm recorded it."""
    strategy = cfg.get("strategy") or {}
    if not isinstance(strategy, dict):
        return strategy
    return (
        strategy.get("streams")
        or strategy.get("max_concurrency")
        or strategy.get("type_")
    )


def completed_requests(benchmark):
    """Count of requests guidellm considered done, whatever shape it used."""
    requests = benchmark.get("requests", {}) or {}
    for key in ("successful", "completed", "total"):
        value = requests.get(key)
        if isinstance(value, list):
            return len(value)
        if isinstance(value, (int, float)):
            return value
    return None


def main(path):
    with open(path, encoding="utf-8") as handle:
        doc = json.load(handle)

    print(HEADER)
    for benchmark in doc["benchmarks"]:
        rate = requested_rate(benchmark.get("config", {}) or {})
        metrics = benchmark.get("metrics", {}) or {}
        reqps = stat(metrics.get("requests_per_second"))
        outtps = stat(metrics.get("output_tokens_per_second"))
        ttft = stat(metrics.get("time_to_first_token_ms"))
        itl = stat(metrics.get("inter_token_latency_ms"))
        tpot = stat(metrics.get("time_per_output_token_ms"))
        completed = completed_requests(benchmark)
        print(
            f"{str(rate):>6} {str(completed):>10} {fmt(reqps):>8} "
            f"{fmt(outtps):>10} {fmt(ttft, 1):>9} {fmt(itl):>8} {fmt(tpot):>8}"
        )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: extract_perf.py <benchmarks.json>", file=sys.stderr)
        sys.exit(2)
    main(sys.argv[1])
