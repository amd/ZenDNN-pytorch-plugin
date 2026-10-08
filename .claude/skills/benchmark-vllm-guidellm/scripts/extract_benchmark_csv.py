#!/usr/bin/env python3

# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

"""Extract benchmark summary CSVs from guidellm benchmarks.json files."""

import argparse
import csv
import json
import os
import sys


def extract_benchmark(json_path, csv_path):
    with open(json_path) as f:
        data = json.load(f)

    rows = []
    rows.append([
        "Benchmark", "Request Latency", "", "TTFT", "", "ITL", "",
        "TPOT", "", "Output Throughput", "Total Throughput",
    ])
    rows.append([
        "Strategy", "Sec", "", "ms", "", "ms", "", "ms", "",
        "tok/s", "tok/s",
    ])
    rows.append([
        "", "Mdn", "p95", "Mdn", "p95", "Mdn", "p95", "Mdn", "p95", "", "",
    ])

    for b in data["benchmarks"]:
        strategy = b["config"]["strategy"]
        streams = strategy.get("streams") or strategy.get("max_concurrency") or ""
        label = f"{strategy['type_']} (streams={streams})"

        m = b["metrics"]
        s = "successful"
        duration = b["duration"]

        rl = m["request_latency"][s]
        ttft = m["time_to_first_token_ms"][s]
        itl = m["inter_token_latency_ms"][s]
        tpot = m["time_per_output_token_ms"][s]
        out_tps = m["output_token_count"][s]["total_sum"] / duration
        tot_tps = m["total_token_count"][s]["total_sum"] / duration

        rows.append([
            label,
            f"{rl['median']:.2f}", f"{rl['percentiles']['p95']:.2f}",
            f"{ttft['median']:.2f}", f"{ttft['percentiles']['p95']:.2f}",
            f"{itl['median']:.2f}", f"{itl['percentiles']['p95']:.2f}",
            f"{tpot['median']:.2f}", f"{tpot['percentiles']['p95']:.2f}",
            f"{out_tps:.2f}", f"{tot_tps:.2f}",
        ])

    with open(csv_path, "w", newline="") as csvf:
        csv.writer(csvf).writerows(rows)

    print(f"  Created: {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract benchmark summary CSVs from guidellm benchmarks.json files."
    )
    parser.add_argument(
        "results_dir",
        help="Top-level results directory containing chat-* subdirectories, "
             "or a single chat-* directory with external-endpoint/benchmarks.json",
    )
    args = parser.parse_args()

    results_dir = os.path.abspath(args.results_dir)

    # Check if the path itself contains benchmarks.json directly
    direct = os.path.join(results_dir, "external-endpoint", "benchmarks.json")
    if os.path.isfile(direct):
        dirs = [results_dir]
    else:
        dirs = sorted(
            os.path.join(results_dir, d)
            for d in os.listdir(results_dir)
            if os.path.isfile(os.path.join(results_dir, d, "external-endpoint", "benchmarks.json"))
        )

    if not dirs:
        print(f"No benchmarks.json found under {results_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(dirs)} benchmark dir(s) under {results_dir}\n")

    for d in dirs:
        json_path = os.path.join(d, "external-endpoint", "benchmarks.json")
        csv_path = os.path.join(d, "benchmark_summary.csv")
        print(f"Processing: {os.path.basename(d)}")
        extract_benchmark(json_path, csv_path)

    print(f"\nDone. {len(dirs)} CSV(s) generated.")


if __name__ == "__main__":
    main()
