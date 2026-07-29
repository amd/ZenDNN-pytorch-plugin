#!/usr/bin/env python3

# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

"""Parse offline `vllm bench throughput` result files into a summary CSV.

Walks a results directory that contains one subdirectory per model, each holding
`result_inst<INST>_<RUN_TAG>.txt` files written by offline_launcher.sh, and
emits `throughput_summary.csv` plus a console summary. Files from different
sweep variants (different N/CPI/batch/len/algo) in the same model directory are
grouped separately, not summed together.

Adapted from ZenDNN_tools/.../offline/offline_parser.py.

    usage: parse_offline.py <results_dir>
"""
import csv
import os
import re
import sys
from collections import defaultdict

# Newer vLLM: "Throughput: ... N total tokens/s, M output tokens/s"
THROUGHPUT_PATTERN = re.compile(
    r"Throughput:.*?([\d\.]+)\s+total tokens/s,\s*([\d\.]+)\s+output tokens/s"
)
# Older vLLM builds print a single figure: "Throughput: X tokens/s" (or
# "requests/s, X tokens/s"). Used as a fallback so a valid run is not silently
# reported as 0 when pointed at an older/newer --image.
THROUGHPUT_FALLBACK = re.compile(r"Throughput:.*?([\d\.]+)\s+tokens/s")

RESULT_FILENAME = re.compile(
    r"^result_inst(?P<inst>\d+)"
    r"(?:_N(?P<n>\d+)_C(?P<cpi>\d+)_bs(?P<bs>\d+)"
    r"_I(?P<i>\d+)_O(?P<o>\d+)_p(?P<p>\d+)"
    r"(?:_algo(?P<algo>[A-Za-z0-9._-]+))?)?"
    r"\.txt$"
)

SKIP_DIRS = {"lsf_logs"}
LEGACY_KEY = (0, 0, 0, 0, 0, 0, "legacy")


def extract_throughput(path):
    total = output = 0.0
    matched = False
    try:
        with open(path, "r", errors="ignore") as f:
            for line in f:
                m = THROUGHPUT_PATTERN.search(line)
                if m:
                    total += float(m.group(1))
                    output += float(m.group(2))
                    matched = True
                    continue
                fm = THROUGHPUT_FALLBACK.search(line)
                if fm:
                    # Single-number format: treat it as total; output unknown.
                    total += float(fm.group(1))
                    matched = True
    except OSError as e:
        print(f"  Error reading {path}: {e}")
        return total, output, False
    return total, output, matched


def group_result_files(model_dir):
    groups = defaultdict(list)
    unmatched = []
    for fname in sorted(os.listdir(model_dir)):
        if not fname.endswith(".txt"):
            continue
        m = RESULT_FILENAME.match(fname)
        if not m:
            unmatched.append(fname)
            continue
        if m.group("n") is None:
            key = LEGACY_KEY
        else:
            key = (
                int(m.group("n")), int(m.group("cpi")), int(m.group("bs")),
                int(m.group("i")), int(m.group("o")), int(m.group("p")),
                m.group("algo") or "none",
            )
        groups[key].append(fname)
    return groups, unmatched


def render_tag(key):
    if key == LEGACY_KEY:
        return "(legacy, no run tag)"
    n, cpi, bs, i_len, o_len, p, algo = key
    return f"N{n}_C{cpi}_bs{bs}_I{i_len}_O{o_len}_p{p}_algo{algo}"


def main(results_dir):
    if not os.path.isdir(results_dir):
        print(f"Invalid directory: {results_dir}")
        sys.exit(1)

    model_dirs = sorted(
        e for e in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, e)) and e not in SKIP_DIRS
    )
    if not model_dirs:
        print(f"No model subdirectories found in {results_dir}")
        return

    rows = []
    for model_name in model_dirs:
        model_path = os.path.join(results_dir, model_name)
        groups, unmatched = group_result_files(model_path)
        if unmatched:
            print(f"  [{model_name}] WARNING: ignoring unrecognized: {unmatched}")
        if not groups:
            continue
        print(f"  [{model_name}]")
        for key, files in sorted(groups.items()):
            tag = render_tag(key)
            sum_t = sum_o = 0.0
            per = []
            no_tp = []
            for fname in sorted(files):
                t, o, matched = extract_throughput(os.path.join(model_path, fname))
                per.append((fname, round(t, 2), round(o, 2)))
                sum_t += t
                sum_o += o
                if not matched:
                    no_tp.append(fname)
            n_files = len(files)
            avg_t = sum_t / n_files if n_files else 0.0
            avg_o = sum_o / n_files if n_files else 0.0
            print(f"    [{tag}] ({n_files} file(s))")
            for fname, t, o in per:
                print(f"      {fname}: total={t} tok/s, output={o} tok/s")
            print(
                f"      SUM: total={round(sum_t, 2)} tok/s, output={round(sum_o, 2)} tok/s "
                f"(avg/inst: total={round(avg_t, 2)}, output={round(avg_o, 2)})"
            )
            if no_tp:
                print(
                    f"      WARNING: no 'Throughput:' line in {len(no_tp)} file(s): "
                    f"{no_tp} -- the run may have failed (e.g. HF Hub unreachable "
                    f"with a cached model; try --hf-offline) or the vLLM build uses "
                    f"an unrecognized throughput format."
                )
            if key == LEGACY_KEY:
                n_c = cpi_c = bs_c = i_c = o_c = p_c = algo_c = ""
            else:
                n_c, cpi_c, bs_c, i_c, o_c, p_c, algo_c = key
            rows.append((
                model_name, n_c, cpi_c, bs_c, i_c, o_c, p_c, algo_c, n_files,
                round(sum_t, 2), round(sum_o, 2), round(avg_t, 2), round(avg_o, 2),
            ))

    if not rows:
        print("No results to summarize.")
        return

    csv_path = os.path.join(results_dir, "throughput_summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "Model", "Instances (N)", "Cores Per Instance", "Batch Size (bs)",
            "Input Tokens (I)", "Output Tokens (O)", "Num Prompts (p)",
            "ZenDNN Algo", "Result Files", "Sum Total Tokens/s",
            "Sum Output Tokens/s", "Avg Total Tokens/s per Instance",
            "Avg Output Tokens/s per Instance",
        ])
        w.writerows(rows)
    print(f"\nCSV saved to: {csv_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: parse_offline.py <results_dir>")
        sys.exit(1)
    main(sys.argv[1])
