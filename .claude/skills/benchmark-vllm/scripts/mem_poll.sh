#!/usr/bin/env bash

# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

# Poll `podman/docker stats` for containers whose name starts with a given
# prefix and record the peak aggregate memory usage. Writes a CSV trace and a
# final PEAK line.
#
#   usage: mem_poll.sh <label> <out_csv> [interval_sec] [name_prefix]
#
# Stop it by deleting the flag file: <out_csv>.run
#
# Adapted from the parse-guidellm skill's mem_poll.sh; the container-name
# filter is a parameter here (default "bench-vllm") so it also covers the
# single-container quadrants started by this skill.
set -uo pipefail
LABEL="${1:?label}"
OUT="${2:?out csv}"
INTERVAL="${3:-2}"
PREFIX="${4:-bench-vllm}"
FLAG="${OUT}.run"

CR="podman"; command -v podman >/dev/null 2>&1 || CR="docker"

touch "$FLAG"
echo "ts,n_containers,agg_mem_bytes,agg_mem_human,per_container" > "$OUT"

to_bytes() {
  local v="$1" num unit
  num="${v//[^0-9.]/}"; unit="${v//[0-9.]/}"
  case "$unit" in
    B)         awk -v n="$num" 'BEGIN{printf "%.0f", n}';;
    kB|KB|KiB) awk -v n="$num" 'BEGIN{printf "%.0f", n*1024}';;
    MB|MiB)    awk -v n="$num" 'BEGIN{printf "%.0f", n*1024*1024}';;
    GB|GiB)    awk -v n="$num" 'BEGIN{printf "%.0f", n*1024*1024*1024}';;
    TB|TiB)    awk -v n="$num" 'BEGIN{printf "%.0f", n*1024*1024*1024*1024}';;
    *)         echo 0;;
  esac
}

peak=0; peak_human="0B"; peak_n=0
while [[ -f "$FLAG" ]]; do
  mapfile -t lines < <("$CR" stats --no-stream --format '{{.Name}} {{.MemUsage}}' 2>/dev/null | grep -E "^${PREFIX}" || true)
  agg=0; n=0; per=""
  for ln in "${lines[@]}"; do
    name="${ln%% *}"; rest="${ln#* }"
    used="${rest%% /*}"; used="${used// /}"
    b=$(to_bytes "$used")
    agg=$(( agg + b )); n=$(( n + 1 ))
    per="${per}${name}=${used};"
  done
  if (( n > 0 )); then
    human=$(awk -v b="$agg" 'BEGIN{ split("B KiB MiB GiB TiB",u," "); i=1; while(b>=1024 && i<5){b/=1024;i++} printf "%.2f%s", b, u[i] }')
    ts=$(date '+%s' 2>/dev/null || cut -d' ' -f1 /proc/uptime 2>/dev/null || echo 0)
    echo "${ts},${n},${agg},${human},${per}" >> "$OUT"
    if (( agg > peak )); then peak=$agg; peak_human=$human; peak_n=$n; fi
  fi
  sleep "$INTERVAL"
done

human=$(awk -v b="$peak" 'BEGIN{ split("B KiB MiB GiB TiB",u," "); i=1; while(b>=1024 && i<5){b/=1024;i++} printf "%.2f%s", b, u[i] }')
echo "PEAK label=${LABEL} containers=${peak_n} agg_mem_bytes=${peak} agg_mem_human=${human}" >> "$OUT"
echo "PEAK label=${LABEL} containers=${peak_n} agg_mem_human=${human}"
