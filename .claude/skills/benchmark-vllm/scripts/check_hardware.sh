#!/usr/bin/env bash

# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

#
# check_hardware.sh -- detect CPU topology via lscpu, validate a requested
# (N instances x CPI cores-per-instance) layout against the machine, and, if
# it doesn't fit, recommend a config that does.
#
# Two ways to call it:
#
#   1. Topology only (print a summary + a recommendation):
#        ./check_hardware.sh
#
#   2. Validate a specific request and emit machine-readable results:
#        ./check_hardware.sh --mode multi --instances 4 --cpi 32
#        ./check_hardware.sh --mode single --cpi 64
#
# Human-readable summary goes to STDERR. Machine-readable KEY=VALUE lines go to
# STDOUT so a caller can capture them, e.g.:
#        eval "$(./check_hardware.sh --mode multi -n 4 -c 32 --quiet)"
#        echo "$CPUSET $FITS"
#
# Emitted keys: PHYS_CORES, LOGICAL_CPUS, SOCKETS, THREADS_PER_CORE,
# NUMA_NODES, NODE0_CORES, REQ_N, REQ_CPI, FITS (yes|no), CPUSET (when FITS=yes),
# REC_N, REC_CPI (recommended layout).

set -uo pipefail

MODE=""          # single | multi | (empty = topology only)
REQ_N=""
REQ_CPI=""
QUIET=0
EXCLUDE=""       # cpuset to remove from the physical-core pool (e.g. nginx cores)
SLICES=0         # when 1, emit per-instance SLICE_i cpusets from the physical list
RESERVE_TAIL=0   # reserve the last N physical cores (e.g. for a host GuideLLM client)

usage() {
  cat >&2 <<EOF
Usage: $0 [--mode single|multi] [--instances N] [--cpi C] [--quiet] [--exclude SET] [--slices] [--reserve-tail N]
  --mode single    force N=1 (only --cpi is used)
  --mode multi     N instances of C cores each
  -n, --instances N
  -c, --cpi C
  --exclude SET    cpuset removed from the physical-core pool before sizing/slicing
                   (e.g. reserve nginx/host cores: --exclude 0-7)
  --reserve-tail N reserve the LAST N physical cores (after --exclude) out of the
                   pool and emit them as TAIL_CPUSET (e.g. pin a host GuideLLM
                   client off the vLLM cores). vLLM sizing/slicing uses what is
                   left after this reservation.
  --slices         also emit per-instance SLICE_1..SLICE_N cpusets (physical cores)
  --quiet          suppress the human summary on stderr
EOF
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)          MODE="$2"; shift 2 ;;
    -n|--instances)  REQ_N="$2"; shift 2 ;;
    -c|--cpi)        REQ_CPI="$2"; shift 2 ;;
    --exclude)       EXCLUDE="$2"; shift 2 ;;
    --reserve-tail)  RESERVE_TAIL="$2"; shift 2 ;;
    --slices)        SLICES=1; shift ;;
    --quiet)         QUIET=1; shift ;;
    -h|--help)       usage ;;
    *) echo "Unknown arg: $1" >&2; usage ;;
  esac
done

log() { [[ "$QUIET" == "1" ]] || echo "$@" >&2; }

command -v lscpu >/dev/null 2>&1 || { echo "ERROR: lscpu not found." >&2; exit 1; }

# --- scalar topology from lscpu ---
CORES_PER_SOCKET=$(lscpu | awk -F: '/^Core\(s\) per socket:/ {gsub(/^[ \t]+/,"",$2); print $2; exit}')
SOCKETS=$(lscpu | awk -F: '/^Socket\(s\):/ {gsub(/^[ \t]+/,"",$2); print $2; exit}')
THREADS_PER_CORE=$(lscpu | awk -F: '/^Thread\(s\) per core:/ {gsub(/^[ \t]+/,"",$2); print $2; exit}')
LOGICAL_CPUS=$(lscpu | awk -F: '/^CPU\(s\):/ {gsub(/^[ \t]+/,"",$2); print $2; exit}')
NUMA_NODES=$(lscpu | awk -F: '/^NUMA node\(s\):/ {gsub(/^[ \t]+/,"",$2); print $2; exit}')
CORES_PER_SOCKET="${CORES_PER_SOCKET:-0}"
SOCKETS="${SOCKETS:-1}"
THREADS_PER_CORE="${THREADS_PER_CORE:-1}"
NUMA_NODES="${NUMA_NODES:-1}"
PHYS_CORES=$(( CORES_PER_SOCKET * SOCKETS ))

# --- build the ordered list of physical-core logical-CPU ids ---
# One logical CPU per physical core (the first thread seen for each
# socket+core), so pinning avoids SMT siblings. Uses `lscpu -p`.
mapfile -t PHYS_CPUS < <(
  lscpu -p=CPU,CORE,SOCKET 2>/dev/null | awk -F, '
    /^#/ {next}
    { key=$3"-"$2; if (!(key in seen)) { seen[key]=1; print $1 } }
  '
)
# Fallback: if lscpu -p produced nothing, assume 0..PHYS_CORES-1.
if [[ "${#PHYS_CPUS[@]}" -eq 0 && "$PHYS_CORES" -gt 0 ]]; then
  for ((i=0;i<PHYS_CORES;i++)); do PHYS_CPUS+=("$i"); done
fi
# Correct PHYS_CORES to the actual detected count if they disagree.
if [[ "${#PHYS_CPUS[@]}" -gt 0 ]]; then PHYS_CORES="${#PHYS_CPUS[@]}"; fi

# Expand a cpuset string ("0-7,16" -> flat ids).
expand_cpuset_ids() {
  local list="$1" part lo hi i
  [[ -z "$list" ]] && return 0
  IFS=',' read -ra _parts <<< "$list"
  for part in "${_parts[@]}"; do
    if [[ "$part" == *-* ]]; then lo="${part%-*}"; hi="${part#*-}"; for ((i=lo;i<=hi;i++)); do echo "$i"; done
    else echo "$part"; fi
  done
}

# AVAIL_CPUS = physical cores minus any --exclude set (e.g. nginx/host reserve).
declare -A _excl=()
if [[ -n "$EXCLUDE" ]]; then
  while read -r _e; do [[ -n "$_e" ]] && _excl["$_e"]=1; done < <(expand_cpuset_ids "$EXCLUDE")
fi
AVAIL_CPUS=()
for _c in "${PHYS_CPUS[@]}"; do
  [[ -n "${_excl[$_c]:-}" ]] && continue
  AVAIL_CPUS+=("$_c")
done
AVAIL_CORES="${#AVAIL_CPUS[@]}"

# node0 physical core count (used for NUMA-aware recommendation)
NODE0_LINE=$(lscpu | grep -m1 "NUMA node0 CPU(s):" | awk -F: '{gsub(/^[ \t]+/,"",$2); print $2}')
node0_count() {
  local list="$1" part lo hi c=0
  [[ -z "$list" ]] && { echo 0; return; }
  IFS=',' read -ra parts <<< "$list"
  for part in "${parts[@]}"; do
    if [[ "$part" == *-* ]]; then lo="${part%-*}"; hi="${part#*-}"; c=$((c + hi - lo + 1)); else c=$((c+1)); fi
  done
  echo "$c"
}
NODE0_LOGICAL=$(node0_count "$NODE0_LINE")
# physical cores per node (divide out SMT)
if [[ "$THREADS_PER_CORE" -gt 0 && "$NODE0_LOGICAL" -gt 0 ]]; then
  NODE0_CORES=$(( NODE0_LOGICAL / THREADS_PER_CORE ))
else
  NODE0_CORES=$(( PHYS_CORES / (NUMA_NODES>0?NUMA_NODES:1) ))
fi
[[ "$NODE0_CORES" -le 0 ]] && NODE0_CORES=$PHYS_CORES

# --- recommendation ---
# Prefer one instance per NUMA node with CPI = node core count (keeps each
# instance on a single NUMA node). Fall back to CPI=64 if node data is odd.
REC_CPI=$NODE0_CORES
[[ "$REC_CPI" -le 0 ]] && REC_CPI=64
if (( REC_CPI > PHYS_CORES )); then REC_CPI=$PHYS_CORES; fi
REC_N=$(( PHYS_CORES / REC_CPI ))
[[ "$REC_N" -le 0 ]] && REC_N=1

# --- compress an explicit list of CPU ids into a cpuset string ---
compress_ids() {
  local -a sel
  mapfile -t sel < <(printf '%s\n' "$@" | sort -n)
  local out="" run_lo="" prev="" x
  for x in "${sel[@]}"; do
    if [[ -z "$prev" ]]; then run_lo="$x"; prev="$x"; continue; fi
    if (( x == prev + 1 )); then prev="$x"; continue; fi
    if [[ "$run_lo" == "$prev" ]]; then out+="${out:+,}${run_lo}"; else out+="${out:+,}${run_lo}-${prev}"; fi
    run_lo="$x"; prev="$x"
  done
  if [[ -n "$prev" ]]; then
    if [[ "$run_lo" == "$prev" ]]; then out+="${out:+,}${run_lo}"; else out+="${out:+,}${run_lo}-${prev}"; fi
  fi
  echo "$out"
}

# compress the first <count> AVAILABLE physical cores into a cpuset string.
compress_cpuset() {
  local count="$1"
  compress_ids "${AVAIL_CPUS[@]:0:count}"
}

# --- reserve tail cores (e.g. for a host GuideLLM client) ---
# Carve the LAST RESERVE_TAIL physical cores out of AVAIL_CPUS and expose them
# as TAIL_CPUSET. vLLM sizing/slicing below then only sees the remaining cores.
TAIL_CPUSET=""
if [[ "$RESERVE_TAIL" =~ ^[0-9]+$ ]] && (( RESERVE_TAIL > 0 )); then
  if (( ${#AVAIL_CPUS[@]} > RESERVE_TAIL )); then
    _tail_ids=( "${AVAIL_CPUS[@]: -RESERVE_TAIL}" )
    TAIL_CPUSET="$(compress_ids "${_tail_ids[@]}")"
    AVAIL_CPUS=( "${AVAIL_CPUS[@]:0:${#AVAIL_CPUS[@]}-RESERVE_TAIL}" )
    AVAIL_CORES="${#AVAIL_CPUS[@]}"
    log "  Reserved tail cores for host client: $TAIL_CPUSET ($AVAIL_CORES cores left for vLLM)"
  else
    log "  WARNING: --reserve-tail $RESERVE_TAIL >= available cores ($AVAIL_CORES); not reserving a tail."
  fi
fi

# --- summary ---
log "==================== hardware ===================="
log "  Sockets:            $SOCKETS"
log "  Cores/socket:       $CORES_PER_SOCKET"
log "  Physical cores:     $PHYS_CORES"
log "  Logical CPUs:       $LOGICAL_CPUS"
log "  Threads/core (SMT): $THREADS_PER_CORE"
log "  NUMA nodes:         $NUMA_NODES"
log "  Cores per NUMA:     $NODE0_CORES (node0)"
log "  Recommended:        N=$REC_N x CPI=$REC_CPI  (one instance per NUMA node)"
log "=================================================="

# --- normalize the request ---
FITS="n/a"
CPUSET=""
SLICE_LINES=()
if [[ -n "$MODE" ]]; then
  if [[ "$MODE" == "single" ]]; then REQ_N=1; fi
  [[ -z "$REQ_N"   ]] && REQ_N="$REC_N"
  [[ -z "$REQ_CPI" ]] && REQ_CPI="$REC_CPI"

  # H1: N and CPI must be strictly positive integers (reject 0).
  if ! [[ "$REQ_N" =~ ^[0-9]+$ && "$REQ_CPI" =~ ^[0-9]+$ ]] || (( REQ_N <= 0 )) || (( REQ_CPI <= 0 )); then
    echo "ERROR: --instances and --cpi must be positive integers (> 0)." >&2
    exit 2
  fi

  NEED=$(( REQ_N * REQ_CPI ))
  POOL_NOTE=""
  [[ -n "$EXCLUDE" ]] && POOL_NOTE=" (of $AVAIL_CORES available after excluding '$EXCLUDE')"
  if (( NEED <= AVAIL_CORES )); then
    FITS="yes"
    CPUSET="$(compress_cpuset "$NEED")"
    log "  Request N=$REQ_N x CPI=$REQ_CPI -> $NEED cores${POOL_NOTE}: FITS (cpuset $CPUSET)"
    # NUMA-alignment soft warning
    if (( REQ_CPI > NODE0_CORES )); then
      log "  WARNING: CPI=$REQ_CPI > cores-per-NUMA=$NODE0_CORES; each instance will span NUMA nodes."
    elif (( NODE0_CORES % REQ_CPI != 0 )); then
      log "  NOTE: CPI=$REQ_CPI does not divide NUMA node size ($NODE0_CORES); instances may straddle nodes."
    fi
    if [[ "$SLICES" == "1" ]]; then
      for ((si=0; si<REQ_N; si++)); do
        off=$(( si * REQ_CPI ))
        slice="$(compress_ids "${AVAIL_CPUS[@]:off:REQ_CPI}")"
        SLICE_LINES+=( "SLICE_$((si+1))=$slice" )
        log "    instance $((si+1)): $slice"
      done
    fi
  else
    FITS="no"
    log "  Request N=$REQ_N x CPI=$REQ_CPI -> $NEED cores${POOL_NOTE}: DOES NOT FIT (only $AVAIL_CORES cores available)."
    log "  RECOMMENDED instead: N=$REC_N x CPI=$REC_CPI (=$(( REC_N * REC_CPI )) cores),"
    log "               or keep CPI=$REQ_CPI and use N=$(( AVAIL_CORES / REQ_CPI ))."
  fi
fi

# --- machine-readable output on stdout ---
echo "PHYS_CORES=$PHYS_CORES"
echo "AVAIL_CORES=$AVAIL_CORES"
echo "LOGICAL_CPUS=$LOGICAL_CPUS"
echo "SOCKETS=$SOCKETS"
echo "THREADS_PER_CORE=$THREADS_PER_CORE"
echo "NUMA_NODES=$NUMA_NODES"
echo "NODE0_CORES=$NODE0_CORES"
echo "REC_N=$REC_N"
echo "REC_CPI=$REC_CPI"
echo "REQ_N=${REQ_N:-}"
echo "REQ_CPI=${REQ_CPI:-}"
echo "FITS=$FITS"
echo "CPUSET=$CPUSET"
echo "TAIL_CPUSET=$TAIL_CPUSET"
if ((${#SLICE_LINES[@]})); then for _l in "${SLICE_LINES[@]}"; do echo "$_l"; done; fi

# Exit non-zero when a concrete request was made and does not fit, so callers
# can branch on the exit status too.
if [[ "$FITS" == "no" ]]; then exit 3; fi
exit 0
