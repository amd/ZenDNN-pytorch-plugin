#!/usr/bin/env bash

# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

#
# incontainer_env.sh -- runs INSIDE the container. Source it to set the
# performance LD_PRELOAD (tcmalloc + libiomp5) that vLLM CPU wants, plus the
# ZenDNN/vLLM tuning vars if not already provided via `-e`.
#
#   source /bench/incontainer_env.sh
#
# The tcmalloc / libiomp5 paths differ per image, so they are discovered at
# runtime via ldconfig rather than hardcoded on the host.

# --- tcmalloc (prefer the minimal build) ---
_tc=$(ldconfig -p 2>/dev/null | grep -E 'libtcmalloc_minimal\.so' | grep -v debug | head -n1 | awk '{print $NF}')
if [[ -z "${_tc}" ]]; then
  _tc=$(ldconfig -p 2>/dev/null | grep -E 'libtcmalloc\.so' | grep -v debug | head -n1 | awk '{print $NF}')
fi

# --- libiomp5 (LLVM OpenMP) ---
_iomp=$(ldconfig -p 2>/dev/null | grep -E 'libiomp5\.so' | head -n1 | awk '{print $NF}')
if [[ -z "${_iomp}" ]]; then
  # Fall back to a filesystem search of common venv/conda locations.
  _iomp=$(find /opt /usr -name 'libiomp5.so' 2>/dev/null | head -n1)
fi

_preload=""
[[ -n "${_tc}" ]]   && _preload="${_tc}"
[[ -n "${_iomp}" ]] && _preload="${_preload:+${_preload}:}${_iomp}"

if [[ -n "${_preload}" ]]; then
  # Merge with any LD_PRELOAD the image already set, de-duplicating entries so
  # tcmalloc/libiomp5 don't appear twice (the amdih image preloads them too).
  _merged="${_preload}${LD_PRELOAD:+:${LD_PRELOAD}}"
  _dedup=""
  _old_ifs="$IFS"; IFS=':'
  for _p in $_merged; do
    [[ -z "$_p" ]] && continue
    case ":${_dedup}:" in
      *":${_p}:"*) ;;                                    # already present
      *) _dedup="${_dedup:+${_dedup}:}${_p}" ;;
    esac
  done
  IFS="$_old_ifs"
  export LD_PRELOAD="${_dedup}"
  echo "[incontainer_env] LD_PRELOAD=${LD_PRELOAD}"
else
  echo "[incontainer_env] WARNING: tcmalloc/libiomp5 not found; running without LD_PRELOAD" >&2
fi

# ZenDNN + vLLM CPU tuning (respect values already passed via -e).
export TORCHINDUCTOR_FREEZING="${TORCHINDUCTOR_FREEZING:-1}"
export VLLM_CPU_KVCACHE_SPACE="${VLLM_CPU_KVCACHE_SPACE:-90}"

echo "[incontainer_env] TORCHINDUCTOR_FREEZING=${TORCHINDUCTOR_FREEZING} VLLM_CPU_KVCACHE_SPACE=${VLLM_CPU_KVCACHE_SPACE}"
