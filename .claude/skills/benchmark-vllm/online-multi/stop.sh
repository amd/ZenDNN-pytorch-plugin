#!/usr/bin/env bash

# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

#
# stop.sh -- tear down the online-multi stack.
#
#   ./stop.sh [--generated-dir DIR] [--purge]
#     --purge   also remove named volumes/networks (compose down -v)
#
set -uo pipefail
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SELF_DIR}/../scripts/common.sh"

GENERATED_DIR="${SELF_DIR}/generated"
PURGE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --generated-dir) GENERATED_DIR="$2"; shift 2 ;;
    --purge) PURGE=1; shift ;;
    -h|--help) echo "Usage: $0 [--generated-dir DIR] [--purge]"; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

detect_runtime || exit 1
[[ -z "$CCOMPOSE" ]] && { echo "ERROR: no compose available." >&2; exit 1; }
[[ -f "${GENERATED_DIR}/docker-compose.yml" ]] || { echo "Nothing to stop (${GENERATED_DIR}/docker-compose.yml missing)."; exit 0; }

if [[ "$PURGE" == "1" ]]; then
  ( cd "$GENERATED_DIR" && $CCOMPOSE down -v )
else
  ( cd "$GENERATED_DIR" && $CCOMPOSE down )
fi
echo "Stack stopped."
