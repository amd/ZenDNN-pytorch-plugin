#!/usr/bin/env python3

# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

"""Print vLLM's unified KV block size for a hybrid model on CPU.

Runs inside the vLLM image (start.sh --lmcache mounts it there). Hybrid models
(Mamba / linear-attention layers plus attention, e.g. Qwen3.5 / Qwen3.6) need
prefix caching on and an LMCache chunk size that is a multiple of this block
size; see test-vllm-lmcache-hybrid. Reads only the model config, no weights.

Prints one line on stdout:
  hybrid <N>   hybrid model, unified block size N tokens
  dense        not a hybrid model
Exits non-zero when vLLM cannot answer (e.g. a private API changed).
"""

import sys


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--kv-cache-dtype", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    from vllm.config import set_current_vllm_config
    from vllm.engine.arg_utils import EngineArgs
    from vllm.platforms import current_platform
    from vllm.v1.attention.backends.cpu_attn import CPUAttentionBackend

    config = EngineArgs(
        model=args.model,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
        kv_cache_dtype=args.kv_cache_dtype,
        trust_remote_code=args.trust_remote_code,
        enable_prefix_caching=True,
    ).create_engine_config()
    if not config.model_config.is_hybrid:
        print("dense")
        return
    # The workers do this once the layers exist (update_block_size_for_backend);
    # on CPU every attention layer uses CPUAttentionBackend, so call it directly.
    with set_current_vllm_config(config):
        type(current_platform)._align_hybrid_block_size(config, CPUAttentionBackend)
    print(f"hybrid {config.cache_config.block_size}")


if __name__ == "__main__":
    sys.exit(main())
