# vLLM-zentorch Plugin

> **Seamless, high-performance LLM inference on CPUs with zentorch and vLLM**
> *Accelerate your large language models on modern x86 CPUs—no code changes required.*

---

## Overview

The **zentorch vLLM plugin** integrates [zentorch](https://github.com/amd/ZenDNN-pytorch-plugin) with [vLLM's V1 engine](https://docs.vllm.ai/en/stable/usage/v1_guide/) to deliver optimized large language model inference on AMD EPYC™ CPUs. By leveraging ZenDNN's highly optimized kernels, this plugin accelerates both attention and non-attention operations in vLLM, providing significant throughput improvements for popular LLMs

The plugin uses vLLM's platform and general plugin entry points to:
- Replace vLLM's attention with zentorch's optimized PagedAttention
- Inject zentorch optimization passes into `torch.compile`
- Disable replacement with Intel oneDNN kernels to enable replacement with zentorch kernels
- Enable CPU-only profiling

---

## Key Features

- **Plug-and-Play Acceleration:** No code modifications required—just install zentorch alongside vLLM for automatic acceleration.
- **Seamless vLLM Integration:** vLLM detects zentorch and transparently uses ZenDNN-optimized attention and non-attention kernels for supported CPUs.
- **Optimized for Modern x86 CPUs:** Delivers best-in-class performance on AMD EPYC™ processors, while supporting a broad range of x86 CPUs with the necessary instruction set.
- **Powered by ZenDNN:** Leverages AMD's ZenDNN library for state-of-the-art, CPU-optimized neural network operations.

---

## Compatibility

| Component | Version | Notes |
|-----------|---------|-------|
| vLLM | 0.11.0 - 0.13.0 | Profiler fixes applied per version |
| Python | 3.10+ | |
| PyTorch | 2.8/2.9.1+ | Auto-installed by vLLM |

---

## Architecture

When both vLLM and the `zentorch` package are installed, vLLM automatically detects the zentorch platform and replaces its default attention mechanism with the highly optimized zentorch PagedAttention kernel.

```
┌─────────────────────────────────────────────────────────────┐
│                        vLLM V1 Engine                       │
├─────────────────────────────────────────────────────────────┤
│  Plugin Loading (runs in ALL processes)                     │
│  ├── vllm.platform_plugins → ZenCPUPlatform                 │
│  └── vllm.general_plugins  → Early monkey-patches           │
├─────────────────────────────────────────────────────────────┤
│  ZenCPUPlatform                                             │
│  ├── Configures torch.compile with inductor backend         │
│  ├── Injects zentorch.optimize_pass for ZenDNN kernels      │
│  └── Patches profiler(version-specific)                     │
├─────────────────────────────────────────────────────────────┤
│  Monkey Patches (applied early)                             │
│  ├── PagedAttention → zentorch implementation               │
│  ├── CompilationConfig.__repr__ → Handle custom passes      │
│  └── _supports_onednn = False → Use zentorch linear         │
├─────────────────────────────────────────────────────────────┤
│  torch.compile (inductor + zentorch optimize_pass)          │
│  └── Replaces aten ops with zentorch ops (mm, attention)    │
└─────────────────────────────────────────────────────────────┘
```

This kernel leverages AMD EPYC specific intrinsics and optimizations to accelerate computations on AMD EPYC CPUs. However, the plugin may also function on other x86 CPUs that meet the required ISA. Further, we use zentorch to compile the LLM with `torch.compile`, replacing the native ops with zentorch's optimized ops.

## Key Components

**ZenCPUPlatform** (`platform.py`)
- Extends vLLM's `CpuPlatform`
- Sets `device_name = "cpu"` and `device_type = "cpu"`
- Configures `CompilationLevel.DYNAMO_ONCE`/`CompilationMode.DYNAMO_TRACE_ONCE` with inductor backend
- Injects `zentorch._compile_backend.optimize_pass` introducing zentorch operators

**Plugin Entry Points** (`__init__.py`)
- Registered via `vllm.platform_plugins` and `vllm.general_plugins`
- Applies patches before model initialization
- Validates vLLM version compatibility

**PagedAttention** (`attention.py`)
- Replaces vLLM's CPU attention backend with zentorch optimized paged attention kernel
- Drop-in replacement, no code changes required

---

## Installation

1. **Create a new Python environment:**

   > Note: To get started with conda, refer to [Miniforge Installation Guide](https://github.com/conda-forge/miniforge?tab=readme-ov-file#unix-like-platforms-macos-linux--wsl)

   **Using conda:**
   ```bash
   conda create -n vllm-env python=3.10
   conda activate vllm-env
   ```

2. **Build vLLM from Source**
   - Follow the official [vLLM Installation Guide](https://docs.vllm.ai/en/stable/getting_started/installation/cpu.html) for detailed, step-by-step instructions.
   - **Important:** Pre-built vLLM CPU binaries are not available. You must build vLLM from source to enable CPU support.
   - Supported versions: 0.11.x, 0.12.0, 0.13.0. Check out the appropriate release tag before building.

3. **Install zentorch:**
   Refer to the [zentorch Installation Guide](https://github.com/amd/ZenDNN-pytorch-plugin?tab=readme-ov-file#2-installation) for detailed instructions.

---

## Usage

No code changes are required. Once installed, simply run your vLLM inference workload as usual. The plugin will be automatically detected and used for inference on supported x86 CPUs that meet the required ISA features. While optimized for AMD EPYC™ CPUs, it may also function on other compatible x86 processors.

> **Note:**
>
> - Upon importing vLLM, you should see the following message in the logs:
>   ```
>   INFO [__init__.py] Platform plugin zentorch is activated
>   ```

### Environment Configuration

### Environment Variables

```bash
# ZenDNN settings
export TORCHINDUCTOR_FREEZING=0 
export ZENTORCH_LINEAR=1 
export USE_ZENDNN_MATMUL_DIRECT=1 
export USE_ZENDNN_SDPA_MATMUL_DIRECT=1 
export ZENDNNL_MATMUL_WEIGHT_CACHE=1 
export ZENDNNL_MATMUL_ALGO=1
# vLLM CPU settings
export VLLM_CPU_KVCACHE_SPACE=90  # GB for KV cache
export VLLM_CPU_OMP_THREADS_BIND=0-95  # CPU cores to use

# Performance libraries
sudo apt-get install libtcmalloc-minimal4
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4:$LD_PRELOAD
# Install llvm-openmp
conda install -c conda-forge llvm-openmp=18.1.8=hf5423f3_1 -y
export LD_PRELOAD="<install_path>/miniconda3/pkgs/llvm-openmp-18.1.8-hf5423f3_1/lib/libiomp5.so:$LD_PRELOAD"
```

### Example

```python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3.1-8B", dtype="bfloat16")
params = SamplingParams(temperature=0.8, top_p=0.95)
output = llm.generate(["Hello, world!"], sampling_params=params)
print(output)
```

### Benchmarking

```bash
vllm bench throughput \
    --model meta-llama/Llama-3.1-8B \
    --input-len 128 \
    --output-len 128 \
    --num-prompts 100
```

### Profiling

```bash
export VLLM_TORCH_PROFILER_DIR="."
vllm bench throughput \
    --model meta-llama/Llama-3.1-8B \
    --input-len 128 \
    --output-len 128 \
    --num-prompts 10 \
    --profile
```

> Note: Avoid benchmarking with the `--profile` flag as it impacts the results. Use profiler to only get the operator level metrics.

---

## Troubleshooting

### Plugin Not Detected

If you don't see "Platform plugin zentorch is activated":
1. Verify zentorch is installed: `python -c "import zentorch"`
2. Check vLLM version: `python -c "import vllm; print(vllm.__version__)"` (must be 0.11.x - 0.13.0)

### Stale Compilation Cache

After updating zentorch, clear the inductor cache:
```bash
rm -rf ~/.cache/torch_inductor /tmp/torchinductor_*
```

## Support

For questions or issues, visit the [ZenDNN PyTorch Plugin GitHub](https://github.com/amd/ZenDNN-pytorch-plugin).
