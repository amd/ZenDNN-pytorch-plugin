# vLLM-zentorch Plugin

> **Seamless, high-performance LLM inference on CPUs with zentorch and vLLM**
> *Accelerate your large language models on modern x86 CPUs—no code changes required.*

---

## Overview

The **zentorch vLLM plugin** integrates [zentorch](https://github.com/amd/ZenDNN-pytorch-plugin) with [vLLM's V1 engine](https://docs.vllm.ai/en/stable/usage/v1_guide/) to deliver optimized large language model inference on AMD EPYC™ CPUs. By leveraging ZenDNN's highly optimized kernels, this plugin accelerates linear and embedding operations in vLLM, providing significant throughput improvements for popular LLMs.

The plugin uses vLLM's platform and general plugin entry points to:
- Inject zentorch optimization passes into `torch.compile`
- Bypass or reroute GEMM dispatch so zentorch linear kernels stay active
- Enable CPU-only profiling

---

## Key Features

- **Plug-and-Play Acceleration:** No code modifications required—just install zentorch alongside vLLM for automatic acceleration.
- **Seamless vLLM Integration:** vLLM detects zentorch and transparently uses ZenDNN-optimized linear and embedding kernels for supported CPUs.
- **Optimized for Modern x86 CPUs:** Delivers best-in-class performance on AMD EPYC™ processors, while supporting a broad range of x86 CPUs with the necessary instruction set.
- **Powered by ZenDNN:** Leverages AMD's ZenDNN library for state-of-the-art, CPU-optimized neural network operations.

---

## Compatibility

| Component | Version | Notes |
|-----------|---------|-------|
| vLLM | 0.15.0 - 0.20.0 | V1 engine fully supported |
| Python | 3.10+ | |
| PyTorch | 2.10.0 (vLLM 0.15.0-0.19.x) / 2.11.0 (vLLM 0.20.0) | Auto-installed by vLLM |
| TorchAO | 0.16.0 (vLLM 0.15.0-0.19.x) / 0.17.0 (vLLM 0.20.0) | |

---

## Architecture

When both vLLM and the `zentorch` package are installed, vLLM automatically detects the zentorch platform and uses zentorch optimizations via `torch.compile`.

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
│  └── Patches profiler (version-specific)                    │
├─────────────────────────────────────────────────────────────┤
│  Monkey Patches (applied early)                             │
│  ├── CompilationConfig.__repr__ → Handle custom passes      │
│  └── GEMM dispatch / oneDNN bypass → Use zentorch linear    │
├─────────────────────────────────────────────────────────────┤
│  torch.compile (inductor + zentorch optimize_pass)          │
│  └── Replaces aten ops with zentorch ops (mm, embedding)    │
└─────────────────────────────────────────────────────────────┘
```

The plugin leverages AMD EPYC specific intrinsics and optimizations to accelerate computations on AMD EPYC CPUs. However, it may also function on other x86 CPUs that meet the required ISA. We use zentorch to compile the LLM with `torch.compile`, replacing the native ops with zentorch's optimized ops.

---

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

---

## Installation

1. **Create a new Python environment:**

   > Note: To get started with conda, refer to [Miniforge Installation Guide](https://github.com/conda-forge/miniforge?tab=readme-ov-file#unix-like-platforms-macos-linux--wsl)

   **Using conda:**
   ```bash
   conda create -n vllm-env python=3.10
   conda activate vllm-env
   ```

2. **Build vLLM from Source:**
   - Follow the official [vLLM Installation Guide](https://docs.vllm.ai/en/stable/getting_started/installation/cpu/#build-wheel-from-source) for detailed, step-by-step instructions.

     > **Important:** Pre-built vLLM CPU binaries are available from [0.13.0](https://docs.vllm.ai/en/stable/getting_started/installation/cpu/#pre-built-wheels), so all currently supported versions can use the published CPU wheels.

   - Supported versions: 0.15.0, 0.15.1, 0.16.0, 0.17.0, 0.17.1, 0.18.0, 0.18.1, 0.19.0, 0.19.1, 0.20.0. Check out the appropriate release tag before building.

3. **Install zentorch:**

   | vLLM version | PyTorch version (auto-installed by vLLM) | zentorch install method |
   |--------------|-----------------|------------------------|
   | 0.15.0 - 0.19.1 | 2.10.0 | PyPI or source |
   | 0.20.0 | 2.11.0 | PyPI or source |

   - **From PyPI** (vLLM 0.15.0+):
     ```bash
     pip install zentorch
     ```

   - **From source** (all supported versions):
     Follow the [zentorch Installation Guide](https://github.com/amd/ZenDNN-pytorch-plugin?tab=readme-ov-file#2-installation).

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

> **Note:** The plugin is recommended to be run with `ZENDNNL_MATMUL_ALGO=1` (the default).

#### Environment Variables

```bash
export TORCHINDUCTOR_FREEZING=1          # Only supported from vLLM version 0.12.0 onwards
export VLLM_CPU_KVCACHE_SPACE=120         # GB for KV cache
export VLLM_CPU_OMP_THREADS_BIND=0-127    # CPU cores to use
export VLLM_USE_AOT_COMPILE=0            # Disable AOT compile - interferes with freezing
export TORCHINDUCTOR_AUTOGRAD_CACHE=0    # Disable AOT compile - interferes with freezing
```

#### Performance Libraries

Install and preload `tcmalloc` and `llvm-openmp` for best performance:

```bash
# tcmalloc
sudo apt-get install libtcmalloc-minimal4
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4:$LD_PRELOAD

# llvm-openmp
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
    --random-input-len 128 \
    --random-output-len 128 \
    --num-prompts 100
```

### Profiling

```bash
export VLLM_TORCH_PROFILER_DIR="."
vllm bench throughput \
    --model meta-llama/Llama-3.1-8B \
    --random-input-len 128 \
    --random-output-len 128 \
    --num-prompts 10 \
    --profile
```

> Note: Avoid benchmarking with the `--profile` flag as it impacts the results. Use profiler to only get the operator level metrics.

---

## Docker

Pre-built Docker images with vLLM and zentorch are available for quick deployment.

### Pulling the Image

```bash
docker pull amdih/zendnn_zentorch:vllm_v<VLLM_VER>_zentorch_v<ZT_VER>_<OS_TAG>_<REL_TAG>
```

Replace `<VLLM_VER>`, `<ZT_VER>`, `<OS_TAG>`, and `<REL_TAG>` with the desired values. Available OS tags: `ubuntu22.04`, `rhel9.5`.

**Ubuntu Example:**

```bash
docker pull amdih/zendnn_zentorch:vllm_v0.17.0_zentorch_v5.2.0_ubuntu22.04_2026_ww11
```

**RHEL Example:**

```bash
docker pull amdih/zendnn_zentorch:vllm_v0.17.0_zentorch_v5.2.0_rhel9.5_2026_ww11
```

### Running the Container

```bash
docker run -d --name vllm_zentorch \
    -v /path/to/models:/models \
    amdih/zendnn_zentorch:vllm_v0.17.0_zentorch_v5.2.0_ubuntu22.04_2026_ww11 \
    tail -f /dev/null

docker exec -it vllm_zentorch bash
cd workspace
```

Mount volumes (`-v`) for model files and any datasets you need inside the container. Environment variables are pre-configured; adjust `VLLM_CPU_OMP_THREADS_BIND` for your machine (e.g., `0-127` for Turin, `0-95` for Genoa).

---

## Troubleshooting

### Plugin Not Detected

If you don't see "Platform plugin zentorch is activated":
1. Verify zentorch is installed: `python -c "import zentorch"`
2. Check vLLM version: `python -c "import vllm; print(vllm.__version__)"` (must be 0.15.0 - 0.20.0)

### Stale Compilation Cache

After updating zentorch, clear the inductor cache:
```bash
rm -rf ~/.cache/torch_inductor /tmp/torchinductor_*
```

---

## Support

For questions or issues, visit the [ZenDNN PyTorch Plugin GitHub](https://github.com/amd/ZenDNN-pytorch-plugin).
