Copyright &copy; 2023-2026 Advanced Micro Devices, Inc. All rights reserved.

_zentorch_: A PyTorch extension for AMD EPYC&trade; CPUs.
=============

Table of Contents
-------------

<!-- toc -->
- [About _zentorch_](#1-about-zentorch)
  - [Overview](#11-overview)
  - [Structure](#12-structure)
  - [Third Party Libraries](#13-third-party-libraries)
  - [Supported OS](#14-supported-os)
- [Installation](#2-installation)
  - [From Binaries](#21-from-binaries)
  - [From Source](#22-from-source)
- [Unit Tests](#3-unit-tests)
  - [Install Unit tests Dependencies](#31-install-unit-tests-dependencies)
  - [Run All Unit Tests](#32-run-all-unit-tests)
  - [Run All Tests](#33-run-all-tests)
  - [Run Individual Tests](#34-run-individual-tests)
- [Usage](#4-usage)
  - [General Usage](#41-general-usage)
  - [CNN Models](#42-cnn-models)
  - [HuggingFace NLP models](#43-huggingface-nlp-models)
  - [HuggingFace Generative LLM models](#44-huggingface-generative-llm-models)
  - [Weight only Quantized models](#45-weight-only-quantized-models)
  - [vLLM Zentorch Plugin](#46-vllm-zentorch-plugin)
- [Logging and Debugging](#5-logging-and-debugging)
  - [ZenDNN logs](#51-zendnn-logs)
  - [_zentorch_ logs](#52-zentorch-logs)
  - [Support for `TORCH_COMPILE_DEBUG`](#53-support-for-torch_compile_debug)
- [Performance tuning and Benchmarking](#6-performance-tuning-and-benchmarking)
- [Additional Utilities](#7-additional-utilities)
  - [_zentorch_ attributes](#71-zentorch-attributes)
<!-- tocstop -->

# 1. About _zentorch_

## 1.1. Overview

__The latest stable ZenDNN Plugin for PyTorch* (zentorch) [5.1](https://github.com/amd/ZenDNN-pytorch-plugin/tree/r5.1).__


__The main branch contains pre-release zentorch 5.2 plugin.__

zentorch 5.2 pre-release plugin is the PyTorch plugin which comes with ZenDNN 5.2 pre-release.
This upgrade continues the focus on optimizing inference with Recommender Systems and Large Language Models on AMD EPYC™ CPUs. It includes AMD EPYC™ enhancements for bfloat16 performance, expanded support for cutting-edge models like Llama 3.1 and 3.2, Microsoft Phi, and more as well as support for INT4 quantized datatype.
This includes the advanced Activation-Aware Weight Quantization (AWQ) algorithm for LLMs and quantized support for the DLRM-v2 model with int8 weights.
This also includes support for running generative models with vLLM.

Under the hood, ZenDNN’s enhanced AMD-specific optimizations operate at every level. In addition to highly optimized operator microkernels, these include comprehensive graph optimizations including pattern identification, graph reordering, and fusions.
They also incorporate optimized embedding bag kernels and enhanced zenMatMul matrix splitting strategies which leverage the AMD EPYC™ microarchitecture to deliver enhanced throughput and latency.

Combined with PyTorch's torch.compile, zentorch transforms deep learning pipelines into finely-tuned, AMD-specific engines, delivering unparalleled efficiency and speed for large-scale inference workloads

The zentorch 5.2 pre-release plugin seamlessly works with PyTorch versions including 2.10.0 and 2.9.1, offering a high-performance experience for deep learning on AMD EPYC™ platforms.

>**Note:** We recommend using Torch 2.9.1 or higher as there is a known [issue](https://github.com/pytorch/pytorch/pull/166338) with Torch 2.9.0 that leads to longer compilation time with zentorch backend. The issue has been fixed in later versions.

## Support

We welcome feedback, suggestions, and bug reports. Should you have any of the these, please kindly file an issue on the ZenDNN Plugin for PyTorch Github page [here](https://github.com/amd/ZenDNN-pytorch-plugin/issues)

## License

AMD copyrighted code in ZenDNN is subject to the [Apache-2.0, MIT, or BSD-3-Clause](LICENSE) licenses; consult the source code file headers for the applicable license.  Third party copyrighted code in ZenDNN is subject to the licenses set forth in the source code file headers of such code.

## 1.2. Structure
_zentorch_ consists of three parts. They are
- ZenDNN Integration Code
- _zentorch_ backend
- Build System

### 1.2.1. ZenDNN Integration Code
ZenDNN is integrated into _zentorch_ using CPP code which interfaces ATen API to ZenDNN's API. This code exports torch compatible API similar to ATen IR of PyTorch. The exported API is made available for usage from python code using TORCH_LIBRARY, TORCH_LIBRARY_FRAGMENT and TORCH_LIBRARY_IMPL. Integration code is linked and compiled into _zentorch_ using CppExtension provided by PyTorch.

The following ops are integrated as of now:
- Embedding bag op
- Embedding op
- Matmul ops
- Custom Fusion ops
- Rope op
- MHA op

### 1.2.2. The _zentorch_ custom backend to torch.compile
We have registered a custom backend to torch.compile called _zentorch_. This backend integrates ZenDNN optimizations after AOTAutograd through a function called optimize. This function operates on the FX based graph at the ATEN IR to produce an optimized FX based graph as the output.

### 1.2.3. Build System

The static library for ZenDNN and the cpp Extension modules that bind the ZenDNN operators with Python are built using `setup.py` script.

#### 1.2.3.1. CMake Based Build
CMake downloads the ZenDNN during configure stage. It generates a config.h with GIT hashes of ZenDNN. It downloads ZenDNN as a static library.

#### 1.2.3.2. Packaging into a Wheel File
The CPP code, being an extension module, is built through CppExtension. It takes various attributes to zentorch. `setup.py` also adds in various attributes to the _zentorch_ for debugging and providing additional information.

## 1.3. Third Party Libraries
_zentorch_ uses following library for its functionality.
  * [ZenDNN](https://github.com/amd/ZenDNN)


## 1.4. Supported OS

Refer to the [support matrix](https://www.amd.com/en/developer/zendnn.html#getting-started) for the list of supported operating systems.

# 2. Installation

_zentorch_ can be installed using binary wheel file or can be built from source itself.
Only stable releases are available as binary wheel files. The latest stable release is _zentorch_ v5.1.0 which supports PyTorch v2.6.0 and v2.7.0. Zentorch 5.2 pre-release can be built from source and supports PyTorch v2.9.1 and v2.10.0.

## 2.1. From Binaries

* Create conda or python environment and activate it.
* Uninstall any existing _zentorch_ installations.
```bash
pip uninstall zentorch
```
* Install PyTorch v2.7.0
```bash
pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu
```
* Use one of two methods to install zentorch:

Using pip utility
```bash
pip install zentorch==5.1.0
```
or

Using the release package.

> Download the package from AMD developer portal from [here](https://www.amd.com/en/developer/zendnn.html).

> Run the following commands to unzip the package and install the binary.

```bash
unzip ZENTORCH_v5.1.0_Python_v3.10.zip
cd ZENTORCH_v5.1.0_Python_v3.10/
pip install zentorch-5.1.0-cp310-cp310-manylinux_2_28_x86_64.whl
```
>**Notes:**
>* Dependent packages 'numpy' and 'torch' will be installed by '_zentorch_' if not already present.
>* If you get the error: ImportError: /lib64/libstdc++.so.6: version `GLIBCXX_.a.b.cc' not found (required by <path_to_conda>/envs/<env_name>/lib/python<py_version>/site-packages/zentorch-5.2.0-pyx.y-linux-x86_64.egg/zentorch/_C.cpython-xy-x86_64-linux-gnu.so), export LD_PRELOAD as: export LD_PRELOAD=<path_to_conda>/envs/<env_name>/lib/libstdc++.so.6:$LD_PRELOAD

## 2.2. From Source
Run the following commands:
```bash
git clone https://github.com/amd/ZenDNN-pytorch-plugin.git
cd ZenDNN-pytorch-plugin
```
>**Notes:**
>* The repository defaults to the master branch.
>* Build from the master branch generates zentorch 5.2 pre-release plugin.
>* ```export ZENDNNL_MANYLINUX_BUILD=1``` is needed for build from source for RHEL/FEDORA/Almalinux/CentOS OS families

### 2.2.1. Preparing third party repositories

Build setup downloads the ZenDNN repo into `third_party` folder.

### 2.2.2. Linux build
#### 2.2.2.1. Create conda environment for the build

```bash
conda create -n pt-zentorch python=3.10 -y
conda activate pt-zentorch
```
#### 2.2.2.2. Install PyTorch v2.10.0
```bash
# Pip command
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
```
>**Notes:**
>* This README uses Python 3.10.
>* Zentorch follows PyTorch’s Python version compatibility. For PyTorch 2.10.0, Zentorch supports Python versions 3.10 through 3.13, and the same range applies to PyTorch 2.9.1. For other PyTorch releases, refer to the PyTorch Release Compatibility Matrix
.
>* Zentorch does not support Python 3.13T or Python 3.14. 

#### 2.2.2.3. Install Dependencies
```bash
pip install -r requirements.txt
```
#### 2.2.2.4. To build & generate wheel file of _zentorch_
```bash
python setup.py bdist_wheel
```
>**Note:** The wheel file will be generated in dist folder in ZenDNN-pytorch-plugin directory

#### 2.2.2.5. To install the wheel file of _zentorch_
```bash
cd dist
pip install zentorch-5.2.0-cp310-cp310-linux_x86_64.whl
```
>**Note:** If you build from the main branch, the generated wheel file will be named:
zentorch-5.2.0-cp310-cp310-linux_x86_64.whl
#### 2.2.2.6. Build Cleanup
```bash
python setup.py clean --all
```

### 2.2.3. Windows build (Experimental)

>**Note:** Windows support is experimental. The build system has been made cross-platform,
>but full functionality depends on ZenDNN and its dependencies being available for Windows.
>An AMD EPYC™ CPU with AVX-512 support is still required.

#### 2.2.3.1. Prerequisites

* [Visual Studio 2019 or later](https://visualstudio.microsoft.com/) with the
  "Desktop development with C++" workload (provides MSVC and CMake).
* [Git for Windows](https://gitforwindows.org/)
* Python 3.10+ (via conda or the official installer)

#### 2.2.3.2. Create conda environment for the build

```cmd
conda create -n pt-zentorch python=3.10 -y
conda activate pt-zentorch
```

#### 2.2.3.3. Install PyTorch v2.10.0

```cmd
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
```

#### 2.2.3.4. Install Dependencies

```cmd
pip install -r requirements.txt
```

#### 2.2.3.5. Build from a Visual Studio Developer Command Prompt

Open a **"x64 Native Tools Command Prompt for VS"** (or run `vcvarsall.bat x64`), then:

```cmd
python setup.py bdist_wheel
```

#### 2.2.3.6. Install the wheel file

```cmd
cd dist
pip install zentorch-5.2.0-cp310-cp310-win_amd64.whl
```

#### 2.2.3.7. Build Cleanup

```cmd
python setup.py clean --all
```
# 3. Unit Tests

## 3.1 Install Unit tests Dependencies
```python
python test/install_requirements.py
```
>**Note:** Before running any unit tests, export the following environment variables to disable ZenDNN caching:
```bash
export ZENDNNL_MATMUL_WEIGHT_CACHE=0
export ZENDNNL_ZP_COMP_CACHE=0
```

## 3.2 Run All Unit Tests
```python
python -m unittest discover -s ./test/unittests
```

## 3.3 Run All Tests
```python
python -m unittest discover -s ./test
```

## 3.4 Run Individual Tests
```python
python -m unittest test/unittests/op_tests/test_bmm.py
```
# 4. Usage
## 4.1 General Usage
```python
# Using torch.compile with 'zentorch' as backend
import torch
import zentorch
--snip--
compiled_model = torch.compile(model, backend='zentorch')
with torch.no_grad():
    output = compiled_model(input)
```

### 4.1.1 Using freeze path with zentorch

Additionally, zentorch supports freezing the model, the example below shows the same:
```python
import torch
import zentorch
--snip--
compiled_model = torch.compile(model, backend='zentorch')
with torch.no_grad(), zentorch.freezing_enabled():
    output = compiled_model(input)
```
>**Notes:** 
>* zentorch.freezing_enabled() is deprecated and will be removed in next release. Please use ```export TORCHINDUCTOR_FREEZING=1``` to enable freezing path for zentorch.
>*  _zentorch_ is able to do the zentorch op replacements in both non-inference and inference modes. But some of the _zentorch_ optimizations are only supported for the inference mode, so it is recommended to use `torch.no_grad()` if you are running the model for inference only.

## 4.2 CNN Models
For CNN models, set `dynamic=False` when calling for `torch.compile` as below:
```python
model = torch.compile(model, backend='zentorch', dynamic=False)
with torch.no_grad():
    output = model(input)
```

## 4.3 HuggingFace NLP models
For HuggingFace NLP models, optimize them as below:
```python
model = torch.compile(model, backend='zentorch')
with torch.no_grad():
    output = model(input)
```
## 4.4 HuggingFace Generative LLM models

>**Note:** The `zentorch.llm.optimize` API has been deprecated. You can run generative models using `torch.compile(model, backend="zentorch")`, but for optimal performance we recommend using vLLM. Please refer to [section 4.5 vLLM Zentorch Plugin](#45-vllm-zentorch-plugin) for more details.

## 4.5 vLLM Zentorch Plugin

The vLLM-ZenTorch plugin enhances the capabilities of the vLLM inference engine, enabling plug-and-play acceleration of large language model inference on AMD EPYC™ CPUs. By incorporating ZenTorch with vLLM, users can experience substantial throughput enhancements for LLM workloads without requiring any modifications to their existing code.

For more details regarding vLLM-ZenTorch Plugin refer to this [Readme](./src/cpu/python/zentorch/vllm/README.md).

# 5. Logging and Debugging
## 5.1 ZenDNN logs
Logging for ZenDNN is disabled by default but can be enabled by using the environment variable **ZENDNN_API_LOG_LEVEL** before running any tests. An example to turn on info logging is given below.
```bash
export ZENDNNL_API_LOG_LEVEL=4
```
To enable the profiling logs, you can use:
```bash
export ZENDNNL_ENABLE_PROFILER=1
```

For further details on ZenDNN logging mechanism, refer to ZenDNN user-guide from [this page](https://www.amd.com/en/developer/zendnn.html#:~:text=Documentation-,ZenDNN%20User%20Guide,-TensorFlow%20%2B%20ZenDNN%20User).

## 5.2 _zentorch_ logs
For _zentorch_, CPP specific logging can be enabled by setting the environment variable `TORCH_CPP_LOG_LEVEL`. This has four levels: **INFO**, **WARNING**, **ERROR** and **FATAL** in decreasing order of verbosity. Similarly, python logging can be enabled by setting the environment variable `ZENTORCH_PY_LOG_LEVEL`, this has five levels: **DEBUG**, **INFO**, **WARNING**, **ERROR** and **CRITICAL**, again in decreasing order of verbosity. An example to enable INFO level logs for cpp and DEBUG level for python (most verbose) is given below:
```bash
export TORCH_CPP_LOG_LEVEL=INFO
export ZENTORCH_PY_LOG_LEVEL=DEBUG
```
The default level of logs is **WARNING** for both cpp and python sources but can be overridden as discussed above.
>**Note:** The log levels are the same as those provided by the python logging module.

>**Info:** Since all OPs implemented in _zentorch_ are registered with torch using the TORCH_LIBRARY(), TORCH_LIBRARY_FRAGMENT() and TORCH_LIBRARY_IMPL() macros in bindings, the PyTorch profiler can be used without any modifications to measure the op level performance.

## 5.3 Support for `TORCH_COMPILE_DEBUG`
PyTorch offers a debugging toolbox that comprises a built-in stats and trace function. This functionality facilitates the display of the time spent by each compilation phase, output code, output graph visualization, and IR dump. `TORCH_COMPILE_DEBUG` invokes this debugging tool that allows for better problem-solving while troubleshooting the internal issues of TorchDynamo and TorchInductor. This functionality works for the models optimized using _zentorch_, so it can be leveraged to debug these models as well. To enable this functionality, users can either set the environment variable `TORCH_COMPILE_DEBUG=1` or specify the environment variable with the runnable file (e.g., test.py) as input.
```bash
# test.py contains model optimized by torch.compile with 'zentorch' as backend
TORCH_COMPILE_DEBUG=1 python test.py
```
For more information about TORCH_COMPILE_DEBUG refer to the official PyTorch documentation available.

# 6. Performance tuning and Benchmarking
zentorch v5.2.0 pre-release plugin is supported with pre-release ZenDNN v5.2.0  plugin. Please see the **Tuning Guidelines** section of ZenDNN User Guide for performance tuning. ZenDNN User Guide can be downloaded from [here](https://developer.amd.com/zendnn)

# 7. Additional Utilities:

## 7.1 _zentorch_ attributes:
To check the version of _zentorch_ use the following command:

```bash
python -c 'import zentorch; print(zentorch.__version__)'
```

To check the build config of _zentorch_ use the following command:
```bash
python -c 'import zentorch; print(*zentorch.__config__.split("\n"), sep="\n")'
```
