Copyright &copy; 2023-2025 Advanced Micro Devices, Inc. All rights reserved.

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

__The latest ZenDNN Plugin for PyTorch* (zentorch) 5.1 is here!__

zentorch 5.1 is the PyTorch plugin which comes with ZenDNN 5.1.
This upgrade continues the focus on optimizing inference with Recommender Systems and Large Language Models on AMD EPYC™ CPUs. includes AMD EPYC™ enhancements for bfloat16 performance, expanded support for cutting-edge models like Llama 3.1 and 3.2, Microsoft Phi, and more as well as support for INT4 quantized datatype.
This includes the advanced Activation-Aware Weight Quantization (AWQ) algorithm for LLMs and quantized support for the DLRM-v2 model with int8 weights.

Under the hood, ZenDNN’s enhanced AMD-specific optimizations operate at every level. In addition to highly optimized operator microkernels, these include comprehensive graph optimizations including pattern identification, graph reordering, and fusions.
They also incorporate optimized embedding bag kernels and enhanced zenMatMul matrix splitting strategies which leverage the AMD EPYC™ microarchitecture to deliver enhanced throughput and latency.

Combined with PyTorch's torch.compile, zentorch transforms deep learning pipelines into finely-tuned, AMD-specific engines, delivering unparalleled efficiency and speed for large-scale inference workloads

The zentorch 5.1 release plugs seamlessly with PyTorch versions from 2.7 and 2.6, offering a high-performance experience for deep learning on AMD EPYC™ platforms.

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

The static libraries for ZenDNN, AOCL BLIS and the cpp Extension modules that bind the ZenDNN operators with Python are built using `setup.py` script.

#### 1.2.3.1. CMake Based Build: ZenDNN , AOCL BLIS and FBGEMM
CMake downloads the ZenDNN , AOCL BLIS and FBGEMM during configure stage. It generates a config.h with GIT hashes of ZenDNN , AOCL BLIS and FBGEMM. It builds ZenDNN , AOCL BLIS and FBGEMM as static libraries.

#### 1.2.3.2. Packaging into a Wheel File
The CPP code, being an extension module, is built through CppExtension. It takes static libraries of the ZenDNN , AOCL BLIS and FBGEMM libraries. `setup.py` also adds in various attributes to the _zentorch_ for debugging and providing additional information.

## 1.3. Third Party Libraries
_zentorch_ uses following libraries for its functionality.
  * [ZenDNN](https://github.com/amd/ZenDNN)
  * [AOCL BLIS](https://github.com/amd/blis)
  * [FBGEMM](https://github.com/pytorch/FBGEMM)

## 1.4. Supported OS

Refer to the [support matrix](https://www.amd.com/en/developer/zendnn.html#getting-started) for the list of supported operating systems.

# 2. Installation

_zentorch_ can be installed using binary wheel file or can be built from source itself.

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
>Notes:
* Zentorch inherits its Python version compatibility from PyTorch. For Torch 2.7 and 2.6, Zentorch supports Python 3.9 to 3.13. For other versions, please refer to the [PyTorch Release Compatibility Matrix](https://github.com/pytorch/pytorch/blob/main/RELEASE.md#release-compatibility-matrix). This README uses Python 3.10.
* Dependent packages 'numpy' and 'torch' will be installed by '_zentorch_' if not already present.
* If you get the error: ImportError: /lib64/libstdc++.so.6: version `GLIBCXX_.a.b.cc' not found (required by <path_to_conda>/envs/<env_name>/lib/python<py_version>/site-packages/zentorch-5.1.0-pyx.y-linux-x86_64.egg/zentorch/_C.cpython-xy-x86_64-linux-gnu.so), export LD_PRELOAD as:
  * export LD_PRELOAD=<path_to_conda>/envs/<env_name>/lib/libstdc++.so.6:$LD_PRELOAD

## 2.2. From Source
Run the following commands:
```bash
git clone https://github.com/amd/ZenDNN-pytorch-plugin.git
cd ZenDNN-pytorch-plugin
```
>Note: The repository defaults to the master branch. To build version 5.1, please check out the r5.1 branch; otherwise, it will build using the master branch.
```bash
git checkout r5.1
```

### 2.2.1. Preparing third party repositories

Build setup downloads the ZenDNN, AOCL BLIS and FBGEMM repos into `third_party` folder.

### 2.2.2. Linux build
#### 2.2.2.1. Create conda environment for the build

```bash
conda create -n pt-zentorch python=3.10 -y
conda activate pt-zentorch
```
#### 2.2.2.2. Install PyTorch v2.7.0
```bash
# Pip command
pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu
```

>Note: The CPU version of torch/pytorch only supports CPU version of torchvision.

>Note: cmake & ninja are required for cpp extension builds, will be installed through build script.

#### 2.2.2.3. Install Dependencies
```bash
pip install -r requirements.txt
```
#### 2.2.2.4. To build & generate wheel file of _zentorch_
```bash
python setup.py bdist_wheel
```
>Note: The wheel file will be generated in dist folder in ZenDNN-pytorch-plugin directory

#### 2.2.2.5. To install the wheel file of _zentorch_
```bash
cd dist
pip install zentorch-5.1.0-cp310-cp310-linux_x86_64.whl
```
#### 2.2.2.6. Build Cleanup
```bash
python setup.py clean --all
```
# 3. Unit Tests

## 3.1 Install Unit tests Dependencies
```python
python test/install_requirements.py
```
>Note: Before running any unit tests, export the following environment variable to disable ZenDNN caching:
```bash
export ZENDNN_ENABLE_CACHE=0
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

>Note: If same model is optimized with `torch.compile` for multiple backends within single script, it is recommended to use `torch._dynamo.reset()` before calling the `torch.compile` on that model. This is applicable if torch version is less than 2.3.

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

>Note: _zentorch_ is able to do the zentorch op replacements in both non-inference and inference modes. But some of the _zentorch_ optimizations are only supported for the inference mode, so it is recommended to use `torch.no_grad()` if you are running the model for inference only.

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
For HuggingFace Generative LLM models, usage of zentorch.llm.optimize is recommended. All the optimizations included in this API are specifically targeted for Generative Large Language Models from HuggingFace. If a model is not a valid Generative Large Language Model from HuggingFace, the following warning will be displayed and zentorch.llm.optimize will act as a dummy function with no optimizations being applied to the model that is passed: “Cannot detect the model transformers family by model.config.architectures. Please pass a valid HuggingFace LLM model to the zentorch.llm.optimize API.” This check confirms the presence of the "config" and "architectures" attributes of the model to get the model id. Considering the check, two scenarios the zentorch.llm.optimize can still act as a dummy function:
1.  HuggingFace has a plethora of models, of which Generative LLMs are a subset of. So, even if the model has the attributes of "config" and "architectures", the model id might not be yet present in the supported models list from zentorch. In this case zentorch.llm.optimize will act as a dummy function.
2. A model can be a valid generative LLM from HuggingFace or not, might miss the "config" and "architectures" attributes. In this case also, the zentorch.llm.optimize API will act as a dummy function.

If the model passed is valid, all the supported optimizations will be applied, and performant execution is ensured.
To check the supported models, run the following command:
```bash
python -c 'import zentorch; print("\n".join([f"{i+1:3}. {item}" for i, item in enumerate(zentorch.llm.SUPPORTED_MODELS)]))'
```

If a model id other than the listed above are passed, zentorch.llm.optimize will not apply the above specific optimizations to the model and a warning will be displayed as follows: “Complete set of optimizations are currently unavailable for this model.” Control will pass to the zentorch custom backend to torch.compile for applying optimizations.

For leveraging the best performance of zentorch_llm_optimize, user has to install IPEX corresponding to the PyTorch version that is installed in the environment.
The PyTorch version for performance execution of supported LLMs should be greater than or equal to 2.6.0. Recommended version for optimal performance is using PyTorch 2.7.
zentorch.llm.optimize requires the dtype to be torch.dtype. Please make sure you pass a valid torch.dtype (such as torch.bfloat16) to optimize your model.

### Case #1. If output is generated through a call to direct `model`, optimize it as below:
```python
model = zentorch.llm.optimize(model, dtype)
model = torch.compile(model, backend='zentorch')
with torch.no_grad():
    output = model(input)
```

### Case #2. If output is generated through a call to `model.forward`, optimize it as below:
```python
model = zentorch.llm.optimize(model, dtype)
model.forward = torch.compile(model.forward, backend='zentorch')
with torch.no_grad():
    output = model.forward(input)
```

### Case #3. If output is generated through a call to `model.generate`, optimize it as below:
    - Optimize the `model.forward` with torch.compile instead of `model.generate`.
    - But still generate the output through a call to `model.generate`.
```python
model = zentorch.llm.optimize(model, dtype)
model.forward = torch.compile(model.forward, backend='zentorch')
with torch.no_grad():
    output = model.generate(input)
```

## 4.5 Weight only Quantized models

Huggingface models are quantized using [AMD's Quark tool](https://github.com/amd/Quark/blob/v0.8/README.md).
After downloading the zip file, install Quark and follow the below steps:

> zentorch v5.1 is compatible with Quark v0.8. Please make sure you download the right version.

### 4.5.1 Go to the examples/torch/language_modeling/llm_ptq/ directory
### 4.5.2 Install the necessary dependencies
```bash
pip install -r requirements.txt
pip install -r ../llm_eval/requirements.txt
```
### 4.5.3 Run the following command to quantize the model
#### 4.5.3.1 For per-channel quantization
```bash
OMP_NUM_THREADS=<physical-cores-num> numactl --physcpubind=<physical-cores-list> python quantize_quark.py --model_dir <hugging_face_model_id> --device cpu --data_type bfloat16 --model_export hf_format --quant_algo awq --quant_scheme w_int4_per_group_sym --group_size -1 --num_calib_data 128 --dataset pileval_for_awq_benchmark --seq_len 128 --output_dir <output_dir> --pack_method order
```
#### 4.5.3.2 For per-group quantization
```bash
OMP_NUM_THREADS=<physical-cores-num> numactl --physcpubind=<physical-cores-list> python quantize_quark.py --model_dir <hugging_face_model_id> --device cpu --data_type bfloat16 --model_export hf_format --quant_algo awq --quant_scheme w_int4_per_group_sym --group_size <group_size> --num_calib_data 128 --dataset pileval_for_awq_benchmark --seq_len 128 --output_dir <output_dir> --pack_method order
```
> Note: The channel/out_features dimension should be divisible by the 'group_size' value.

As currently HF does not support AWQ format for CPU, an additional codeblock needs to be added to your inference script for loading the WOQ models.
```python
config = AutoConfig.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_config(config, trust_remote_code=True, torch_dtype=torch.bfloat16)
model = zentorch.load_quantized_model(model, safetensor_path)
```


> Note:
> * From zentorch 5.0.1 load_woq_model() API is deprecated and will be removed in future releases. Please use load_quantized_model() API instead.

Here, safetensor_path refers to the "<output_dir>" path of the quantized model.

After the loading steps, the model can be executed in a similar fashion as the cases# 1-3 listed in [section 4.4](#44-huggingface-generative-llm-models).

## 4.6 vLLM Zentorch Plugin

The vLLM-ZenTorch plugin enhances the capabilities of the vLLM inference engine, enabling plug-and-play acceleration of large language model inference on AMD EPYC™ CPUs. By incorporating ZenTorch with vLLM, users can experience substantial throughput enhancements for LLM workloads without requiring any modifications to their existing code.

For more details regarding vLLM-ZenTorch Plugin refer to this [Readme](./src/cpu/python/zentorch/vllm/README.md).

# 5. Logging and Debugging
## 5.1 ZenDNN logs
Logging for ZenDNN is disabled by default but can be enabled by using the environment variable **ZENDNN_LOG_OPTS** before running any tests. Its behavior can be specified by setting **ZENDNN_LOG_OPTS** to a comma-delimited list of **ACTOR:DBGLVL** pairs. An example to turn on info logging is given below.
```bash
export ZENDNN_LOG_OPTS=ALL:2
```
To enable the profiling logs **zendnn_primitive_create** and **zendnn_primitive_execute**, you can use:
```bash
export ZENDNN_PRIMITIVE_LOG_ENABLE=1
```

For further details on ZenDNN logging mechanism, refer to ZenDNN user-guide from [this page](https://www.amd.com/en/developer/zendnn.html#:~:text=Documentation-,ZenDNN%20User%20Guide,-TensorFlow%20%2B%20ZenDNN%20User).

## 5.2 _zentorch_ logs
For _zentorch_, CPP specific logging can be enabled by setting the environment variable `TORCH_CPP_LOG_LEVEL`. This has four levels: **INFO**, **WARNING**, **ERROR** and **FATAL** in decreasing order of verbosity. Similarly, python logging can be enabled by setting the environment variable `ZENTORCH_PY_LOG_LEVEL`, this has five levels: **DEBUG**, **INFO**, **WARNING**, **ERROR** and **CRITICAL**, again in decreasing order of verbosity. An example to enable INFO level logs for cpp and DEBUG level for python (most verbose) is given below:
```bash
export TORCH_CPP_LOG_LEVEL=INFO
export ZENTORCH_PY_LOG_LEVEL=DEBUG
```
The default level of logs is **WARNING** for both cpp and python sources but can be overridden as discussed above.
>NOTE: The log levels are the same as those provided by the python logging module.

>INFO: Since all OPs implemented in _zentorch_ are registered with torch using the TORCH_LIBRARY(), TORCH_LIBRARY_FRAGMENT() and TORCH_LIBRARY_IMPL() macros in bindings, the PyTorch profiler can be used without any modifications to measure the op level performance.

## 5.3 Support for `TORCH_COMPILE_DEBUG`
PyTorch offers a debugging toolbox that comprises a built-in stats and trace function. This functionality facilitates the display of the time spent by each compilation phase, output code, output graph visualization, and IR dump. `TORCH_COMPILE_DEBUG` invokes this debugging tool that allows for better problem-solving while troubleshooting the internal issues of TorchDynamo and TorchInductor. This functionality works for the models optimized using _zentorch_, so it can be leveraged to debug these models as well. To enable this functionality, users can either set the environment variable `TORCH_COMPILE_DEBUG=1` or specify the environment variable with the runnable file (e.g., test.py) as input.
```bash
# test.py contains model optimized by torch.compile with 'zentorch' as backend
TORCH_COMPILE_DEBUG=1 python test.py
```
For more information about TORCH_COMPILE_DEBUG refer to the official PyTorch documentation available.

# 6. Performance tuning and Benchmarking
zentorch v5.1 is supported with ZenDNN v5.1. Please see the **Tuning Guidelines** section of ZenDNN User Guide for performance tuning. ZenDNN User Guide can be downloaded from [here](https://developer.amd.com/zendnn)

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
