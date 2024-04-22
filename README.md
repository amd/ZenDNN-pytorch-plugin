Copyright &copy; 2023-2024 Advanced Micro Devices, Inc. All rights reserved.

_zentorch_: A PyTorch extension for AMD EPYC&trade; CPUs.
=============

Table of Contents
-------------

<!-- toc -->
- [About _zentorch_](#1-about-zentorch)
  - [Overview](#11-overview)
  - [Structure](#12-structure)
  - [Third Party Libraries](#13-third-party-libraries)
- [Installation](#2-installation)
  - [From Binaries](#21-from-binaries)
  - [From Source](#22-from-source)
- [Usage](#3-usage)
- [Logging, Profiling and Debugging](#4-logging-profiling-and-debugging)
  - [ZenDNN logs](#41-zendnn-logs)
  - [_zentorch_ logs](#42-zentorch-logs)
  - [Saving the graph](#43-saving-the-graph)
  - [Support for `TORCH_COMPILE_DEBUG`](#44-support-for-torch_compile_debug)
- [Additional Utilities](#5-additional-utilities)
  - [Disabling Inductor](#51-disabling-inductor)
  - [_zentorch_ attributes](#52-zentorch-attributes)
<!-- tocstop -->

# 1. About _zentorch_

## 1.1. Overview

**EARLY ACCESS:** The ZenDNN PyTorch* Plugin (zentorch) extends PyTorch* with an innovative upgrade that's set to revolutionize performance on AMD hardware.

As of version 4.2, AMD is unveiling a game-changing upgrade to ZenDNN, introducing a cutting-edge plug-in mechanism and an enhanced architecture under the hood. This isn't just about extensions; ZenDNN's aggressive AMD-specific optimizations operate at every level. It delves into comprehensive graph optimizations, including pattern identification, graph reordering, and seeking opportunities for graph fusions. At the operator level, ZenDNN boasts enhancements with microkernels, mempool optimizations, and efficient multi-threading on the large number of AMD EPYC cores. Microkernel optimizations further exploit all possible low-level math libraries, including AOCL BLIS.

The result? Enhanced performance with respect to baseline PyTorch*. zentorch leverages torch.compile, the latest PyTorch enhancement for accelerated performance. torch.compile makes PyTorch code run faster by JIT-compiling PyTorch code into optimized kernels, all while requiring minimal code changes and unlocking unprecedented speed and efficiency.

The _zentorch_ extension to PyTorch enables inference optimizations for deep learning workloads on AMD EPYC&trade; CPUs. It uses  the ZenDNN library, which contains deep learning operators tailored for high performance on AMD EPYC&trade; CPUs. Multiple passes of graph level optimizations run on the torch.fx graph provide further performance acceleration. All _zentorch_ optimizations can be enabled by a call to torch.compile with zentorch as the backend.

The ZenDNN PyTorch plugin is compatible with PyTorch version 2.1.2.

## Support

Please note that zentorch is currently in “Early Access” mode. We welcome feedback, suggestions, and bug reports. Should you have any of the these, please contact us on zendnn.maintainers@amd.com

## License

zentorch is licensed under [Apache-2.0, MIT, BSD-3-Clause](LICENSE).

## 1.2. Structure
_zentorch_ consists of three parts. They are
- ZenDNN Integration Code
- _zentorch_ backend
- Build System

### 1.2.1. ZenDNN Integration Code
ZenDNN is integrated into _zentorch_ using CPP code which interfaces ATen API to ZenDNN's API. This code exports torch compatible API similar to ATen IR of PyTorch. The exported API is made avaiable for usage from python code using TORCH_LIBRARY and TORCH_LIBRARY_IMPL. Integration code is linked and compiled into _zentorch_ using CppExtension provided by PyTorch.

The following ops are integrated as of now:
- Embedding bag op
- Embedding op
- Matmul ops
- Custom Fusion ops

### 1.2.2. The _zentorch_ custom backend to torch.compile
We have registered a custom backend to torch.compile called _zentorch_. This backend integrates ZenDNN optimizations after AOTAutograd through a function called optimize. This function operates on the FX based graph at the ATEN IR to produce an optimized FX based graph as the output.

### 1.2.3. Build System

The static libraries for ZenDNN, AOCL BLIS and the cpp Extension modules that bind the ZenDNN operators with Python are built using `setup.py` script.

#### 1.2.3.1. CMake Based Build: ZenDNN , AOCL BLIS and FBGEMM
CMake downloads the ZenDNN , AOCL BLIS and FBGEMM during configure stage. It generates a config.h with GIT hashes of ZenDNN , AOCL BLIS and FBGEMM. It builds ZenDNN , AOCL BLIS and FBGEMM as static libraries.

#### 1.2.3.2. Packaging into a Wheel File
The CPP code, being an extension module, is built through CppExtension. It takes static libraries of the ZenDNN , AOCL BLIS and FBGEMM libraries. `setup.py` also adds in various attributes to the _zentorch_ for debugging and providing additional information.


#### 1.2.3.3. build.sh
It installs packages necessary for the _zentorch_ build and completes all other steps needed for building the _zentorch_.

Wheel file can be generated solely through the `setup.py` script, without the need for any additional scripts. The build.sh is a tiny wrapper around the Python setup script.

> NOTE: Alternatively PyPA build can be used instead of `setup.py`. Currently minimal support is added for PyPA build. When using PyPA Build, ensure to run `pip install build` prior to building.

## 1.3. Third Party Libraries
_zentorch_ uses following libraries for its functionality.
  * [ZenDNN](https://github.com/amd/ZenDNN)
  * [AOCL BLIS](https://github.com/amd/blis)
  * [FBGEMM](https://github.com/pytorch/FBGEMM)

# 2. Installation

_zentorch_ can be installed using binary wheel file or can be built from source itself.

## 2.1. From Binaries
* Create conda or python environment and activate it.
* Uninstall any existing _zentorch_ installations.
```bash
pip uninstall zentorch
```
* Download the wheel file and install it using pip or conda install command.

>INFO: Please find the latest Nightly wheel file [here](http://atlvjksapp02:8080/job/ZenDNN_Nightly_Build_for_PT_PLUGIN/lastSuccessfulBuild/)

```bash
pip install zentorch-*-linux_x86_64.whl
```
>Note: 
* Dependent packages 'numpy' and 'torch' will be installed by '_zentorch_' if not already present.
* Torch Version should be greater than or equal to 2.0

## 2.2. From Source
Run the following commands:
```bash
git clone "ssh://gerritgit/amd/ec/ZenDNN_PyTorch_Plugin"
cd ZenDNN_PyTorch_Plugin/
```

### 2.2.1. Preparing third party repositories

Build setup downloads the AOCL BLIS and ZenDNN repos into `third_party` folder. It can alternatively use local copies of ZenDNN and AOCL BLIS. This is very useful for day to day development scenarios, where developer may be interested in using recent version of repositories. Build setup will switch between local and remote copies of ZenDNN , AOCL BLIS and FBGEMM with environmental variables `ZENDNN_PT_USE_LOCAL_ZENDNN` , `ZENDNN_PT_USE_LOCAL_BLIS` and `ZENDNN_PT_USE_LOCAL_FBGEMM` respectively. To use local copies of ZenDNN , AOCL BLIS or FBGEMM, set `ZENDNN_PT_USE_LOCAL_ZENDNN` , `ZENDNN_PT_USE_LOCAL_BLIS` or `ZENDNN_PT_USE_LOCAL_FBGEMM` to 1 respectively. The source repositories should be downloaded/cloned in the directory where `ZenDNN_PyTorch_Plugin` is cloned for local setting. Folder structure may look like below.

```
<parent folder>
    |
    |------><AOCL BLIS repo>
    |
    |------><ZenDNN repo>
    |
    |------><FBGEMM repo>
    |
    |------><ZenDNN_PyTorch_Plugin>
```
>NOTE:
> 1. The recommended values of `ZENDNN_PT_USE_LOCAL_ZENDNN` , `ZENDNN_PT_USE_LOCAL_BLIS` and `ZENDNN_PT_USE_LOCAL_FBGEMM` are 1 , 0 and 0 respectively. Default values are the same as recommended values.
>```bash
>export ZENDNN_PT_USE_LOCAL_ZENDNN=1
>export ZENDNN_PT_USE_LOCAL_BLIS=0
>export ZENDNN_PT_USE_LOCAL_FBGEMM=0
>```
> 2. ZenDNN repository can be cloned using command<br> `git clone "ssh://gerritgit/amd/ec/ZenDNN"`
> 3. AOCL BLIS can be cloned using command<br> `git clone "ssh://gerritgit/cpulibraries/er/blis"`
> 4. FBGEMM can be cloned using command<br> `git clone https://github.com/pytorch/FBGEMM.git`

### 2.2.2. Linux build
#### 2.2.2.1. Create conda environment for the build
```bash
conda create -n pt-zentorch python=3.8
conda activate pt-zentorch
```
#### 2.2.2.2. You can install torch using 'conda' or 'pip'
```bash
# Pip command
pip install torch --index-url https://download.pytorch.org/whl/cpu
```
or
```bash
# Conda command
conda install pytorch cpuonly -c pytorch
```

>Note: The CPU version of torch/pytorch only supports CPU version of torchvision.

>Note: cmake & ninja are required for cpp extension builds, will be installed through build script.

#### 2.2.2.3. To build & install the _zentorch_
```bash
bash build.sh
```

### 2.2.3. manylinux build
'manylinux_setup' script allows you to create a docker container, you need to install conda and proceed the build process

```bash
sudo bash manylinux_setup.sh
export SCRIPT_TYPE=many_linux
```

#### 2.2.3.1. Create conda environment for the build
```bash
conda create -n pt-zentorch python=3.8
conda activate pt-zentorch
conda install pytorch cpuonly -c pytorch
```
#### 2.2.3.2. To build & install the _zentorch_
>Note: To build in debug mode 'export DEBUG=1'
```bash
bash build.sh
```
#### 2.2.3.3. Build Cleanup
```bash
python setup.py clean --all
```
# 3. Usage
General Usage
```python
# Using torch.compile with 'zentorch' as backend
import torch
import zentorch
compiled_model = torch.compile(model, backend='zentorch')
output = compiled_model(input)

# Using make_fx (Deprecated)
from torch.fx.experimental.proxy_tensor import make_fx
import zentorch
# The model should be a fx graph which can be generated with model and input fed to make_fx
model_fx = make_fx(model)(input)
model_optim = zentorch.optimize(model_fx)
output = model_optim(input)
```

>Note: If same model is optimized with `torch.compile` for multiple backends within single script, it is recommended to use `torch._dynamo.reset()` before calling the `torch.compile` on that model.

>Note: _zentorch_ is able to do the zentorch op replacements in both non-inference and inference modes. But some of the _zentorch_ optimizations are only supported for the inference mode, so it is recommended to use `torch.no_grad()` if you are running the model for inference only.

For torchvision CNN models, set `dynamic=False` when calling for `torch.compile` as below:
```python
with torch.no_grad():
  model = torch.compile(model, backend='zentorch', dynamic=False)

  output = model(input)
```

For hugging face NLP models, optimize them as below:
```python
with torch.no_grad():
  model = torch.compile(model, backend='zentorch')

  output = model(input)
```

For hugging face LLM models, optimize them as below:
1. If output is generated through a call to direct `model`, optimize it as below:
```python
with torch.no_grad():
  model = torch.compile(model, backend='zentorch')

  output = model(input)
```

2. If output is generated through a call to `model.forward`, optimize it as below:
```python
with torch.no_grad():
  model.forward = torch.compile(model.forward, backend='zentorch')

  output = model.forward(input)
```

3. If output is generated through a call to `model.generate`, optimize it as below:
    - Optimize the `model.forward` with torch.compile instead of `model.generate`.
    - But still generate the output through a call to `model.generate`.
```python
with torch.no_grad():
  model.forward = torch.compile(model.forward, backend='zentorch')

  output = model.generate(input)
```

# 4. Logging, Profiling and Debugging
## 4.1 ZenDNN logs
Logging for ZenDNN is disabled by default but can be enabled by using the environment variable **ZENDNN_LOG_OPTS** before running any tests. Its behavior can be specified by setting **ZENDNN_LOG_OPTS** to a comma-delimited list of **ACTOR:DBGLVL** pairs. An example to turn on info logging is given below.
```bash
export ZENDNN_LOG_OPTS=ALL:2
```
To enable the profiling logs **zendnn_primitive_create** and **zendnn_primitive_execute**, you can use:
```bash
export ZENDNN_PRIMITIVE_LOG_ENABLE=1
```

For further details on ZenDNN logging mechanism, refer to ZenDNN user-guide from [this page](https://www.amd.com/en/developer/zendnn.html#:~:text=Documentation-,ZenDNN%20User%20Guide,-TensorFlow%20%2B%20ZenDNN%20User).

## 4.2 _zentorch_ logs
For _zentorch_, CPP specific logging can be enabled by setting the environment variable `TORCH_CPP_LOG_LEVEL`. This has four levels: **INFO**, **WARNING**, **ERROR** and **FATAL** in decreasing order of verbosity. Similarly, python logging can be enabled by setting the environment variable `ZENTORCH_PY_LOG_LEVEL`, this has five levels: **DEBUG**, **INFO**, **WARNING**, **ERROR** and **CRITICAL**, again in decreasing order of verbosity. An example to enable INFO level logs for cpp and DEBUG level for python (most verbose) is given below:
```bash
export TORCH_CPP_LOG_LEVEL=INFO
export ZENTORCH_PY_LOG_LEVEL=DEBUG
```
The default level of logs is **WARNING** for both cpp and python sources but can be overridden as discussed above.
>NOTE: The log levels are the same as those provided by the python logging module.

>INFO: Since all OPs implemented in _zentorch_ are registered with torch using the TORCH_LIBRARY() and TORCH_LIBRARY_IMPL() macros in bindings, the PyTorch profiler can be used without any modifications to measure the op level performance.


## 4.3 Saving the graph
Saving of the fx graphs before and after optimization in svg format can be enabled by setting the environment variable `ZENTORCH_SAVE_GRAPH` to 1.
```bash
export ZENTORCH_SAVE_GRAPH=1
```
The graphs will be saved by the names 'native_model.svg' and 'zen_optimized_model.svg', in the parent directory of the script in which the optimize function provided by the _zentorch_ is used.

## 4.4 Support for `TORCH_COMPILE_DEBUG`
PyTorch offers a debugging toolbox that comprises a built-in stats and trace function. This functionality facilitates the display of the time spent by each compilation phase, output code, output graph visualization, and IR dump. `TORCH_COMPILE_DEBUG` invokes this debugging tool that allows for better problem-solving while troubleshooting the internal issues of TorchDynamo and TorchInductor. This functionality works for the models optimized using _zentorch_, so it can be leveraged to debug these models as well. To enable this functionality, users can either set the environment variable `TORCH_COMPILE_DEBUG=1` or specify the environment variable with the runnable file (e.g., test.py) as input.
```bash
# test.py contains model optimized by torch.compile with 'zentorch' as backend
TORCH_COMPILE_DEBUG=1 python test.py
```
For more information about TORCH_COMPILE_DEBUG refer to the official PyTorch documentaion available.

# 5. Additional Utilities:

## 5.1 Disabling Inductor:

This feature is intended for use whenever fx_graphs generated from torch.compile needs to be compared with and without Inductor compilation.

disable_inductor() API takes in a boolean input to disable Inductor. Once disabled, to re-enable inductor, pass "False" to the same API.

```python
import torch
import zentorch

# To disable Torch Inductor
zentorch.disable_inductor(True)
compiled_model = torch.compile(model, backend='zentorch')
output = compiled_model(input)

# To re-enable Torch Inductor
torch._dynamo.reset()
zentorch.disable_inductor(False)
compiled_model = torch.compile(model, backend='zentorch')

```

Fx graphs are sent to AOT Autograd using aot_module_simplified and thus Inductor is not used at all.

## 5.2 _zentorch_ attributes:
To check the version of _zentorch_ use the following command:

```bash
python -c 'import zentorch; print(zentorch.__version__)'
```

To check the build config of _zentorch_ use the following command:
```bash
python -c 'import zentorch; print(*zentorch.__config__.split("\n"), sep="\n")'
```
