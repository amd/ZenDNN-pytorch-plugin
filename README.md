Copyright &copy; 2023-2024 Advanced Micro Devices, Inc. All rights reserved.

zentorch: A PyTorch Add-on for AMD CPUs
=============

Table of Contents
-------------

<!-- toc -->
- [About zentorch](#1-about-zentorch)
  - [Overview](#11-overview)
  - [Structure](#12-structure)
  - [Third Party Libraries](#13-third-party-libraries)
- [Installation](#2-installation)
  - [From Binaries](#21-from-binaries)
  - [From Source](#22-from-source)
- [Usage](#3-usage)
- [Logging and Profiling](#4-logging-and-profiling)
  - [ZenDNN logs](#41-zendnn-logs)
  - [zentorch logs](#42-zentorch-logs)
  - [Saving the graph](#43-saving-the-graph)
- [Contributing](#5-contributing)
<!-- tocstop -->

# 1. About zentorch

## 1.1. Overview
zentorch enables ZenDNN library on top of PyTorch. This plugin enables inference optimizations for deep learning workloads on AMD CPUs. It uses ZenDNN for accelerating basic deep learning ops. ZenDNN is a library which enables performance improvements on AMD CPU architectures. Find the repo for more details [here](https://github.com/amd/ZenDNN). Plugin further accelerates the performance by adding more optimizations at graph(model) level using multiple passes on torch.fx graph. All plugin optimizations can be enabled with a call to optimize function on models.

## 1.2. Structure
zentorch consists of three parts. They are
- ZenDNN Integration Code
- Optimize Function
- Build System

### 1.2.1. ZenDNN Integration Code
ZenDNN is integrated into plugin using CPP code which interfaces ATen API to ZenDNN's API. This code exports torch compatible API similar to ATen IR of PyTorch. The exported API is made avaiable for usage from python code using PYBIND11 library. PYBIND11 is a header only library. Integration code is linked and compiled into plugin using CppExtension provided by PyTorch.

The following ops are integrated as of now:
- Embedding bag op

### 1.2.2. Optimize Function
`optimize` is a Python function that acts as an interface to leverage CPU related optimizations of the plugin. It takes in the FX based graph in ATen IR and adds on all optimizations of ZenDNN and produces FX based graph as output.

### 1.2.3. Build System

The static libraries for ZenDNN, AOCL BLIS and the cpp Extension modules that bind the ZenDNN operators with Python are built using `setup.py` script.

#### 1.2.3.1. CMake Based Build: ZenDNN and AOCL BLIS
CMake downloads the ZenDNN and AOCL BLIS during configure stage. It generates a config.h with GIT hashes of ZenDNN and AOCL BLIS. It builds both ZenDNN and AOCL BLIS as static libraries.

#### 1.2.3.2. Packaging into a Wheel File
The CPP code, being an extension module, is built through CppExtension. It takes static libraries of the ZenDNN and AOCL BLIS static libraries. `setup.py` also adds in various attributes to the plugin for debugging and providing additional information.


#### 1.2.3.3. build.sh
It installs packages necessary for the plugin build and completes all other steps needed for building the plugin.

Wheel file can be generated solely through the `setup.py` script, without the need for any additional scripts. The build.sh is a tiny wrapper around the Python setup script.

> NOTE: Alternatively PyPA build can be used instead of `setup.py`. Currently minimal support is added for PyPA build. When using PyPA Build, ensure to run `pip install build` prior to building.

## 1.3. Third Party Libraries
Plugin uses following libraries for its functionality.
  * [ZenDNN](https://github.com/amd/ZenDNN)
  * [AOCL BLIS](https://github.com/amd/blis)

# 2. Installation

Zentorch can be installed using binary wheel file or can be built from source itself.

## 2.1. From Binaries
* Create conda or python environment and activate it.
* Uninstall any existing plugin installations.
```bash
pip uninstall zentorch
```
* Download the wheel file and install it using pip or conda install command.

>INFO: Please find the latest Nightly wheel file [here](http://atlvjksapp02:8080/job/ZenDNN_Nightly_Build_for_PT_PLUGIN/lastSuccessfulBuild/)

```bash
pip install zentorch-*-linux_x86_64.whl
```
>Note: 
* Dependent packages 'numpy' and 'torch' will be installed by 'zentorch' if not already present.
* Torch Version should be greater than or equal to 2.0

## 2.2. From Source
Run the following commands:
```bash
git clone "ssh://gerritgit/amd/ec/ZenDNN_PyTorch_Plugin"
cd ZenDNN_PyTorch_Plugin/
```

### 2.2.1. Preparing third party repositories

Build setup downloads AOCL BLIS and ZenDNN repos into `third_party` folder. It can alternatively use local copies of ZenDNN and AOCL BLIS. This is very useful for day to day development scenarios, where developer may be interested in using recent version of repositories. Build setup will switch between local and remote copies of ZenDNN and AOCL BLIS with environmental variables `ZENDNN_PT_USE_LOCAL_ZENDNN` and `ZENDNN_PT_USE_LOCAL_BLIS` respectively. To use local copies of ZenDNN or AOCL BLIS, set `ZENDNN_PT_USE_LOCAL_ZENDNN` or `ZENDNN_PT_USE_LOCAL_BLIS` to 1 respectively. The source repositories should be downloaded/cloned in the directory where plugin is cloned for local setting. Folder structure may look like below.

```
<parent folder>
    |
    |------><AOCL BLIS repo>
    |
    |------><ZenDNN repo>
    |
    |------><ZenDNN_PyTorch_Plugin>
```
>NOTE:
> 1. The recommended values of `ZENDNN_PT_USE_LOCAL_ZENDNN` and `ZENDNN_PT_USE_LOCAL_BLIS` are 1 and 0 respectively. Default values are the same as recommended values.
>```bash
>export ZENDNN_PT_USE_LOCAL_ZENDNN=1
>export ZENDNN_PT_USE_LOCAL_BLIS=0
>```
> 2. ZenDNN repository can be cloned using command<br> `git clone "ssh://gerritgit/amd/ec/ZenDNN"`
> 3. AOCL BLIS can be cloned using command<br> `git clone "ssh://gerritgit/cpulibraries/er/blis"`

### 2.2.2. Linux build
#### 2.2.2.1. Create conda environment for the build
```bash
conda create -n pt-plugin python=3.8
conda activate pt-plugin
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

>Note: cmake & ninja are required for cpp extension builds, will be installed through build script.

#### 2.2.2.3. To build & install the plugin
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
conda create -n pt-plugin python=3.8
conda activate pt-plugin
conda install pytorch cpuonly -c pytorch
```
#### 2.2.3.2. To build & install the plugin
>Note: To build in debug mode 'export DEBUG=1'
```bash
bash build.sh
```

# 3. Usage
```python
#Using torch.compile
import torch
import zentorch
compiled_model = torch.compile(model, backend='zentorch')
output = compiled_model(input)

#Using make_fx (Deprecated)
from torch.fx.experimental.proxy_tensor import make_fx
import zentorch
# The model should be a fx graph which can be generated with model and input fed to make_fx
model_fx = make_fx(model)(input)
model_optim = zentorch.optimize(model_fx)
output = model_optim(input)
```

# 4. Logging and Profiling
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

## 4.2 zentorch logs
For zentorch, CPP specific logging can be enabled by setting the environment variable `TORCH_CPP_LOG_LEVEL`. This has four levels: **INFO**, **WARNING**, **ERROR** and **FATAL** in decreasing order of verbosity. Similarly, python logging can be enabled by setting the environment variable `ZENTORCH_PY_LOG_LEVEL`, this has five levels: **DEBUG**, **INFO**, **WARNING**, **ERROR** and **CRITICAL**, again in decreasing order of verbosity. An example to enable INFO level logs for cpp and DEBUG level for python (most verbose) is given below:
```bash
export TORCH_CPP_LOG_LEVEL=INFO
export ZENTORCH_PY_LOG_LEVEL=DEBUG
```
The default level of logs is **WARNING** for both cpp and python sources but can be overridden as discussed above.
>NOTE: The log levels are the same as those provided by the python logging module.

>INFO: Since all OPs implemented in zentorch are registered with torch using the TORCH_LIBRARY() and TORCH_LIBRARY_IMPL() macros in bindings, the PyTorch profiler can be used without any modifications to measure the op level performance.


## 4.3 Saving the graph
Saving of the fx graphs before and after optimization in svg format can be enabled by setting the environment variable `ZENTORCH_SAVE_GRAPH` to 1.
```bash
export ZENTORCH_SAVE_GRAPH=1
```
The graphs will be saved by the names 'native_model.svg' and 'zen_optimized_model.svg', in the parent directory of the script in which the optimize function provided by the plugin is used.

# 5. Additional Utilities:

## 5.1 Disabling Inductor:

This feature is intended for use whenever fx_graphs generated from torch.compile needs to be compared with and without Inductor compilation. 

disable_inductor() API takes in a boolean input to disable Inductor. Once disabled, to re-enable inductor, pass "False" to the same API.

```python
import torch
import zentorch

#To disable Torch Inductor
zentorch.disable_inductor(True)
compiled_model = torch.compile(model, backend='zentorch')
output = compiled_model(input)

#To re-enable Torch Inductor
torch._dynamo.reset()
zentorch.disable_inductor(False)
compiled_model = torch.compile(model, backend='zentorch')

```

Fx graphs are sent to AOT Autograd using aot_module_simplified and thus Inductor is not used at all.

# 6. Contributing
Any contribution to zentorch can be done by following the guidelines given [here](CONTRIBUTING.md).
