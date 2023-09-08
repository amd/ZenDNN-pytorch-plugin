<style type="text/css">
body { counter-reset: h0counter h1counter h2counter h3counter h4counter h5counter h6counter; }

h0 { counter-reset: h1counter; }
h1 { counter-reset: h2counter; }
h2 { counter-reset: h3counter; }
h3 { counter-reset: h4counter; }
h4 { counter-reset: h5counter; }
h5 { counter-reset: h6counter; }
h6 {}

h1:before {
    counter-increment: h1counter;
    content: counter(h1counter) ".\0000a0\0000a0";
}

h2:before {
    counter-increment: h2counter;
    content: counter(h1counter) "." counter(h2counter) ".\0000a0\0000a0";
}

h3:before {
    counter-increment: h3counter;
    content: counter(h1counter) "." counter(h2counter) "." counter(h3counter) ".\0000a0\0000a0";
}

h4:before {
    counter-increment: h4counter;
    content: counter(h1counter) "." counter(h2counter) "." counter(h3counter) "." counter(h4counter) ".\0000a0\0000a0";
}

h5:before {
    counter-increment: h5counter;
    content: counter(h1counter) "." counter(h2counter) "." counter(h3counter) "." counter(h4counter) "." counter(h5counter) ".\0000a0\0000a0";
}

h6:before {
    counter-increment: h6counter;
    content: counter(h1counter) "." counter(h2counter) "." counter(h3counter) "." counter(h4counter) "." counter(h5counter) "." counter(h6counter) ".\0000a0\0000a0";
}
</style>

Copyright &copy; 2023 Advanced Micro Devices, Inc. All rights reserved.

<font size="+3"><center>
**zentorch: A PyTorch Add-on for AMD CPUs**
___
</center></font>

<!-- toc -->
- [About zentorch](#about-zentorch)
  - [Overview](#overview)
  - [Structure](#structure)
  - [Third Party Libraries](#third-party-libraries)
- [Installation](#installation)
  - [From Binaries](#from-binaries)
  - [From Source](#from-source)
- [Usage](#usage)
- [Contributing](#contributing)
<!-- tocstop -->

# About zentorch

## Overview
zentorch enables ZenDNN library on top of PyTorch. This plugin enables inference optimizations for deep learning workloads on AMD CPUs. It uses ZenDNN for accelerating basic deep learning ops. ZenDNN is a library which enables performance improvements on AMD CPU architectures. Find the repo for more details [here](https://github.com/amd/ZenDNN). Plugin further accelerates the performance by adding more optimizations at graph(model) level using multiple passes on torch.fx graph. All plugin optimizations can be enabled with a call to optimize function on models.

## Structure
zentorch consists of three parts. They are
- ZenDNN Integration Code
- Optimize Function
- Build System

### ZenDNN Integration Code
ZenDNN is integrated into plugin using CPP code which interfaces ATen API to ZenDNN's API. This code exports torch compatible API similar to ATen IR of PyTorch. The exported API is made avaiable for usage from python code using PYBIND11 library. PYBIND11 is a header only library. Integration code is linked and compiled into plugin using CppExtension provided by PyTorch.

The following ops are integrated as of now:
- Embedding bag op

### Optimize Function
`optimize` is a Python function that acts as an interface to leverage CPU related optimizations of the plugin. It takes in the FX based graph in ATen IR and adds on all optimizations of ZenDNN and produces FX based graph as output.

### Build System

The static libraries for ZenDNN, AOCL BLIS and the cpp Extension modules that bind the ZenDNN operators with Python are built using `setup.py` script.

#### CMake Based Build: ZenDNN and AOCL BLIS
CMake downloads the ZenDNN and AOCL BLIS during configure stage. It generates a config.h with GIT hashes of ZenDNN and AOCL BLIS. It builds both ZenDNN and AOCL BLIS as static libraries.

#### Packaging into a Wheel File
The CPP code, being an extension module, is built through CppExtension. It takes static libraries of the ZenDNN and AOCL BLIS static libraries. `setup.py` also adds in various attributes to the plugin for debugging and providing additional information.


#### build.sh
It installs packages necessary for the plugin build and completes all other steps needed for building the plugin.

Wheel file can be generated solely through the `setup.py` script, without the need for any additional scripts. The build.sh is a tiny wrapper around the Python setup script.

> NOTE: Alternatively PyPA build can be used instead of `setup.py`. Currently minimal support is added for PyPA build. When using PyPA Build, ensure to run `pip install build` prior to building.

## Third Party Libraries
Plugin uses following libraries for its functionality.
  * [ZenDNN](https://github.com/amd/ZenDNN)
  * [AOCL BLIS](https://github.com/amd/blis)

# Installation

Zentorch can be installed using binary wheel file or can be built from source itself.

## From Binaries
* Create conda or python environment and activate it.
* Uninstall any existing plugin installations.
```bash
pip uninstall torch_zendnn_plugin
```
* Download the wheel file and install it using pip or conda install command.

```bash
pip install torch_zendnn_plugin-*-linux_x86_64.whl
```
>Note: Dependent packages 'numpy' and 'torch' will be installed by 'torch_zendnn_plugin' if not already present.

## From Source

### preparing third party repositories

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

### Linux build
#### Create conda environment for the build
```bash
conda create -n pt-plugin python=3.8
conda activate pt-plugin
```
#### You can install torch using 'conda' or 'pip'
```bash
# Pip command
pip install torch --index-url https://download.pytorch.org/whl/cpu
```
or
```bash
# Conda command
conda install pytorch cpuonly -c pytorch
```

>Note: cmake & ninja are required for cpp extension builds, will be installed through build script

#### To build & install the plugin
```bash
bash build.sh
```

### manylinux build
'manylinux_setup' script allows you to create a docker container, you need to install conda and proceed the build process

```bash
sudo bash manylinux_setup.sh
export SCRIPT_TYPE=many_linux
```

#### Create conda environment for the build
```bash
conda create -n pt-plugin python=3.8
conda activate pt-plugin
conda install pytorch cpuonly -c pytorch
```
#### To build & install the plugin
>Note: Change the current directory to Plugin directory i.e. `cd ZenDNN_PyTorch_Plugin`
```bash
bash build.sh
```

# Usage

```python
import torch_zendnn_plugin as zentorch
model_optim = zentorch.optimize(model_fx)
output = model_optim(input)
```

The model should be a fx graph which can be generated with model and input fed to make_fx, sample code below:

```python
from torch.fx.experimental.proxy_tensor import make_fx
model_fx = make_fx(model)(input)
```

# Contributing
Contributing guidelines for zentorch can be found [here](CONTRIBUTING.md).
