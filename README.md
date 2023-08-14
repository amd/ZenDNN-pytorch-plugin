Copyright &copy; 2023 Advanced Micro Devices, Inc. All rights reserved.

# zentorch: A PyTorch Add-on for AMD  CPUs

<!-- toc -->
- [About zentorch](#about-zentorch)
  - [Overview](#overview)
  - [Structure](#structure)
  - [Third Party Libraries](#third-party-libraries)
- [Installation](#installation)
  - [From Binaries](#from-binaries)
  - [From Source](#from-source)
- [Usage](#usage)
- [Testing](#testing)
- [Linting](#linting)
<!-- tocstop -->

# About zentorch

## Overview
zentorch enables ZenDNN library on top of PyTorch. This plugin enables inference optimizations for deep learning workloads on AMD CPUs. It uses ZenDNN for accelerating basic deep learning ops. ZenDNN is a library which enables performance improvements on AMD CPU architectures. Find the repo for more details [here](https://github.com/amd/ZenDNN). Plugin further accelerates the performance by adding more optimizations at graph(model) level using multiple passes on torch.fx graph. All plugin optimizations can be enabled with a call to optimize function on your model.

## Structure
zentorch consists of three parts. They are
- ZenDNN Integration Code
- Optimize Function
- Build System

### ZenDNN Integration Code
ZenDNN is integrated into plugin using CPP code which interfaces ATen API to ZenDNN's API. This code exports torch compatible API similar to ATen IR of PyTorch. The exported API is made avaiable for usage from python code using PYBIND11 library. PYBIND11 is a header only library. Integration code is linked and compiled into plugin using CppExtension provided by PyTorch.

Following ops are integrated as of now
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

## From Binaries
Create conda or python environment and activate it. Download the wheel file and install it using pip or conda install command

Note: Dependent packages 'numpy' and 'torch' packages will be installed by 'torch_zendnn_plugin'
```bash
pip install torch_zendnn_plugin-*-linux_x86_64.whl
```

## From Source

Note: To use local copies of ZenDNN/BLIS, export ZENDNN_PT_USE_LOCAL_ZENDNN=1, export ZENDNN_PT_USE_LOCAL_BLIS=1.
The libraries should be present/downloaded in the parent directory of current working directory.

### non-manylinux build
#### Create conda environment for the build
```bash
conda create -n pt-plugin python=3.8
conda activate pt-plugin
conda install pytorch cpuonly -c pytorch
```

Note: cmake & ninja are required for cpp extension builds, will be installed through build script

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
Note: Change the current directory to Plugin directory i.e. cd PLUGIN_DIRECTORY
```bash
bash build.sh
```

# Usage

```python
import torch_zendnn_plugin as zentorch
model = zentorch.optimize(model)
output = model(input)
```

# Testing

### To run tests
```bash
python test/test_zendnn.py
```

# Linting

## To run coding format checks with linter script
```bash
bash linter/py_cpp_linter.sh
```
