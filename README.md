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
<!-- tocstop -->

# About zentorch

## Overview
zentorch enables ZenDNN library on top of PyTorch. This plugin will enable inference optimizations for deep learning workloads on AMD CPUs. It uses ZenDNN for accelerating basic deep learning ops. ZenDNN is a library which enables performance improvements on AMD CPU architectures. Find the repo for more details [here](https://github.com/amd/ZenDNN). Plugin further accelerates the performance by adding more optimizations at graph(model) level using multiple passes on torch.fx graph. All plugin optimizations can be enabled with a call to optimize function on your model.

## Structure
zentorch consists of three parts. They are
- ZenDNN integration code
- optimize function
- build system

### ZenDNN integration code
ZenDNN is integrated into plugin using CPP code which interfaces ATen API to ZenDNN's API. This code exports torch compatible API similar to ATen IR of PyTorch. The exported API is made avaiable for usage from python code using PYBIND11 library. PYBIND11 is a header only library. Integration code is linked and compiled into plugin using CppExtension provided by PyTorch.

Following ops are integrated as of now
- embedding bag op

### optimize function
optimize function is written in Python. It acts as an interface to leverage all CPU related optimizations of the plugin. It takes in the fx based graph in ATen IR and adds on all optimizations of ZenDNN and produces FX based graph as output.

### build system
#### CMake based build of ZenDNN and AOCL BLIS
CMake downloads the ZenDNN and AOCL BLIS during configure stage. It generates a config.h with GIT hashes of ZenDNN and AOCL BLIS. It builds both ZenDNN and AOCL BLIS as static libraries.
#### Setup.py based plugin build
It builds CPP code through CppExtension and python code into the plugin. It takes static libraries of the ZenDNN and AOCL BLIS static libraries. It also adds in various attributes to the plugin for debugigng and information purposes.
#### build.sh
It installs packages necessary for the plugin build and completes all other steps needed for building the plugin.

## Third Party Libraries
Plugin uses following libraries for its functionality.
  * [ZenDNN](https://github.com/amd/ZenDNN)
  * [AOCL BLIS](https://github.com/amd/blis)

# Installation

## From Binaries
Create conda or python environment and activate it. Download the wheel file and install it using pip or conda install command
```
pip install numpy
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install torch_zendnn_plugin-*-linux_x86_64.whl
```

## From Source
### Create conda environment for the build
```
conda create -n pt-plugin python=3.8
conda activate pt-plugin
conda install pytorch cpuonly -c pytorch
```

Note: cmake & ninja are required for cpp extension builds, will be installed through build script

### To build & install the plugin
```
bash build.sh
```
# Usage

```
import torch_zendnn_plugin as zentorch
model = zentorch.optimize(model)
output = model(input)
```

# Testing

### To run tests
```
python test/test_zendnn.py
```