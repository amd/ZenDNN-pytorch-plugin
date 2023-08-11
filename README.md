Copyright &copy; 2023 Advanced Micro Devices, Inc. All rights reserved.

# PyTorch + ZenDNN: AMD Add-on for CPUs

<!-- toc -->

- [Installation](#installation)
- [Testing](#testing)

<!-- tocstop -->

## Installation

### Create conda environment
```
conda create -n pt-plugin python=3.8
conda activate pt-plugin
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

Note: cmake & ninja are required for cpp extension builds, will be installed through build script

### To build & install the plugin
```
cd pt_plugin
bash build.sh
```

## Testing

### To run tests
```
python test/test_zendnn.py
```