# ******************************************************************************
# Copyright (c) 2024 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import pip
import torch
import importlib.util as importutil

_all_ = [
    "transformers",
    "expecttest==0.1.6",
    "parameterized"
]


def install(packages):
    for package in packages:
        pip.main(['install', package])


if __name__ == '__main__':

    install(_all_)
    torch_version = torch.__version__
    torch_version = torch_version.split('+')[0]
    torchvision_compatibilty = {
        "2.0.0" : 'torchvision==0.15.0',
        "2.0.1" : 'torchvision==0.15.2',
        "2.1.0" : 'torchvision==0.16.0',
        "2.1.1" : 'torchvision==0.16.1',
        "2.1.2" : 'torchvision==0.16.2',
        "2.2.0" : 'torchvision==0.17.0',
        "2.2.1" : 'torchvision==0.17.1'
    }
    if importutil.find_spec('torchvision') is not None:
        print("Warning: Torchvision already installed, skipping installing it")
        exit(1)
    elif torch_version in torchvision_compatibilty.keys():
        install([torchvision_compatibilty[torch_version]])
    else:
        print("Couldnot find the valid torchvision version which is \
compatibility with installed torch version. Supported Torch versions \
are 2.0.*/2.1.*/2.2.*")
        exit(1)
