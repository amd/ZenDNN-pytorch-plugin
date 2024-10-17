# ******************************************************************************
# Copyright (c) 2023-2024 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import importlib.util as importutil

import pip
import torch

_all_ = ["transformers", "expecttest==0.1.6", "parameterized"]


def install(packages):
    for package in packages:
        pip.main(["install", package])


if __name__ == "__main__":

    install(_all_)

    extra_args = (
        ["--index-url", "https://download.pytorch.org/whl/cpu"]
        if not torch.version.cuda
        else []
    )
    torch_version = torch.__version__
    torch_version = torch_version.split("+")[0]
    torchvision_compatibilty = {
        "2.0.0": "torchvision==0.15.0",
        "2.0.1": "torchvision==0.15.2",
        "2.1.0": "torchvision==0.16.0",
        "2.1.1": "torchvision==0.16.1",
        "2.1.2": "torchvision==0.16.2",
        "2.2.0": "torchvision==0.17.0",
        "2.2.1": "torchvision==0.17.1",
        "2.2.2": "torchvision==0.17.2",
        "2.3.0": "torchvision==0.18.0",
        "2.3.1": "torchvision==0.18.1",
        "2.4.0": "torchvision==0.19.0",
        "2.4.1": "torchvision==0.19.1",
        "2.5.0": "torchvision==0.20.0",
    }
    if importutil.find_spec("torchvision") is not None:
        print("Warning: Torchvision already installed, skipping installing it")
        exit(1)
    elif torch_version in torchvision_compatibilty.keys():
        pip.main(["install", torchvision_compatibilty[torch_version], *extra_args])
    else:
        print(
            "Couldnot find the valid torchvision version which is \
compatibility with installed torch version. Supported Torch versions \
are 2.0.*/2.1.*/2.2.*/2.3.*/2.4.*/2.5.*"
        )
        exit(1)
