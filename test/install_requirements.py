# ******************************************************************************
# Copyright (c) 2023-2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import importlib.util as importutil
import torch
import subprocess

_all_ = ["transformers==4.48.0", "expecttest==0.1.6", "parameterized"]


def install_package(cmd):
    """
    Installs a package using the provided command.

    This function uses subprocess.Popen to run the given command in a new
    subprocess, which helps to avoid conflicts that might arise from running
    pip.main() within the current Python process.
    Args:
        cmd (str): The command to run for installing the package, typically a
        pip install command.
    Returns:
        tuple: A tuple containing the return code (int), standard output (str),
        and standard error (str).
    """
    p1 = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p1.communicate()
    rc = p1.returncode
    return rc, out.decode("ascii", "ignore"), err.decode("ascii", "ignore")


def install(packages):
    for package in packages:
        rc, out, err = install_package("pip install %s" % package)
        if rc != 0:
            print("Issue while installing the package=%s" % package)
            print(err)
            exit(1)
        else:
            print(out)


if __name__ == "__main__":

    install(_all_)
    extra_args = (
        " --index-url https://download.pytorch.org/whl/cpu"
        if not torch.version.cuda
        else ""
    )
    torch_version = torch.__version__
    torch_version = torch_version.split("+")[0]
    torchvision_compatibilty = {
        "2.6.0": "torchvision==0.21.0",
        "2.7.0": "torchvision==0.22.0",
    }
    if importutil.find_spec("torchvision") is not None:
        print("Warning: Torchvision already installed, skipping installing it")
        exit(1)
    elif torch_version in torchvision_compatibilty:
        torchvision_cmd = [torchvision_compatibilty[torch_version] + extra_args]
        install(torchvision_cmd)
    else:
        print(
            "Couldnot find the valid torchvision version which is \
compatibility with installed torch version. Supported Torch versions \
are 2.6.*/2.7.*"
        )
        exit(1)
