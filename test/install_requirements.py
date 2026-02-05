# ******************************************************************************
# Copyright (c) 2023-2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

from importlib.util import find_spec
from importlib.metadata import version
import torch
import subprocess
import sys

BASE_REQUIREMENTS = ["transformers", "expecttest==0.1.6", "parameterized"]


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

    install(BASE_REQUIREMENTS)
    extra_args = (
        " --index-url https://download.pytorch.org/whl/cpu"
        if not torch.version.cuda
        else ""
    )
    torch_version = torch.__version__
    torch_version = torch_version.split("+")[0]
    torchvision_compatibility = {
        # "2.0.0": "torchvision==0.15.0",
        # "2.0.1": "torchvision==0.15.2",
        # "2.1.0": "torchvision==0.16.0",
        # "2.1.1": "torchvision==0.16.1",
        # "2.1.2": "torchvision==0.16.2",
        # "2.2.0": "torchvision==0.17.0",
        # "2.2.1": "torchvision==0.17.1",
        # "2.2.2": "torchvision==0.17.2",
        # "2.3.0": "torchvision==0.18.0",
        # "2.3.1": "torchvision==0.18.1",
        # "2.4.0": "torchvision==0.19.0",
        # "2.4.1": "torchvision==0.19.1",
        # "2.5.0": "torchvision==0.20.0",
        # "2.5.1": "torchvision==0.20.1",
        "2.6.0": "torchvision==0.21.0",
        "2.7.0": "torchvision==0.22.0",
        "2.8.0": "torchvision==0.23.0",
        "2.9.0": "torchvision==0.24.0",
        "2.9.1": "torchvision==0.24.1",
        "2.10.0": "torchvision==0.25.0",
    }

    torchao_compatibility = {
        "2.9.1": "torchao==0.15.0",
        # Torch 2.10 might give a warning suggesting to install torchao of latest version, but 0.15.0 is the latest
        # version currently.
        "2.10.0": "torchao==0.15.0",
    }

    if find_spec("torchao") is not None:
        if torch_version not in torchao_compatibility:
            print("Torch version not supported for torchao compatibility check")
            sys.exit(1)
        pkg_version = version("torchao").split("+")[0]
        torchao_compatible_version = torchao_compatibility[torch_version].split("==")[1]
        if pkg_version != torchao_compatible_version:
            print(
                f"Torchao version is not compatible with the installed torch version. \
                    Expected: {torchao_compatible_version}, Found: {pkg_version}"
            )
            sys.exit(1)
    else:
        if torch_version in torchao_compatibility:
            torchao_cmd = [torchao_compatibility[torch_version] + extra_args]
            install(torchao_cmd)
        else:
            print("Could not find the valid torchao version which is \
                compatible with installed torch version. Supported Torch versions \
                are 2.9.1 and 2.10.0")
            sys.exit(1)

    if find_spec("torchvision") is not None:
        if torch_version not in torchvision_compatibility:
            print("Torch version not supported for torchvision compatibility check")
            sys.exit(1)
        pkg_version = version("torchvision").split("+")[0]
        torchvision_compatible_version = torchvision_compatibility[torch_version].split(
            "=="
        )[1]
        if pkg_version != torchvision_compatible_version:
            print(
                f"Torchvision version is not compatible with the installed torch version. \
                    Expected: {torchvision_compatible_version}, Found: {pkg_version}"
            )
            sys.exit(1)
    else:
        if torch_version in torchvision_compatibility:
            torchvision_cmd = [torchvision_compatibility[torch_version] + extra_args]
            install(torchvision_cmd)
        else:
            print("Could not find the valid torchvision version which is \
                compatible with installed torch version. Supported Torch versions \
                are 2.6.0/2.7.0/2.8.0/2.9.0/2.9.1/2.10.0")
            sys.exit(1)
