#******************************************************************************
# Copyright (c) 2023 Advanced Micro Devices, Inc.
# All rights reserved.
#******************************************************************************

from setuptools import setup, Command
from torch.utils.cpp_extension import BuildExtension, CppExtension
import os, shutil, torch
import glob, subprocess


#   ZenTorch_BUILD_VERSION
#     specify the version of torch_zendnn_plugin, rather than the hard-coded version
#     in this file; used when we're building binaries for distribution

# Define env values
PACKAGE_NAME = "torch_zendnn_plugin"
PACKAGE_VERSION = "1.0.0"

def get_build_version(base_dir): 
    git_sha = (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=base_dir)
        .decode("ascii")
        .strip()
    )
    zen_version = os.getenv("ZenTorch_BUILD_VERSION", PACKAGE_VERSION + "+git" + git_sha[:7])
    return zen_version


project_root_dir = os.path.abspath(os.path.dirname(__file__))
sources = glob.glob(os.path.join(project_root_dir, 'src/cpu/cpp/*.cpp'))
include_dirs = [os.path.join(project_root_dir, "third_party/ZenDNN/inc"), os.path.join(project_root_dir, "third_party/blis/include/amdzen")]
extra_objects = [os.path.join(project_root_dir, "build/lib/libamdZenDNN.a"), os.path.join(project_root_dir, "build/lib/libblis-mt.a")]
torch_zendnn_plugin_build_version = get_build_version(project_root_dir)
wheel_file_dependencies = []

long_description = ""
with open(os.path.join(project_root_dir, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

def main():
    setup(
        name=PACKAGE_NAME,
        version=torch_zendnn_plugin_build_version,
        description = "ZenDNN plugin for PyTorch*",
        long_description = long_description,
        long_description_content_type="text/markdown",
        author_email= "",
        author = "",
        # URL needs to be updates once the plugin is open sourced
        url = "",
        # license needs to be added when the source code gets the license
        license  = "",
        install_requires=wheel_file_dependencies,
        ext_modules=[
            CppExtension(
                name=f'{PACKAGE_NAME}._C',
                sources=sources,
                extra_objects=extra_objects,
                include_dirs=include_dirs,
                extra_compile_args=['-Werror']
                )],
            cmdclass={'build_ext': BuildExtension,},
            packages=[PACKAGE_NAME],
            package_dir={"":"src/cpu/python"}
            )

if __name__ == '__main__':
    main()
