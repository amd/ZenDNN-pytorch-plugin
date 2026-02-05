# ******************************************************************************
# Copyright (c) 2023-2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension
from packaging.version import parse
from torch.torch_version import __version__
from os.path import join as Path
import os
import subprocess
import sys
import torch
import warnings

IS_WINDOWS = sys.platform == "win32"

if parse(__version__) < parse("2.9.1"):
    raise ImportError(
        "zentorch Plugin requires torch version \
     2.9.1 or higher. Please upgrade your torch version \
        and retry the build."
    )

if parse(__version__) < parse("2.10"):
    warnings.warn(
        "Consider upgrading to torch version 2.10 for improved performance.",
        stacklevel=1,
    )


class CustomBuildExtension(BuildExtension):
    def run(self) -> None:
        """
        Invoke the CMAKE compilation commands.
        """
        # Env variables set to copy ZenDNN from local
        # if variables not set: then use default values
        if "ZENTORCH_USE_LOCAL_ZENDNN" not in os.environ:
            os.environ["ZENTORCH_USE_LOCAL_ZENDNN"] = "0"

        #  self.build_temp is created as part of following line
        os.makedirs(os.path.join(self.build_temp, "lib"), exist_ok=True)
        os.makedirs(os.path.join(self.build_lib), exist_ok=True)

        rc, out, err = subproc_communicate(
            "where python" if IS_WINDOWS else "which python"
        )
        if rc == 0:
            out = out.split("\n")[0]
            os.environ["PYTHON_PATH"] = out.strip()
        else:
            print("Issue with getting the python path")
            exit(1)

        build_type = "Debug" if os.getenv("DEBUG", 0) == "1" else "Release"

        # Finding torch cmake path that will be used to find Torch.cmake
        torch_cmake_prefix_path = torch.utils.cmake_prefix_path
        working_dir = os.path.abspath(os.path.dirname(__file__))

        build_shared_libs = "OFF"

        cmake_cmd = [
            "cmake",
            "-S",
            working_dir,
            "-B",
            self.build_temp,
            f"-DCMAKE_BUILD_TYPE={build_type}",
            f"-DBUILD_SHARED_LIBS={build_shared_libs}",
            f"-DCMAKE_PREFIX_PATH={torch_cmake_prefix_path}",
            f"-DINSTALL_LIB_DIR={self.build_lib}",
        ]

        # Add compile flags to cmake
        cmake_cmd.append(f"-DCMAKE_CXX_FLAGS={' '.join(zentorch_compile_args)}")

        self.spawn(cmake_cmd)

        build_cmd = [
            "cmake",
            "--build",
            self.build_temp,
            "--config",
            build_type,
            "--parallel",
            str(os.cpu_count()),
        ]
        self.spawn(build_cmd)

        super().run()

    def build_extensions(self) -> None:
        """
        Dynamically add the static libraries generated during compile phase.
        """
        project_root_dir = os.path.abspath(os.path.dirname(__file__))

        extension = self.extensions[0]

        if IS_WINDOWS:
            zentorch_lib = "zentorch.dll"
        else:
            zentorch_lib = "libzentorch.so"

        extra_objects = [
            Path(project_root_dir, self.build_lib, PACKAGE_NAME, zentorch_lib),
        ]

        extension.extra_objects.extend(extra_objects)

        super().build_extensions()


def subproc_communicate(cmd):

    p1 = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p1.communicate()
    rc = p1.returncode
    return rc, out.decode("ascii", "ignore"), err.decode("ascii", "ignore")


def get_commit_hash(base_dir):
    cwd = os.getcwd()
    os.chdir(base_dir)
    rc, out, err = subproc_communicate("git rev-parse HEAD")
    if rc == 0:
        git_sha = out.strip()
    else:
        print("Issue with getting the GIT hash of %s" % base_dir)
        exit(1)
    os.chdir(cwd)
    return git_sha


#   ZenTorch_BUILD_VERSION
#     specify the version of zentorch, rather than the hard-coded version
#     in this file; used when we're building binaries for distribution

# Define env values
PACKAGE_NAME = "zentorch"
PACKAGE_VERSION = "5.2.0"
PT_VERSION = __version__
ZENTORCH_VLLM_PLUGIN_BUILD = os.getenv("ZENTORCH_VLLM_PLUGIN_BUILD", "1") != "0"

# Initializing all the parameters for the setup function
project_root_dir = os.path.abspath(os.path.dirname(__file__))
sources = [Path(project_root_dir, "src", "cpu", "cpp", "Bindings.cpp")]

include_dirs = [
    Path(project_root_dir, "third_party", "ZenDNN", "inc"),
    Path(project_root_dir, "third_party", "ZenDNN", "third_party", "fbgemm", "include"),
]

zentorch_build_version = os.getenv("ZenTorch_BUILD_VERSION", PACKAGE_VERSION)
git_sha = get_commit_hash(project_root_dir)
wheel_file_dependencies = [
    "numpy",
    "torch",
    "deprecated",
    "safetensors",
]
# -Wno-unknown-pragma is for [unroll pragma], to be removed
# -fopenmp is needed for omp related pragmas (simd etc.)
zentorch_compile_args = [
    "-DZENTORCH_VERSION_HASH=" + git_sha,
    "-DZENTORCH_VERSION=" + PACKAGE_VERSION,
    "-DPT_VERSION=" + PT_VERSION,
]

if IS_WINDOWS:
    zentorch_compile_args += [
        "/W3",
        "/WX",
        "/openmp",
    ]
else:
    zentorch_compile_args += [
        "-Wall",
        "-Werror",
        "-fopenmp",
        "-Wno-unknown-pragmas",
    ]

# Enable C++11 ABI compilation for zentorch
# if PyTorch was built with ABI support.
if not IS_WINDOWS:
    zentorch_compile_args += [
        f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}"
    ]

# add the optimization flag only when we are doing release build
# check for release build
if not os.getenv("DEBUG", 0):
    zentorch_compile_args += ["/O2"] if IS_WINDOWS else ["-O2"]


long_description = ""
with open(Path(project_root_dir, "DESCRIPTION.md"), encoding="utf-8") as f:
    long_description = f.read()

config_file = "_build_info.py"

_build_info_path = os.path.join(
    project_root_dir, "src", "cpu", "python", PACKAGE_NAME, config_file
)

_build_config = "# PyTorch Build Version:\n"
_build_config += '__torchversion__ = "{}"\n'.format(PT_VERSION)

packages = [
    PACKAGE_NAME,
    PACKAGE_NAME + ".llm",
]
if ZENTORCH_VLLM_PLUGIN_BUILD:
    packages.append(PACKAGE_NAME + ".vllm")
extras_require = {}

with open(_build_info_path, "w") as f:
    f.write(_build_config)
    f.close()


def main():
    setup(
        name=PACKAGE_NAME,
        version=zentorch_build_version,
        description="zentorch : A PyTorch* extension for AMD EPYC CPUs.",
        long_description=long_description,
        long_description_content_type="text/markdown",
        author_email="zendnn.maintainers@amd.com",
        author="AMD",
        # URL needs to be updates once the plugin is open sourced
        url="https://developer.amd.com/zendnn",
        # license needs to be added when the source code gets the license
        license="MIT",
        keywords="pytorch tensor machine learning plugin ZenDNN AMD",
        install_requires=wheel_file_dependencies,
        ext_modules=[
            CppExtension(
                name=f"{PACKAGE_NAME}._C",
                sources=sources,
                include_dirs=include_dirs,
                extra_compile_args=zentorch_compile_args,
                extra_link_args=[] if IS_WINDOWS else ['-Wl,-rpath,$ORIGIN'],
            )
        ],
        cmdclass={
            "build_ext": CustomBuildExtension,
        },
        packages=packages,
        package_dir={"": Path("src", "cpu", "python")},
        extras_require=extras_require,
        entry_points={
            # vLLM will import this automatically when the wheel is present
            "vllm.platform_plugins": [
                "zentorch = zentorch.vllm:register",
            ],
            "vllm.general_plugins": [
                # same callable, but this group is invoked for runtime patches
                "zentorch_general = zentorch.vllm:register",
            ],
        } if ZENTORCH_VLLM_PLUGIN_BUILD else {},
    )


if __name__ == "__main__":
    main()
