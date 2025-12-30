# ******************************************************************************
# Copyright (c) 2023-2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension
from packaging.version import parse
from torch.torch_version import __version__, TorchVersion
from os.path import join as Path
import os
import subprocess
import torch
import warnings

if parse(__version__) < parse("2.8"):
    raise ImportError(
        "zentorch Plugin requires torch version \
     2.8 or higher. Please upgrade your torch version \
        and retry the build."
    )

if parse(__version__) < parse("2.9.1"):
    warnings.warn(
        "Consider upgrading to torch version 2.9.1 for improved performance.",
        stacklevel=1,
    )


class CustomBuildExtension(BuildExtension):
    def run(self) -> None:
        """
        Invoke the CMAKE compilation commands.
        """
        # Env variables set to copy ZenDNN/BLIS from local
        # if variables not set: then use default values
        if "ZENTORCH_USE_LOCAL_BLIS" not in os.environ:
            os.environ["ZENTORCH_USE_LOCAL_BLIS"] = "0"
        if "ZENTORCH_USE_LOCAL_ZENDNN" not in os.environ:
            os.environ["ZENTORCH_USE_LOCAL_ZENDNN"] = "0"

        #  self.build_temp is created as part of following line
        os.makedirs(os.path.join(self.build_temp, "lib"), exist_ok=True)
        os.makedirs(os.path.join(self.build_lib), exist_ok=True)

        rc, out, err = subproc_communicate("which python")
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

        self.spawn(["make", "-j", str(os.cpu_count()), "-C", self.build_temp])

        super().run()

    def build_extensions(self) -> None:
        """
        Dynamically add the static libraries generated during compile phase.
        """
        project_root_dir = os.path.abspath(os.path.dirname(__file__))

        extension = self.extensions[0]

        extra_objects = [
            Path(project_root_dir, self.build_lib, PACKAGE_NAME, "libzentorch.so"),
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


def get_required_ipex_version(pt_version):
    # This function returns the most suitable version of
    # intel_extension_for_pytorch to be installed, which is required for
    # zentorch.llm.optimize to work. This version is based on the
    # version of torch that is being used.

    torch_version = str(pt_version).split(".")
    torch_major_version = torch_version[0]
    torch_minor_version = torch_version[1]
    torch_major_minor_version = TorchVersion(
        ".".join([torch_major_version, torch_minor_version])
    )

    # Check for minimum torch version of 2.7. This is required for
    # intel_extension_for_pytorch to work under the hood.
    if torch_major_minor_version >= TorchVersion("2.7.0"):
        required_ipex_version = ".".join(
            [torch_major_version, torch_minor_version, "0"]
        )

        # Behaviour of tilda in the installation process
        # "torch~=2.x" installs PT 2.8. i.e Latest in 2.x
        # "torch~=2.7.x" installs PT 2.7.0 i.e Latest in 2.7.x
        return f"intel_extension_for_pytorch~={required_ipex_version}"


#   ZenTorch_BUILD_VERSION
#     specify the version of zentorch, rather than the hard-coded version
#     in this file; used when we're building binaries for distribution

# Define env values
PACKAGE_NAME = "zentorch"
PACKAGE_VERSION = "5.2.0"
PT_VERSION = __version__

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
    "-Wall",
    "-Werror",
    "-fopenmp",
    "-Wno-unknown-pragmas",
    "-DZENTORCH_VERSION_HASH=" + git_sha,
    "-DZENTORCH_VERSION=" + PACKAGE_VERSION,
    "-DPT_VERSION=" + PT_VERSION,
]

# Enable C++11 ABI compilation for zentorch
# if PyTorch was built with ABI support.
zentorch_compile_args += [
    f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}"
]

# add the "-O2" optimization only when we are doing release build
# check for release build
if not os.getenv("DEBUG", 0):
    zentorch_compile_args += ["-O2"]


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
    PACKAGE_NAME + ".vllm"
]
extras_require = {}

# maybe_valid_ipex_version will contain either the valid ipex version
# if torch version is greater than or equal to 2.3.0.
# If not, maybe_valid_ipex_version will be None.
maybe_valid_ipex_version = get_required_ipex_version(PT_VERSION)
if maybe_valid_ipex_version:
    # pip install zentorch[llm] # ipex
    # pip install zentorch
    extras_require["llm"] = maybe_valid_ipex_version

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
                extra_link_args=['-Wl,-rpath,$ORIGIN'],
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
        },
    )


if __name__ == "__main__":
    main()
