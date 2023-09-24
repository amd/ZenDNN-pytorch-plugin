# ******************************************************************************
# Copyright (c) 2023 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension
import os
import glob
import shutil
import subprocess


class CustomBuildExtension(BuildExtension):
    def run(self) -> None:
        """
        Invoke the CMAKE compilation commands.
        """
        # Env variables set to copy ZenDNN/BLIS from local
        # After ZenDNN4.1 release ZENDNN_PT_USE_LOCAL_ZENDNN should be set to 0

        # if variables not set: then use default values
        if "ZENDNN_PT_USE_LOCAL_BLIS" not in os.environ:
            os.environ["ZENDNN_PT_USE_LOCAL_BLIS"] = "0"
        if "ZENDNN_PT_USE_LOCAL_ZENDNN" not in os.environ:
            os.environ["ZENDNN_PT_USE_LOCAL_ZENDNN"] = "1"

        if os.path.exists(self.build_temp):
            shutil.rmtree(self.build_temp)
        os.makedirs(self.build_temp, exist_ok=True)

        build_type = "Debug" if os.getenv("DEBUG", 0) == "1" else "Release"
        working_dir = os.path.abspath(os.path.dirname(__file__))
        cmake_cmd = ["cmake", "-S", working_dir, "-B", self.build_temp,
                     f"-DCMAKE_BUILD_TYPE={build_type}"]
        self.spawn(cmake_cmd)
        self.spawn(["make", "-j", "-C", self.build_temp])

        super().run()

    def build_extensions(self) -> None:
        """
        Dynamically add the static libraries generated during compile phase.
        """
        project_root_dir = os.path.abspath(os.path.dirname(__file__))

        extension = self.extensions[0]

        extra_objects = [
            os.path.join(project_root_dir, self.build_temp, "lib/libamdZenDNN.a"),
            os.path.join(project_root_dir, self.build_temp, "lib/libblis-mt.a"),
        ]

        extension.extra_objects.extend(extra_objects)

        super().build_extensions()


#   ZenTorch_BUILD_VERSION
#     specify the version of torch_zendnn_plugin, rather than the hard-coded version
#     in this file; used when we're building binaries for distribution

# Define env values
PACKAGE_NAME = "torch_zendnn_plugin"
PACKAGE_VERSION = "1.0.0"


def get_build_version(base_dir):
    git_sha = (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=base_dir)
        .decode("ascii")
        .strip()
    )
    zen_version = os.getenv(
        "ZenTorch_BUILD_VERSION", PACKAGE_VERSION + "+git" + git_sha[:7]
    )
    return zen_version, git_sha


project_root_dir = os.path.abspath(os.path.dirname(__file__))
sources = glob.glob(os.path.join(project_root_dir, "src/cpu/cpp/*.cpp"))
include_dirs = [
    os.path.join(project_root_dir, "third_party/ZenDNN/inc"),
    os.path.join(project_root_dir, "third_party/blis/include/amdzen"),
]

torch_zendnn_plugin_build_version, git_sha = get_build_version(project_root_dir)
wheel_file_dependencies = ["numpy", "torch"]

long_description = ""
with open(os.path.join(project_root_dir, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


def main():
    setup(
        name=PACKAGE_NAME,
        version=torch_zendnn_plugin_build_version,
        description="ZenDNN plugin for PyTorch*",
        long_description=long_description,
        long_description_content_type="text/markdown",
        author_email="",
        author="",
        # URL needs to be updates once the plugin is open sourced
        url="",
        # license needs to be added when the source code gets the license
        license="",
        install_requires=wheel_file_dependencies,
        ext_modules=[
            CppExtension(
                name=f"{PACKAGE_NAME}._C",
                sources=sources,
                include_dirs=include_dirs,
                extra_compile_args=["-Werror",
                                    "-DZENTORCH_VERSION_HASH=" + git_sha],
            )
        ],
        cmdclass={
            "build_ext": CustomBuildExtension,
        },
        packages=[PACKAGE_NAME],
        package_dir={"": "src/cpu/python"},
        data_files=[
            (
                "build/src/cpu/python/torch_zendnn_plugin/",
                ["src/cpu/python/torch_zendnn_plugin/logging.conf"],
            )
        ],
    )


if __name__ == "__main__":
    main()
