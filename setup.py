# ******************************************************************************
# Copyright (c) 2023-2024 Advanced Micro Devices, Inc.
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
        # if variables not set: then use default values
        if "ZENDNN_PT_USE_LOCAL_BLIS" not in os.environ:
            os.environ["ZENDNN_PT_USE_LOCAL_BLIS"] = "0"
        if "ZENDNN_PT_USE_LOCAL_ZENDNN" not in os.environ:
            os.environ["ZENDNN_PT_USE_LOCAL_ZENDNN"] = "0"
        if "ZENDNN_PT_USE_LOCAL_FBGEMM" not in os.environ:
            os.environ["ZENDNN_PT_USE_LOCAL_FBGEMM"] = "0"

        if os.path.exists(self.build_temp):
            shutil.rmtree(self.build_temp)
        os.makedirs(self.build_temp, exist_ok=True)

        rc, out, err = subproc_communicate("which python")
        if rc == 0:
            out = out.split("\n")[0]
            os.environ["PYTHON_PATH"] = out.strip()
        else:
            print("Issue with getting the python path")
            exit(1)

        build_type = "Debug" if os.getenv("DEBUG", 0) == "1" else "Release"
        working_dir = os.path.abspath(os.path.dirname(__file__))
        cmake_cmd = [
            "cmake",
            "-S",
            working_dir,
            "-B",
            self.build_temp,
            f"-DCMAKE_BUILD_TYPE={build_type}",
        ]
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
            os.path.join(project_root_dir, self.build_temp, "lib/libfbgemm.a"),
            os.path.join(project_root_dir, self.build_temp, "lib/libasmjit.a"),
        ]

        extension.extra_objects.extend(extra_objects)

        super().build_extensions()


def subproc_communicate(cmd):
    p1 = subprocess.Popen(
        [cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
    )
    out, err = p1.communicate()
    rc = p1.returncode
    return rc, out.decode("ascii", "ignore"), err.decode("ascii", "ignore")


#   ZenTorch_BUILD_VERSION
#     specify the version of zentorch, rather than the hard-coded version
#     in this file; used when we're building binaries for distribution

# Define env values
PACKAGE_NAME = "zentorch"
PACKAGE_VERSION = "4.2.0"


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


project_root_dir = os.path.abspath(os.path.dirname(__file__))
sources = glob.glob(os.path.join(project_root_dir, "src/cpu/cpp/*.cpp"))
include_dirs = [
    os.path.join(project_root_dir, "third_party/ZenDNN/inc"),
    os.path.join(project_root_dir, "third_party/FBGEMM/include"),
    os.path.join(project_root_dir, "third_party/blis/include/amdzen"),
]

zentorch_build_version = os.getenv("ZenTorch_BUILD_VERSION", PACKAGE_VERSION)
git_sha = get_commit_hash(project_root_dir)
wheel_file_dependencies = ["numpy", "torch"]

long_description = ""
with open(os.path.join(project_root_dir, "DESCRIPTION.md"), encoding="utf-8") as f:
    long_description = f.read()


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
        keywords='pytorch tensor machine learning plugin ZenDNN AMD',
        install_requires=wheel_file_dependencies,
        ext_modules=[
            CppExtension(
                name=f"{PACKAGE_NAME}._C",
                sources=sources,
                include_dirs=include_dirs,
                extra_compile_args=[
                    "-Werror",
                    "-DZENTORCH_VERSION_HASH=" + git_sha,
                    "-DZENTORCH_VERSION=" + PACKAGE_VERSION,
                ],
            )
        ],
        cmdclass={
            "build_ext": CustomBuildExtension,
        },
        packages=[PACKAGE_NAME],
        package_dir={"": "src/cpu/python"},
    )


if __name__ == "__main__":
    main()
