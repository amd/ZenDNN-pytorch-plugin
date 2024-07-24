# ******************************************************************************
# Copyright (c) 2023-2024 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension
from torch.torch_version import __version__
from os.path import join as Path
import os
import glob
import subprocess


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
        if "ZENTORCH_USE_LOCAL_FBGEMM" not in os.environ:
            os.environ["ZENTORCH_USE_LOCAL_FBGEMM"] = "0"

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
            Path(project_root_dir, self.build_temp, "lib", "libamdZenDNN.a"),
            Path(project_root_dir, self.build_temp, "lib", "libblis-mt.a"),
            Path(project_root_dir, self.build_temp, "lib", "libfbgemm.a"),
            Path(project_root_dir, self.build_temp, "lib", "libasmjit.a"),
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
PACKAGE_VERSION = "5.0.0"
PT_VERSION = __version__


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
sources = glob.glob(Path(project_root_dir, "src", "cpu", "cpp", "*.cpp"))
include_dirs = [
    Path(project_root_dir, "third_party", "ZenDNN", "inc"),
    Path(project_root_dir, "third_party", "FBGEMM", "include"),
    Path(project_root_dir, "third_party", "blis", "include", "amdzen"),
]

zentorch_build_version = os.getenv("ZenTorch_BUILD_VERSION", PACKAGE_VERSION)
git_sha = get_commit_hash(project_root_dir)
wheel_file_dependencies = ["numpy", "torch"]

long_description = ""
with open(Path(project_root_dir, "DESCRIPTION.md"), encoding="utf-8") as f:
    long_description = f.read()


config_file = "_build_info.py"

_build_info_path = os.path.join(
    project_root_dir, "src", "cpu", "python", PACKAGE_NAME, config_file
)

_build_config = "# PyTorch Build Version:\n"
_build_config += '__torchversion__ = "{}"\n'.format(PT_VERSION)

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
                    "-DPT_VERSION=" + PT_VERSION,
                ],
            )
        ],
        cmdclass={
            "build_ext": CustomBuildExtension,
        },
        packages=[PACKAGE_NAME],
        package_dir={"": Path("src", "cpu", "python")},
    )


if __name__ == "__main__":
    main()
