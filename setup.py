# ******************************************************************************
# Copyright (c) 2023-2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension
from packaging.version import parse
from torch.torch_version import __version__ as torch_version
from os.path import join as Path
import datetime
import os
import shutil
import subprocess
import torch
import warnings

if parse(torch_version) < parse("2.11.0"):
    raise ImportError(
        "zentorch Plugin requires torch version "
        "2.11.0 or higher. Please upgrade your torch version "
        "and retry the build."
    )

if parse(torch_version) < parse("2.12"):
    warnings.warn(
        "Consider upgrading to torch version 2.12 for improved performance.",
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


def get_source_tag(base_dir):
    """Return the exact source tag on HEAD, or empty string if untagged.

    Resolution order:
    1. ZENTORCH_SOURCE_TAG env var (explicit, always wins).
    2. Exact git tag on HEAD.
    Returns empty string when HEAD is not tagged.
    """
    tag = os.getenv("ZENTORCH_SOURCE_TAG", "")
    if tag:
        return tag
    cwd = os.getcwd()
    os.chdir(base_dir)
    rc, out, err = subproc_communicate("git describe --tags --exact-match HEAD")
    os.chdir(cwd)
    if rc == 0:
        return out.strip()
    return ""


def get_tag_commit(base_dir, tag):
    """Return the commit hash that a tag points to."""
    if not tag:
        return ""
    cwd = os.getcwd()
    os.chdir(base_dir)
    rc, out, err = subproc_communicate(f"git rev-list -n 1 {tag}")
    os.chdir(cwd)
    if rc == 0:
        return out.strip()
    return "unknown"


#   ZenTorch_BUILD_VERSION
#     specify the version of zentorch, rather than the hard-coded version
#     in this file; used when we're building binaries for distribution

# Define env values
PACKAGE_NAME = "zentorch"
_PLUGIN_PATCH = "1"
_pt_ver = parse(torch_version)
PACKAGE_VERSION = f"{_pt_ver.major}.{_pt_ver.minor}.{_pt_ver.micro}.{_PLUGIN_PATCH}"
PT_VERSION = torch_version

_release_type_env = os.getenv("ZENTORCH_RELEASE_TYPE", "ga").lower()
_allowed_release_types = {"weekly", "ga"}
if _release_type_env not in _allowed_release_types:
    raise ValueError(
        f"Invalid ZENTORCH_RELEASE_TYPE={_release_type_env!r}. "
        f"Allowed values are: {sorted(_allowed_release_types)}."
    )
RELEASE_TYPE = _release_type_env

if RELEASE_TYPE == "weekly":
    DIST_NAME = "zentorch-weekly"
    _date_str = (
        os.getenv("ZENTORCH_WEEKLY_DATE", "")
        or datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d")
    )
    # Validate that ZENTORCH_WEEKLY_DATE is in the expected YYYYMMDD format
    try:
        # Ensure it is exactly 8 digits and represents a valid calendar date
        if not (_date_str.isdigit() and len(_date_str) == 8):
            raise ValueError
        datetime.datetime.strptime(_date_str, "%Y%m%d")
    except ValueError:
        raise RuntimeError(
            f"Invalid ZENTORCH_WEEKLY_DATE value: {_date_str!r}. "
            "It must be an 8-digit date in YYYYMMDD format (e.g., 20250318)."
        ) from None
    PACKAGE_VERSION = f"{PACKAGE_VERSION}.dev{_date_str}"
elif RELEASE_TYPE == "ga":
    DIST_NAME = PACKAGE_NAME

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
source_tag = get_source_tag(project_root_dir)
tag_commit = get_tag_commit(project_root_dir, source_tag)
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
    "-DZENTORCH_SOURCE_TAG=" + (source_tag or "untagged"),
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


_desc_file = "DESCRIPTION_WEEKLY.md" if RELEASE_TYPE == "weekly" else "DESCRIPTION.md"
long_description = ""
with open(Path(project_root_dir, _desc_file), encoding="utf-8") as f:
    long_description = f.read()
long_description = long_description.replace("{{PYTORCH_VERSION}}", PT_VERSION)

_build_info_section = "\n## Build Information\n\n"
_build_info_section += "| Field | Value |\n|---|---|\n"
if source_tag:
    _build_info_section += f"| Source Tag | `{source_tag}` |\n"
    _build_info_section += f"| Tag Commit | `{tag_commit}` |\n"
_build_info_section += f"| Build Commit | `{git_sha}` |\n"
_build_info_section += f"| PyTorch Version | `{PT_VERSION}` |\n"
_build_info_section += f"| Release Type | `{RELEASE_TYPE}` |\n"
if source_tag:
    _tag_url = f"https://github.com/amd/ZenDNN-pytorch-plugin/releases/tag/{source_tag}"
    _build_info_section += f"\nBuilt from [{source_tag}]({_tag_url})\n"
long_description += _build_info_section

config_file = "_build_info.py"

_build_info_path = os.path.join(
    project_root_dir, "src", "cpu", "python", PACKAGE_NAME, config_file
)

_build_config = "# Auto-generated build information - DO NOT EDIT\n"
_build_config += '__version__ = "{}"\n'.format(PACKAGE_VERSION)
_build_config += '__torchversion__ = "{}"\n'.format(PT_VERSION)
_build_config += '__source_tag__ = "{}"\n'.format(source_tag)
_build_config += '__tag_commit__ = "{}"\n'.format(tag_commit)
_build_config += '__build_commit__ = "{}"\n'.format(git_sha)
_build_config += '__release_type__ = "{}"\n'.format(RELEASE_TYPE)

packages = [
    PACKAGE_NAME,
    PACKAGE_NAME + ".llm",
]
if ZENTORCH_VLLM_PLUGIN_BUILD:
    packages.append(PACKAGE_NAME + ".vllm")
    packages.append(PACKAGE_NAME + ".vllm.layers")
    packages.append(PACKAGE_NAME + ".vllm.layers.gdn")
extras_require = {}

with open(_build_info_path, "w") as f:
    f.write(_build_config)
    f.close()

# Copy the AOTI shim header into the package tree so it ships with the wheel
# and can be located at runtime via the package directory.
_aoti_header_src = Path(project_root_dir, "src", "cpu", "cpp", "shim_cpu_zentorch.hpp")
_aoti_include_dir = Path(
    project_root_dir, "src", "cpu", "python", PACKAGE_NAME, "include"
)
os.makedirs(_aoti_include_dir, exist_ok=True)
shutil.copy2(_aoti_header_src, _aoti_include_dir)


def main():
    setup(
        name=DIST_NAME,
        version=zentorch_build_version,
        description="zentorch : A PyTorch* extension for AMD EPYC CPUs.",
        long_description=long_description,
        long_description_content_type="text/markdown",
        author_email="zendnn.maintainers@amd.com",
        author="AMD",
        # URL needs to be updates once the plugin is open sourced
        url="https://developer.amd.com/zendnn",
        project_urls={
            **({"Source Tag": f"https://github.com/amd/ZenDNN-pytorch-plugin/releases/tag/{source_tag}"}
               if source_tag else {}),
            "Source": "https://github.com/amd/ZenDNN-pytorch-plugin",
        },
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
        package_data={PACKAGE_NAME: ["include/*.hpp"]},
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
