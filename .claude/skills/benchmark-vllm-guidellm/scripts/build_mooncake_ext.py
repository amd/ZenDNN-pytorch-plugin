# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

"""In-container build of lmcache.lmcache_mooncake (CPU).

Bind mounts (RO): /mooncake (Mooncake source+build), /lmcache-src (LMCache source).
C++ deps are apt-installed into /usr; libasio.so is copied to /usr/local/lib
before this runs, so the only runtime .so deps resolve from default paths (the
Mooncake static libs are baked into the extension).
"""
# Third Party
from setuptools import setup
from torch.utils import cpp_extension

MC = "/mooncake"
MCB = f"{MC}/build"
LM = "/lmcache-src"

includes = [
    f"{LM}/csrc/storage_backends",
    f"{LM}/csrc/storage_backends/mooncake",
    f"{MC}/mooncake-common/etcd",
    f"{MC}/mooncake-common/k8s-lease",
    f"{MC}/mooncake-common/include",
    f"{MC}/mooncake-transfer-engine/include",
    f"{MC}/mooncake-store/include/cachelib_memory_allocator/include",
    f"{MC}/mooncake-store/include/cachelib_memory_allocator/fake_include",
    f"{MC}/mooncake-store/include/cachelib_memory_allocator",
    f"{MC}/mooncake-store/include/mooncake-store/proto",
    f"{MC}/mooncake-store/include",
    f"{MCB}/mooncake-store/include",
    f"{MC}/extern/yalantinglibs/include",
    f"{MC}/extern/yalantinglibs/include/ylt/standalone",
    f"{MC}/extern/yalantinglibs/include/ylt/thirdparty",
    "/usr/include/jsoncpp",
]

static_libs = [
    f"{MCB}/mooncake-transfer-engine/src/libtransfer_engine.a",
    f"{MCB}/mooncake-store/src/libmooncake_store.a",
    f"{MCB}/mooncake-store/src/cachelib_memory_allocator/libcachelib_memory_allocator.a",
    f"{MCB}/mooncake-transfer-engine/src/common/base/libbase.a",
    f"{MCB}/mooncake-common/src/libmooncake_common.a",
]

extra_link = (
    ["-Wl,--start-group", *static_libs, "-Wl,--end-group"]
    + ["/usr/local/lib/libasio.so"]
    + ["-lglog", "-lgflags", "-lyaml-cpp", "-lcurl", "-ljsoncpp", "-libverbs",
       "-lnuma", "-ldl", "-lunwind", "-lpthread", "-lxxhash", "-lzstd",
       "-luring", "-lrdmacm"]
    + ["-Wl,-rpath,/usr/local/lib"]
)

ext = cpp_extension.CppExtension(
    "lmcache.lmcache_mooncake",
    sources=[
        f"{LM}/csrc/storage_backends/mooncake/connector.cpp",
        f"{LM}/csrc/storage_backends/mooncake/pybind.cpp",
    ],
    include_dirs=includes,
    extra_compile_args=["-O3", "-std=c++20", "-DYLT_ENABLE_IBV"],
    extra_link_args=extra_link,
)

setup(
    name="lmcache_mooncake_ext",
    ext_modules=[ext],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
    script_args=["build_ext", "--build-lib", "/tmp/lmm_out"],
)
