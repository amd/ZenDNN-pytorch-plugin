#*******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#*******************************************************************************
#
# Ubuntu build of vLLM (CPU) + zentorch (ZenDNN backend).
# Vendored into the build-vllm-zentorch skill so the skill is self-contained.
#
# Repos default to the PUBLIC upstreams. For private/internal branches, override
# the *_REPO args and pass a token via GIT_TOKEN (used only at build time; not
# baked into the image). Never hardcode a PAT in this file.
#
# Example:
#   docker build -t zentorch-ubuntu \
#     --build-arg VLLM_VERSION=<confirmed-version> \
#     --build-arg ZENTORCH_REF=main --build-arg ZENDNN_REF=main \
#     -f Dockerfile .
# Private repos:
#   docker build -t zentorch-ubuntu \
#     --build-arg VLLM_VERSION=<confirmed-version> \
#     --build-arg ZENTORCH_REPO=github.com/<org>/ZenDNN_PyTorch_Plugin.git \
#     --build-arg ZENDNN_REPO=github.com/<org>/ZenDNN.git \
#     --secret id=git_token,env=GIT_TOKEN -f Dockerfile .

FROM ubuntu:22.04

# vLLM version — REQUIRED, no default. Confirm it with the user and validate it
# against zentorch's supported range (see SKILL.md) before building. Leaving it
# unset fails the build rather than silently using a version that has drifted
# out of the supported range.
ARG VLLM_VERSION

# Source repos (default: public upstreams) + refs.
ARG ZENTORCH_REPO=github.com/amd/ZenDNN-pytorch-plugin.git
ARG ZENDNN_REPO=github.com/amd/ZenDNN.git
ARG ZENTORCH_REF=main
ARG ZENDNN_REF=main
# Optional build-time token for private repos (prefer --secret over --build-arg).
ARG GIT_TOKEN=

ARG ZENTORCH_RELEASE_TYPE=ga
ARG ZENTORCH_WEEKLY_DATE=

RUN test -n "${VLLM_VERSION}" || { \
        echo "ERROR: --build-arg VLLM_VERSION=<confirmed-version> is required." >&2; \
        echo "       Validate it against zentorch's supported range first (see SKILL.md)." >&2; \
        exit 1; \
    }

# ---------------------------------------------------
# Step 1: System dependencies + Python 3.12 + gcc-13
# ---------------------------------------------------
RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    add-apt-repository -y ppa:ubuntu-toolchain-r/test && \
    apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
        python3.12 python3.12-dev python3.12-venv \
        gcc-13 g++-13 gfortran-13 make libatomic1 libnuma-dev numactl wget git curl \
        libtcmalloc-minimal4 unzip zstd && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 10 \
                        --slave /usr/bin/g++ g++ /usr/bin/g++-13 \
                        --slave /usr/bin/c++ c++ /usr/bin/g++-13 && \
    ln -sf /usr/bin/gcc-13 /usr/bin/x86_64-linux-gnu-gcc && \
    ln -sf /usr/bin/g++-13 /usr/bin/x86_64-linux-gnu-g++

# ---------------------------------------------------
# Step 2: Python 3.12 virtual environment
# ---------------------------------------------------
RUN python3.12 -m venv /opt/venv/py312 && \
    /opt/venv/py312/bin/pip install --upgrade pip setuptools wheel

ENV PATH="/opt/venv/py312/bin:$PATH"
ENV VIRTUAL_ENV="/opt/venv/py312"

# ---------------------------------------------------
# Step 3: llvm-openmp (manual, no conda)
# ---------------------------------------------------
RUN mkdir -p /opt/llvm-openmp && \
    cd /opt/llvm-openmp && \
    wget https://conda.anaconda.org/conda-forge/linux-64/llvm-openmp-18.1.8-hf5423f3_1.conda && \
    unzip llvm-openmp-18.1.8-hf5423f3_1.conda && \
    tar --use-compress-program=zstd -xvf pkg-*.tar.zst && \
    rm -f llvm-openmp-18.1.8-hf5423f3_1.conda pkg-*.tar.zst

# ---------------------------------------------------
# Step 4: Build vLLM (CPU mode)
# ---------------------------------------------------
RUN git clone https://github.com/vllm-project/vllm.git && \
    cd vllm && git checkout v${VLLM_VERSION}

RUN cd vllm && \
    pip install -v -r requirements/build/cpu.txt \
        --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip install -v -r requirements/cpu.txt \
        --extra-index-url https://download.pytorch.org/whl/cpu && \
    VLLM_TARGET_DEVICE=cpu pip install . --no-build-isolation && \
    cd .. && rm -rf vllm

# ---------------------------------------------------
# Step 5: Install ZenDNN PyTorch Plugin (zentorch) against local ZenDNN
# ---------------------------------------------------
ENV ZENTORCH_RELEASE_TYPE=${ZENTORCH_RELEASE_TYPE}
ENV ZENTORCH_WEEKLY_DATE=${ZENTORCH_WEEKLY_DATE}
ENV ZENTORCH_USE_LOCAL_ZENDNN=1

# Build a credentialed URL prefix only if GIT_TOKEN is set; otherwise clone public.
RUN AUTH="" && [ -n "${GIT_TOKEN}" ] && AUTH="${GIT_TOKEN}@" || true; \
    git clone https://${AUTH}${ZENTORCH_REPO} /ZenDNN_PyTorch_Plugin && \
    git clone https://${AUTH}${ZENDNN_REPO} /ZenDNN

RUN cd /ZenDNN && git checkout ${ZENDNN_REF}

RUN cd /ZenDNN_PyTorch_Plugin && \
    git checkout ${ZENTORCH_REF} && \
    export CC=$(which gcc) CXX=$(which g++) && \
    rm -rf build dist *.egg-info && \
    python setup.py bdist_wheel && \
    pip install dist/*.whl && \
    pip install deprecated scipy sympy torchao==0.17.0 && \
    pip install lm-eval[vllm]==0.4.12 ray && \
    cd / && rm -rf /ZenDNN_PyTorch_Plugin /ZenDNN

# ---------------------------------------------------
# Step 6: Runtime environment
# ---------------------------------------------------
# vLLM >=0.18.0 needs libiomp + tcmalloc in LD_PRELOAD for CPU inference.
ENV LD_PRELOAD="/opt/llvm-openmp/lib/libiomp5.so:/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4"

# HF token supplied at runtime (docker run -e HF_TOKEN=...). Do NOT bake secrets.
ENV HF_TOKEN=""

WORKDIR /workspace
CMD ["/bin/bash"]
