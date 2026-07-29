#*******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#*******************************************************************************
#
# RHEL/UBI9 build of vLLM (CPU) + zentorch (ZenDNN backend).
# Vendored into the build-vllm-zentorch skill so the skill is self-contained.
#
# Repos default to the PUBLIC upstreams. For private/internal branches, override
# the *_REPO args and pass a token via GIT_TOKEN (used only at build time; not
# baked into the image). Never hardcode a PAT in this file.
#
# Example:
#   docker build -t zentorch-rhel \
#     --build-arg VLLM_VERSION=<confirmed-version> \
#     --build-arg ZENTORCH_REF=main --build-arg ZENDNN_REF=main \
#     -f Dockerfile .

FROM registry.access.redhat.com/ubi9/ubi:9.5

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

# Use bash for RUN commands
SHELL ["/bin/bash", "-c"]

RUN test -n "${VLLM_VERSION}" || { \
        echo "ERROR: --build-arg VLLM_VERSION=<confirmed-version> is required." >&2; \
        echo "       Validate it against zentorch's supported range first (see SKILL.md)." >&2; \
        exit 1; \
    }

# ---------------------------------------------------
# Enable CRB, EPEL, CentOS Stream repos
# ---------------------------------------------------
RUN dnf update -y && \
    dnf install -y dnf-plugins-core && \
    dnf install -y libxcb && \
    dnf install -y https://dl.fedoraproject.org/pub/epel/epel-release-latest-9.noarch.rpm && \
    /usr/bin/crb enable && \
    rpm --import https://www.centos.org/keys/RPM-GPG-KEY-CentOS-Official && \
    dnf config-manager --add-repo https://mirror.stream.centos.org/9-stream/CRB/x86_64/os/ && \
    dnf config-manager --add-repo https://mirror.stream.centos.org/9-stream/BaseOS/x86_64/os/ && \
    dnf clean all

# GCC 13 toolset
RUN dnf install -y gcc-toolset-13 gcc-toolset-13-gcc gcc-toolset-13-gcc-c++ gcc-toolset-13-libatomic-devel && \
    echo 'source /opt/rh/gcc-toolset-13/enable' >> /etc/profile.d/gcc-toolset.sh

# tcmalloc
RUN dnf install -y gperftools-libs

# ---------------------------------------------------
# Python 3.12 + build dependencies
# ---------------------------------------------------
RUN dnf install -y --allowerasing \
    python3.12 python3.12-pip python3.12-devel \
    git wget cmake ninja-build \
    gcc gcc-c++ make \
    numactl ccache libomp-devel \
    openssl-devel bzip2-devel libffi-devel zlib-devel xz-devel \
    && dnf clean all

# ---------------------------------------------------
# Python 3.12 virtual environment
# ---------------------------------------------------
RUN python3.12 -m venv /opt/venv/py312 && \
    /opt/venv/py312/bin/pip install --upgrade pip setuptools wheel

ENV PATH="/opt/venv/py312/bin:$PATH"
ENV VIRTUAL_ENV="/opt/venv/py312"

# numactl-devel from CentOS Stream repo (required for vLLM CPU)
RUN wget https://rpmfind.net/linux/centos-stream/9-stream/AppStream/x86_64/os/Packages/numactl-devel-2.0.19-3.el9.x86_64.rpm && \
    dnf install ./numactl-devel-*.rpm -y

# ---------------------------------------------------
# Build vLLM (CPU mode)
# ---------------------------------------------------
RUN git clone https://github.com/vllm-project/vllm.git && \
    cd vllm && git checkout v${VLLM_VERSION}

RUN source /opt/rh/gcc-toolset-13/enable && \
    cd vllm && \
    pip install --upgrade pip setuptools && \
    pip install -v -r requirements/build/cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip install -v -r requirements/cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu && \
    VLLM_TARGET_DEVICE=cpu pip install . --no-build-isolation && \
    cd .. && rm -rf vllm

# ---------------------------------------------------
# Install ZenDNN PyTorch Plugin (zentorch) against local ZenDNN
# ---------------------------------------------------
ENV ZENTORCH_RELEASE_TYPE=${ZENTORCH_RELEASE_TYPE}
ENV ZENTORCH_WEEKLY_DATE=${ZENTORCH_WEEKLY_DATE}
ENV ZENDNNL_MANYLINUX_BUILD=1
ENV ZENTORCH_USE_LOCAL_ZENDNN=1
ENV CXX_PREFIX=""

RUN AUTH="" && [ -n "${GIT_TOKEN}" ] && AUTH="${GIT_TOKEN}@" || true; \
    git clone https://${AUTH}${ZENTORCH_REPO} /ZenDNN_PyTorch_Plugin && \
    git clone https://${AUTH}${ZENDNN_REPO} /ZenDNN

RUN cd /ZenDNN && git checkout ${ZENDNN_REF}

RUN source /opt/rh/gcc-toolset-13/enable && \
    cd /ZenDNN_PyTorch_Plugin && \
    git checkout ${ZENTORCH_REF} && \
    pip install -r requirements.txt && \
    export CC=$(which gcc) CXX=$(which g++) && \
    rm -rf build dist *.egg-info && \
    python setup.py bdist_wheel && \
    pip install dist/*.whl && \
    pip install deprecated scipy sympy torchao==0.17.0 && \
    pip install lm-eval[vllm]==0.4.12 ray && \
    cd / && rm -rf /ZenDNN_PyTorch_Plugin /ZenDNN

RUN echo "Default Python: $(python --version)"

# HF token supplied at runtime (docker run -e HF_TOKEN=...). Do NOT bake secrets.
ENV HF_TOKEN=""

# vLLM >=0.18.0 needs libiomp5 + tcmalloc in LD_PRELOAD for CPU inference.
ENV LD_PRELOAD=/opt/venv/py312/lib/libiomp5.so:/usr/lib64/libtcmalloc_minimal.so.4
