#  *****************************************************************************
#  * Modifications Copyright (c) 2025 Advanced Micro Devices, Inc.
#  * All rights reserved.
#  *
#  * Was sourced from
#  * https://github.com/mlcommons/inference_results_v3.1/blob/main/closed/Intel/code/dlrm-v2-99/pytorch-cpu-int8/prepare_env.sh
#  * commit ID: eaf622a
#  ******************************************************************************

# Install required libraries
pip install scikit-learn pybind11 iopath==0.1.10 pyre_extensions==0.0.30
pip install "git+https://github.com/mlperf/logging.git@3.0.0-rc2"
conda install -c conda-forge gperftools llvm-openmp -y

# Install torch and required libraries
pip3 install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
pip install fbgemm-gpu==1.0.0 --index-url https://download.pytorch.org/whl/cpu
pip install torchrec==0.7.0
pip install torchsnapshot==0.1.0

# Install mlperf loadgen
git clone https://github.com/mlcommons/inference.git
pushd inference
git checkout v4.1
git submodule update --init --recursive
pushd loadgen
CFLAGS="-std=c++14" python setup.py install
popd
cp -r mlperf.conf ..
popd


