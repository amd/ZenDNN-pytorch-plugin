#  *****************************************************************************
#  * Modifications Copyright (c) 2025 Advanced Micro Devices, Inc.
#  * All rights reserved.
#  *
#  * Was sourced from
#  * https://github.com/mlcommons/inference_results_v3.1/blob/main/closed/Intel/code/dlrm-v2-99/pytorch-cpu-int8/run_main.sh
#  * commit ID: eaf622a
#  ******************************************************************************

#!/bin/bash

dtype="fp32"
batch_size=$(($BATCH_SIZE + 0))
if [ $# -ge 2 ]; then
    if [[ $2 == "accuracy" ]]; then
        test_type="accuracy"
    fi
    if [[ $2 == "int8" ]] || [[ $3 == "int8" ]]; then
        dtype="int8"
    fi
else
    test_type="performance"
fi

export ZENDNN_EB_THREAD_TYPE=2
export ZENDNN_MATMUL_ALGO=INT8:2
export OMP_NUM_THREADS=$CPUS_PER_INSTANCE
export ZENDNN_PRIMITIVE_CACHE_CAPACITY=20971520 # https://oneapi-src.github.io/oneDNN/dev_guide_primitive_cache.html. Kindly refer this link for more details
export LD_PRELOAD="${CONDA_PREFIX}/lib/libtcmalloc.so:${CONDA_PREFIX}/lib/libomp.so:$LD_PRELOAD"
export TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=30469645312 # https://github.com/gperftools/gperftools/issues/360. Kindly refer this link for more details

# echo  $LD_PRELOAD
# export LD_PRELOAD="/usr/local/lib/libjemalloc.so:/home/amd/anaconda3/envs/zentorch_dinesh/lib/libomp.so:$LD_PRELOAD"
# echo $LD_PRELOAD

mode="Offline"
extra_option="--samples-per-query-offline=204800"
if [ $1 == "server" ]; then
    mode="Server"
    extra_option=""
fi

# sudo ./run_clean.sh
echo "Running $mode bs=$batch_size int8 $test_type"
./run_local.sh pytorch dlrm multihot-criteo cpu $dtype $test_type --scenario $mode --max-ind-range=40000000  --samples-to-aggregate-quantile-file=${PWD}/tools/dist_quantile.txt --max-batchsize=$batch_size $extra_option
