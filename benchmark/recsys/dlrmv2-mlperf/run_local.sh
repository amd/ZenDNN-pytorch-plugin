#  *****************************************************************************
#  * Modifications Copyright (c) 2025 Advanced Micro Devices, Inc.
#  * All rights reserved.
#  *
#  * Was sourced from
#  * https://github.com/mlcommons/inference_results_v3.1/blob/main/closed/Intel/code/dlrm-v2-99/pytorch-cpu-int8/run_local.sh
#  * commit ID: eaf622a
#  ******************************************************************************
#!/bin/bash

source ./run_common.sh

common_opt="--config ./mlperf.conf"
OUTPUT_DIR=$PWD/output/$name/$mode/$test_type
if [[ $test_type == "performance" ]]; then
    OUTPUT_DIR=$OUTPUT_DIR/run_1
    # OUTPUT_DIR="$OUTPUT_DIR/$(date +%Y-%m-%d-%H:%M:%S)"
fi
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi

set -x # echo the next command

profiling=0
if [ $profiling == 1 ]; then
    EXTRA_OPS="$EXTRA_OPS --enable-profiling=True"
fi

## multi-instance
python -u python/runner.py --profile $profile $common_opt --model $model --int8-model-path $MODEL_DIR \
                           --dataset $dataset --dataset-path $DATA_DIR --output $OUTPUT_DIR $EXTRA_OPS $@
