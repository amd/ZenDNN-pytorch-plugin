#  *****************************************************************************
#  * Modifications Copyright (c) 2025 Advanced Micro Devices, Inc.
#  * All rights reserved.
#  *
#  * Was sourced from
#  * https://github.com/mlcommons/inference_results_v3.1/blob/main/closed/Intel/code/dlrm-v2-99/pytorch-cpu-int8/run_common.sh
#  * commit ID: eaf622a
#  ******************************************************************************
#!/bin/bash

if [ "x$DATA_DIR" == "x" ]; then
    echo "DATA_DIR not set" && exit 1
fi
if [ "x$MODEL_DIR" == "x" ]; then
    echo "MODEL_DIR not set" && exit 1
fi

# defaults
backend=pytorch
model=dlrm
dataset=dlrm-multihot-pytorch
device="cpu"
mode="Offline"
dtype="fp32"
test_type="performance"

for i in $* ; do
    case $i in
       pytorch) backend=$i; shift;;
       dlrm) model=$i; shift;;
       multihot-criteo) dataset=$i; shift;;
       cpu) device=$i; shift;;
       fp32|int8-fp32|int8-bf16) dtype=$i; shift;;
       performance|accuracy) test_type=$i; shift;;
       Server|Offline) mode=$i;
    esac
done
# debuging
# echo $backend
# echo $model
# echo $dataset
# echo $device
# echo $MODEL_DIR
# echo $DATA_DIR
# echo $DLRM_DIR
# echo $EXTRA_OPS

if [[ $dtype == "int8-fp32" ]] ; then
    extra_args="$extra_args --use-int8-fp32"
elif [[ $dtype == "int8-bf16" ]] ; then
    extra_args="$extra_args --use-int8-bf16"
fi

if [[ $test_type == "accuracy" ]] ; then
    extra_args="$extra_args --accuracy"
fi

name="$model-$dataset-$backend"

echo $name
#
# pytorch
#
if [ $name == "dlrm-multihot-criteo-pytorch" ] ; then
    model_path="$MODEL_DIR/dlrm-multihot-pytorch.pt"
    profile=dlrm-multihot-pytorch
fi
# debuging
# echo $model_path
# echo $profile
# echo $extra_args

name="$backend-$device/$model"
EXTRA_OPS="$extra_args $EXTRA_OPS"
