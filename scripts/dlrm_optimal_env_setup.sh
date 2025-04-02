#!/bin/bash

# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

script_path=$(dirname "${BASH_SOURCE[0]}")
script_name=$(basename "${BASH_SOURCE[0]}")

# Function to display help information
display_help(){
    echo "Usage: Activate your working conda environment other than base"
    echo "Usage: source $script_path/$script_name --threads num_threads --precision bf16/fp32/int8"
    echo "Options:"
    echo " --threads, -t              Specify the num of threads. (if not specified this option, by default set to number of CPUs available on your system.)"
    echo " --precision, -p            Specify the precision ['bf16','fp32','int8'] (if not specified this option, by default set to combination of fp32, bf16, and int8)"
    echo " --help, -h                 Display this help message."
    return
}

if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    display_help
    return
fi

# Check if inside a valid conda environment
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" = "base" ]; then
    echo "You are either not in a conda environment or you are in the base environment."
    echo "Please create and activate a valid conda environment before proceeding."
    return
else
    echo "You are inside the conda environment: $CONDA_DEFAULT_ENV"
fi

# Initialize variables
framework=""
model=""
threads=""
precision=""

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
    --threads|-t) threads="$2"; shift ;;
        --precision|-p) precision="$2"; shift ;;
        -h|--help) display_help ;;
        *) echo "Unknown parameter passed: $1"; display_help; return ;;
    esac
    shift
done


# Set default to DLRM if no input is provided (also convert input to lowercase for case-insensitive comparison)
model=${model:-dlrm}
model=$(echo "$model" | tr '[:upper:]' '[:lower:]')

# Set default to number of CPUs available on your system if no input is provided.
if [ -z "$threads" ]; then
    threads=$(lscpu | grep "^Core(s) per socket:" | awk '{print $4}')
fi

threads=$(echo "$threads")

# Validate the input for threads
if ! [[ "$threads" =~ ^[0-9]+$ ]]; then
    echo "Invalid input. Please enter a valid number for threads."
    display_help
    return
fi

# Set default precision if no input is provided (also convert input to lowercase for case-insensitive comparison)
precision=${precision:-default}
precision=$(echo "$precision" | tr '[:upper:]' '[:lower:]')

# Validate the input for precision
if ! ( [ "$precision" = "fp32" ] || [ "$precision" = "int8" ] || [ "$precision" = "bf16" ] || [ "$precision" = "default" ]; ); then
    echo "Invalid combination of model = $model and precision = $precision. Please choose a valid combination."
    display_help
    exit
fi

# Output the selected model
echo "Model: $model"
echo "Threads: $threads"
echo "Precision: $precision"

# Get the path of the conda executable
# path=$(which conda)
path=$(conda info --base)
echo "Path to anaconda is $path"

# Use regex to find the path right before 'anaconda3'
if [[ $path =~ (.*\/)(anaconda3)(\/.*|$) ]]; then
    extracted_path=${BASH_REMATCH[1]%/}
    condavar="anaconda3"
    # echo "Path of Anaconda3 is $extracted_path"
elif [[ $path =~ (.*\/)(miniconda3)(\/.*|$) ]]; then
    extracted_path=${BASH_REMATCH[1]%/}
    # echo "Path of Anaconda3 is $extracted_path"
    condavar="miniconda3"
else
    echo "No path found to anaconda3 or miniconda3"
    return
fi

echo "Extracted path to $condavar is $extracted_path/$condavar"


if [ "$precision" = "fp32" ]; then
   export ZENDNN_MATMUL_ALGO=FP32:2
   export ZENDNN_EB_THREAD_TYPE=1
elif [ "$precision" = "bf16" ]; then
   export ZENDNN_MATMUL_ALGO=BF16:2
   export ZENDNN_EB_THREAD_TYPE=1
elif [ "$precision" = "int8" ]; then
   export ZENDNN_MATMUL_ALGO=INT8:2
   export ZENDNN_EB_THREAD_TYPE=2
elif [ "$precision" = "default" ]; then
   export ZENDNN_MATMUL_ALGO=FP32:2,BF16:2,INT8:2
   export ZENDNN_EB_THREAD_TYPE=2
fi



echo "ZENDNN_EB_THREAD_TYPE = $ZENDNN_EB_THREAD_TYPE"
echo "ZENDNN_MATMUL_ALGO = $ZENDNN_MATMUL_ALGO"
