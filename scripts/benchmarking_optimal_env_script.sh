#*******************************************************************************
# Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#*******************************************************************************

#!/bin/bash

script_name=$(basename "${BASH_SOURCE[0]}")

# Function to display help information
display_help(){
    echo "Usage: Activate your working conda environment other than base"
    echo "Usage: source $script_name --framework zentorch/ipex --model llm/recsys/cnn/nlp --threads num_threads --precision bf16/fp32/woq/bf16_amp/int8"
    echo "Options:"
    echo " --framework, -f            Specify the framework ['zentorch', 'ipex'] (if not specified this option, by default set to zentorch)"
    echo " --model, -m                Specify the model ['llm', 'recsys', 'cnn', 'nlp'] (if not specified this option, by default set to llm)"
    echo " --threads, -t              Specify the num of threads. (if not specified this option, by default set to number of CPUs available on your system.)"
    echo " --precision, -p            Specify the precision ['bf16_amp', 'bf16','fp32','woq','int8'] (if not specified this option, by default set to bf16)"
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

# LLVM Package details
package_name="llvm-openmp"
package_version="18.1.8"
package_build="hf5423f3_1"

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --framework|-f) framework="$2"; shift ;;
        --model|-m) model="$2"; shift ;;
	    --threads|-t) threads="$2"; shift ;;
        --precision|-p) precision="$2"; shift ;;
        -h|--help) display_help ;;
        *) echo "Unknown parameter passed: $1"; display_help; return ;;
    esac
    shift
done

# Set default to zentorch if no input is provided (also convert input to lowercase for case-insensitive comparison)
framework=${framework:-zentorch}
framework=$(echo "$framework" | tr '[:upper:]' '[:lower:]')

# Validate the input for framework
if [ "$framework" != "zentorch" ] && [ "$framework" != "ipex" ]; then
    echo "Invalid framework. Please choose either 'zentorch' or 'ipex'."
    display_help
    return
fi

# Set default to llm if no input is provided (also convert input to lowercase for case-insensitive comparison)
model=${model:-llm}
model=$(echo "$model" | tr '[:upper:]' '[:lower:]')

# Set default to number of CPUs available on your system if no input is provided.
if [ -z "$threads" ]; then
    threads=$(lscpu | awk '/^CPU\(s\):/ {print $2}')
fi

threads=$(echo "$threads")

# Validate the input for threads
if ! [[ "$threads" =~ ^[0-9]+$ ]]; then
    echo "Invalid input. Please enter a valid number for threads."
    display_help
    return
fi

# Set default to bf16 if no input is provided (also convert input to lowercase for case-insensitive comparison)
precision=${precision:-bf16}
precision=$(echo "$precision" | tr '[:upper:]' '[:lower:]')

# Validate the input for precision
if ! ( ( [[ "$model" = "cnn" ]] && { [ "$precision" = "fp32" ] || [ "$precision" = "int8" ] || [ "$precision" = "bf16_amp" ]; } ) \
       || ( [[ "$model" = "nlp" ]] && { [ "$precision" = "fp32" ] || [ "$precision" = "bf16_amp" ]; } ) \
       || ( [[ "$model" = "recsys" ]] && { [ "$precision" = "fp32" ] || [ "$precision" = "bf16" ] || [ "$precision" = "bf16_amp" ]; } ) \
       || ( [[ "$model" = "llm" ]] && { [ "$precision" = "bf16" ] || [ "$precision" = "woq" ]; } ) ); then
    echo "Invalid combination of model = $model and precision = $precision. Please choose a valid combination."
    display_help
    exit
fi

# Output the selected framework and model
echo "Framework: $framework"
echo "Model: $model"
echo "Threads: $threads"
if [ "$framework" = "zentorch" ]; then
    echo "Precision: $precision"
fi

export OMP_WAIT_POLICY=ACTIVE
export OMP_DYNAMIC=FALSE
export OMP_NUM_THREADS=$threads
export KMP_BLOCKTIME=1
export KMP_TPAUSE=0
export KMP_FORKJOIN_BARRIER_PATTERN=dist,dist
export KMP_PLAIN_BARRIER_PATTERN=dist,dist
export KMP_REDUCTION_BARRIER_PATTERN=dist,dist
export KMP_AFFINITY=granularity=fine,compact,1,0


echo "OMP_WAIT_POLICY=$OMP_WAIT_POLICY"
echo "OMP_DYNAMIC=$OMP_DYNAMIC"
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "KMP_BLOCKTIME=$KMP_BLOCKTIME"
echo "KMP_TPAUSE=$KMP_TPAUSE"
echo "KMP_FORKJOIN_BARRIER_PATTERN=$KMP_FORKJOIN_BARRIER_PATTERN"
echo "KMP_PLAIN_BARRIER_PATTERN=$KMP_PLAIN_BARRIER_PATTERN"
echo "KMP_REDUCTION_BARRIER_PATTERN=$KMP_REDUCTION_BARRIER_PATTERN"
echo "KMP_AFFINITY=$KMP_AFFINITY"


if [ -f "/usr/local/lib/libjemalloc.so" ]; then
    export LD_PRELOAD=/usr/local/lib/libjemalloc.so:$LD_PRELOAD
    echo "LD_PRELOAD of libjemalloc is success"
else
    echo "Downloading & Building jemalloc"
    conda install autoconf
    git clone https://github.com/jemalloc/jemalloc.git --quiet
    cd $(pwd)/jemalloc
    bash ./autogen.sh > jemalloc_build.log 2>&1
    make >>jemalloc_build.log 2>&1
    sudo make install LIBDIR=/usr/local/lib >>jemalloc_build.log 2>&1
    if [ -f "/usr/local/lib/libjemalloc.so" ]; then
        export LD_PRELOAD=/usr/local/lib/libjemalloc.so:$LD_PRELOAD
        echo "Build & LD_PRELOAD of libjemalloc is success"
    else
        echo "Installing of libjemalloc is failed"
        echo "Please explicitly install libjemalloc in your environment"
    fi
    cd ../
fi

export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:-1,muzzy_decay_ms:-1"
echo "MALLOC_CONF =$MALLOC_CONF"

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

# According to framework and model set different environment variables

if [ "$framework" = "zentorch" ]; then
    # export ZENDNN_WEIGHT_CACHING=1
    # echo "ZENDNN_WEIGHT_CACHING = $ZENDNN_WEIGHT_CACHING"
    export ZENDNN_WEIGHT_CACHE_CAPACITY=1024
    echo "ZENDNN_WEIGHT_CACHE_CAPACITY=$ZENDNN_WEIGHT_CACHE_CAPACITY"

    if [ "$model" = "cnn" ] && { [ "$precision" = "fp32" ] || [ "$precision" = "int8" ] || [ "$precision" = "bf16_amp" ]; }; then
        export ZENDNN_MATMUL_ALGO=FP32:4,BF16:3
    elif [ "$model" = "nlp" ]; then
        if [ "$precision" = "fp32" ]; then
            export ZENDNN_MATMUL_ALGO=FP32:3,BF16:0
        elif [ "$precision" = "bf16_amp" ]; then
            export ZENDNN_MATMUL_ALGO=FP32:4,BF16:3
        fi
    elif [ "$model" = "recsys" ]; then
        if [ "$precision" = "fp32" ]; then
            export ZENDNN_MATMUL_ALGO=FP32:3,BF16:0
        elif [ "$precision" = "bf16" ]; then
            export ZENDNN_MATMUL_ALGO=FP32:4,BF16:2
        elif [ "$precision" = "bf16_amp" ]; then
            export ZENDNN_MATMUL_ALGO=FP32:4,BF16:3
        fi
    elif [ "$model" = "llm" ] && { [ "$precision" = "bf16" ] || [ "$precision" = "woq" ]; }; then
        export ZENDNN_MATMUL_ALGO=FP32:3,BF16:0
    fi

    echo "ZENDNN_MATMUL_ALGO = $ZENDNN_MATMUL_ALGO"
    export ZENDNN_PRIMITIVE_CACHE_CAPACITY=1024
    echo "ZENDNN_PRIMITIVE_CACHE_CAPACITY = $ZENDNN_PRIMITIVE_CACHE_CAPACITY"
    conda install -c conda-forge $package_name=$package_version=$package_build --no-deps -y

    if [ -f "$extracted_path/$condavar/pkgs/$package_name-$package_version-$package_build/lib/libiomp5.so" ]; then
        echo "Installation Successful of libiomp5.so for framework '$framework' and model '$model'"
        export LD_PRELOAD="$extracted_path/$condavar/pkgs/$package_name-$package_version-$package_build/lib/libiomp5.so:$LD_PRELOAD"
        echo "LD_PRELOAD=$LD_PRELOAD"
    else
        echo "Installation Unsuccessful for libiomp5.so for framework '$framework' and model '$model'"
        echo "Please explicitly install libiomp5.so for framework '$framework' and model '$model' in your conda environment"
        echo "export LD_PRELOAD=\"$extracted_path/$condavar/pkgs/$package_name-$package_version-$package_build/lib/libiomp5.so:$LD_PRELOAD\""
    fi

# for ipex framework
else
    export ONEDNN_PRIMITIVE_CACHE_CAPACITY=1024
    echo "ONEDNN_PRIMITIVE_CACHE_CAPACITY = $ONEDNN_PRIMITIVE_CACHE_CAPACITY"
    pip install intel-openmp

    if pip show intel-openmp > /dev/null 2>&1; then
        echo "intel-openmp is installed."
    else
        echo "intel-openmp is not installed."
        echo "Please explicitly install intel-openmp in your environment"
        return
    fi

    if [ -f "$extracted_path/$condavar/pkgs/$package_name-$package_version-$package_build/lib/libiomp5.so" ]; then
        echo "Installation Successful of libiomp5.so for framework '$framework' and model '$model'"
        export LD_PRELOAD="$extracted_path/$condavar/envs/$CONDA_DEFAULT_ENV/lib/libiomp5.so:$LD_PRELOAD"
        echo "LD_PRELOAD=$LD_PRELOAD"
    else
        echo "Installation Unsuccessful of libiomp5.so for framework '$framework' and model '$model'"
        echo "Please explicitly install libiomp5.so for framework '$framework' and model '$model' in your conda environment"
        echo "export LD_PRELOAD=\"$extracted_path/$condavar/envs/$CONDA_DEFAULT_ENV/lib/libiomp5.so:$LD_PRELOAD\""
    fi

fi
