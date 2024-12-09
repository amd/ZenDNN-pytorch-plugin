This scripts setps the optimal env settings for zentorch/ipex llm / recsys runs

Usage:
Create a conda environment where you run the benchmarks.(Don't use any conda base environment)
Before you run the benchmarks , activate the conda environment and run the benchmarking_optimal_env_script.sh file

source benchmarking_optimal_env_script.sh --help
source benchmarking_optimal_env_script.sh --framework zentorch/ipex --model llm/recsys/cnn/nlp --threads 96/128/192 --precision bf16_amp/bf16/fp32/woq/int8

Its sets all the necessary variables for respective runs based on the options provided

Please feel free to provide your further inputs
