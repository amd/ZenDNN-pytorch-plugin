### For CNN/RECSYS/LLM/NLP models:
This scripts setups the optimal env settings for zentorch/ipex llm/recsys/cnn/nlp runs

#### Usage:
Create a conda environment where you run the benchmarks.(Don't use any conda base environment)

Before you run the benchmarks, activate the conda environment and run the zentorch_env_setup.sh file

source zentorch_env_setup.sh --help

source zentorch_env_setup.sh --framework zentorch/ipex --model llm/recsys/cnn/nlp --threads 96/128/192 --precision bf16_amp/bf16/fp32/woq/int8

It sets all the necessary variables for respective runs based on the options provided.

### For DLRM model:
This scripts setups the optimal env settings for zentorch dlrm runs

#### Usage:
Create a conda environment where you run the benchmarks.(Don't use any conda base environment)

Before you run the benchmarks, activate the conda environment and run the dlrm_optimal_env_setup.sh file

source dlrm_optimal_env_setup.sh --help

source dlrm_optimal_env_setup.sh --threads 96/128/192 --precision bf16/fp32/int8

It sets all the necessary variables for respective runs based on the options provided.

