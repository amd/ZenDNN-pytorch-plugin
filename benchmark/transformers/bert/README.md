# Running the Bert Model with Zentorch

## 1. Environment Setup

### 1.1. Create a New Conda Environment

```bash
conda create -n zentorch-env-py3.10 python=3.10 -y
conda activate zentorch-env-py3.10
```

### 1.2. Install Zentorch

Ensure GCC version is 12.2 or higher.

Follow the zentorch installation steps in the [README](https://github.com/amd/ZenDNN-pytorch-plugin?tab=readme-ov-file#2-installation) file.

## 2. Execute DLRMv2

### 2.1 Dependency Installation

```shell
pip install -r requirements.txt
```

### 2.2. Setup

Modify the configuration, to match your machine's specifications. The script currently supports `Genoa-96`, `Genoa-80` and `Milan-64`

```shell
export ENV_OMP_NUM_THREADS=2  # the number of cores used per instance (1, 2, ...)
export ENV_ZENDNN_MATMUL_ALGO="FP32:2,BF16:2" # for float32 # "FP32:1,BF16:1" - for bfloat16
export NOF_INSTANCES=48 # - number of instances to be run (1, 2, ...)
export SEQUENCE_LENGTH=256 # token length (32, 64, 128, 256)
export DATA_TYPE=float32 # data type (float32, bfloat16)
export SYSTEM_TYPE="baremetal" # "vm" is using virtual machine 
export NOF_HOST_LOGICAL_CPU=0 # - the number of logical cores used by Hyper-V, 0 for baremetal. 
```

### 2.3. Single instance

To generate the performance numbers for single instance, please execute the following command.

```shell
./run_tests_1-inst_VM_ZenDNN-510.sh Genoa-96 results.txt
```

### 2.4. Multi-instance

To generate the accuracy numbers for multi instance, please execute the following command.

```shell
./run_tests_M-inst_VM_ZenDNN-510.sh Genoa-96 results.txt
```

