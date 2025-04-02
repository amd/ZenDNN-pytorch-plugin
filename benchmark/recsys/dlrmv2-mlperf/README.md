# Running the Quantized DLRMv2 Model with Zentorch

> **_NOTE:_** The following paths are relative to the directory this file is located in.

## 1. Environment Setup

### 1.1. Create a New Conda Environment

```bash
conda create -n zentorch-env-py3.10 python=3.10 -y
conda activate zentorch-env-py3.10
```

### 1.2. Install Zentorch

Ensure GCC version is 12 or higher.

Follow the zentorch installation steps in the [README](https://github.com/amd/ZenDNN-pytorch-plugin?tab=readme-ov-file#2-installation) file.

## 2. Data Preparation

### 2.1 To prepare the data, refer to [MLPerf DLRMv2](https://github.com/mlcommons/training/tree/master/recommendation_v2/torchrec_dlrm#create-the-synthetic-multi-hot-dataset). The data structure should be as follows

```shell
.
├── terabyte_input
│   ├── day_23_dense.npy
│   ├── day_23_labels.npy
│   └── day_23_sparse_multi_hot.npz
```

Set the directory path to `$DATA_DIR`

```bash
export DATA_DIR=/path/to/terabyte_input/
```

## 3. Model Preparation

Download and install Quark v0.8. Installation instructions can be found [here](https://quark.docs.amd.com/release-0.8/install.html).
We suggest downloading the "zip release".

> zentorch v5.0.2 is compatible with Quark v0.8. Please make sure you download the right version.

Follow the steps in the README file at "examples/torch/rm" directory to download, prepare and quantize the model.

Set the path for the quantized DLRM model directory.

```bash
export MODEL_DIR=/path/to/dlrm_quark
```

## 4. Execute DLRMv2

### 4.1 Dependency Installation

```shell
bash prepare_env.sh
```

### 4.2. Setup

Ensure `$DATA_DIR` and `$MODEL_DIR` are set. Use the most optimal setup for performance using

```shell
source ../../../scripts/dlrm_optimal_env_setup.sh
```

And modify the configuration files, `setup_env_offline.sh`, to match your machine's specifications.

```shell
export NUM_SOCKETS=2         # e.g., 2
export CPUS_PER_SOCKET=128   # e.g., 128
export CPUS_PER_PROCESS=128  # determines the number of processes used
                                # process-per-socket = CPUS_PER_SOCKET/CPUS_PER_PROCESS
export CPUS_PER_INSTANCE=2   # instance-per-process number=CPUS_PER_PROCESS/CPUS_PER_INSTANCE
                                # total-instance = instance-per-process * process-per-socket
export CPUS_FOR_LOADGEN=1    # number of CPUs for loadgen
                                # finally used in our code is max(CPUS_FOR_LOADGEN, remaining cores for instances)
export BATCH_SIZE=100
```

### 4.3. Offline Performance

To generate the performance numbers in offline mode, please execute the following command.

```shell
source setup_env_offline.sh && ./run_main.sh offline int8-bf16
```

### 4.4. Offline Accuracy

To generate the accuracy numbers in offline mode, please execute the following command.

```shell
source setup_env_offline.sh && ./run_main.sh offline accuracy int8-bf16
```
