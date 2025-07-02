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
## 3. Model Preparation

Download and install Quark v0.8. Installation instructions can be found [here](https://github.com/amd/Quark/blob/v0.8/README.md).
We suggest downloading the "zip release".

> zentorch v5.1 is compatible with Quark v0.8. Please make sure you download the right version.

Follow the steps in the README file at "examples/torch/rm" directory to download, prepare and quantize the model.


## 4. Execute DLRMv2

### 4.1 Dependency Installation

```shell
pip install scikit-learn
```

### 4.2. Setup

Use the most optimal setup for performance using

```shell
source ../../../scripts/dlrm_optimal_env_setup.sh
```

And modify the configuration to match your machine's specifications.

```shell
## This configuration is for a 128 Core, 2 socket, turin machine.
export NUM_SOCKETS=2         # e.g., 2
export CPUS_PER_SOCKET=128   # e.g., 128
export CPUS_PER_INSTANCE=2   # instance-per-process number=CPUS_PER_PROCESS/CPUS_PER_INSTANCE
                                # total-instance = instance-per-process * process-per-socket
export CPUS_FOR_LOADGEN=1    # number of CPUs for loadgen
                                # finally used in our code is max(CPUS_FOR_LOADGEN, remaining cores for instances)
```

### 4.3. Benchmarking

To generate the performance numbers please execute the following command.

```shell
python main.py --dataset_path=<DATA_DIR> --model_path=<MODEL_DIR>
```
where DATA_DIR is the path of the data directory and MODEL_DIR is the path of model directory
The expected performace during optimal configurationon 128 Core 2 Socket turin machine is around 2Mil Samples per second

To generate the accuracy numbers add the optional argument `--accuracy_mode`

```shell
python main.py --dataset_path=<DATA_DIR> --model_path=<MODEL_DIR> --accuracy_mode
```

The expected accuracy for quant bf16 model is 80.273, and for quant fp32 is 80.274
For all available options `--help`

```shell
python main.py --help
```


