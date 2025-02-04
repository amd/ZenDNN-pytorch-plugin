## 1. Setup ENV:

1. Create a new conda env
    ```shell
    conda create -n zentorch-env-py3.10 python=3.10 -y
    conda activate zentorch-env-py3.10
    ```

2. Install/Check GCC 12 or higher version
    Verify if you have GCC>=12 if not please use below command to install GCC 12
    ```shell
    sudo apt install gcc-12 g++-12
    ```
3. Install zentorch

    * Uninstall any existing _zentorch_ installations.
    ```bash
    pip uninstall zentorch
    ```
    * Install Pytorch v2.5.0
    ```bash
    pip install torch==2.5.0 --index-url https://download.pytorch.org/whl/cpu
    ```
    * Use one of two methods to install zentorch:

    Using pip utility
    ```bash
    pip install zentorch==5.0.1
    ```
    or

    Using the release package.

    > Download the package from AMD developer portal from [here](https://www.amd.com/en/developer/zendnn.html).

    > Run the following commands to unzip the package and install the binary.

    ```bash
    unzip ZENTORCH_v5.0.1_Python_v3.10.zip
    cd ZENTORCH_v5.0.1_Python_v3.10/
    pip install zentorch-5.0.1-cp310-cp310-manylinux_2_28_x86_64.whl
    ```
    >Notes:
    * In above steps, we have taken an example for release package with Python version 3.10.
    * Dependent packages 'numpy' and 'torch' will be installed by '_zentorch_' if not already present.
    * If you get the error: ImportError: /lib64/libstdc++.so.6: version `GLIBCXX_.a.b.cc' not found (required by <path_to_conda>/envs/<env_name>/lib/python<py_version>/site-packages/zentorch-5.0.1-pyx.y-linux-x86_64.egg/zentorch/_C.cpython-xy-x86_64-linux-gnu.so), export LD_PRELOAD as:
    * export LD_PRELOAD=<path_to_conda>/envs/<env_name>/lib/libstdc++.so.6:$LD_PRELOAD

4. Prepare the environment:
    ```shell
    bash prepare_env.sh
    ```

## 2. Prepare Data and Model

1. To prepare the data, check [MLPerf DLRMv2](https://github.com/mlcommons/training/tree/master/recommendation_v2/torchrec_dlrm#create-the-synthetic-multi-hot-dataset). The data should look like this:
    ```shell
    .
    ├── terabyte_input
    │   ├── day_23_dense.npy
    │   ├── day_23_labels.npy
    │   └── day_23_sparse_multi_hot.npz
    ```
    Set the directory to `$DATA_DIR`
    ```
    export DATA_DIR=/path/to/terabyte_input/
    ```

2. Download Quantized model weights.
    Download the Quantized model weights from Hugging Face Repo [here]()

3. Unzip the file using below command.
    ```bash
    unzip quantized_weights_dlrmv2.zip
    export MODEL_DIR=/path/to/quantized_weights_dlrmv2
    ```

## 3. Run DLRMv2

1. Setup

    Make sure to set `$DATA_DIR` and `$MODEL_DIR`, and edit the config files, `setup_env_offline.sh` to suit the machine.
    ```shell
    export NUM_SOCKETS=2        # i.e. 2
    export CPUS_PER_SOCKET=128   # i.e. 128
    export CPUS_PER_CONSUMER=128  # which determine how much processes will be used
                                # consumer-per-socket = CPUS_PER_SOCKET/CPUS_PER_CONSUMER
    export CPUS_PER_INSTANCE=2  # instance-per-consumer number=CPUS_PER_CONSUMER/CPUS_PER_INSTANCE
                                # total-instance = instance-per-consumer * consumer-per-socket
    export CPUS_FOR_LOADGEN=1   # number of cpus for loadgen
                                # finally used in our code is max(CPUS_FOR_LOADGEN, left cores for instances)
    export BATCH_SIZE=100
    ```

2. Offline Performance
    ```shell
    source setup_env_offline.sh && ./run_main.sh offline int8
    ```
3. Offline Accuracy
    ```shell
    source setup_env_offline.sh && ./run_main.sh offline accuracy int8
    ```