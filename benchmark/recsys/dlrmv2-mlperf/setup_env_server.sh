#  *****************************************************************************
#  * Modifications Copyright (c) 2025 Advanced Micro Devices, Inc.
#  * All rights reserved.
#  *
#  * Was sourced from
#  * https://github.com/mlcommons/inference_results_v3.1/blob/main/closed/Intel/code/dlrm-v2-99/pytorch-cpu-int8/setup_env_server.sh
#  * commit ID: eaf622a
#  ******************************************************************************
set -x
export NUM_SOCKETS=2        # i.e. 2
export CPUS_PER_SOCKET=128   # i.e. 128
export CPUS_PER_CONSUMER=128  # which determine how much processes will be used
                            # consumer-per-socket = CPUS_PER_SOCKET/CPUS_PER_CONSUMER
export CPUS_PER_INSTANCE=2  # instance-per-consumer number=CPUS_PER_CONSUMER/CPUS_PER_INSTANCE
                            # total-instance = instance-per-consumer * consumer-per-socket
export CPUS_FOR_LOADGEN=1   # number of cpus for loadgen
                            # finally used in our code is max(CPUS_FOR_LOADGEN, left cores for instances)
export BATCH_SIZE=200
set +x
