# ******************************************************************************
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

PLATFORM=$1
RES_FILE=$2

# ---------------------------
# Check Command Line Options
# ---------------------------

script_usage() {
    echo ""
    echo "Usage: $(basename "$0") <PLATFORM> <RESULT_FILE>"
    echo "       PLATFORM: Milan-64 | Genoa-96 | Genoa-80"
    echo ""
    echo "--------------------------------------------------------------------------------------------------"
    echo "Current Platfrom:"
    echo "--------------------------------------------------------------------------------------------------"
    lscpu | grep -e "Model name:" -e "Thread(s) per core:" -e "Core(s) per socket:" -e "Socket(s):"
    echo ""
    lscpu | grep -e "NUMA"
    echo "--------------------------------------------------------------------------------------------------"
    echo ""
}


if [[ $PLATFORM == "" ]]; then
    echo ""
    echo "ERROR: Platform is not specified"
    script_usage
    exit
else
    if [[ $PLATFORM != "Milan-64" && $PLATFORM != "Genoa-96" && $PLATFORM != "Genoa-80" ]]; then
        echo ""
        echo "ERROR: Platfrom  $PLATFORM is not supported"
        script_usage
        exit
    fi
fi

if [[ $RES_FILE == "" ]]; then
    echo ""
    echo "ERROR: Result file is not specified"
    script_usage
    exit
else
    if [ -e $RES_FILE ]; then
        echo ""
        echo "ERROR: Result file $RES_FILE already exists"
        echo "       Remove it before continue"
        echo ""
        exit
    fi
fi

echo ""

# ---------------------------
# Configure Tests
# ---------------------------
case $PLATFORM in
    "Milan-64")
        LST_OMP_NUM_THREADS_INST1=(1 $(seq 4 4 16)); LST_DATA_TYPE=(float32)
        NOF_PHYSICAL_CORES=64
        ;;
    "Genoa-96")
        LST_OMP_NUM_THREADS_INST1=(1 $(seq 4 4 16)); LST_DATA_TYPE=(float32 bfloat16)
        NOF_PHYSICAL_CORES=96
        ;;
    "Genoa-80")
        LST_OMP_NUM_THREADS_INST1=(1 $(seq 4 4 16)); LST_DATA_TYPE=(float32 bfloat16)
        NOF_PHYSICAL_CORES=80
        ;;
esac

# -----------------------------------------------
# Define:
#   -) System type (Baremetal of VM) and
#   -) Nb of physical and logical cores used by
#      Hypervisor in VM environment
# Calculate Nb of usable physical cores
# ----------------------------------------------
SYSTEM_TYPE="vm"
NOF_HOST_PHYSICAL_CPU='8'
NOF_HOST_LOGICAL_CPU='16'

NOF_USABLE_PYSICAL_CORES=$((NOF_PHYSICAL_CORES - NOF_HOST_PHYSICAL_CPU))
#echo "Number of Usable Physical cores: $NOF_USABLE_PYSICAL_CORES"


LST_SEQUENCE_LENGTH=(32 64 128 256)
LST_NOF_INSTANCES=(2 4 6 8)

NB_OF_TESTS=$(( ${#LST_NOF_INSTANCES[@]} * ${#LST_OMP_NUM_THREADS_INST1[@]} * ${#LST_SEQUENCE_LENGTH[@]} * ${#LST_DATA_TYPE[@]} ))
#echo "Total Number of Tests: $NB_OF_TESTS"

echo "PLATFORM: $PLATFORM" > $RES_FILE
echo "INDX NOF_INSTANCES DATA_TYPE SEQUENCE_LENGTH ENV_OMP_NUM_THREADS Avg_proc_time(ms) Duration(sec)" >> $RES_FILE


GLOBAL_INDEX=0
START_INDEX=0 #84
for NOF_INSTANCES in ${LST_NOF_INSTANCES[@]}; do
    LST_OMP_NUM_THREADS=${LST_OMP_NUM_THREADS_INST1[@]}

    for DATA_TYPE in ${LST_DATA_TYPE[@]}; do

        if [[ $DATA_TYPE == "float32" ]]; then
            ENV_ZENDNN_MATMUL_ALGO="FP32:2,BF16:2"
        else
            ENV_ZENDNN_MATMUL_ALGO="FP32:1,BF16:1"
        fi

        for SEQUENCE_LENGTH in ${LST_SEQUENCE_LENGTH[@]}; do
            for ENV_OMP_NUM_THREADS in ${LST_OMP_NUM_THREADS[@]}; do

                GLOBAL_INDEX=$((GLOBAL_INDEX+1))
                if [ "$GLOBAL_INDEX" -lt "$START_INDEX" ]; then
                    continue
                fi

                NOF_REQUIRED_PHYSICAL_CORES=$((NOF_INSTANCES * ENV_OMP_NUM_THREADS))

                echo "==============================================================================="
                echo "*** ($PLATFORM) Test $GLOBAL_INDEX / $NB_OF_TESTS : #inst(${NOF_INSTANCES}) dtype(${DATA_TYPE}) sl(${SEQUENCE_LENGTH}) #cores(${ENV_OMP_NUM_THREADS})"
                echo "*** #Required / #Usable Physical cores: $NOF_REQUIRED_PHYSICAL_CORES / $NOF_USABLE_PYSICAL_CORES"
                echo "==============================================================================="
                echo -n "${GLOBAL_INDEX} ${NOF_INSTANCES} ${DATA_TYPE} ${SEQUENCE_LENGTH} ${ENV_OMP_NUM_THREADS} " >> $RES_FILE

                CMD="OMP_NUM_THREADS=${ENV_OMP_NUM_THREADS} OMP_WAIT_POLICY=ACTIVE OMP_DYNAMIC=FALSE \
                     KMP_BLOCKTIME=1 KMP_TPAUSE=0 KMP_FORKJOIN_BARRIER_PATTERN=dist,dist \
                     KMP_PLAIN_BARRIER_PATTERN=dist,dist KMP_REDUCTION_BARRIER_PATTERN=dist,dist \
                     KMP_AFFINITY=granularity=fine,compact,1,0 \
                     LD_PRELOAD=${CONDA_PREFIX}/lib/libtcmalloc.so:${CONDA_PREFIX}/lib/libiomp5.so \
                     ZENDNN_WEIGHT_CACHE_CAPACITY=1024 ZENDNN_MATMUL_ALGO=${ENV_ZENDNN_MATMUL_ALGO} ZENDNN_PRIMITIVE_CACHE_CAPACITY=1024 ZENDNN_WEIGHT_CACHING=1 \
                     python bert_base_scripts/run_bert_ds_mp.py -mt 'bert-base-uncased' -zt -ws 20 -nr ${NOF_INSTANCES} -sl ${SEQUENCE_LENGTH} -dt ${DATA_TYPE} \
                     -sys ${SYSTEM_TYPE} -hcpu ${NOF_HOST_LOGICAL_CPU} \
                     | grep -e \"Average processing time:\" -e \"Duration:\" "

                echo "---------------------------------------------------------------------------"
                echo $CMD
                echo "---------------------------------------------------------------------------"

                if [ "$NOF_REQUIRED_PHYSICAL_CORES" -gt "$NOF_USABLE_PYSICAL_CORES" ]; then
                    echo "WARNING: The number of Required physical cores exceeds the number of Usable ones: skipping this configuration"
                    echo ""
                    echo "- -" >> $RES_FILE
                else
                    RESULT=$(eval ${CMD})
                    RESULT_LST=($RESULT)
                    RESULT_TO_FILE="${RESULT_LST[3]} ${RESULT_LST[6]}"
                    echo $RESULT
                    echo $RESULT_TO_FILE >> $RES_FILE
                fi
                #exit
            done
        done
    done
done



