#!/bin/bash

MODE=$1 # pass 'cpu' or 'gpu' as argument
EVICTION_PERIOD=$4
PREFETCH_FRACTION=$3
ALPHA=$5
HIT_RATE=$6
DATASET_NAME=$7
NUM_NODES=$8
NUM_TRAINERS=$9
NUM_SAMPLER_PROCESSES=${10}
MODEL=${11}  
QUEUE=${12}
LOGS_DIR=${13}
DATA_DIR=${14}
PROJ_PATH=${15}
PARTITION_DIR=${16}
PARTITION_METHOD=${17}
PREFETCHER_INIT=${18}
DECISION_MODEL=${19}
BATCH_SIZE=${20} # batch size for training
BATCHSIZE_EXP=${21} # whether to run experiments with different batch sizes
ENABLE_FINETUNE=${22} # whether to enable finetuning of the decision model
FINETUNE_INTERVAL=${23} # finetune interval, if finetuning is enabled
ML_MODEL_DIR=${24} # optional override for non-LLM model directory
OLLAMA_BIN=${25} # optional override for ollama executable
OLLAMA_MODELS_DIR=${26} # optional override for ollama models directory
# echo "Decision model: $DECISION_MODEL"
if [ "$MODE" == "cpu" ]; then
    BACKEND=$2
    # if backend is not passed as argument, throw error
    if [ -z "$BACKEND" ]; then
        echo "Backend not passed as argument"
        exit 1
    fi
    SCRIPT="cpu.sh"
    LOGNAME="cpu_${BACKEND}"
    
elif [ "$MODE" == "gpu" ]; then
    BACKEND=$2
    if [ -z "$BACKEND" ]; then
        echo "Backend not passed as argument"
        exit 1
    fi
    SCRIPT="gpu.sh"
    LOGNAME="gpu_${BACKEND}"
else
    echo "Invalid mode: choose either 'cpu' or 'gpu'"
    exit 1
fi

for DATASET in $DATASET_NAME; do
    # if finetune is enabled, use a different logs directory.
    if [ "$ENABLE_FINETUNE" == "true" ]; then
        # assert that the finetune interval is set
        if [ -z "$FINETUNE_INTERVAL" ]; then
            echo "Finetune interval is not set. Please set it in the script."
            exit 1
        fi
        if [ "$BATCHSIZE_EXP" == "true" ]; then
            LOGS_DIR="$LOGS_DIR/${DATASET}/${LOGNAME}/${MODEL}/pf_${PREFETCH_FRACTION}/${EVICTION_PERIOD}_period_${PREFETCH_FRACTION}_fraction_${ALPHA}_alpha/finetune_interval_${FINETUNE_INTERVAL}/batchsize_${BATCH_SIZE}"
        else
            LOGS_DIR="$LOGS_DIR/${DATASET}/${LOGNAME}/${MODEL}/pf_${PREFETCH_FRACTION}/${EVICTION_PERIOD}_period_${PREFETCH_FRACTION}_fraction_${ALPHA}_alpha/finetune_interval_${FINETUNE_INTERVAL}"
        fi
    else
        if [ "$BATCHSIZE_EXP" == "true" ]; then
            LOGS_DIR="$LOGS_DIR/${DATASET}/${LOGNAME}/${MODEL}/pf_${PREFETCH_FRACTION}/${EVICTION_PERIOD}_period_${PREFETCH_FRACTION}_fraction_${ALPHA}_alpha/batchsize_${BATCH_SIZE}"
        else
            LOGS_DIR="$LOGS_DIR/${DATASET}/${LOGNAME}/${MODEL}/pf_${PREFETCH_FRACTION}/${EVICTION_PERIOD}_period_${PREFETCH_FRACTION}_fraction_${ALPHA}_alpha"
        fi
    fi
   
    # Create the logs directory if it doesn't exist
    IP_CONFIG_DIR="${LOGS_DIR}/ip_config"
    echo "Creating logs directory: $LOGS_DIR with batch size: $BATCH_SIZE and batchsize exp flag: $BATCHSIZE_EXP"
    mkdir -p $LOGS_DIR
    mkdir -p $IP_CONFIG_DIR
    for PARTITION in $PARTITION_METHOD; do
        for NODES in $NUM_NODES; do
            for SAMPLER_PROCESSES in $NUM_SAMPLER_PROCESSES; do
                for TRAINERS in $NUM_TRAINERS; do
                    NAME="${DATASET}_${PARTITION}_n${NODES}_samp${SAMPLER_PROCESSES}_trainer${TRAINERS}"
                    JOBNAME="as-${DECISION_MODEL}_${MODE}_PF${PREFETCH_FRACTION}_P${EVICTION_PERIOD}_a${ALPHA}_${HIT_RATE}_${MODEL}_${NAME}_finetune_${ENABLE_FINETUNE}_interval_${FINETUNE_INTERVAL}_batchsize_${BATCH_SIZE}"
                    OUTFILE="${LOGS_DIR}/${NAME}_%j.out"
                    ERRFILE="${LOGS_DIR}/${NAME}_%j.err"
                    SUMMARYFILE="${LOGS_DIR}/${NAME}"
                    IP_CONFIG_FILE="${IP_CONFIG_DIR}/ip_config_${NAME}"

                    if [ "$QUEUE" == "debug" ]; then
                        TIME="00:30:00"
                    elif [ "$QUEUE" == "regular" ]; then
                        if [ "$DATASET" == "ogbn-papers100M" ] || [ "$DATASET" == "friendster" ]; then
                            case "$MODEL-$NODES" in
                                gat-2)  TIME="06:00:00" ;;
                                gat-4)  TIME="03:30:00" ;;
                                gat-8)  TIME="01:00:00" ;;
                                gat-16) TIME="01:30:00" ;;
                                gat-32) TIME="00:45:00" ;;
                                gat-64) TIME="00:50:00" ;;
                                sage-2)  TIME="03:00:00" ;;
                                sage-4)  TIME="05:00:00" ;;
                                sage-8)  TIME="03:00:00" ;;
                                sage-16) TIME="03:00:00" ;;
                                sage-32) TIME="00:59:00" ;;
                                sage-64) TIME="00:50:00" ;;
                                *) TIME="01:00:00" ;;  # default fallback
                            esac
                        elif [ "$DATASET" == "ogbn-products" ] || [ "$DATASET" == "orkut" ]; then
                            case "$MODEL-$NODES" in
                                gat-4)  TIME="03:00:00" ;;
                                sage-4)  TIME="00:30:00" ;;
                                sage-8)  TIME="00:30:00" ;;
                                sage-16) TIME="00:30:00" ;;
                                sage-32) TIME="00:30:00" ;;
                                sage-64) TIME="00:30:00" ;;
                            esac
                        elif [ "$DATASET" == "reddit" ]; then
                            case "$MODEL-$NODES" in
                                sage-2)  TIME="01:30:00" ;;
                                sage-4)  TIME="00:40:00" ;;
                                sage-8)  TIME="01:00:00" ;;
                                sage-16) TIME="00:15:00" ;;
                            esac
                        elif [ "$DATASET" == "ogbn-arxiv" ] && [ "$MODEL" == "gat" ]; then
                            TIME="01:30:00"
                        elif [ "$DATASET" == "ogbn-arxiv" ] && [ "$MODEL" == "sage" ]; then
                            TIME="00:15:00"
                        else
                            case "$MODEL" in
                                gat) TIME="01:00:00" ;;
                                sage) TIME="00:45:00" ;;
                            esac
                        fi
                    fi
                    echo "-----------------------------------------------------"
                    echo "Submitting job $JOBNAME with the following parameters:"
                    echo "Dataset: $DATASET"
                    echo "Number of Nodes: $NODES"
                    echo "Summary file: $SUMMARYFILE"
                    echo "Eviction Period: $EVICTION_PERIOD"
                    echo "Prefetch Fraction: $PREFETCH_FRACTION"
                    echo "Alpha: $ALPHA"
                    echo "Time: $TIME"
                    if [ "$MODE" == "gpu" ]; then
                        sbatch -N "$NODES" -q "$QUEUE" --job-name "$JOBNAME" -o "$OUTFILE" -e "$ERRFILE" --time="$TIME" "$SCRIPT" \
                            "$DATASET" "$PARTITION" "$NODES" "$SAMPLER_PROCESSES" "$SUMMARYFILE" "$IP_CONFIG_FILE" "$TRAINERS" "$BACKEND" \
                            "$EVICTION_PERIOD" "$PREFETCH_FRACTION" "$ALPHA" "$HIT_RATE" "$MODEL" "$DATA_DIR" "$PROJ_PATH" "$PARTITION_DIR" \
                            "$PREFETCHER_INIT" "$DECISION_MODEL" "$ENABLE_FINETUNE" "$BATCH_SIZE" "$FINETUNE_INTERVAL" "$ML_MODEL_DIR" \
                            "$OLLAMA_BIN" "$OLLAMA_MODELS_DIR"
                    elif [ "$MODE" == "cpu" ]; then
                        sbatch -N "$NODES" -q "$QUEUE" --job-name "$JOBNAME" -o "$OUTFILE" -e "$ERRFILE" --time="$TIME" "$SCRIPT" \
                            "$DATASET" "$PARTITION" "$NODES" "$SAMPLER_PROCESSES" "$SUMMARYFILE" "$IP_CONFIG_FILE" "$BACKEND" "$TRAINERS" \
                            "$EVICTION_PERIOD" "$PREFETCH_FRACTION" "$ALPHA" "$HIT_RATE" "$MODEL" "$DATA_DIR" "$PROJ_PATH" "$PARTITION_DIR" \
                            "$PREFETCHER_INIT" "$DECISION_MODEL" "$ENABLE_FINETUNE" "$BATCH_SIZE" "$FINETUNE_INTERVAL" "$ML_MODEL_DIR" \
                            "$OLLAMA_BIN" "$OLLAMA_MODELS_DIR"
                    fi
                done
            done
        done
    done
done
