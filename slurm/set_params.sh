#!/bin/bash

# Function to display help message
show_help() {
    echo "Usage:"
    echo "  $0 --config <config_file>"
    echo "  $0 [MODE] [HIT_RATE] [MODEL] [FP] [DELTA] [ALPHAS] [DATASET_NAME] [NUM_NODES] [NUM_TRAINERS] [NUM_SAMPLER_PROCESSES] [QUEUE] [LOGS_DIR] [DATA_DIR] [PROJ_PATH] [PARTITION_DIR] [PARTITION_METHOD] ..."
    echo
    echo "Arguments:"
    echo "  --config <file>      Source parameters from a shell config file."
    echo "  MODE                 Execution mode, either 'cpu' or 'gpu'."
    echo "  HIT_RATE             Hit rate flag, 'true' or 'false'."
    echo "  MODEL                Model name to be used. Currently accepts 'sage' or 'gat'."
    echo "  FP                   % halo nodes to prefetch while initializing buffer (e.g., '0.5')."
    echo "  DELTA                Eviction interval."
    echo "  ALPHAS               Alpha value (e.g., '0.05'). Alpha is calculated as 1-delta."
    echo "  DATASET_NAME         Name of the dataset (e.g., 'ogbn-products')."
    echo "  NUM_NODES            Number of nodes to be used (e.g., '2 4 8')."
    echo "  NUM_TRAINERS         Number of trainers to be used."
    echo "  NUM_SAMPLER_PROCESSES Number of sampler processes to be used."
    echo "  QUEUE                SLURM queue name (e.g., 'regular' or 'debug')."
    echo "  LOGS_DIR             Path to SLURM logs."
    echo "  DATA_DIR             Directory where the input graph data is stored."
    echo "  PROJ_PATH            Path to the project directory."
    echo "  PARTITION_DIR        Directory where the partitioned graphs are stored."
    echo "  PARTITION_METHOD     Method to partition the dataset (e.g., 'metis')."
    echo
    echo "Example:"
    echo "  $0 gpu true sage 0.25 32 0.005 ogbn-products '2 4 8' 2 4 regular '~/MassiveGNN' '~/MassiveGNN/dataset' '~/MassiveGNN' '~/MassiveGNN/partitions' 'metis'"
    echo
}


# Check if help is requested
if [[ $1 == "-h" || $1 == "--help" ]]; then
    show_help
    exit 0
fi

# Optional config-file mode.
# Example: bash set_params.sh --config slurm/example_config.sh
if [[ $1 == "--config" ]]; then
    CONFIG_FILE=$2
    if [[ -z "$CONFIG_FILE" || ! -f "$CONFIG_FILE" ]]; then
        echo "Error: config file not found: $CONFIG_FILE"
        exit 1
    fi
    source "$CONFIG_FILE"
    shift 2
fi

# CMD arguments
MODE=${MODE:-$1}
HIT_RATE=${HIT_RATE:-$2}
MODEL=${MODEL:-$3}
FP=${FP:-$4}
DELTA=${DELTA:-$5}
ALPHAS=${ALPHAS:-$6}
DATASET_NAME=${DATASET_NAME:-$7}
NUM_NODES=${NUM_NODES:-$8}
NUM_TRAINERS=${NUM_TRAINERS:-$9}
NUM_SAMPLER_PROCESSES=${NUM_SAMPLER_PROCESSES:-${10}}
QUEUE=${QUEUE:-${11}}
LOGS_DIR=${LOGS_DIR:-${12}}
DATA_DIR=${DATA_DIR:-${13}}
PROJ_PATH=${PROJ_PATH:-${14}}
PARTITION_DIR=${PARTITION_DIR:-${15}}
PARTITION_METHOD=${PARTITION_METHOD:-${16}}
PREFETCHER_INIT=${PREFETCHER_INIT:-${17}}
DECISION_MODEL=${DECISION_MODEL:-${18}}
ENABLE_FINETUNE=${ENABLE_FINETUNE:-${19}} # whether to enable finetuning of the decision model
FINETUNE_INTERVAL=${FINETUNE_INTERVAL:-${20}} # finetune interval, if finetuning is enabled
BATCH_SIZE=${BATCH_SIZE:-${21}} # batch size for training
BATCHSIZE_EXP=${BATCHSIZE_EXP:-${22}} # whether to run experiments with different batch sizes
ML_MODEL_DIR=${ML_MODEL_DIR:-${23}} # optional override for non-LLM model directory
OLLAMA_BIN=${OLLAMA_BIN:-${24}} # optional override for ollama executable
OLLAMA_MODELS_DIR=${OLLAMA_MODELS_DIR:-${25}} # optional override for ollama models directory
COLLECT_TRAINING_FOR_CLASSIFIER=${COLLECT_TRAINING_FOR_CLASSIFIER:-${26}} # whether to collect classifier-training data
TRAINING_DATA_FILEPATH=${TRAINING_DATA_FILEPATH:-${27}} # optional output CSV path for collected training data

# Validate that all required arguments are provided
if [ -z "$MODE" ] || [ -z "$HIT_RATE" ] || [ -z "$MODEL" ] || [ -z "$FP" ] || [ -z "$DELTA" ] || [ -z "$ALPHAS" ] || [ -z "$DATASET_NAME" ] || [ -z "$NUM_NODES" ] || [ -z "$NUM_TRAINERS" ] || [ -z "$NUM_SAMPLER_PROCESSES" ] || [ -z "$QUEUE" ] || [ -z "$LOGS_DIR" ] || [ -z "$DATA_DIR" ] || [ -z "$PROJ_PATH" ] || [ -z "$PARTITION_DIR" ] || [ -z "$PARTITION_METHOD" ]; then
    echo "Error: One or more required arguments are missing."
    show_help
    exit 1
fi

echo "FP: $FP"
echo "DELTA: $DELTA"
echo "ALPHAS: $ALPHAS"
echo "Decision model: $DECISION_MODEL"
echo "Enable finetuning: $ENABLE_FINETUNE"
echo "Ollama bin: $OLLAMA_BIN"
echo "Ollama models dir: $OLLAMA_MODELS_DIR"
echo "Collect training data for classifier: $COLLECT_TRAINING_FOR_CLASSIFIER"
echo "Training data filepath: $TRAINING_DATA_FILEPATH"


if [ "$ENABLE_FINETUNE" = "true" ]; then
  INTERVALS=( $FINETUNE_INTERVAL )
else
  INTERVALS=( "" )
fi


# if mode is cpu, use backend gloo, else use nccl
if [ "$MODE" == "cpu" ]; then
    BACKEND="gloo"
elif [ "$MODE" == "gpu" ]; then
    BACKEND="nccl"
else
    echo "Invalid mode: choose either 'cpu' or 'gpu'"
    exit 1
fi

# Loop over each combination of prefetch fraction, delta, and alpha
for n in $NUM_NODES; do
    for bs in $BATCH_SIZE; do
        for fp in $FP; do
            for delta in $DELTA; do
                for alpha in $ALPHAS; do
                    for finetune_interval in "${INTERVALS[@]}"; do
                        NEW_PARTITION_DIR="${PARTITION_DIR}/${PARTITION_METHOD}/${DATASET_NAME}/${n}_parts/${DATASET_NAME}.json"
                        echo "Submitting job for $NUM_NODES nodes with prefetch fraction $fp, delta $delta, alpha $alpha, finetune interval $finetune_interval"

                        bash submit.sh \
                            "$MODE" "$BACKEND" "$fp" "$delta" "$alpha" "$HIT_RATE" "$DATASET_NAME" "$n" \
                            "$NUM_TRAINERS" "$NUM_SAMPLER_PROCESSES" "$MODEL" "$QUEUE" "$LOGS_DIR" "$DATA_DIR" \
                            "$PROJ_PATH" "$NEW_PARTITION_DIR" "$PARTITION_METHOD" "$PREFETCHER_INIT" \
                            "$DECISION_MODEL" "$bs" "$BATCHSIZE_EXP" "$ENABLE_FINETUNE" "$finetune_interval" \
                            "$ML_MODEL_DIR" "$OLLAMA_BIN" "$OLLAMA_MODELS_DIR" \
                            "$COLLECT_TRAINING_FOR_CLASSIFIER" "$TRAINING_DATA_FILEPATH"
                        done
                done
            done
        done
    done
done
