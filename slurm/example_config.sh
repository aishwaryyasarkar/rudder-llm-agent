#!/bin/bash

# Example config for:
#   bash slurm/set_params.sh --config slurm/example_config.sh

# Required core params
MODE="gpu"                         # cpu | gpu
HIT_RATE="false"                   # true | false
MODEL="sage"                       # sage | gat
FP="0.5"                           # prefetch fraction(s), space-separated allowed
DELTA="25"                         # eviction period(s), space-separated allowed
ALPHAS="0.05"                      # alpha value(s), space-separated allowed
DATASET_NAME="ogbn-products"       # dataset name(s), space-separated allowed
NUM_NODES="2"                      # node counts, space-separated allowed (e.g., "2 4 8")
NUM_TRAINERS="4"                   # trainers per node
NUM_SAMPLER_PROCESSES="1"          # sampler processes per trainer
QUEUE="regular"                    # SLURM queue

# Paths
LOGS_DIR="/path/to/logs"
DATA_DIR="/path/to/datasets"
PROJ_PATH="/path/to/rudder-gnn"
PARTITION_DIR="/path/to/partitions"
PARTITION_METHOD="metis"

# Optional knobs
PREFETCHER_INIT="empty"           # degree | empty | random 
DECISION_MODEL="gemma"                # llm model name or mlp/tabnet/lr/rf/xgb/svm
ENABLE_FINETUNE="false"
FINETUNE_INTERVAL="50"
BATCH_SIZE="2000"
BATCHSIZE_EXP="false"

# Optional explicit model directory override for non-LLM agents.
# If empty, scripts fallback to: ${PROJ_PATH}/classifier_models/${DECISION_MODEL}/trained_model
ML_MODEL_DIR=""
