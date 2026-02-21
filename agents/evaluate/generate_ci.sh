#!/bin/bash

# A small directory to store logs, for example:
DIR="logs_passk_non-llm"
mkdir -p "$DIR"

# Arrays for your models and datasets:
# MODELS=("gemma3-nocutoff" "gemma3:1B-nocutoff" "llama3.2-nocutoff" "smollm2-360M-nocutoff" "smollm2-1.7B-nocutoff" "qwen-1.5b-nocutoff")
MODELS=("mlp-nocutoff" "tabnet-nocutoff" "xgb-nocutoff" "gemma3-nocutoff")
# DATASETS=("ogbn-arxiv" "yelp")
DATASETS=("ogbn-arxiv")
# DATASETS=("ogbn-products" "orkut" "reddit")

for model in "${MODELS[@]}"; do
  for ds in "${DATASETS[@]}"; do
    # Name of the output file for this combination:
    out_file="${DIR}/${model}_${ds}.txt"

    echo "Launching passk-ci.py for model=${model}, dataset=${ds}, writing to ${out_file}"

    # For LLMs
    # The command (wrapped in background with &):
    # python passk-ci.py \
    #   --buffer-size 0.20 \
    #   --datasets "${ds}" \
    #   --agent_models "${model}" \
    #   --agent_dir /global/homes/s/sark777/llm-agents/slurm/conda/logs/agents \
    #   --node-config n4 \
    #   > "${out_file}" 2>&1 &
    # For non-LLMs
    python passk-ci-non-llm.py \
      --datasets "${ds}" \
      --buffer-size 0.05 0.25 \
      --agent_models "${model}" \
      --agent_dir /global/homes/s/sark777/llm-agents/slurm/conda/logs/agents \
      --node-config n2 n4 n8 \
      > "${out_file}" 2>&1 &

  done
done

# wait for all background tasks to finish
wait
echo "All runs completed."
