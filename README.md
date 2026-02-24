# Rudder
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.4.0%2Bcu121-ee4c2c)
![DGL](https://img.shields.io/badge/DGL-2.5-green)
![Ollama](https://img.shields.io/badge/LLM-Ollama-black)

Rudder is an adaptive prefetch-and-replacement system for distributed GNN training, implemented in DistDGL. During neighborhood sampling, it continuously decides what to keep in a fixed-size persistent buffer and when to replace stale remote-node features, so communication overhead is reduced while training progresses.

Rudder supports two decision backends:
- LLM-based decision agents (served through Ollama)
- non-LLM classifiers (MLP, TabNet, Logistic Regression, Random Forest, XGBoost, SVM)

The implementation in this repository corresponds to the Rudder paper artifact (see `ICS26_Rudder.pdf` in your release bundle).

## Core Features

- Distributed DistDGL + PyTorch training pipeline.
- Two prefetchers:
  - `Prefetch` (standard)
  - `MemoryEfficientPrefetcher` (memory-optimized) (`--use_memory_efficient_prefetcher`)
- Configurable buffer initialization (`--prefetcher_init`):
  - `empty` starts with an empty/sentinel buffer
  - `degree` warm-starts with high-degree halo nodes (often better early hit rate)
  - `random` starts from a random halo subset
- Agent-based eviction decisions with both LLM and classical ML models.
- Optional online finetuning hooks for ML.

## Repository Layout

```text
rudder-gnn/
├── launch.py                  # DistDGL launcher utility
├── dist_gnn/                  # Distributed training runtime
│   ├── main.py                # Main entry point
│   ├── trainer.py             # Training loop + prefetch integration
│   ├── utils.py               # Helper utilities
│   └── prefetch/
│       ├── prefetch.py        # Standard prefetcher
│       └── lookup.py
├── models/                    # GNN models
│   ├── graphsage.py
│   └── gat.py
├── agents/                    # Eviction agents + Ollama server helper
│   ├── local_agents.py
│   ├── classifiers.py
│   └── start_ollama.py
├── classifier_models/                 # Offline training scripts for non-LLM classifiers
│   ├── mlp/mlp.py
│   ├── tabnet/tabnet.py
│   ├── lr/lr.py
│   ├── rf/rf.py
│   ├── xgb/xgb.py
│   └── svm/svm.py
├── slurm/                     # SLURM workflow scripts
│   ├── example_config.sh      # Example config file for experiments
│   ├── set_params.sh          # Parameter entrypoint (config/CLI)
│   ├── submit.sh              # Job submission orchestrator
│   ├── cpu.sh                 # CPU job script
│   ├── gpu.sh                 # GPU job script
└── README.md
```

## Environment Setup

Minimum expected stack:
- Python 3.10+
- PyTorch (distributed build)
- DGL with distributed support
- NumPy, pandas, scikit-learn
- XGBoost
- PyTorch TabNet
- Numba
- Ollama (for LLM agents)

System/runtime expectations:
- Multi-process training environment for DistDGL.
- Valid `ip_config` and graph partition config (`part_config`).
- Graph partitions with required node fields: `features`, `labels`, `train_mask`, `val_mask`, `test_mask` (and optionally `trainer_id`).

### 1) Create Python Environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 2) Install Python Dependencies

Install framework versions compatible with your CUDA/cluster setup first (PyTorch + DGL), then install the rest:

```bash
pip install numpy pandas scikit-learn numba xgboost pytorch-tabnet joblib tqdm
```

### 3) Install Ollama (LLM Agents Only)

macOS:

```bash
brew install ollama
```

Linux:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### 4) Pull Required Ollama Model(s)

Do this before launching SLURM jobs, so model weights are already present in your Ollama model store.

```bash
ollama pull gemma
```

You can replace `gemma` with any model string you plan to pass to `--decision_model`.

Model selection guidance:
- Prefer Ollama-native tags or GGUF-based instruct models when using external model identifiers.
- In practice, instruct-tuned models are usually better for decision prompts than base models.
- GGUF variants are generally the most reliable format for Ollama runtime compatibility/performance.
- If sourcing from Hugging Face, verify the model is available in an Ollama-compatible form (or has a GGUF variant/tag usable by Ollama).

### 5) Optional Ollama Environment Variables

```bash
export OLLAMA_BIN="$(which ollama)"
export OLLAMA_MODELS="/path/to/ollama/models"
```

These are optional because Rudder also accepts `--ollama_bin` and `--ollama_models_dir`.

## Running with SLURM Scripts

The SLURM workflow in this repo is:

1. `example_config.sh` (set experiment values)
2. `set_params.sh` (expands combinations and calls `submit.sh`)
3. `submit.sh` (dispatches to `cpu.sh` or `gpu.sh`)
4. `cpu.sh` / `gpu.sh` (launch distributed training via `launch.py`)

### Step 1: Partition the input graph

Before submitting Rudder jobs, generate DistDGL partitions for your dataset using the MassiveGNN partitioning intructions : [MassiveGNN: Partition Graph](https://github.com/pnnl/MassiveGNN?tab=readme-ov-file#partition-graph)

Use the partition artifacts from that step as your `PARTITION_DIR` / `--part_config` inputs in this repository.

### Step 2: Edit config file  [`slurm/example_config.sh`](slurm/example_config.sh) 

#### Key fields:
- `MODE`, `MODEL`, `DECISION_MODEL`
- `DATASET_NAME`, `NUM_NODES`, `NUM_TRAINERS`
- `PROJ_PATH`, `PARTITION_DIR`, `LOGS_DIR`, `DATA_DIR`
- Optional: `ML_MODEL_DIR` (if empty, defaults to `"$PROJ_PATH/classifier_models/$DECISION_MODEL/trained_model"`)

### Step 3: Submit jobs

[`slurm/set_params.sh`](slurm/set_params.sh) calls [`slurm/submit.sh`](slurm/submit.sh) to launch either [`slurm/cpu.sh`](slurm/cpu.sh) or [`slurm/gpu.sh`](slurm/gpu.sh) based on `MODE`:

```bash
cd slurm
bash set_params.sh --config example_config.sh
```

### Optional: CLI Args Instead of Config

```bash
bash set_params.sh --help
```

### SLURM Notes

- [`submit.sh`](slurm/submit.sh) chooses backend automatically from mode:
  - `cpu` -> `gloo`
  - `gpu` -> `nccl`
- [`slurm/cpu.sh`](slurm/cpu.sh) / [`slurm/gpu.sh`](slurm/gpu.sh) build an `ip_config` file from allocated nodes.
- Job time and log paths are set in [`slurm/submit.sh`](slurm/submit.sh) from dataset/model/node settings.

## Running Pytorch Distributed Training

Entry point: [`dist_gnn/main.py`](dist_gnn/main.py)

### CLI Options (`dist_gnn/main.py`)

Note: args marked `(launcher)` are usually injected by DistDGL launch wrappers/scripts rather than set manually.

- `--graph_name`: dataset/graph identifier.
- `--backend`: torch distributed backend (`gloo` for CPU, `nccl` for GPU).
- `--part_config` `(launcher)`: DistDGL partition JSON path.
- `--ip_config` `(launcher)`: DistDGL IP config file path.
- `--local-rank` / `--local_rank` `(launcher)`: local process rank.
- `--num_epochs`: number of training epochs.
- `--batch_size`: mini-batch size for training.
- `--num_gpus`: GPUs visible per node/process setup (`0` means CPU mode).
- `--summary_filepath`: output summary file path (rank-level and aggregate stats appended).
- `--prefetch_fraction`: initial fraction of halo nodes to prefetch.
- `--eviction_period`: decision interval for eviction.
- `--alpha`: decay factor used in eviction scoring.
- `--eviction`: enable/disable eviction logic.
- `--num_numba_threads`: Numba thread count used by prefetch/score update paths.
- `--hit_rate_flag`: toggles hit-rate-based decision input behavior.
- `--model`: GNN model (`sage` or `gat`).
- `--prefetcher_init`: prefetch buffer initialization mode (`degree`, `empty`, `random`).
- `--decision_model`: eviction decision model.
  - Classifier models: `mlp`, `tabnet`, `lr`, `rf`, `xgb`, `svm`
  - Agent models (LLM): any Ollama model name string
- `--ml_model_dir`: path to directory that contained trained ML Classifiers.
- `--enable_finetune`: enable/disable online finetuning for supported agents.
- `--finetune_interval`: interval used when finetuning is enabled.
- `--fan_out`: neighbor sampling fanout per layer.
- `--batch_size_eval`: batch size for evaluation/inference pass.
- `--num_hidden`, `--num_layers`, `--num_heads`: model architecture knobs.
- `--lr`, `--dropout`: optimizer/model hyperparameters.
- `--num_trainer_threads`: trainer thread count.
- `--use_memory_efficient_prefetcher {true,false}`:
  - If omitted, defaults to `true` for `ogbn-papers100M`, otherwise `false`.
- `--ollama_bin`: optional Ollama executable override.
- `--ollama_models_dir`: optional Ollama model-store path override.

Note: Why [`launch.py`](launch.py) is required:
- DistDGL training is not a single-process run. It needs coordinated server, trainer, and sampler processes across nodes.
- [`launch.py`](launch.py) sets the required distributed environment (role/process layout, partition/ip configs, torch distributed launcher args, and OMP settings) and launches all processes consistently.
- Calling [`dist_gnn/main.py`](dist_gnn/main.py) directly bypasses that orchestration and will not correctly initialize a multi-node DistDGL job.

## Ollama Behavior

[`agents/start_ollama.py`](`agents/start_ollama.py`) starts a rank-local Ollama server with:
- `OLLAMA_HOST=127.0.0.1:<11434 + local_rank>`
- Optional model directory from `--ollama_models_dir` or env `OLLAMA_MODELS`
- Optional binary from `--ollama_bin` or env `OLLAMA_BIN` (default: `ollama`)
- In `dist_gnn/main.py`, Ollama startup is skipped for classifier models: `mlp`, `tabnet`, `lr`, `rf`, `xgb`, `svm`.

### Example run cmd using LLM Agents

```bash
if [ "$MODEL" == "sage" ]; then
  echo "Running SAGE model..."
  $PYTHON_PATH $PROJ_PATH/launch.py \
    --workspace $PROJ_PATH \
    --num_trainers $GPUS_PER_NODE \
    --num_samplers $SAMPLER_PROCESSES \
    --num_servers 1 \
    --part_config $PARTITION_DIR \
    --ip_config $IP_CONFIG_FILE \
    --num_omp_threads $OMP_THREADS \
    "$PYTHON_PATH dist_gnn/main.py --graph_name $DATASET_NAME \
      --backend $BACKEND \
      --ip_config $IP_CONFIG_FILE --num_epochs $EPOCHS --batch_size $BATCH_SIZE \
      --num_gpus $GPUS_PER_NODE --summary_filepath $SUMMARYFILE \
      --prefetch_fraction $PREFETCH_FRACTION --eviction_period $EVICTION_PERIOD --alpha $ALPHA \
      --eviction $EVICTION \
      --num_numba_threads $NUMBA_THREADS \
      --hit_rate_flag $HIT_RATE \
      --model $MODEL \
      --prefetcher_init $PREFETCHER_INIT \
      --decision_model $DECISION_MODEL \
      --enable_finetune $ENABLE_FINETUNE \
      --finetune_interval $FINETUNE_INTERVAL"
fi
```

## Training Non-LLM Classifier Models
Each script in `classifier_models` supports: `--train_csv`, `--test_csv`, and  `--model_dir`. You will need to collect training data.

### Example: Logistic Regression

```bash
python classifier_models/lr/lr.py \
  --train_csv /path/to/training_dataset.csv \
  --test_csv /path/to/test_dataset.csv \
  --model_dir /path/to/classifier_models/lr_trained
```

### Example: MLP

```bash
python classifier_models/mlp/mlp.py \
  --train_csv /path/to/training_dataset.csv \
  --test_csv /path/to/test_dataset.csv \
  --model_dir /path/to/classifier_models/mlp_trained
```

## Expected Non-LLM CSV Schema

Training scripts use the following fields:

Features:
- `Rank`
- `Batch_Size`
- `Num_Total_Nodes`
- `Num_Partition_Nodes`
- `Num_Remote_Nodes`
- `buffer_size`
- `Eviction_Interval_ID`
- `Num_Evicted_Nodes`
- `Pre_Avg_Hitrate`
- `Pre_Avg_T_rpc`
- `Dataset` (categorical)

Label:
- `eviction_label_norm`

## Outputs

Training runtime:
- Summary metrics appended to `--summary_filepath`
- Rank-specific runtime logs under a directory derived from `summary_filepath`

Non-LLM classifier training:
- Preprocessor: `eviction_preprocessor.joblib`
- Model artifacts (e.g., `lr_eviction.joblib`, `rf_eviction.joblib`, `svm_eviction.joblib`, `xgb_eviction.json`, `tabnet_eviction.zip`, `mlp_eviction.pth`)

## Reproducibility Notes

- Random seeds are set in several model scripts.
- Performance and behavior vary with partitioning strategy, launcher config, hardware topology, and Ollama model choice.
- For release, pin package versions in your environment file and include exact dataset partition artifacts used in experiments.

## Paper / Citation

Add the final citation metadata from your camera-ready paper artifact here.

```bibtex
% TODO: replace with final BibTeX entry for the Rudder paper.
```
