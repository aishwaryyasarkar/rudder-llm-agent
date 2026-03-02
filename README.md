<h1 align="center">Rudder</h1>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.4.0%2Bcu121-ee4c2c" alt="PyTorch">
  <img src="https://img.shields.io/badge/DGL-2.5-green" alt="DGL">
  <img src="https://img.shields.io/badge/LLM-Ollama-black" alt="Ollama">
</p>

Rudder is a multi-agent system embedded in AWS DistDGL that dynamically manages local fixed-size persistent buffers of remote node features to accelerate distributed mini-batch GNN training on large partitioned graphs. During training, each trainer (GPU) runs a co-located LLM agent that uses in-context learning (ICL) to determine replacement strategies for the local buffers. This adaptive buffer management mitigates the communication bottleneck from frequent, irregular remote feature fetches and reduces cross-partition communication and improves end-to-end training performance.


For full details, see our paper: [Rudder: Steering Prefetching in Distributed GNN Training using LLM Agents](https://arxiv.org/abs/2602.23556) (ICS 2026). This repository contains the full implementation.

Rudder supports two decision backends:
- LLM-based decision agents (served through Ollama)
- ML Classifiers (MLP, TabNet, Logistic Regression, Random Forest, XGBoost, SVM)

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
├── classifier_models/         # Offline training scripts for non-LLM classifiers
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
├── env.yml  
└── README.md
```

## Environment Setup

Requirements:
- Python 3.10+
- PyTorch (distributed build)
- DGL with distributed support
- NumPy, pandas, scikit-learn
- XGBoost
- PyTorch TabNet
- Numba
- Ollama (for LLM agents)

### Step 1: Create Conda Environment using provided `env.yml`

```bash
git clone https://github.com/aishwaryyasarkar/rudder-gnn.git
cd rudder-gnn
conda env create -f env.yml
conda activate llm-dgl-cu121
```

#### Note: If your CUDA version differs from 12.1, install DGL and PyTorch separately
  - Follow the instructions on the [DGL official installation page](https://www.dgl.ai/pages/start.html).
  - Visit the [PyTorch official site](https://pytorch.org/get-started/locally/) and select the appropriate configuration based on your CUDA version.

### Step 2: Install Ollama (LLM Agents Only)

macOS:

```bash
brew install ollama
```

Linux:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### Step 3: Pull Required Ollama Model(s)

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

### Step 4: Optional Ollama Environment Variables

```bash
export OLLAMA_BIN="$(which ollama)"
export OLLAMA_MODELS="/path/to/ollama/models"
```

These are optional because Rudder also accepts `--ollama_bin` and `--ollama_models_dir`.
## Rudder CLI arguments

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

## Running Rudder-based GNN training

We provide 4 scripts in [`slurm/`](slurm):

1. `example_config.sh` (to set parameters using a config file)
2. `set_params.sh` (calls `submit.sh`)
3. `submit.sh` (submits job using allocations in `cpu.sh` or `gpu.sh`)
4. `cpu.sh` / `gpu.sh` (launch distributed training via `launch.py`)

### Step 1: Partition the input graph

Before submitting Rudder jobs, generate DistDGL partitions for your dataset using the provided script in `partition/`. 
```
cd partition
sbatch partition.sh <dataset_name> <partition_method> "<num_parts_list>" <DATA_DIR> <PARTITION_DIR>
```

### Step 2: Edit config file  [`slurm/example_config.sh`](slurm/example_config.sh) 

#### Key fields:
- `MODE`, `MODEL`, `DECISION_MODEL`
- `DATASET_NAME`, `NUM_NODES`, `NUM_TRAINERS`
- `PROJ_PATH`, `PARTITION_DIR`, `LOGS_DIR`, `DATA_DIR`
- Optional: `ML_MODEL_DIR` (if empty, defaults to `"$PROJ_PATH/classifier_models/$DECISION_MODEL/trained_model"`)

### Step 3: Submit jobs

[`slurm/set_params.sh`](slurm/set_params.sh) calls [`slurm/submit.sh`](slurm/submit.sh) to launch either [`slurm/cpu.sh`](slurm/cpu.sh) or [`slurm/gpu.sh`](slurm/gpu.sh) based on `MODE` selected. Set your project account by replacing `<account>` in `#SBATCH -A` in both scripts. To change job wall time, update in [`slurm/submit.sh`](slurm/submit.sh).

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

### Step 4 (Optional): Collect runtime samples to train the classifiers 

If you prefer to use Rudder with ML Classifier backend instead of LLM agents, you need to generate training data and train the classifiers offline.

#### 4.1 Set the following in [`slurm/example_config.sh`](slurm/example_config.sh):

- `COLLECT_TRAINING_FOR_CLASSIFIER="true"`
- `TRAINING_DATA_FILEPATH="<output_dir_or_csv_path>"`

Notes: In collection mode, eviction decisions are policy-driven (no LLM/classifier inference). This mode is only for generating training samples. You must collect enough data to train the classifiers encompassing multiple datasets, hyperparameters (both GNN and eviction parameters) and partition combinations.

#### 4.2 Collect samples by running Rudder in collection mode:

```bash
cd slurm
bash set_params.sh --config example_config.sh
```

#### 4.3 Merge rank CSV files and build train/test datasets:

```bash
python collect_samples/process_csv.py \
  --base_dir <TRAINING_DATA_FILEPATH> \
  --output_dir classifier_models
```

This generates:
- `classifier_models/merged_with_labels_normalized.csv`
- `classifier_models/training_dataset.csv`
- `classifier_models/test_dataset.csv`

#### 4.4 Train any classifier using those train/test CSVs. Example (Logistic Regression):

```bash
python classifier_models/lr/lr.py \
  --train_csv classifier_models/training_dataset.csv \
  --test_csv classifier_models/test_dataset.csv \
  --model_dir classifier_models/lr/trained_model
```

#### 4.5 Switch back for classifier-based runtime decisions in [`slurm/example_config.sh`](slurm/example_config.sh):
- `COLLECT_TRAINING_FOR_CLASSIFIER="false"`
- `DECISION_MODEL="lr"` (or `mlp/tabnet/rf/xgb/svm`)
- `ML_MODEL_DIR="<path-to-trained-model-dir>"`

## Reproducibility Notes

- Random seeds are set in several model scripts.
- Performance and behavior vary with partitioning strategy, launcher config, hardware topology, and Ollama model choice.

## Citation

```bibtex
@misc{sarkar2026ruddersteeringprefetchingdistributed,
      title={Rudder: Steering Prefetching in Distributed GNN Training using LLM Agents}, 
      author={Aishwarya Sarkar and Sayan Ghosh and Nathan Tallent and Aman Chadha and Tanya Roosta and Ali Jannesari},
      year={2026},
      eprint={2602.23556},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2602.23556}, 
}
```
