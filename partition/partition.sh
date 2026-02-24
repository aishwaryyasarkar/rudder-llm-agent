#!/bin/bash
#SBATCH --account=<your_account>
#SBATCH -t 00:30:00
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH -q debug
ulimit -c unlimited

# Activate your env
conda activate <your_env>

# Assign command-line arguments to variables
DATASET_NAME=$1       # Name of the dataset to partition
PARTITION_METHOD=$2   # Method to partition the dataset
PARTS_VALUE=$3        # Number of parts to partition the dataset into
DATA_DIR=$4           # Directory where the input graph data is stored
PARTITION_DIR=$5      # Directory where the partitioned output will be saved

# Loop through each part value to perform the partitioning
for PART in $PARTS_VALUE; do
    echo "Partitioning $DATASET_NAME into $PART parts with $PARTITION_METHOD..."
    srun python3 partition_graph.py --dataset $DATASET_NAME \
    --num_parts $PART --part_method $PARTITION_METHOD --undirected --balance_edges \
    --output "${PARTITION_DIR}/${PART}_parts" \
    --dataset_dir $DATA_DIR \
    --balance_train \
    --num_trainers_per_machine 1
done

echo "Partitioning complete for $DATASET_NAME using $PARTITION_METHOD."