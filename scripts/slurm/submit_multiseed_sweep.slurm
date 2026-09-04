#!/bin/bash
#SBATCH --job-name=yolo_multiseed
#SBATCH --partition=gpu.q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-4
#SBATCH --output=runs/slurm_logs/multiseed_%A_%a.out
#SBATCH --error=runs/slurm_logs/multiseed_%A_%a.err

# Peer Review Multi-Seed Replication Array (seeds: 42, 123, 456, 789, 1011)
# Addresses TASK-R3-05 & TASK-MAJ-07 (Reviewers 1 & 3)

set -eo pipefail

SEEDS=(42 123 456 789 1011)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

CONFIG=${1:-"config/models/yolov8n_random.yaml"}

echo "=========================================================="
echo "SCIAMA HPC: YOLO MULTI-SEED RUNNER"
echo "Job ID:            $SLURM_JOB_ID"
echo "Array Task ID:     $SLURM_ARRAY_TASK_ID"
echo "Assigned Node:     $SLURMD_NODENAME"
echo "Random Seed:       $SEED"
echo "Config:            $CONFIG"
echo "Start Time:        $(date)"
echo "=========================================================="

# Load cluster modules
module purge
module load system/sciama services/slurm anaconda3 uv git

# Working directory on Lustre
REPO_DIR="/mnt/lustre2/mres/ghahrem/yolo-threat-detection-benchmark"
cd "$REPO_DIR"

mkdir -p runs/slurm_logs

# Confirm GPU
nvidia-smi

# Execute incremental training with enforced seed
uv run python scripts/train_model.py "$CONFIG" --seed "$SEED" --device 0

echo "=========================================================="
echo "Task $SLURM_ARRAY_TASK_ID (Seed $SEED) finished at $(date)"
echo "=========================================================="
