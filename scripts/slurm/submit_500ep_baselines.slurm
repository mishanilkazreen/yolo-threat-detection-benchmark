#!/bin/bash
#SBATCH --job-name=yolo_baseline_500ep
#SBATCH --partition=gpu.q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-5
#SBATCH --output=runs/slurm_logs/baseline_500ep_%A_%a.out
#SBATCH --error=runs/slurm_logs/baseline_500ep_%A_%a.err

# 500-Epoch Baseline Recovery with early stopping patience=50
# Addresses TASK-MAJ-12 & TASK-MAJ-11 (Reviewer 1 Major 11, 12)

set -eo pipefail

CONFIGS=(
    "config/models/yolov8n_baseline_random.yaml"
    "config/models/yolov8n_baseline_pretrained.yaml"
    "config/models/yolo11n_baseline_random.yaml"
    "config/models/yolo11n_baseline_pretrained.yaml"
    "config/models/yolo12n_baseline_random.yaml"
    "config/models/yolo12n_baseline_pretrained.yaml"
)

CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}

echo "=========================================================="
echo "SCIAMA HPC: 500-EPOCH BASELINE RECOVERY RUNNER"
echo "Job ID:            $SLURM_JOB_ID"
echo "Array Task ID:     $SLURM_ARRAY_TASK_ID"
echo "Assigned Node:     $SLURMD_NODENAME"
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

# Execute baseline training: 500 epochs, early stopping patience=50, seed 42
uv run python scripts/train_baseline.py "$CONFIG" --epochs 500 --patience 50 --seed 42 --device 0

echo "=========================================================="
echo "Task $SLURM_ARRAY_TASK_ID ($CONFIG) finished at $(date)"
echo "=========================================================="
