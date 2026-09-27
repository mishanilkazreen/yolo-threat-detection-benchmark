#!/bin/bash
#SBATCH --job-name=yolo_xai_eval
#SBATCH --partition=gpu.q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=runs/slurm_logs/xai_eval_%j.out
#SBATCH --error=runs/slurm_logs/xai_eval_%j.err

# Deterministic XAI Evaluation for Table 8 & Section 5
# Addresses TASK 10 / TASK 19 (Reviewer 1 Finding 4, Major 9 & 10)

set -eo pipefail

echo "=========================================================="
echo "SCIAMA HPC: XAI EVALUATION RUNNER"
echo "Job ID:            $SLURM_JOB_ID"
echo "Assigned Node:     $SLURMD_NODENAME"
echo "Start Time:        $(date)"
echo "=========================================================="

# Load cluster modules
module purge
module load system/sciama services/slurm anaconda3 uv git

# Working directory on Lustre
REPO_DIR="/mnt/lustre2/mres/ghahrem/yolo-threat-detection-benchmark"
cd "$REPO_DIR"

mkdir -p runs/slurm_logs outputs

# Set cache directories on Lustre
export HF_HOME=/mnt/lustre2/mres/ghahrem/.cache/huggingface
export KAGGLEHUB_CACHE=/mnt/lustre2/mres/ghahrem/.cache/kagglehub
export TORCH_HOME=/mnt/lustre2/mres/ghahrem/.cache/torch
export UV_CACHE_DIR=/mnt/lustre2/mres/ghahrem/.cache/uv

# Confirm GPU
nvidia-smi

# Execute deterministic XAI evaluation on held-out test split
uv run python scripts/evaluate_xai_test_set.py --device 0

echo "=========================================================="
echo "XAI evaluation completed at $(date)"
echo "Output saved to outputs/xai_evaluation_results.json"
echo "=========================================================="
