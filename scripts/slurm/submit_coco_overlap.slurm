#!/bin/bash
#SBATCH --job-name=yolo_coco_overlap
#SBATCH --partition=gpu.q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=runs/slurm_logs/coco_overlap_%j.out
#SBATCH --error=runs/slurm_logs/coco_overlap_%j.err

# COCO Knife Semantic Overlap Zero-Shot Benchmark on GPU Node
# Addresses TASK-MAJ-04 & TASK-R3-03 (Reviewers 1 & 3)

set -eo pipefail

echo "=========================================================="
echo "SCIAMA HPC: COCO KNIFE SEMANTIC OVERLAP EVALUATION"
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

# Confirm GPU
nvidia-smi

# Execute zero-shot evaluation on full test set (all 508 images) across v8n, 11n, 12n
uv run python scripts/evaluate_coco_overlap.py --device 0

echo "=========================================================="
echo "COCO overlap evaluation completed at $(date)"
echo "=========================================================="
