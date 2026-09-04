#!/bin/bash
#SBATCH --job-name=yolo_latency_benchmark
#SBATCH --partition=gpu.q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=runs/slurm_logs/latency_benchmark_%j.out
#SBATCH --error=runs/slurm_logs/latency_benchmark_%j.err

# Real-Time Latency Benchmark on GPU Node
# Addresses TASK-MAJ-01 & TASK-MAJ-10 (Reviewers 1 & 3)

set -eo pipefail

echo "=========================================================="
echo "SCIAMA HPC: DECOMPOSED REAL-TIME LATENCY BENCHMARK"
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

# Execute decomposed latency benchmark with 100 warmup and 500 iterations on GPU 0
uv run python scripts/benchmark_inference.py --device 0 --warmup 100 --iterations 500

echo "=========================================================="
echo "Latency benchmark completed at $(date)"
echo "=========================================================="
