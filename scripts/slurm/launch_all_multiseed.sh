#!/bin/bash
# Master driver script to submit multi-seed sweeps across all 12 core conditions.
# Each submission launches an array of 5 seeds (42, 123, 456, 789, 1011).

set -eo pipefail

CONFIGS=(
    # YOLOv8n
    "config/models/yolov8n_random.yaml"
    "config/models/yolov8n_pretrained.yaml"
    "config/models/yolov8n_random_100ep.yaml"
    "config/models/yolov8n_pretrained_100ep.yaml"

    # YOLO11n
    "config/models/yolo11n_random.yaml"
    "config/models/yolo11n_pretrained.yaml"
    "config/models/yolo11n_random_100ep.yaml"
    "config/models/yolo11n_pretrained_100ep.yaml"

    # YOLOv12n
    "config/models/yolo12n_random.yaml"
    "config/models/yolo12n_pretrained.yaml"
    "config/models/yolo12n_random_100ep.yaml"
    "config/models/yolo12n_pretrained_100ep.yaml"
)

echo "Submitting multi-seed arrays for ${#CONFIGS[@]} configurations..."

for cfg in "${CONFIGS[@]}"; do
    echo "Submitting 5-seed array for: $cfg"
    sbatch scripts/slurm/submit_multiseed_sweep.slurm "$cfg"
done

echo "All multi-seed sweeps successfully queued!"
