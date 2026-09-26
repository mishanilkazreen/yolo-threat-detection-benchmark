# Outputs Directory Provenance and Inventory Guide

## Run Inventory Summary
The `outputs/` directory contains 76 completed experimental run folders:
- **60 Core Factorial Runs**: 3 architectures (`yolov8n`, `yolo11n`, `yolo12n`) x 2 initialisation strategies (`pretrained`, `random`) x 2 epoch budgets (10 epochs, 100 epochs) x 5 seeds (`42`, `123`, `456`, `789`, `1011`).
- **12 Seed 42 Alias Runs**: Created to align historical unseeded runs (`yolov8n_random`, etc.) with the 5-seed directory convention (`seed_42`). Logs are preserved verbatim from the initial single-seed experiments.
- **4 Freeze Variant Runs**: `yolov8n` and `yolo12n` under `frozen_backbone` and `headonly` fine-tuning strategies.

## Provenance of `training_set_evolution.json`
`training_set_evolution.json` in each run directory is a derived summary file reconstructed directly from the raw per-round metric artifacts (`round_1_metrics.json` through `round_5_metrics.json`).

### Explanation of Legacy Spurious Entry
In earlier versions of the active acquisition loop, when the unlabelled pool was fully consumed by Round 5 (`unlabeled_pool_remaining = 0`), an extra logging call appended a duplicate 6th entry (`training_set_size = 3545`, `unlabeled_remaining = 0`). This spurious entry has been cleaned across all evolution files, producing a strictly consistent 5-round trajectory.

## Checkpoint & Weight Retention Policy
- **Full Metric Logs**: `round_N_metrics.json`, `results.csv`, and `args.yaml` are permanently tracked in Git across all 76 runs.
- **Model Checkpoints (`best.pt`)**: Retained under `runs/` for representative reference runs due to repository storage limits.
