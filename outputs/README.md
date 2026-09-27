# Outputs Directory Provenance and Inventory Guide

## Run Inventory Summary
The `outputs/` directory contains 76 completed experimental run folders:
- **60 Core Factorial Runs**: 3 architectures (`yolov8n`, `yolo11n`, `yolo12n`) x 2 initialisation strategies (`pretrained`, `random`) x 2 epoch budgets (10 epochs, 100 epochs) x 5 seeds (`42`, `123`, `456`, `789`, `1011`).
- **12 Seed 42 Alias Runs**: Created to align initial seed-42 runs (executed with runner-level random_seed=42) (`yolov8n_random`, etc.) with the 5-seed directory convention (`seed_42`). Metric summaries in `round_N_metrics.json` reflect the completed 5-round experiments.
- **4 Freeze Variant Runs**: `yolov8n` and `yolo12n` under `frozen_backbone` and `headonly` fine-tuning strategies.

## Provenance of `training_set_evolution.json`
`training_set_evolution.json` in each run directory is a derived summary file reconstructed directly from the raw per-round metric artifacts (`round_1_metrics.json` through `round_5_metrics.json`).

### Root Cause of Legacy Spurious 6th Entry
In `src/training/runner.py` (lines 806–818), a Round 5 post-training cleanup step was executed: after completing Round 5 training, the runner unconditionally moved all remaining unlabelled pool images into the cumulative training set array for final accounting and logged an extra 6th trajectory entry (`training_set_size = 3545`, `unlabeled_remaining = 0`). Because this cleanup step ran after Round 5 training had finished, no model training, validation, or held-out test metric ever consumed or evaluated those post-Round 5 appended samples. All 76 `training_set_evolution.json` files have been standardized to represent the actual 5 training rounds.

## Checkpoint & Logging Retention Policy
- **Run Metric Logs**: All 60 core factorial runs and 4 freeze variants preserve full step-by-step training and evaluation logs in `round_1_metrics.json` through `round_5_metrics.json` (containing optimizer steps, images processed, GFLOPs/TFLOPs, losses, mAP@0.5, and mAP@0.5:0.95). Initial seed-42 legacy and baseline directories also include raw Ultralytics `results.csv` and `args.yaml`.
- **Model Checkpoints (`best.pt`)**: Trained weights for all 300 round checkpoints are stored on the SCIAMA HPC cluster storage volume to comply with GitHub repository file size limits. A planned Zenodo release (v1.2) will archive representative checkpoints once minted; the current Zenodo deposit (v1.1) contains source code and split configurations only.
