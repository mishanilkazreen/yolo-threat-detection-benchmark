---
inclusion: auto
description: Pretrained vs random weight comparison experiment plan and task tracking
---

# Experiment Plan — Baseline-Centric YOLO Evaluation

## Background

The Kutlu & Emiroglu (2025) MDPI paper claims mAP@0.5 = 0.886 with YOLOv8n
using random initialization and 5 incremental rounds of 10 epochs each. Our
reproduction yields mAP@0.5 = 0.561 with truly random weights. That
discrepancy motivated this project, but the broader goal is now:

> Evaluate YOLO architectures (v8, v11, v12; nano only) on weapon
> detection along several dimensions, using a **per-architecture one-shot
> baseline** as the reference every other experiment is measured against
> (metrics + wall-clock time). See issue #13.

The MDPI reproduction (YOLOv8n, 5 rounds x 10 epochs, random init, no freeze)
is one cell in that matrix, not the central study.

## Current status (May 2026)

### Completed nano experiments (main branch)

| Config | R1 | R5 | Test mAP |
|---|---|---|---|
| yolov8n_random | 0.197 | 0.534 | 0.561 |
| yolov8n_pretrained | 0.707 | 0.892 | 0.894 |
| yolo12n_random | 0.170 | 0.416 | 0.398 |
| yolo12n_pretrained | 0.660 | 0.905 | 0.890 |
| yolov8n_random_100ep | 0.603 | 0.827 | 0.843 |
| yolov8n_pretrained_100ep | 0.801 | 0.919 | 0.910 |
| yolo12n_random_100ep | 0.448 | 0.821 | 0.812 |
| yolo12n_pretrained_100ep | 0.803 | 0.895 | 0.906 |

### Completed frozen experiments (frozen_embeddings branch)

| Config | R1 | R5 | Test mAP |
|---|---|---|---|
| yolov8n_pretrained_headonly (freeze=22) | 0.581 | 0.762 | 0.776 |
| yolov8n_pretrained_frozen_backbone (freeze=10) | 0.746 | running | running |
| yolo12n_pretrained_headonly | pending | pending | pending |
| yolo12n_pretrained_frozen_backbone | pending | pending | pending |

### MDPI paper reference values

| Round | mAP@0.5 | knife mAP | pistol mAP |
|---|---|---|---|
| R1 | 0.518 | 0.263 | 0.772 |
| R2 | 0.600 | 0.325 | 0.876 |
| R3 | 0.833 | 0.800 | 0.867 |
| R4 | 0.881 | 0.884 | 0.878 |
| R5 | 0.886 | 0.884 | 0.889 |

## Experiment design (full matrix)

### Axis 0: Training protocol
- One-shot baseline (single run, no rounds) — reference point for everything else
- MDPI-style incremental (5 rounds x epochs_per_round)

### Axis 1: YOLO version
- YOLOv8 (nano)
- YOLOv11 (baselines planned; architecture stub at config/models/yolo11n.yaml)
- YOLOv12 (nano)

Note: only **nano** sizes are in scope. Small models (v8s, v12s) were cut — nano
runs already take significant wall-clock time, and the MDPI paper does not use
small models either.

### Axis 2: Weight initialisation
- Random (`.yaml` architecture only)
- Pretrained (`.pt` COCO weights)

### Axis 3: Fine-tuning strategy
- Full model (no freeze)
- Frozen backbone (freeze=10): backbone frozen, neck+head train
- Head-only (freeze=22): backbone+neck frozen, only detect head trains

### Axis 4: Epoch budget
- 10 epochs/round (MDPI protocol) — incremental only
- 100 epochs/round (early stopping, patience=10) — incremental or one-shot
- 50 epochs — one-shot baseline with compute matched to 5x10 MDPI protocol

## Next tasks (priority order)

### 1. Run the one-shot baselines (GitHub issue #13)
Sixteen baseline configs live under `config/models/*_baseline_*.yaml` for v8n/s
and v12n/s (random + pretrained x 50ep + 100ep). Run them via
`scripts/train_baseline.py` and append results to
`outputs/full_evaluation_results.csv` (placeholder rows with `phase=baseline`
and empty metric columns already exist there). These anchor every downstream
comparison — do them first.

### 2. Baseline vs incremental comparison (GitHub issue #13)
For every existing incremental experiment, produce a comparison row against
its matching one-shot baseline. Report mAP, F1, precision, per-class metrics,
and wall-clock time delta.

### 3. Literature search (GitHub issue #10)
Find papers using the same Joshi weapon detection dataset from Roboflow.
Compare their results to ours and the MDPI paper. Use Google Scholar
search prompts documented in the issue.

### 4. Complete frozen experiments (GitHub issue #11)
Wait for frozen_backbone and yolo12n headonly to finish. Compare all
frozen configs against main branch results, the one-shot baselines, and the
MDPI paper trajectory.

### 5. Structure paper Results section (GitHub issue #12)
Report all experiments in a clear factorial design anchored on the one-shot
baselines. Tables needed:
- Baseline reference table (per-architecture mAP, F1, time)
- Per-round trajectory (all configs vs MDPI paper, vs baseline)
- Per-class breakdown (knife/pistol)
- Extended training with early stopping
- Cross-condition factorial analysis

### 6. YOLOv11 baselines
`config/models/yolo11n.yaml` exists as a stub; add `yolo11n_baseline_*` configs
once the v8 / v12 comparisons are stable. Small sizes are out of scope.

## Key findings so far

1. **Random init cannot reach 0.886 in 50 epochs.** Best: 0.561 (v8n), 0.398 (v12n).
2. **Pretrained matches MDPI paper almost exactly.** v8n pretrained R5=0.892 vs paper 0.886.
3. **Head-only plateaus at ~0.76.** Frozen features limit adaptation.
4. **Frozen backbone R1 = 0.746** — higher than fully pretrained R1 (0.707) due to regularization.
5. **Extended training (100ep) closes the gap partially.** Random 100ep reaches 0.843.
6. **The MDPI paper's results are inconsistent with random init as claimed.**
