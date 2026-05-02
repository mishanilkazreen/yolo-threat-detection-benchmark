---
inclusion: auto
description: Pretrained vs random weight comparison experiment plan and task tracking
---

# Experiment Plan — Pretrained vs Random Weight Comparison

## Background

The Kutlu & Emiroğlu (2025) MDPI paper claims mAP@0.5 = 0.886 with YOLOv8n
using random initialization and 5 incremental rounds of 10 epochs each. Our
reproduction yields mAP@0.5 ≈ 0.54 with truly random weights. This project
tests whether pretrained (COCO) weights explain the discrepancy.

## MDPI paper parameters (verified from paper text)

- Architecture: YOLOv8n (3.2M params)
- Rounds: 5 incremental
- Epochs per round: 10
- Batch size: 16
- Optimizer: AdamW
- Learning rate: 0.001, step decay ×0.1 every 5 epochs
- Dataset: 5064 images, 70/20/10 split, 20% of train for Round 1
- Augmentation: mosaic, scale ±50%, horizontal flip 50%, HSV jitter

## Metrics to report (matching MDPI paper)

For every experiment, report:

1. **Per-round (rounds 1–5):**
   - F1-score (all classes, at optimal confidence threshold)
   - Precision (all classes, at conf=1.0)
   - mAP@0.5 (all classes)
   - Per-class F1-score (knife, pistol)
   - Per-class Precision (knife, pistol)
   - Per-class mAP@0.5 (knife, pistol)

2. **Final test set evaluation:**
   - Same metrics as above on the held-out test set

3. **For early stopping runs only:**
   - Actual epoch the training stopped on (per round)

## Task list

### Phase 1 — Nano models

- [ ] Run `yolov8n_random.yaml` (MDPI protocol, random init)
- [ ] Run `yolov8n_pretrained.yaml` (MDPI protocol, COCO pretrained)
- [ ] Run `yolo12n_random.yaml` (MDPI protocol, random init)
- [ ] Run `yolo12n_pretrained.yaml` (MDPI protocol, COCO pretrained)
- [ ] Run `yolov8n_random_100ep.yaml` (100 ep/round, early stopping, random)
- [ ] Run `yolov8n_pretrained_100ep.yaml` (100 ep/round, early stopping, COCO)
- [ ] Run `yolo12n_random_100ep.yaml` (100 ep/round, early stopping, random)
- [ ] Run `yolo12n_pretrained_100ep.yaml` (100 ep/round, early stopping, COCO)
- [ ] Compile Phase 1 results into comparison table

### Phase 2 — Small models

- [ ] Run `yolov8s_random.yaml` (MDPI protocol, random init)
- [ ] Run `yolov8s_pretrained.yaml` (MDPI protocol, COCO pretrained)
- [ ] Run `yolo12s_random.yaml` (MDPI protocol, random init)
- [ ] Run `yolo12s_pretrained.yaml` (MDPI protocol, COCO pretrained)
- [ ] Run `yolov8s_random_100ep.yaml` (100 ep/round, early stopping, random)
- [ ] Run `yolov8s_pretrained_100ep.yaml` (100 ep/round, early stopping, COCO)
- [ ] Run `yolo12s_random_100ep.yaml` (100 ep/round, early stopping, random)
- [ ] Run `yolo12s_pretrained_100ep.yaml` (100 ep/round, early stopping, COCO)
- [ ] Compile Phase 2 results into comparison table

### Analysis

- [ ] Compare random vs pretrained for each architecture/size
- [ ] Determine if pretrained v8n results match MDPI paper claims
- [ ] Write up findings for the latex paper
