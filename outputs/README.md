# Peer Review Revision: Experimental Benchmark Outputs

This directory contains the experimental artifacts and verified metrics for the paper:
**"Benchmarking YOLO Generations for Explainable Incremental Threat Detection: Weight Initialisation, Architecture, and Training Budget"** (*Journal of Real-Time Image Processing*).

---

## 1. Directory Contents & Artifacts Tracked on GitHub

1. **Multi-Seed Replication Sweeps ($N=5$ independent seeds):**
   - Evaluated across seeds `42`, `123`, `456`, `789`, and `1011` for all 12 core factorial conditions (`yolov8n`, `yolo11n`, `yolo12n`; Random vs Pretrained; 10 vs 100 epochs/round).
   - Every condition includes per-round metrics (`round_1_metrics.json` through `round_5_metrics.json`), final evaluation on the fixed test set (`final_test_metrics.json`), and sample split audits (`split_metadata.json`, `training_set_evolution.json`).
   - Standardised directory naming: explicit `_seed_42`, `_seed_123`, `_seed_456`, `_seed_789`, `_seed_1011` folders.

2. **Recovered 500-Epoch Reference Baselines:**
   - Full convergence baselines with early stopping (patience = 50) for `yolov8n`, `yolo11n`, and `yolo12n` under random and pretrained initialisation.
   - Contains stopping epoch, best epoch (peak validation fitness), optimizer steps, images processed, and wall-clock times (`baseline_summary.json`).

3. **COCO Semantic Overlap Control:**
   - `coco_knife_overlap_evaluation.json`: Zero-shot Round-0 evaluation testing knife prior transfer from MS COCO.

4. **Decomposed Inference Latency:**
   - `decomposed_latency_benchmark.json` and `decomposed_latency_benchmark.csv`: Microsecond-precision benchmarking (preprocessing, forward inference, NMS postprocessing, percentiles) across FP32 and FP16 with warmup and CUDA synchronization.

5. **Consolidated Results Matrix:**
   - `full_evaluation_results.csv`: Complete row-by-row accounting of all 476 evaluation steps across rounds, seeds, and baselines.

---

## 2. Raw Per-Box Prediction Dumps & Backup Archive

To keep the git repository lightweight and responsive, per-box bounding box validation coordinates (`round_*_validations.json` and `round_*_detections.json`, totalling ~400 MB) are archived off-git.

- **Full Raw Archive Location (Local Personal Google Drive):**
  ```text
  /Users/manigh/Library/CloudStorage/GoogleDrive-mashapicasso@gmail.com/My Drive/JRTIP_Revision_Outputs/
  ```
- **HPC Cluster Origin:**
  ```text
  sciama:/mnt/lustre2/mres/ghahrem/yolo-threat-detection-benchmark/outputs/
  ```
