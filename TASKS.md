# Tasks

Task list for completing the paper, ordered: literature review first, paper write-up last.

Scope reminder: nano only (v8n, v11n, v12n). Small models (v8s, v12s) are out of scope —
the MDPI paper does not use them, and nano runs already take significant wall-clock time.

## 1. Literature review — papers using the same dataset

Related issue: [#10](https://github.com/mishanilkazreen/yolo-improvement-detection-moderation-paper/issues/10)

**✅ DONE** — All literature review tasks completed.

- [x] Run the Google Scholar prompts listed in #10
  (e.g. `"weapon_detection" "roboflow" "knife" "pistol" YOLO`)
- [x] Record each hit in a spreadsheet with: citation, dataset used, model, reported
  mAP@0.5, weight init, fine-tune strategy
- [x] Park the most relevant PDFs under `Journal-of-Real-Time-Image-Processing/literature/`
- [x] Cross-check: do any independent papers reproduce the MDPI paper's claimed 0.886
  with random init? That answer drives the Related Work narrative

Baseline reference numbers to compare against (already on `main`):

- Random: `outputs/yolov8n_random/final_test_metrics.json` (mAP@0.5 = 0.561)
- Pretrained: `outputs/yolov8n_pretrained/final_test_metrics.json` (mAP@0.5 = 0.894)
- MDPI claim: mAP@0.5 = 0.886

## 2. Run the one-shot baselines

Related issue: [#13](https://github.com/mishanilkazreen/yolo-improvement-detection-moderation-paper/issues/13)

**✅ DONE** — All 8 nano baselines completed. Results in `outputs/full_evaluation_results.csv` (phase=baseline).

Summary:

| Config | mAP@0.5 | F1 | Epochs | Time |
|---|---|---|---|---|
| yolov8n_baseline_pretrained | 0.9065 | 0.8712 | 50 | 68 min |
| yolov8n_baseline_random | 0.8286 | 0.7925 | 50 | 64 min |
| yolov8n_baseline_pretrained_100ep | 0.9143 | 0.8744 | 500 (stopped ~101) | 135 min |
| yolov8n_baseline_random_100ep | 0.8663 | 0.8382 | 500 (stopped ~136) | 193 min |
| yolo12n_baseline_pretrained | 0.9143 | 0.8735 | 50 | 89 min |
| yolo12n_baseline_random | 0.7251 | 0.6874 | 50 | 85 min |
| yolo12n_baseline_pretrained_100ep | 0.8853 | 0.8515 | 500 (stopped ~59) | 105 min |
| yolo12n_baseline_random_100ep | 0.8630 | 0.8390 | 500 (stopped ~222) | 353 min |

Run each via the baseline runner:

```bash
# 50-epoch baselines (compute-matched to MDPI 5 x 10)
uv run python scripts/train_baseline.py config/models/yolov8n_baseline_pretrained.yaml
uv run python scripts/train_baseline.py config/models/yolov8n_baseline_random.yaml
uv run python scripts/train_baseline.py config/models/yolo12n_baseline_pretrained.yaml
uv run python scripts/train_baseline.py config/models/yolo12n_baseline_random.yaml

# 100-epoch + early stopping baselines
uv run python scripts/train_baseline.py config/models/yolov8n_baseline_pretrained_100ep.yaml
uv run python scripts/train_baseline.py config/models/yolov8n_baseline_random_100ep.yaml
uv run python scripts/train_baseline.py config/models/yolo12n_baseline_pretrained_100ep.yaml
uv run python scripts/train_baseline.py config/models/yolo12n_baseline_random_100ep.yaml
```

Then inspect:

```bash
uv run python scripts/report_results.py --phase baseline
```

Fill in the metrics columns in `outputs/full_evaluation_results.csv` for the 8
`yolo*_baseline_*` rows.

## 3. Run the missing frozen config

Related issue: [#11](https://github.com/mishanilkazreen/yolo-improvement-detection-moderation-paper/issues/11)

**✅ DONE** — `yolo12n_pretrained_headonly` (freeze=21) completed.
Result: test mAP@0.5 = 0.574, F1 = 0.568. Trajectory: R1=0.387, R2=0.483, R3=0.551, R4=0.566, R5=0.594.

Note: YOLOv12n head-only performs worse than v8n head-only (0.776) because v12n's
attention-enhanced backbone produces features that are less transferable when frozen.
The detect head alone cannot compensate.

## 4. Add YOLOv11 baseline configs and run them

Related issue:
[#13](https://github.com/mishanilkazreen/yolo-improvement-detection-moderation-paper/issues/13)
(follow-up checklist)

**✅ DONE** — All yolo11n baseline and incremental configs created and runs completed.

**✅ Configs created** — 4 yolo11n baseline configs are now in `config/models/`:

- `yolo11n_baseline_random.yaml` (50 epochs, no early stopping)
- `yolo11n_baseline_pretrained.yaml` (50 epochs, no early stopping)
- `yolo11n_baseline_random_100ep.yaml` (500 epochs, patience=10)
- `yolo11n_baseline_pretrained_100ep.yaml` (500 epochs, patience=10)

`BASELINE_CONFIGS` in `scripts/report_results.py` has been updated to include them.

### Still to do

- [x] Run the 4 yolo11n baselines ✅ Complete:
  - yolo11n_baseline_random: mAP@0.5 = 0.7879, F1 = 0.752 (50 ep, 69 min)
  - yolo11n_baseline_pretrained: mAP@0.5 = 0.924, F1 = 0.8818 (50 ep, 70 min)
  - yolo11n_baseline_random_100ep: mAP@0.5 = 0.8827, F1 = 0.8417 (500 ep, 247 min)
  - yolo11n_baseline_pretrained_100ep: mAP@0.5 = 0.9243, F1 = 0.8896 (500 ep, 195 min)

- [x] Optional (if needed for Related Work parity): create yolo11n MDPI-style
  incremental configs (`yolo11n_random.yaml`, `yolo11n_pretrained.yaml`, etc.)
  and run them ✅ Complete:
  - yolo11n_random: mAP@0.5 = 0.5281, F1 = 0.5449 (36 min)
  - yolo11n_pretrained: mAP@0.5 = 0.8997, F1 = 0.8670 (55 min)
  - yolo11n_random_100ep: mAP@0.5 = 0.8686, F1 = 0.8403 (418 min)
  - yolo11n_pretrained_100ep: mAP@0.5 = 0.9043, F1 = 0.8595 (264 min)

Existing architecture stub: `config/models/yolo11n.yaml`.
Pretrained weights: `yolo11n.pt` (already downloaded at project root).

## 5. Build the comparison tables (baseline-anchored)

Related issue: [#11](https://github.com/mishanilkazreen/yolo-improvement-detection-moderation-paper/issues/11)

**✅ DONE** — All 5 tables generated by `scripts/build_tables.py`. Output at
`paper_figures/tables.tex`. Run with `uv run python scripts/build_tables.py`.

- [x] **Baseline reference table** (Table A) — 12 configs, mAP@0.5, mAP@0.5:0.95,
  F1, precision, recall, knife/pistol mAP, training time
- [x] **Per-round trajectory table** (Table B) — R1–R5 + final test for all 16
  incremental configs, plus MDPI reference row
- [x] **Per-class breakdown** at Round 1 and Round 5 (Table C)
- [x] **Extended training (100 ep)** comparison (Table D)
- [x] **Wall-clock time comparison** (Table E)

**Key question answered:** does `frozen_backbone` match the MDPI R3–R5 trajectory
better than head-only or fully-pretrained?

| Config | R1 | R3 | R5 | vs MDPI R5 |
|---|---|---|---|---|
| MDPI (random init) | 0.518 | 0.838 | 0.886 | — |
| v8n-P full | 0.708 | 0.862 | 0.892 | +0.006 |
| v8n-P frozen backbone | 0.746 | 0.875 | **0.884** | **−0.002** |
| v8n-P head-only | 0.581 | 0.736 | 0.762 | −0.124 |

**Answer:** `frozen_backbone` matches MDPI's R5 almost exactly (0.884 vs 0.886,
Δ = −0.002) and stays within ±0.04 of MDPI across R3–R5. Full pretrained is
marginally higher (+0.006). Head-only falls well short (−0.124). All pretrained
trajectories start higher at R1 than MDPI (pretrained weights dominate early rounds),
confirming the trajectory shape difference is driven by weight initialisation, not
the freeze strategy.

## 6. Structure the paper Results section

Related issue: [#12](https://github.com/mishanilkazreen/yolo-improvement-detection-moderation-paper/issues/12)

Draft the Results section around the baseline-centric framing:

```text
4. Results
  4.1 One-shot baselines (per architecture)
  4.2 Effect of training protocol: incremental vs baseline
  4.3 Effect of weight init: random vs pretrained
    4.3.1 On the baselines
    4.3.2 On the incremental protocol (MDPI reproduction)
  4.4 Effect of freeze strategy (pretrained only)
  4.5 Extended training with early stopping (100 ep)
  4.6 Cross-condition factorial analysis + time / compute cost
```

Files to update:

- [ ] `README.md` (top-level narrative is already baseline-centric; numbers can be
  filled in as runs complete)
- [ ] `.kiro/steering/experiment-plan.md` (status tables)
- [ ] Paper manuscript in `Journal-of-Real-Time-Image-Processing/` (if present)

## 7. Write the paper

- [ ] Draft Introduction, Related Work (from task 1), Methods, Results, Discussion,
  Conclusions using the tables from task 5
- [ ] Figures: per-round trajectory plots, baseline vs MDPI bar charts, class-wise
  breakdowns, time-vs-accuracy scatter
- [ ] Align on target venue: *Journal of Real-Time Image Processing* (folder exists)
  or re-submission target
- [ ] Final pass: check all numbers against `outputs/full_evaluation_results.csv`;
  every claim cites a row + metric

## 8. Add missing citations to the paper

Related GitHub issues:

- [#1 YOLO models and COCO dataset citations](https://github.com/manighahrmani/Journal-of-Real-Time-Image-Processing/issues/1)
- [#2 Partial model freezing citation](https://github.com/manighahrmani/Journal-of-Real-Time-Image-Processing/issues/2)
- [#3 Data augmentation citations](https://github.com/manighahrmani/Journal-of-Real-Time-Image-Processing/issues/3)

- [x] YOLOv8 official Ultralytics citation — added `Jocher2023` (@misc, closes #1)
- [x] YOLOv11 official Ultralytics citation — added `Jocher2024` (@misc, closes #1)
- [x] YOLOv12 citation — added `Tian2025yolov12` (arXiv:2502.12524, closes #1)
- [x] COCO dataset citation — added `Lin2014` (ECCV 2014, closes #1)
- [x] Citation for partial layer freezing / frozen backbone fine-tuning technique (§3.3)
- [x] Citations for data augmentation: mosaic augmentation, HSV jitter (§3.2)
- [x] Expand Table 1 (literature review) with additional weapon detection papers

## Reference commands

```bash
# Lint / format / type-check before every commit
uv run pre-commit run --all-files

# Just formatting / linting
uv run ruff check .
uv run ruff format .

# Tests
uv run pytest tests/ -v

# Results report
uv run python scripts/report_results.py --phase nano
uv run python scripts/report_results.py --phase baseline
uv run python scripts/report_results.py                 # everything
```

See the issues linked above for full context and sub-checklists.
