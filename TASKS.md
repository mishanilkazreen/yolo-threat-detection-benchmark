# Tasks

Task list for completing the paper, ordered: literature review first, paper write-up last.

Scope reminder: nano only (v8n, v11n, v12n). Small models (v8s, v12s) are out of scope —
the MDPI paper does not use them, and nano runs already take significant wall-clock time.

## 1. Literature review — papers using the same dataset

Related issue: [#10](https://github.com/mishanilkazreen/yolo-improvement-detection-moderation-paper/issues/10)

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

- [ ] Optional (if needed for Related Work parity): create yolo11n MDPI-style
  incremental configs (`yolo11n_random.yaml`, `yolo11n_pretrained.yaml`, etc.)
  and run them

Existing architecture stub: `config/models/yolo11n.yaml`.
Pretrained weights: `yolo11n.pt` (already downloaded at project root).

## 5. Build the comparison tables (baseline-anchored)

Related issue: [#11](https://github.com/mishanilkazreen/yolo-improvement-detection-moderation-paper/issues/11)

Once tasks 2 — 4 are done, produce:

- [ ] **Baseline reference table** — rows: each baseline config; cols: mAP@0.5,
  mAP@0.5:0.95, F1, precision, per-class (knife, pistol), total training time.
  Source: `outputs/full_evaluation_results.csv` rows with `phase = baseline`
- [ ] **Per-round trajectory table** — rows: all nano incremental configs (random /
  pretrained x full / frozen_backbone / head_only); cols: R1, R2, R3, R4, R5, final
  test, matching baseline. Include MDPI paper row (R1=0.518, R5=0.886)
- [ ] **Per-class breakdown** at Round 1 and Round 5 — knife vs pistol mAP
- [ ] **Extended training (100 ep)** comparison — random vs pretrained with actual
  stopped epoch
- [ ] **Wall-clock time comparison** — incremental vs one-shot at the same epoch
  budget

Generate via `uv run python scripts/report_results.py` and refine in a notebook
or script if more post-processing is needed.

Key question to answer: does `frozen_backbone` (freeze=10, neck+head trainable) match
the MDPI paper's R3 — R5 trajectory better than head-only or fully-pretrained?

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
