# Tasks

Task list for completing the paper, ordered: literature review first, paper write-up last.

Scope reminder: nano only (v8n, v11n, v12n). Small models (v8s, v12s) are out of scope —
the MDPI paper does not use them, and nano runs already take significant wall-clock time.

## 1. Literature review — papers using the same dataset

Related issue: [#10](https://github.com/mishanilkazreen/yolo-improvement-detection-moderation-paper/issues/10)

- [ ] Run the Google Scholar prompts listed in #10
  (e.g. `"weapon_detection" "roboflow" "knife" "pistol" YOLO`)
- [ ] Record each hit in a spreadsheet with: citation, dataset used, model, reported
  mAP@0.5, weight init, fine-tune strategy
- [ ] Park the most relevant PDFs under `Journal-of-Real-Time-Image-Processing/literature/`
- [ ] Cross-check: do any independent papers reproduce the MDPI paper's claimed 0.886
  with random init? That answer drives the Related Work narrative

Baseline reference numbers to compare against (already on `main`):

- Random: `outputs/yolov8n_random/final_test_metrics.json` (mAP@0.5 = 0.561)
- Pretrained: `outputs/yolov8n_pretrained/final_test_metrics.json` (mAP@0.5 = 0.894)
- MDPI claim: mAP@0.5 = 0.886

## 2. Run the one-shot baselines

Related issue: [#13](https://github.com/mishanilkazreen/yolo-improvement-detection-moderation-paper/issues/13)

These anchor every other comparison (metrics + wall-clock time). Placeholder rows with
`phase=baseline` are already reserved in `outputs/full_evaluation_results.csv` with
empty metric columns.

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

Config `yolo12n_pretrained_headonly` is committed but its run has not been executed.
Every other nano frozen variant is already on `main`.

```bash
uv run python scripts/train_model.py config/models/yolo12n_pretrained_headonly.yaml
```

## 4. Add YOLOv11 baseline configs and run them

Related issue:
[#13](https://github.com/mishanilkazreen/yolo-improvement-detection-moderation-paper/issues/13)
(follow-up checklist)

- [ ] Copy `config/models/yolov8n_baseline_*.yaml` to `yolo11n_baseline_*.yaml`,
  swap the `name` and `weights` fields to `yolov11n` / `yolo11n.pt` or `yolo11n.yaml`
- [ ] Mirror the same for the MDPI-style incremental configs if needed
  (e.g. `yolo11n_random.yaml`, `yolo11n_pretrained.yaml`, `_100ep` variants)
- [ ] Reserve rows in `outputs/full_evaluation_results.csv` (8 new baseline rows)
- [ ] Run each via `scripts/train_baseline.py`
- [ ] Update `NANO_CONFIGS` / `BASELINE_CONFIGS` in `scripts/report_results.py`

Existing architecture stub: `config/models/yolo11n.yaml`.

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
