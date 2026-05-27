# Tasks — Revision Phase

All experiments are complete. All GitHub issues are closed.
This file tracks paper revision tasks.

## Status summary

- **36 experiments completed** (16 incremental + 20 one-shot baselines)
- **3 architectures**: YOLOv8n, YOLO11n, YOLOv12n
- **2 init strategies**: random, COCO-pretrained
- **3 freeze strategies**: none, frozen backbone, head-only
- **2 epoch budgets**: 50/10-per-round, 500/100-per-round
- **2 protocols**: incremental (5-round), one-shot baseline
- **Results CSV**: `outputs/full_evaluation_results.csv` (116 rows)
- **Training curves/plots**: `runs/` (results.csv, confusion matrices, PR curves)
- **Paper repo**: `Journal-of-Real-Time-Image-Processing/submission/sn-article.tex`

## Revision tasks

### 1. Review and address reviewer comments

- [ ] Read reviewer feedback (when received)
- [ ] List each comment and required action
- [ ] Assign each action to a team member

### 2. Verify paper numbers match CSV

- [ ] Cross-check every number in the paper against `outputs/full_evaluation_results.csv`
- [ ] Ensure per-class metrics (knife/pistol) are correctly reported
- [ ] Verify training times match what's in the JSON files

### 3. Figures and plots

- [ ] Generate publication-quality figures from `runs/` data
- [ ] Per-round mAP trajectory plot (incremental configs vs MDPI paper)
- [ ] Baseline comparison bar chart
- [ ] Frozen vs non-frozen comparison
- [ ] Confusion matrices for key configs (pretrained vs random)
- [ ] Training time comparison chart

### 4. Discussion section

- [ ] Ensure the frozen-baseline findings are discussed:
  - v8n frozen backbone baseline: 0.885 (vs non-frozen 0.907)
  - v8n headonly baseline: 0.783 (vs non-frozen 0.907)
  - v12n headonly baseline: 0.625 (severe degradation)
- [ ] Discuss why v12n head-only performs much worse than v8n head-only
- [ ] Address the MDPI paper's R1 anomaly (0.518 doesn't match any config)

### 5. Final checks before resubmission

- [ ] Compile paper: `cd Journal-of-Real-Time-Image-Processing/submission && pdflatex sn-article.tex && bibtex sn-article && pdflatex sn-article.tex && pdflatex sn-article.tex`
- [ ] Check page count (12-page limit for JRTIP)
- [ ] Spell check: `npx cspell "**/*.tex" --no-progress`
- [ ] Verify all references compile without warnings
- [ ] Push final PDF to journal repo

## Key files for revision

| File | Purpose |
|---|---|
| `outputs/full_evaluation_results.csv` | All metrics, all experiments |
| `outputs/*/final_test_metrics.json` | Detailed per-config test results |
| `outputs/*/round_N_metrics.json` | Per-round incremental trajectories |
| `outputs/*_baseline/train/final_test_metrics.json` | Baseline results |
| `runs/**/results.csv` | Per-epoch training curves |
| `runs/**/*_curve.png` | PR/F1/P/R curves |
| `runs/**/confusion_matrix*.png` | Confusion matrices |
| `Journal-of-Real-Time-Image-Processing/submission/sn-article.tex` | Paper manuscript |
| `Journal-of-Real-Time-Image-Processing/submission/sn-bibliography.bib` | References |

## Reference commands

```bash
# Regenerate CSV from all outputs
uv run python scripts/regenerate_csv.py

# Lint / format
uv run ruff check --fix .
uv run ruff format .

# Compile paper
cd Journal-of-Real-Time-Image-Processing/submission
pdflatex sn-article.tex && bibtex sn-article && pdflatex sn-article.tex && pdflatex sn-article.tex
```
