# Tasks — Pre-submission Revision

Pre-submission feedback from Morgan (co-author). Tracked as GitHub issues.

## Paper (manighahrmani — `Journal-of-Real-Time-Image-Processing`)

- [x] Expand "YOLO" to "You Only Look Once (YOLO)" on first use — issue [#6](https://github.com/manighahrmani/Journal-of-Real-Time-Image-Processing/issues/6) (closed)
- [ ] Add better citation for YOLOv8 edge-device deployment — issue [#7](https://github.com/manighahrmani/Journal-of-Real-Time-Image-Processing/issues/7) (open; Rey et al. 2025 added but may want a stronger peer-reviewed source — see issue for details on what to update in Sections 2.1 and 4.7)
- [x] Clarify epoch labels in Tables 2 and 4 — issue [#8](https://github.com/manighahrmani/Journal-of-Real-Time-Image-Processing/issues/8) (closed; relabelled "100 ep" → "500 ep" for baselines — no retraining needed)
- [x] Add 30 fps human-eye baseline reference for real-time claim — issue [#9](https://github.com/manighahrmani/Journal-of-Real-Time-Image-Processing/issues/9) (closed; cites Haarlem et al. 2024)
- [x] Discuss model size (parameters / GFLOPs) throughout paper — issue [#10](https://github.com/manighahrmani/Journal-of-Real-Time-Image-Processing/issues/10) (closed; new Section 4.7 "Inference Speed and Model Complexity")

## Code (mishanil — `yolo-threat-detection-benchmark`)

- [x] Measure inference speed (FPS / ms per image) for all configs — issue [#14](https://github.com/mishanilkazreen/yolo-threat-detection-benchmark/issues/14) (closed)
- [x] Record GFLOPs and parameter count per config — issue [#15](https://github.com/mishanilkazreen/yolo-threat-detection-benchmark/issues/15) (closed)
- [x] Possibly re-run one-shot baselines at 50/500 epochs — issue [#16](https://github.com/mishanilkazreen/yolo-threat-detection-benchmark/issues/16) (closed; no retraining needed, was a labelling issue)

## Held — awaiting input

- XAI: paper says "future work" but repo has explainability results. Wait for Anna before deciding whether to remove from future work or include results.

## Reference commands

```bash
# Compile paper
cd Journal-of-Real-Time-Image-Processing/submission
pdflatex sn-article.tex && bibtex sn-article && pdflatex sn-article.tex && pdflatex sn-article.tex

# Re-run inference benchmark
uv run python scripts/benchmark_inference.py

# Lint / format
uv run ruff check --fix .
uv run ruff format .
```
