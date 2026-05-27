# Tasks — Pre-submission Revision

Colleague review comments to address before submission. Tracked as GitHub issues.

## Paper (manighahrmani — `Journal-of-Real-Time-Image-Processing`)

- [x] Expand "YOLO" to "You Only Look Once (YOLO)" on first use — issue [#6](https://github.com/manighahrmani/Journal-of-Real-Time-Image-Processing/issues/6)
- [ ] Add citation for YOLOv8 edge-device deployment in related works — issue [#7](https://github.com/manighahrmani/Journal-of-Real-Time-Image-Processing/issues/7)
- [ ] Clarify epoch labels in Tables 2 and 4 (one-shot 50/100 vs incremental 10/100 mismatch) — issue [#8](https://github.com/manighahrmani/Journal-of-Real-Time-Image-Processing/issues/8)
- [ ] Add 30 fps human-eye baseline reference for real-time claim — issue [#9](https://github.com/manighahrmani/Journal-of-Real-Time-Image-Processing/issues/9)
- [ ] Discuss model size (parameters / GFLOPs) throughout paper — issue [#10](https://github.com/manighahrmani/Journal-of-Real-Time-Image-Processing/issues/10)

## Code (mishanil — `yolo-threat-detection-benchmark`)

- [ ] Measure inference speed (FPS / ms per image) for all configs — issue [#14](https://github.com/mishanilkazreen/yolo-threat-detection-benchmark/issues/14)
- [ ] Record GFLOPs and parameter count per config — issue [#15](https://github.com/mishanilkazreen/yolo-threat-detection-benchmark/issues/15)
- [ ] Possibly re-run one-shot baselines at 50/500 epochs — issue [#16](https://github.com/mishanilkazreen/yolo-threat-detection-benchmark/issues/16) (blocked on #8)

## Held — awaiting input

- XAI: paper says "future work" but repo has explainability results. Wait for Anna before deciding whether to remove from future work or include results.

## Reference commands

```bash
# Compile paper
cd Journal-of-Real-Time-Image-Processing/submission
pdflatex sn-article.tex && bibtex sn-article && pdflatex sn-article.tex && pdflatex sn-article.tex

# Lint / format
uv run ruff check --fix .
uv run ruff format .
```
