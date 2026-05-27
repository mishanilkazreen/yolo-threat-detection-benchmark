# Tasks — Pre-submission Revision

Pre-submission feedback from Morgan (co-author).
Tracked as GitHub issues.

## Paper (`Journal-of-Real-Time-Image-Processing`)

- [x] Expand YOLO abbreviation — [#6][i6] (closed)
- [ ] Better edge-device citation — [#7][i7] (open, for other agent)
- [x] Epoch label fix — [#8][i8] (closed, relabelled to 500)
- [x] 30 fps human-eye reference — [#9][i9] (closed)
- [x] Model size discussion — [#10][i10] (closed)
- [x] XAI section added (EigenCAM + SHAP)

[i6]: https://github.com/manighahrmani/Journal-of-Real-Time-Image-Processing/issues/6
[i7]: https://github.com/manighahrmani/Journal-of-Real-Time-Image-Processing/issues/7
[i8]: https://github.com/manighahrmani/Journal-of-Real-Time-Image-Processing/issues/8
[i9]: https://github.com/manighahrmani/Journal-of-Real-Time-Image-Processing/issues/9
[i10]: https://github.com/manighahrmani/Journal-of-Real-Time-Image-Processing/issues/10

## Code (`yolo-threat-detection-benchmark`)

- [x] Inference speed — [#14][c14] (closed)
- [x] GFLOPs/params — [#15][c15] (closed)
- [x] Epoch re-runs — [#16][c16] (closed, not needed)

[c14]: https://github.com/mishanilkazreen/yolo-threat-detection-benchmark/issues/14
[c15]: https://github.com/mishanilkazreen/yolo-threat-detection-benchmark/issues/15
[c16]: https://github.com/mishanilkazreen/yolo-threat-detection-benchmark/issues/16

## Held

- XAI: ~~wait for Anna~~ — done, section added.

## Commands

```bash
# Compile paper
cd Journal-of-Real-Time-Image-Processing/submission
pdflatex sn-article.tex && bibtex sn-article \
  && pdflatex sn-article.tex && pdflatex sn-article.tex

# Lint / format
uv run ruff check --fix .
uv run ruff format .
```
