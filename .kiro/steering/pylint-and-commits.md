---
inclusion: auto
description: Pylint rules, commit conventions, and experiment context
---

# Pylint and Commit Standards

## Pylint is mandatory

Pylint runs as a pre-commit hook on every commit. The minimum score threshold
is **9.0/10**. Do not relax any pylint rules — fix the code instead.

### Common pylint fixes

| Rule | Fix |
|------|-----|
| `logging-fstring-interpolation` | `logger.info("val: %s", x)` not `logger.info(f"val: {x}")` |
| `unspecified-encoding` | Always pass `encoding="utf-8"` to `open()` |
| `invalid-name` | Classes use PascalCase (`MyClass`), not underscores (`My_Class`) |
| `raise-missing-from` | `raise X from exc` inside `except` blocks |
| `broad-exception-caught` | Catch specific exceptions (`OSError`, `ValueError`, etc.) |
| `import-outside-toplevel` | Move imports to the top of the file |
| `unnecessary-pass` | Remove `pass` from non-empty class/function bodies |
| `superfluous-parens` | `if condition:` not `if (condition):` |
| `no-else-return` | Remove `else` after a `return` statement |
| `unused-argument` | Prefix with underscore: `_unused_arg` |
| `too-many-arguments` | Group related params into a dataclass |
| `too-many-locals/branches/statements` | Extract helper methods |

### Never add `# pylint: disable` comments

If pylint flags something, refactor the code to fix it properly. The only
exception is `# pylint: disable=duplicate-code` for test files that
intentionally share setup patterns.

## Pre-commit workflow

Before committing, all of these must pass:

```bash
uv run pre-commit run --all-files
```

This runs: trailing-whitespace, ruff, markdownlint, mypy, and **pylint**.

On `git push`, pytest also runs.

## Commit message format

Use conventional commits:

- `feat:` new feature
- `fix:` bug fix
- `refactor:` code restructuring
- `docs:` documentation
- `style:` formatting only
- `test:` test changes
- `chore:` maintenance

## Project context — pretrained vs random comparison

This project investigates whether the Kutlu & Emiroğlu (2025) MDPI paper
actually used pretrained weights despite claiming random initialization.
See GitHub issue #9 for the original discussion.

### Key facts from the MDPI paper

- **Architecture:** YOLOv8n (nano, 3.2M params)
- **Protocol:** 5 incremental rounds, 10 epochs per round (50 total)
- **Optimizer:** AdamW, lr=0.001, step decay ×0.1 every 5 epochs
- **Claimed init:** Random (Algorithm 1 line 8, Section 3.2)
- **Claimed results:** mAP@0.5 = 0.886, F1 = 0.83

### Our reproduction results (random init)

- mAP@0.5 ≈ 0.54, F1 ≈ 0.55 — far below the paper's claims

### Experiment design

We run a 2×2×2×2 comparison matrix:

- **Architecture:** YOLOv8 vs YOLOv12
- **Size:** nano (n) only (small was considered, cut due to compute cost)
- **Init:** random (`.yaml`) vs pretrained (`.pt`)
- **Epochs:** 10/round (MDPI protocol) vs 100/round (early stopping, patience=10)

### Config naming convention

`{arch}_{init}[_{epochs}].yaml` — e.g. `yolov8n_pretrained_100ep.yaml`

### Weight initialization rules

- `weights: "yolov8n.yaml"` → random init (architecture only, no pretrained weights)
- `weights: "yolov8n.pt"` → COCO-pretrained transfer learning
- Same pattern for v12 and s variants

Do NOT change a config's weights field without understanding the experiment
it belongs to. Each config is a specific cell in the comparison matrix.
