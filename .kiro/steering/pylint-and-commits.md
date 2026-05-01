---
inclusion: auto
description: Pylint rules, commit conventions, and weight initialization context
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

## Weight initialization — important context

The Kutlu & Emiroğlu (2025) paper this repo implements states that Round 1
uses **random initialization** (Algorithm 1 line 8, Section 3.2). The config
file `config/models/yolov8n.yaml` uses `weights: "yolov8n.yaml"` (architecture
only, random init). Do NOT change this to `yolov8n.pt` (COCO pretrained)
unless the paper text is also updated. See GitHub issue #9 for context.

For the newer models (yolo11n, yolo12n, yolo26n) in our extension paper,
pretrained weights (`.pt`) are used because our paper's methodology section
will document this as transfer learning.
