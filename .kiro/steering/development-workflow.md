---
inclusion: auto
description: Standard development workflow including setup, linting, testing, and CI/CD
---

# Development Workflow

This document outlines the standard development workflow for this project.

## Environment Setup

Always use `uv run` to run commands inside the virtual environment:

```bash
uv run <command>
```

Or activate the venv manually:

```bash
# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

## Running Experiments

### Phase 1 — Nano models (run first)

```bash
# MDPI protocol (10 epochs/round, 5 rounds)
uv run python scripts/train_model.py config/models/yolov8n_random.yaml
uv run python scripts/train_model.py config/models/yolov8n_pretrained.yaml
uv run python scripts/train_model.py config/models/yolo12n_random.yaml
uv run python scripts/train_model.py config/models/yolo12n_pretrained.yaml

# Extended training (100 epochs/round, early stopping patience=10)
uv run python scripts/train_model.py config/models/yolov8n_random_100ep.yaml
uv run python scripts/train_model.py config/models/yolov8n_pretrained_100ep.yaml
uv run python scripts/train_model.py config/models/yolo12n_random_100ep.yaml
uv run python scripts/train_model.py config/models/yolo12n_pretrained_100ep.yaml
```

Small-size variants (yolov8s, yolo12s) are explicitly out of scope — see the
experiment-plan steering.

## On Every Change

### 1. Lint and Format

```bash
uv run ruff check --fix .
uv run ruff format .
```

### 2. Pylint

```bash
uv run pylint src/ scripts/ tests/ --fail-under=9.0
```

### 3. Type Check

```bash
uv run mypy src/ --ignore-missing-imports
```

### 4. Run Tests

```bash
uv run pytest tests/ -v
```

### 5. Spell Check

```bash
npx -y cspell "**/*.md" "**/*.py" "**/*.yaml" "**/*.yml" "**/*.toml" --no-progress
```

### 6. Run Pre-commit

```bash
uv run pre-commit run --all-files
```

## Commit Conventions

Use conventional commit messages:

- `feat:` — New feature
- `fix:` — Bug fix
- `docs:` — Documentation changes
- `style:` — Formatting, whitespace
- `refactor:` — Code refactoring
- `test:` — Adding or updating tests
- `chore:` — Maintenance tasks
- `ci:` — CI/CD changes
- `build:` — Build system changes

## CI/CD

GitHub Actions runs automatically on push/PR:

- **CI** (`ci.yml`): pre-commit, ruff, mypy, pytest across Python 3.10 and 3.11
- **CD** (`cd.yml`): builds and validates the distribution package

## Quick Reference

```bash
# Setup
uv venv --python 3.11
uv pip install -e ".[dev]"
uv run pre-commit install

# Dev loop
uv run ruff check --fix .
uv run ruff format .
uv run pylint src/ scripts/ tests/ --fail-under=9.0
uv run mypy src/ --ignore-missing-imports
uv run pytest tests/ -v
uv run pre-commit run --all-files
```

## VS Code Extensions Recommendation

This workspace consists of three repositories, each with their own IDE recommendations in `.vscode/extensions.json`:
- **Outer Repo (Benchmarks):** Python (`ms-python.python`), pylint (`ms-python.pylint`), mypy (`ms-python.mypy-type-checker`), ruff (`charliermarsh.ruff`), markdownlint (`DavidAnson.vscode-markdownlint`), cspell (`streetsidesoftware.code-spell-checker`), gitlens (`eamodio.gitlens`).
- **Journal Paper Repo (`Journal-of-Real-Time-Image-Processing`):** LaTeX Workshop (`James-Yu.latex-workshop`), cspell (`streetsidesoftware.code-spell-checker`), cspell scientific terms (`streetsidesoftware.code-spell-checker-scientific-terms`), markdownlint (`DavidAnson.vscode-markdownlint`), gitlens (`eamodio.gitlens`).
- **YOLO Cam Repo (`yolo_cam`):** Jupyter (`ms-toolsai.jupyter`), Jupyter Renderers (`ms-toolsai.jupyter-renderers`), Python (`ms-python.python`), debugpy (`ms-python.debugpy`), ruff (`charliermarsh.ruff`), cspell (`streetsidesoftware.code-spell-checker`), gitlens (`eamodio.gitlens`).

## Overleaf Sync (Journal Paper)

In the `Journal-of-Real-Time-Image-Processing/` repo:
- Pull changes from Overleaf: `git pull overleaf master --no-rebase`
- Push changes to Overleaf: `./push_to_overleaf.sh`
