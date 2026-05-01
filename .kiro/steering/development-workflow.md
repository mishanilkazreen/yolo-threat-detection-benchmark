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

## Running Training

```bash
uv run python scripts/train_model.py config/models/yolov8n.yaml
uv run python scripts/train_model.py config/models/yolo11n.yaml
uv run python scripts/train_model.py config/models/yolo12n.yaml
uv run python scripts/train_model.py config/models/yolo26n.yaml
```

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
