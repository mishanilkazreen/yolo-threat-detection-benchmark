# Contributing

## Prerequisites

Before cloning and committing to this repo, make sure you have the following
installed on your system.

### CLI tools

| Tool | Purpose | Install |
|------|---------|---------|
| **Python 3.10–3.11** | Runtime | <https://www.python.org/downloads/> |
| **uv** | Package manager | `pip install uv` or see <https://docs.astral.sh/uv/getting-started/installation/> |
| **git** | Version control | <https://git-scm.com/downloads> |
| **pre-commit** | Git hook manager | `uv tool install pre-commit` or `pip install pre-commit` |
| **Node.js 18+** | markdownlint-cli (pre-commit) | <https://nodejs.org/> |

### VS Code / Kiro extensions

Open the repo in VS Code or Kiro and accept the workspace extension
recommendations (`.vscode/extensions.json`), or install them manually:

- `ms-python.python` — Python language support
- `ms-python.pylint` — Pylint integration
- `ms-python.mypy-type-checker` — Mypy type checking
- `charliermarsh.ruff` — Ruff linter and formatter
- `ms-python.debugpy` — Python debugger
- `DavidAnson.vscode-markdownlint` — Markdown linting
- `streetsidesoftware.code-spell-checker` — Spell checking
- `eamodio.gitlens` — Git history and blame

## First-time setup

```bash
# Clone the repo
git clone https://github.com/mishanilkazreen/yolo-improvement-detection-moderation-paper.git
cd yolo-improvement-detection-moderation-paper

# Install dependencies (creates .venv automatically)
uv sync --all-extras

# Install pre-commit hooks
pre-commit install
pre-commit install --hook-type pre-push
```

## Pre-commit checks

Every commit is checked by the following hooks (run automatically):

1. **Trailing whitespace / EOF fixer** — formatting hygiene
2. **Ruff** — linting and auto-formatting (replaces black + isort + flake8)
3. **markdownlint** — Markdown style
4. **Mypy** — static type checking (src/ only)
5. **Pylint** — code quality (fail-under 9.0)

On `git push`, **pytest** also runs.

To run all hooks manually:

```bash
pre-commit run --all-files
```

## Code style

- **Line length**: 100 characters
- **Formatter**: Ruff (double quotes, spaces)
- **Linter**: Ruff + Pylint (no relaxed rules)
- **Type checker**: Mypy with strict settings
- **Naming**: PascalCase for classes, snake_case for functions/variables
- **Logging**: Use lazy `%`-style formatting, not f-strings
  (`logger.info("value: %s", val)` not `logger.info(f"value: {val}")`)
- **File I/O**: Always specify `encoding="utf-8"` in `open()` calls
- **Exceptions**: Catch specific exceptions, not bare `Exception`
- **Re-raising**: Use `raise ... from exc` in except blocks
