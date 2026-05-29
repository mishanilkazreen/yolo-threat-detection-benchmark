# YOLO Threat Detection Benchmark - Copilot Instructions

All agents (current and future) collaborating on this repository must strictly
adhere to the following workflow guidelines to ensure code quality and prevent
CI/CD pipeline breakage.

## Mandatory CI/CD Verification Before Commit/Push

Before committing or pushing any changes to the remote repositories (both GitHub
and Overleaf), you **MUST** run the full validation suite locally and ensure all
checks pass.

### Local Check Commands

Ensure that the following commands exit successfully with no warnings or errors:

1. **Ruff Linter**:

   ```bash
   uv run ruff check .
   ```

2. **Ruff Formatter Check**:

   ```bash
   uv run ruff format --check .
   ```

3. **Mypy Type Checking**:

   ```bash
   uv run mypy src/ --ignore-missing-imports
   ```

4. **Pytest Suite**:

   ```bash
   uv run pytest tests/ -v
   ```

Alternatively, you can run the unified pre-commit checks:

```bash
uv run pre-commit run --all-files
```

### Hook Bypass Guidelines

- Do **not** bypass git hooks (e.g., using `--no-verify`) for custom source
  code edits or tests.
- Bypassing hooks is **only** permitted when committing/pushing imported
  third-party assets (such as downloaded skills/instructions in `.github/`)
  that may not conform to the repository's strict formatting standards.
