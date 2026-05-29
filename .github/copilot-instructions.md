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

## Markdown Lint Standards

All `.md` files authored in this repo (excluding `.github/instructions/` and
`.github/skills/` which contain third-party content) **must** pass these rules:

- **MD013**: Line length ≤ 120 characters (excluding code blocks and tables).
- **MD022**: Every heading must have a blank line above **and** below it.
- **MD040**: Every fenced code block must specify a language identifier
  (use `` `text` `` for plain-text/diagram blocks with no better language).
- **MD041**: First line of every markdown file must be a top-level heading.

To check before committing, run:

```bash
uv run pre-commit run markdownlint-fix --all-files
```

### Excluded Directories

The following directories are excluded from markdownlint CI (set in
`.pre-commit-config.yaml`). Do not add new `.md` files here that violate the
rules above:

- `.github/instructions/` — installed Copilot instruction files
- `.github/skills/` — installed Copilot skill files

## Agent File Standards (`.github/agents/*.agent.md`)

When authoring `.agent.md` files, use only **recognized VS Code Copilot tool
names** in the `tools:` frontmatter list. Unknown tool names cause IDE
validation warnings and may silently disable capabilities.

### Valid Tool Names

| Purpose | Tool Name |
|---------|-----------|
| Read files | `read` |
| Edit/write files | `editFiles` |
| Search across files | `searchFiles` |
| Fetch web content | `fetch` |
| Run terminal commands | `terminal` |
| Create new files | `createFile` |
| Delete files | `deleteFile` |

**Do not use:** `web_search`, `web_fetch`, `edit`, `search`, `bgpt/*` —
these cause "Unknown tool" IDE warnings and are not valid built-in names.

For MCP server tools, declare the server in `mcp-servers:` and call tools
via the MCP server name (e.g., `bgpt.search_papers` in your prompts).

## Automated Agent Check Workflow

Every agent collaborating on this repository **must** perform the following
checks automatically before any `git push`:

1. Run `uv run pre-commit run --all-files` — all hooks must pass.
2. Verify no untracked modified files remain (`git status`).
3. Confirm the IDE problem panel is clear for all authored `.md` files.
4. If any markdownlint violations are found in authored files, fix them before
   committing (do not add exceptions to `.pre-commit-config.yaml` for authored
   files — only for third-party content).
