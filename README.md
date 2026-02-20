# XplainRescue

> The rapid proliferation of harmful visual content has introduced novel challenges for content
> moderation on online platforms such as Discord and Reddit, where hateful or harmful imagery can be
> disseminated at scale. Existing threat detection approaches for flagging such material have
> predominantly relied on YOLOv8 with limited explainability, leaving moderators without actionable
> insight into model decisions. This study addresses two open gaps in the literature: (1) the lack of
> comparative evaluation of post-v8 YOLO architectures for hateful content detection, and (2) the
> insufficient integration of advanced explainable AI (XAI) techniques beyond coarse Grad-CAM
> heatmaps. We benchmark YOLOv11, YOLOv12, and YOLO26 on a multiclass harmful-content dataset and
> systematically apply complementary XAI methods, including Layer-wise Relevance Propagation (LRP)
> and SHAP, alongside Grad-CAM to provide fine-grained, instance-level attribution maps. Our
> experimental pipeline evaluates detection accuracy (mAP@0.5, precision, recall, F1-score),
> inference latency, and explainability fidelity across all architectures. Preliminary results
> indicate that attention-enhanced backbones in YOLOv11 and YOLOv12 improve localisation of subtle
> hateful symbols, while YOLO26's NMS-free prediction head reduces post-processing overhead. The
> hybrid XAI framework yields richer explanations that enable moderators to understand why specific
> image regions are flagged, supporting transparent and accountable automated moderation. This work
> provides a practical, deployable framework for platform trust and safety teams seeking accurate,
> interpretable, and efficient AI-driven content moderation.

## Prerequisites

- Python 3.10 or 3.11
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (Python package and project manager)
- [pre-commit](https://pre-commit.com/) (installed automatically as a dev dependency)
- A Roboflow API key (stored in `.env`) for dataset download
- Node.js (optional, for markdownlint in pre-commit hooks)

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/mishanilkazreen/yolo-improvement-detection-moderation-paper.git
cd yolo-improvement-detection-moderation-paper
```

### 2. Install uv

Follow the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/):

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 3. Create virtual environment and install dependencies

```bash
# Create venv and install the project with dev dependencies
uv venv --python 3.11
uv pip install -e ".[dev]"
```

To also install optional explainability dependencies (SHAP):

```bash
uv pip install -e ".[dev,xai]"
```

### 4. Install pre-commit hooks

```bash
uv run pre-commit install
uv run pre-commit install --hook-type pre-push
```

### 5. Download dataset

Create a `.env` file with your Roboflow credentials:

```text
ROBOFLOW_API_KEY=your_api_key
ROBOFLOW_WORKSPACE=your_workspace
ROBOFLOW_PROJECT=your_project
ROBOFLOW_VERSION=1
```

Then run:

```bash
uv run python scripts/download_dataset.py
```

### 6. Validate dataset

```bash
uv run python scripts/validate_dataset.py
```

## Usage

### Quick test (3 epochs)

```bash
uv run python scripts/train_model.py config/models/yolov8n_test.yaml
```

### Full training

```bash
uv run python scripts/train_model.py config/models/yolov8n.yaml
uv run python scripts/train_model.py config/models/yolov11n.yaml
uv run python scripts/train_model.py config/models/yolov12n.yaml
uv run python scripts/train_model.py config/models/yolo26n.yaml
```

## Development

### Linting and formatting

```bash
# Check linting
uv run ruff check .

# Auto-fix lint issues
uv run ruff check --fix .

# Check formatting
uv run ruff format --check .

# Auto-format
uv run ruff format .

# Type checking
uv run mypy src/ --ignore-missing-imports
```

### Running tests

```bash
# Run all tests
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ -v --cov=src --cov-report=html --cov-report=term
```

### Pre-commit

Pre-commit hooks run automatically on `git commit` and include:

- Trailing whitespace and end-of-file fixes
- YAML, TOML, and JSON validation
- Ruff linting and formatting
- Markdownlint for Markdown files
- Mypy type checking on `src/`
- Pytest runs on `git push` (pre-push hook)

To run all hooks manually:

```bash
uv run pre-commit run --all-files
```

### Editor setup

The `.vscode/` folder is committed with recommended extensions and workspace settings. When you open
the project in VS Code or Kiro, you'll get a prompt to install the recommended extensions:

- Ruff (linter and formatter)
- Python
- Mypy type checker
- Markdownlint
- Even Better TOML
- Code Spell Checker (uses `.cspell.json` dictionary)
- YAML

These ensure consistent formatting, linting, and spell checking across all contributors.

## CI/CD

This project uses GitHub Actions for continuous integration and delivery:

- **CI** (`ci.yml`): Runs on push to `main`/`develop` and on pull requests. Executes pre-commit
  hooks, ruff linting, ruff formatting checks, mypy type checking, and pytest with coverage across
  Python 3.10 and 3.11.
- **CD** (`cd.yml`): Builds and validates the distribution package on push to `main` or version
  tags.

## Project structure

```text
├── .github/workflows/     # CI/CD pipelines
├── .vscode/               # Recommended extensions and editor settings
├── .cspell.json           # Spell-check dictionary
├── .pre-commit-config.yaml
├── .markdownlint.json
├── .python-version
├── pyproject.toml         # Project metadata, dependencies, and tool config
├── config/
│   ├── data/              # Dataset configurations
│   └── models/            # Model training configurations
├── src/
│   ├── config/            # Configuration parsing
│   ├── data/              # Dataset loading and validation
│   ├── training/          # Training and evaluation
│   ├── explainability/    # Explainability methods (future)
│   └── aggregation/       # Results aggregation (future)
├── scripts/               # Training and utility scripts
├── tests/                 # Unit and property-based tests
├── runs/                  # Training outputs (gitignored)
├── outputs/               # Evaluation results (gitignored)
└── explanations/          # Explainability outputs (gitignored)
```

## Configuration

Model configs are in `config/models/`. Each config specifies:

- Model architecture and pretrained weights
- Training parameters (epochs, image size, patience)
- Dataset path
- Multi-run settings (seeds, number of runs)

Dataset configs are in `config/data/`:

- `weapon_detection_data.yaml` — Relative paths (for version control)
- `weapon_detection_data.local.yaml` — Absolute paths (gitignored)

## Results

Training results are saved to:

- `runs/detect/{model_name}/` — Training logs, plots, checkpoints
- `outputs/{model_name}/` — Evaluation metrics (JSON)
- `explanations/{model_name}/` — Explainability visualizations (future)
