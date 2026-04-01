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

### 3. Clone YOLO-CAM (EigenCAM for YOLO)

YOLO-CAM provides EigenCAM visualization for explainability. Clone it into the project root:

```bash
git clone https://github.com/rigvedrs/YOLO-26-CAM.git yolo_cam
```

This provides [rigvedrs/YOLO-26-CAM](https://github.com/rigvedrs/YOLO-26-CAM), an EigenCAM implementation
supporting YOLO26, YOLOv12, YOLOv11, and YOLOv8. The library will be automatically available to Python when
running scripts or tests from the project root.

To update YOLO-CAM to the latest version:

```bash
cd yolo_cam
git pull origin main
cd ..
```

### 6. Download dataset

#### Get your Roboflow credentials

1. Go to [roboflow.com](https://roboflow.com) and sign in
2. Navigate to your project
3. Look at the URL in your browser — it will be in this format:

   ```text
   https://app.roboflow.com/{WORKSPACE}/{PROJECT}/...
   ```

   For example: `https://app.roboflow.com/example-workspace/example-project/1`
   - `WORKSPACE` = `example-workspace`
   - `PROJECT` = `example-project`

4. Get your API key:
   - Click your profile icon (top right)
   - Go to "Settings" or visit [app.roboflow.com/settings](https://app.roboflow.com/settings)
   - Scroll to the "API Key" section
   - Copy your private API key

5. Create a `.env` file in the project root with these values (you can copy `.env.example` as a starting point):

   ```bash
   cp .env.example .env
   ```

   Then edit `.env` with your actual values.

#### Download the dataset

Then run:

```bash
uv run python scripts/download_dataset.py
```

### 7. Validate dataset

```bash
uv run python scripts/validate_dataset.py
```

## Usage

### Full training

```bash
uv run python scripts/train_model.py config/models/yolov8n.yaml
uv run python scripts/train_model.py config/models/yolo11n.yaml
uv run python scripts/train_model.py config/models/yolo12n.yaml
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
├── yolo_cam/              # YOLO-CAM (EigenCAM) library (vendored from rigvedrs/YOLO-26-CAM@f380b07)
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
