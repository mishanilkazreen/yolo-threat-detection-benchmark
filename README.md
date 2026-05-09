# XplainRescue

> **Pretrained vs Random Weight Initialization in Incremental YOLO-Based Threat Detection**
>
> The Kutlu & Emiroğlu (2025) MDPI paper
> ([Computers 14(12):511](https://www.mdpi.com/2073-431X/14/12/511)) reports
> mAP@0.5 = 0.886 using YOLOv8n with "random initialization" across five
> incremental learning rounds (10 epochs each). Our reproduction with truly
> random weights yields mAP@0.5 ≈ 0.54 — far below the claimed result
> ([issue #9](https://github.com/mishanilkazreen/yolo-improvement-detection-moderation-paper/issues/9)).
>
> **Hypothesis:** The MDPI paper may have inadvertently used COCO-pretrained
> weights rather than random initialization.
>
> This project performs a systematic comparison to test that hypothesis:
>
> | Experiment axis | Values |
> |---|---|
> | **Architecture** | YOLOv8, YOLOv12 |
> | **Model size** | nano (n), small (s) |
> | **Weight init** | Random (`.yaml`) vs Pretrained (`.pt`) |
> | **Epoch budget** | 10 epochs/round (MDPI protocol) vs 100 epochs/round (early stopping, patience=10) |
>
> For every combination we report the same metrics the MDPI paper reports:
> F1-score, Precision, mAP@0.5, and per-class (knife / pistol) breakdowns.
> For the 100-epoch early stopping runs we also report the actual stopping epoch.

## Experiment matrix

### Phase 1 — Nano models

| Config | Arch | Init | Epochs/round | Early stop |
|---|---|---|---|---|
| `yolov8n_random.yaml` | v8n | random | 10 | no |
| `yolov8n_pretrained.yaml` | v8n | COCO | 10 | no |
| `yolo12n_random.yaml` | v12n | random | 10 | no |
| `yolo12n_pretrained.yaml` | v12n | COCO | 10 | no |
| `yolov8n_random_100ep.yaml` | v8n | random | 100 | patience=10 |
| `yolov8n_pretrained_100ep.yaml` | v8n | COCO | 100 | patience=10 |
| `yolo12n_random_100ep.yaml` | v12n | random | 100 | patience=10 |
| `yolo12n_pretrained_100ep.yaml` | v12n | COCO | 100 | patience=10 |

### Phase 2 — Small models

| Config | Arch | Init | Epochs/round | Early stop |
|---|---|---|---|---|
| `yolov8s_random.yaml` | v8s | random | 10 | no |
| `yolov8s_pretrained.yaml` | v8s | COCO | 10 | no |
| `yolo12s_random.yaml` | v12s | random | 10 | no |
| `yolo12s_pretrained.yaml` | v12s | COCO | 10 | no |
| `yolov8s_random_100ep.yaml` | v8s | random | 100 | patience=10 |
| `yolov8s_pretrained_100ep.yaml` | v8s | COCO | 100 | patience=10 |
| `yolo12s_random_100ep.yaml` | v12s | random | 100 | patience=10 |
| `yolo12s_pretrained_100ep.yaml` | v12s | COCO | 100 | patience=10 |

### Metrics reported (per experiment)

- **Overall:** F1-score (at optimal confidence), Precision (at conf=1.0), mAP@0.5
- **Per-class:** F1-score, Precision, mAP@0.5 for knife and pistol
- **Per-round:** All of the above for each of the 5 incremental rounds
- **Early stopping runs:** Actual epoch the training stopped on

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

### 4. Download dataset

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

### 5. Validate dataset

```bash
uv run python scripts/validate_dataset.py
```

## Usage

### Running experiments

#### Phase 1 — Nano models (run first)

```bash
# MDPI protocol — 10 epochs/round, no early stopping
uv run python scripts/train_model.py config/models/yolov8n_random.yaml
uv run python scripts/train_model.py config/models/yolov8n_pretrained.yaml
uv run python scripts/train_model.py config/models/yolo12n_random.yaml
uv run python scripts/train_model.py config/models/yolo12n_pretrained.yaml

# Extended — 100 epochs/round, early stopping patience=10
uv run python scripts/train_model.py config/models/yolov8n_random_100ep.yaml
uv run python scripts/train_model.py config/models/yolov8n_pretrained_100ep.yaml
uv run python scripts/train_model.py config/models/yolo12n_random_100ep.yaml
uv run python scripts/train_model.py config/models/yolo12n_pretrained_100ep.yaml
```

After each run (or all of them), generate the results table:

```bash
uv run python scripts/report_results.py --phase nano
```

#### Phase 2 — Small models (heavier compute)

Run these on a machine with more GPU memory — same pattern, just swap `n` → `s`:

```bash
# MDPI protocol
uv run python scripts/train_model.py config/models/yolov8s_random.yaml
uv run python scripts/train_model.py config/models/yolov8s_pretrained.yaml
uv run python scripts/train_model.py config/models/yolo12s_random.yaml
uv run python scripts/train_model.py config/models/yolo12s_pretrained.yaml

# Extended with early stopping
uv run python scripts/train_model.py config/models/yolov8s_random_100ep.yaml
uv run python scripts/train_model.py config/models/yolov8s_pretrained_100ep.yaml
uv run python scripts/train_model.py config/models/yolo12s_random_100ep.yaml
uv run python scripts/train_model.py config/models/yolo12s_pretrained_100ep.yaml
```

```bash
# Report
uv run python scripts/report_results.py --phase small
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
├── .kiro/steering/        # Agent steering docs (project context and workflow)
├── .vscode/               # Recommended extensions and editor settings
├── .cspell.json           # Spell-check dictionary
├── .pre-commit-config.yaml
├── .python-version
├── pyproject.toml         # Project metadata, dependencies, and tool config
├── yolo_cam/              # YOLO-CAM (EigenCAM) library (vendored from rigvedrs/YOLO-26-CAM@f380b07)
├── config/
│   ├── data/              # Dataset configurations
│   └── models/            # Model training configurations (one per experiment)
├── src/
│   ├── config/            # Configuration parsing
│   ├── data/              # Dataset loading and validation
│   ├── training/          # Training and evaluation
│   ├── explainability/    # Explainability methods
│   └── aggregation/       # Results aggregation
├── scripts/               # Training and utility scripts
├── tests/                 # Unit and property-based tests
├── runs/                  # Training outputs (gitignored)
├── outputs/               # Evaluation results (gitignored)
└── explanations/          # Explainability outputs (gitignored)
```

## Configuration

Model configs are in `config/models/`. Each config specifies:

- Model architecture and weight initialization (`.yaml` = random, `.pt` = pretrained)
- Training parameters (epochs per round, patience for early stopping)
- Dataset path
- Multi-run settings (seeds, number of runs)

Naming convention: `{arch}_{init}[_{epochs}].yaml`

- `yolov8n_random.yaml` — YOLOv8 nano, random init, MDPI protocol
- `yolov8n_pretrained.yaml` — YOLOv8 nano, COCO pretrained, MDPI protocol
- `yolov8n_random_100ep.yaml` — YOLOv8 nano, random init, 100 epochs + early stopping
- `yolo12s_pretrained_100ep.yaml` — YOLO12 small, COCO pretrained, 100 epochs + early stopping

Dataset configs are in `config/data/`:

- `weapon_detection_data.yaml` — Relative paths (for version control)
- `weapon_detection_data.local.yaml` — Absolute paths (gitignored)

## Results

Training results are saved to:

- `runs/detect/{model_name}/` — Training logs, plots, checkpoints
- `outputs/{model_name}/` — Evaluation metrics (JSON)
- `explanations/{model_name}/` — Explainability visualizations

## References

- Kutlu, Z.; Emiroğlu, B.G. Image-Based Threat Detection and Explainability Investigation Using
  Incremental Learning and Grad-CAM with YOLOv8. *Computers* **2025**, *14*, 511.
  [doi:10.3390/computers14120511](https://doi.org/10.3390/computers14120511)
