# Benchmarking YOLO Generations for Explainable Incremental Threat Detection: Weight Initialisation, Architecture, and Training Budget

## Abstract

Incremental learning frameworks for real-time object detection have attracted growing interest in security-critical
applications, yet benchmarks comparing You Only Look Once (YOLO) generations under controlled initialisation conditions
remain scarce.
Kutlu and Emiroğlu (2025) report mAP@0.5 of 0.886 for YOLOv8-nano under reported random initialisation on a
two-class (pistol and knife) weapon dataset; a direct replication yields only 0.54, a result more consistent with
COCO-pretrained initialisation than with the random protocol reported by the authors.
We evaluate YOLOv8, YOLOv11, and YOLOv12 nano variants under both random and COCO-pretrained initialisation with
two epoch budgets (10 and 100 per round) across five data-incremental rounds, yielding twelve core conditions.
For each condition we report mAP@0.5, mAP@0.5:0.95, F1-score, precision, recall, and per-class mAP@0.5.
Weight initialisation emerges as the dominant factor: pretrained architectures reach Round-5 mAP@0.5 above 0.89 at
10 epochs per round; random initialisation peaks at 0.534 under the same protocol.
Extended training reduces but does not eliminate this gap, and freeze-strategy ablations confirm that backbone-only
feature transfer reproduces the reference trajectory to within 0.002 mAP@0.5.
Grad-CAM, LRP, and SHAP analyses assess how initialisation influences interpretability across architectures and rounds.
All configurations, scripts, and metrics are made publicly available.

---

## Experiment matrix

All experiments use **nano-only** variants (YOLOv8n, YOLOv11n, YOLOv12n).

### Phase 0 — One-shot baselines (reference)

Single-pass training, no incremental rounds. These serve as reference points for all incremental experiments
(metrics and wall-clock time).

| Config | Arch | Init | Epochs | Early stop |
|---|---|---|---|---|
| `yolov8n_baseline_pretrained.yaml` | v8n | COCO | 50 | no |
| `yolov8n_baseline_random.yaml` | v8n | random | 50 | no |
| `yolov8n_baseline_pretrained_100ep.yaml` | v8n | COCO | 100 | patience=10 |
| `yolov8n_baseline_random_100ep.yaml` | v8n | random | 100 | patience=10 |
| `yolo11n_baseline_pretrained.yaml` | v11n | COCO | 50 | no |
| `yolo11n_baseline_random.yaml` | v11n | random | 50 | no |
| `yolo11n_baseline_pretrained_100ep.yaml` | v11n | COCO | 100 | patience=10 |
| `yolo11n_baseline_random_100ep.yaml` | v11n | random | 100 | patience=10 |
| `yolo12n_baseline_*.yaml` | v12n | random / COCO | 50 / 100 | no / patience=10 |

### Phase 1 — Nano models (incremental protocol)

| Config | Arch | Init | Epochs/round | Early stop |
|---|---|---|---|---|
| `yolov8n_random.yaml` | v8n | random | 10 | no |
| `yolov8n_pretrained.yaml` | v8n | COCO | 10 | no |
| `yolo11n_random.yaml` | v11n | random | 10 | no |
| `yolo11n_pretrained.yaml` | v11n | COCO | 10 | no |
| `yolo12n_random.yaml` | v12n | random | 10 | no |
| `yolo12n_pretrained.yaml` | v12n | COCO | 10 | no |
| `yolov8n_random_100ep.yaml` | v8n | random | 100 | patience=10 |
| `yolov8n_pretrained_100ep.yaml` | v8n | COCO | 100 | patience=10 |
| `yolo11n_random_100ep.yaml` | v11n | random | 100 | patience=10 |
| `yolo11n_pretrained_100ep.yaml` | v11n | COCO | 100 | patience=10 |
| `yolo12n_random_100ep.yaml` | v12n | random | 100 | patience=10 |
| `yolo12n_pretrained_100ep.yaml` | v12n | COCO | 100 | patience=10 |

### Metrics reported (per experiment)

- **Overall:** F1-score (at optimal confidence), Precision (at conf=1.0), mAP@0.5
- **Per-class:** F1-score, Precision, mAP@0.5 for knife and pistol
- **Per-round** (incremental configs only): all of the above for each of the 5 rounds
- **Early stopping runs:** actual epoch training stopped on
- **Wall-clock time:** training and validation seconds per round / per baseline run, so
  comparisons against the one-shot baseline are fair on compute as well as accuracy

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

#### Phase 0 — One-shot baselines (run first, anchors all other comparisons)

```bash
# 50-epoch baselines
uv run python scripts/train_baseline.py config/models/yolov8n_baseline_pretrained.yaml
uv run python scripts/train_baseline.py config/models/yolov8n_baseline_random.yaml
uv run python scripts/train_baseline.py config/models/yolo11n_baseline_pretrained.yaml
uv run python scripts/train_baseline.py config/models/yolo11n_baseline_random.yaml
uv run python scripts/train_baseline.py config/models/yolo12n_baseline_pretrained.yaml
uv run python scripts/train_baseline.py config/models/yolo12n_baseline_random.yaml

# 100-epoch + early stopping baselines
uv run python scripts/train_baseline.py config/models/yolov8n_baseline_pretrained_100ep.yaml
uv run python scripts/train_baseline.py config/models/yolov8n_baseline_random_100ep.yaml
uv run python scripts/train_baseline.py config/models/yolo11n_baseline_pretrained_100ep.yaml
uv run python scripts/train_baseline.py config/models/yolo11n_baseline_random_100ep.yaml
uv run python scripts/train_baseline.py config/models/yolo12n_baseline_pretrained_100ep.yaml
uv run python scripts/train_baseline.py config/models/yolo12n_baseline_random_100ep.yaml
```

Then report:

```bash
uv run python scripts/report_results.py --phase baseline
```

Baseline rows are reserved in `outputs/full_evaluation_results.csv` with empty metric
columns; the runners and `report_results.py` populate them on completion.

#### Phase 1 — Nano models (incremental)

```bash
# Standard protocol — 10 epochs/round, no early stopping
uv run python scripts/train_model.py config/models/yolov8n_random.yaml
uv run python scripts/train_model.py config/models/yolov8n_pretrained.yaml
uv run python scripts/train_model.py config/models/yolo11n_random.yaml
uv run python scripts/train_model.py config/models/yolo11n_pretrained.yaml
uv run python scripts/train_model.py config/models/yolo12n_random.yaml
uv run python scripts/train_model.py config/models/yolo12n_pretrained.yaml

# Extended — 100 epochs/round, early stopping patience=10
uv run python scripts/train_model.py config/models/yolov8n_random_100ep.yaml
uv run python scripts/train_model.py config/models/yolov8n_pretrained_100ep.yaml
uv run python scripts/train_model.py config/models/yolo11n_random_100ep.yaml
uv run python scripts/train_model.py config/models/yolo11n_pretrained_100ep.yaml
uv run python scripts/train_model.py config/models/yolo12n_random_100ep.yaml
uv run python scripts/train_model.py config/models/yolo12n_pretrained_100ep.yaml
```

After each run (or all of them), generate the results table:

```bash
uv run python scripts/report_results.py --phase nano
```

```bash
# Report all phases (baselines + nano incremental)
uv run python scripts/report_results.py
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

- `yolov8n_random.yaml` — YOLOv8 nano, random init, JRTIP incremental protocol
- `yolov8n_pretrained.yaml` — YOLOv8 nano, COCO pretrained, JRTIP incremental protocol
- `yolov8n_random_100ep.yaml` — YOLOv8 nano, random init, 100 epochs + early stopping
- `yolo11n_random.yaml` — YOLOv11 nano, random init, standard incremental protocol
- `yolo11n_pretrained.yaml` — YOLOv11 nano, COCO pretrained, standard incremental protocol
- `yolo11n_random_100ep.yaml` — YOLOv11 nano, random init, 100 epochs + early stopping
- `yolo11n_pretrained_100ep.yaml` — YOLOv11 nano, COCO pretrained, 100 epochs + early stopping
- `yolo12n_pretrained_100ep.yaml` — YOLO12 nano, COCO pretrained, 100 epochs + early stopping

Dataset configs are in `config/data/`:

- `weapon_detection_data.yaml` — Relative paths (for version control)
- `weapon_detection_data.local.yaml` — Absolute paths (gitignored)

## Results

Training results are saved to:

- `runs/detect/{model_name}/` — Training logs, plots, checkpoints
- `outputs/{model_name}/` — Evaluation metrics (JSON)
- `explanations/{model_name}/` — Explainability visualizations

## Citation

*This paper is currently under review. Citation details will be updated upon publication.*

If you use this work, please contact the corresponding author at `mani.ghahremani@port.ac.uk` for citation details
until the paper is published.
