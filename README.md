# Multi-YOLO Evaluation Framework

A modular framework for evaluating multiple YOLO architectures (YOLOv8, YOLOv11, YOLOv12, YOLO26) on weapon detection datasets with explainability analysis.

## Features

- Multi-architecture YOLO training and evaluation
- Automated dataset validation
- Reproducible experiments with seed management
- Comprehensive metrics collection (mAP, precision, recall, inference time, model size)
- CUDA auto-detection with CPU fallback
- Property-based testing for correctness validation

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Dataset

```bash
python scripts/download_dataset.py
```

This downloads the weapon detection dataset from Roboflow (requires API key in `.env`).

### 3. Validate Dataset

```bash
python scripts/validate_dataset.py
```

## Usage

### Quick Test (3 epochs)

```bash
python scripts/train_model.py config/models/yolov8n_test.yaml
```

### Full Training

```bash
python scripts/train_model.py config/models/yolov8n.yaml
python scripts/train_model.py config/models/yolov11n.yaml
python scripts/train_model.py config/models/yolov12n.yaml
python scripts/train_model.py config/models/yolo26n.yaml
```

## Project Structure

```
├── config/
│   ├── data/              # Dataset configurations
│   └── models/            # Model training configurations
├── src/
│   ├── data/              # Dataset loading and validation
│   ├── config/            # Configuration parsing
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
- `weapon_detection_data.yaml` - Relative paths (for version control)
- `weapon_detection_data.local.yaml` - Absolute paths (gitignored)

## Testing

Run all tests:
```bash
python run_tests.py
```

Run with coverage:
```bash
python run_tests.py --coverage
```

Or use pytest directly:
```bash
python -m pytest tests/ -v
```

## Results

Training results are saved to:
- `runs/detect/{model_name}/` - Training logs, plots, checkpoints
- `outputs/{model_name}/` - Evaluation metrics (JSON)
- `explanations/{model_name}/` - Explainability visualizations (future)