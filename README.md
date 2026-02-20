# Asbtract

>The rapid proliferation of harmful visual content has introduced novel challenges for content moderation on online platforms such as Discord and Reddit, where hateful or harmful imagery can be disseminated at scale. Existing threat detection approaches for flagging such material have predominantly relied on YOLOv8 with limited explainability, leaving moderators without actionable insight into model decisions. This study addresses two open gaps in the literature: (1) the lack of comparative evaluation of post-v8 YOLO architectures for hateful content detection, and (2) the insufficient integration of advanced explainable AI (XAI) techniques beyond coarse Grad-CAM heatmaps. We benchmark YOLOv11, YOLOv12, and YOLO26 on a multiclass harmful-content dataset and systematically apply complementary XAI methods, including Layer-wise Relevance Propagation (LRP) and SHAP, alongside Grad-CAM to provide fine-grained, instance-level attribution maps. Our experimental pipeline evaluates detection accuracy (mAP@0.5, precision, recall, F1-score), inference latency, and explainability fidelity across all architectures. Preliminary results indicate that attention-enhanced backbones in YOLOv11 and YOLOv12 improve localisation of subtle hateful symbols, while YOLO26's NMS-free prediction head reduces post-processing overhead. The hybrid XAI framework yields richer explanations that enable moderators to understand why specific image regions are flagged, supporting transparent and accountable automated moderation. This work provides a practical, deployable framework for platform trust and safety teams seeking accurate, interpretable, and efficient AI-driven content moderation.

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