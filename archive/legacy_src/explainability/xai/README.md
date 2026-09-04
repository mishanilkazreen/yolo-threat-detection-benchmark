# XAI (Explainable AI) Module

This module provides explainable AI capabilities for YOLO object detection models.
It implements three complementary XAI methods as modular, independent components.

## Architecture

The XAI module follows a modular architecture where each method can be used independently:

```text
src/explainability/xai/
├── __init__.py          # Module exports
├── interfaces.py        # Core data models and interfaces
├── config.py           # Configuration classes and architecture mappings
├── manager.py          # XAI orchestration and batch processing
├── gradcam.py          # Grad-CAM implementation
├── lrp.py              # Layer-wise Relevance Propagation
├── shap.py             # SHAP implementation
└── README.md           # This file
```

## Supported XAI Methods

### 1. Grad-CAM (Gradient-weighted Class Activation Mapping)

- **Fast and efficient** - Good for initial analysis
- Generates heatmaps showing important regions for model predictions
- Supports HFS (Heatmap Focus Score) computation
- Works with all YOLO architectures

### 2. LRP (Layer-wise Relevance Propagation)

- **Computationally expensive** - Disabled by default
- Provides pixel-level relevance scores
- Supports bbox-weighted relevance computation
- Uses zennit library (preferred) or captum as fallback

### 3. SHAP (SHapley Additive exPlanations)

- **Very computationally expensive** - Disabled by default
- Provides theoretically grounded attributions
- Requires background set of validation images
- Does not support HFS-like computation

## Configuration

XAI functionality is configured through YAML files:

```yaml
xai:
  enabled: true
  methods:
    gradcam: true    # Fast, good for initial analysis
    lrp: false       # Expensive, disabled by default
    shap: false      # Very expensive, disabled by default
  sample_limit: 5    # Limit processing for performance
  target_layers:     # Architecture-specific layers
    yolov11: "model.22"
    yolov12: "model.22"
    yolo26: "model.21"
  background_set_size: 75
  output_overlays: true
```

## Architecture Support

The module supports multiple YOLO architectures with configurable target layers:

- **YOLOv11**: `model.22` (final conv layer before head)
- **YOLOv12**: `model.22`
- **YOLO26**: `model.21` (NMS-free head has different structure)
- **YOLOv8**: `model.22` (backward compatibility)

## Usage

### Basic Usage

```python
from src.explainability.xai import XAIManager, XAIConfig

# Configure XAI
config = XAIConfig(
    enabled=True,
    methods={"gradcam": True, "lrp": False, "shap": False},
    sample_limit=10
)

# Create manager
manager = XAIManager(config)

# Process batch of images
results = manager.process_batch(
    model=yolo_model,
    image_paths=["img1.jpg", "img2.jpg"],
    detections=detection_results,
    gt_boxes=ground_truth_boxes,
    round_num=1,
    output_dir="outputs/"
)
```

### Individual Method Usage

```python
from src.explainability.xai import generate_gradcam_attribution

# Use Grad-CAM independently
attribution_map, hfs_score, output_path = generate_gradcam_attribution(
    model=yolo_model,
    image_path="test.jpg",
    detections=detections,
    gt_boxes=[(0.5, 0.5, 0.4, 0.6)],
    target_layer="model.22"
)
```

## Performance Considerations

- **Sample Limiting**: Use `sample_limit` to process only first N images per round
- **Method Selection**: Enable only necessary methods (Grad-CAM is fastest)
- **Zero Overhead**: When `enabled: false`, no performance impact on detection pipeline
- **Background Sets**: SHAP requires pre-computed background set of validation images

## Output Structure

XAI results are saved in method-specific directories:

```text
outputs/
└── xai/
    ├── gradcam/
    │   ├── image1_gradcam.png
    │   └── image2_gradcam.png
    ├── lrp/
    │   └── image1_lrp.png
    └── shap/
        └── image1_shap.png
```

## Requirements

- **Core**: torch, numpy, opencv-python, PIL
- **LRP**: zennit (preferred) or captum
- **SHAP**: shap
- **HFS**: Uses existing `src.explainability.hfs_scorer`

## Integration

The XAI module integrates with the existing YOLO pipeline through:

1. **Configuration Extension**: XAI settings in model YAML files
2. **Post-Inference Processing**: Called after model inference, before results written
3. **Metrics Collection**: HFS scores aggregated with existing metrics
