"""Run Grad-CAM XAI comparison on best incremental vs best one-shot model.

Produces side-by-side heatmaps for a sample of test images, comparing:
  - Best incremental: YOLO11n pretrained, Round 5 (mAP@0.5 = 0.900)
  - Best one-shot:    YOLO11n pretrained 500ep baseline (mAP@0.5 = 0.924)

Usage:
    uv run python scripts/run_xai_comparison.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "yolo_cam"))

from src.explainability.xai.gradcam import generate_gradcam_attribution
from src.explainability.xai.output_manager import XAIOutputManager

# --- Configuration ---
BEST_INCREMENTAL = (
    PROJECT_ROOT / "runs" / "yolo11n_pretrained" / "incremental_round_5" / "weights" / "best.pt"
)
BEST_BASELINE = (
    PROJECT_ROOT
    / "runs"
    / "yolo11n_baseline_pretrained_100ep"
    / "yolo11n_baseline_pretrained_100ep_baseline"
    / "weights"
    / "best.pt"
)

# Use test set images
TEST_LIST = PROJECT_ROOT / "config" / "data" / "test_fixed.txt"
SAMPLE_SIZE = 20  # number of test images to process
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "xai_comparison"
TARGET_LAYER = "model.22"  # YOLO11n neck P5


def _load_test_images(n: int) -> list[str]:
    """Load first n test image paths from local dataset."""
    dataset_root = PROJECT_ROOT / "Weapon_Detection-1"
    # Gather all images from both train and valid dirs
    images = []
    for subdir in ("valid/images", "train/images"):
        d = dataset_root / subdir
        if d.exists():
            images.extend(sorted(str(f) for f in d.glob("*.jpg")))
    # Use a deterministic subset (every Nth image to get variety)
    step = max(1, len(images) // n)
    return images[::step][:n]


def _get_detections_and_gt(
    model: YOLO, image_path: str
) -> tuple[dict, list[tuple[float, float, float, float]]]:
    """Run inference and load ground truth for one image."""
    # Run inference
    results = model.predict(image_path, imgsz=640, device="cuda", verbose=False, conf=0.25)
    r = results[0]

    boxes_xyxy = r.boxes.xyxy.cpu().numpy().tolist() if len(r.boxes) > 0 else []
    classes = r.boxes.cls.cpu().numpy().tolist() if len(r.boxes) > 0 else []
    scores = r.boxes.conf.cpu().numpy().tolist() if len(r.boxes) > 0 else []

    detections = {"boxes": boxes_xyxy, "classes": classes, "scores": scores}

    # Load ground truth from label file
    label_path = Path(
        str(image_path).replace("\\images\\", "\\labels\\").replace("/images/", "/labels/")
    ).with_suffix(".txt")

    gt_boxes = []
    if label_path.exists():
        for line in label_path.read_text(encoding="utf-8").strip().splitlines():
            parts = line.strip().split()
            if len(parts) >= 5:
                gt_boxes.append(
                    (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
                )

    return detections, gt_boxes


def run_xai_for_model(model: YOLO, image_paths: list[str], output_subdir: str, label: str) -> dict:
    """Run Grad-CAM on all images for one model."""
    out_dir = OUTPUT_DIR / output_subdir / "gradcam"
    out_dir.mkdir(parents=True, exist_ok=True)

    output_manager = XAIOutputManager(
        base_output_dir=str(OUTPUT_DIR / output_subdir),
        save_overlays=True,
        save_raw=False,
    )

    hfs_scores = []
    for i, img_path in enumerate(image_paths):
        print(f"  [{i + 1:2d}/{len(image_paths)}] {Path(img_path).name}", flush=True)
        detections, gt_boxes = _get_detections_and_gt(model, img_path)

        try:
            _cam, hfs, _path = generate_gradcam_attribution(
                model=model,
                image_path=img_path,
                detections=detections,
                gt_boxes=gt_boxes,
                target_layer=TARGET_LAYER,
                _device="cuda",
                output_manager=output_manager,
                target_class_names=["knife", "pistol"],
            )
            if hfs is not None:
                hfs_scores.append(hfs)
        except Exception as exc:
            print(f"    ! Failed: {exc}")

    mean_hfs = float(np.mean(hfs_scores)) if hfs_scores else 0.0
    print(f"  {label}: mean HFS = {mean_hfs:.4f} ({len(hfs_scores)} images)")
    return {"label": label, "mean_hfs": mean_hfs, "n": len(hfs_scores)}


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Verify checkpoints exist
    if not BEST_INCREMENTAL.exists():
        print(f"ERROR: {BEST_INCREMENTAL} not found", file=sys.stderr)
        return 1
    if not BEST_BASELINE.exists():
        print(f"ERROR: {BEST_BASELINE} not found", file=sys.stderr)
        return 1

    # Load test images
    image_paths = _load_test_images(SAMPLE_SIZE)
    if not image_paths:
        print("ERROR: No test images found", file=sys.stderr)
        return 1
    print(f"Processing {len(image_paths)} test images\n")

    # --- Best incremental model ---
    print("=== Best Incremental: YOLO11n-P Round 5 ===")
    model_inc = YOLO(str(BEST_INCREMENTAL))
    model_inc.to(device)
    inc_result = run_xai_for_model(
        model_inc, image_paths, "incremental_yolo11n_pretrained_r5", "Incremental R5"
    )
    del model_inc
    torch.cuda.empty_cache()

    # --- Best one-shot baseline ---
    print("\n=== Best One-Shot: YOLO11n-P 500ep Baseline ===")
    model_base = YOLO(str(BEST_BASELINE))
    model_base.to(device)
    base_result = run_xai_for_model(
        model_base, image_paths, "baseline_yolo11n_pretrained_500ep", "Baseline 500ep"
    )
    del model_base
    torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 60)
    print("XAI Comparison Summary")
    print("=" * 60)
    print(f"  Incremental (YOLO11n-P R5):  mean HFS = {inc_result['mean_hfs']:.4f}")
    print(f"  Baseline (YOLO11n-P 500ep):  mean HFS = {base_result['mean_hfs']:.4f}")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
