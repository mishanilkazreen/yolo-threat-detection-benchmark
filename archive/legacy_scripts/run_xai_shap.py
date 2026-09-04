"""Run SHAP XAI on best incremental vs best one-shot model.

Usage:
    uv run python scripts/run_xai_shap.py
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

from src.explainability.xai.shap import generate_shap_attribution

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

SAMPLE_SIZE = 20
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "xai_comparison"


def _load_test_images(n: int) -> list[str]:
    """Load first n test image paths from local dataset."""
    dataset_root = PROJECT_ROOT / "Weapon_Detection-1"
    images: list[str] = []
    for subdir in ("valid/images", "train/images"):
        d = dataset_root / subdir
        if d.exists():
            images.extend(sorted(str(f) for f in d.glob("*.jpg")))
    step = max(1, len(images) // n)
    return images[::step][:n]


def _get_gt_boxes(image_path: str) -> list[tuple[float, float, float, float]]:
    """Load ground truth boxes for one image."""
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
    return gt_boxes


def _get_validation_images(n: int = 75) -> list[str]:
    """Get validation images for SHAP background set."""
    dataset_root = PROJECT_ROOT / "Weapon_Detection-1"
    valid_dir = dataset_root / "valid" / "images"
    if valid_dir.exists():
        imgs = sorted(str(f) for f in valid_dir.glob("*.jpg"))
        return imgs[:n]
    return []


def run_shap_for_model(
    model: YOLO,
    image_paths: list[str],
    output_subdir: str,
    label: str,
    background_images: list[np.ndarray],
) -> dict:
    """Run SHAP on all images for one model."""
    out_base = OUTPUT_DIR / output_subdir / "shap"
    out_base.mkdir(parents=True, exist_ok=True)

    hfs_scores = []
    for i, img_path in enumerate(image_paths):
        print(f"  [{i + 1:2d}/{len(image_paths)}] {Path(img_path).name}", flush=True)
        gt_boxes = _get_gt_boxes(img_path)

        try:
            _attr, hfs, _path = generate_shap_attribution(
                model=model,
                image_path=img_path,
                detections={},
                gt_boxes=gt_boxes,
                background_set=background_images,
                device="cuda",
                output_dir=str(out_base),
            )
            if hfs is not None:
                hfs_scores.append(hfs)
                print(f"    HFS = {hfs:.4f}")
            else:
                print("    HFS = N/A")
        except Exception as exc:
            print(f"    ! Failed: {exc}")

    mean_hfs = float(np.mean(hfs_scores)) if hfs_scores else 0.0
    print(f"  {label}: mean SHAP HFS = {mean_hfs:.4f} ({len(hfs_scores)} images)")
    return {"label": label, "mean_hfs": mean_hfs, "n": len(hfs_scores)}


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if not BEST_INCREMENTAL.exists():
        print(f"ERROR: {BEST_INCREMENTAL} not found", file=sys.stderr)
        return 1
    if not BEST_BASELINE.exists():
        print(f"ERROR: {BEST_BASELINE} not found", file=sys.stderr)
        return 1

    image_paths = _load_test_images(SAMPLE_SIZE)
    if not image_paths:
        print("ERROR: No test images found", file=sys.stderr)
        return 1
    print(f"Processing {len(image_paths)} test images with SHAP")

    # Create background set from validation images
    print("Loading SHAP background set (75 validation images)...")
    from PIL import Image

    val_paths = _get_validation_images(75)
    background_images = []
    for vp in val_paths:
        try:
            img = Image.open(vp).convert("RGB")
            background_images.append(np.array(img))
        except Exception:
            pass
    print(f"  Background set: {len(background_images)} images\n")

    # --- Best incremental model ---
    print("=== Best Incremental: YOLO11n-P Round 5 (SHAP) ===")
    model_inc = YOLO(str(BEST_INCREMENTAL))
    model_inc.to(device)
    inc_result = run_shap_for_model(
        model_inc,
        image_paths,
        "incremental_yolo11n_pretrained_r5",
        "Incremental R5",
        background_images,
    )
    del model_inc
    torch.cuda.empty_cache()

    # --- Best one-shot baseline ---
    print("\n=== Best One-Shot: YOLO11n-P 500ep Baseline (SHAP) ===")
    model_base = YOLO(str(BEST_BASELINE))
    model_base.to(device)
    base_result = run_shap_for_model(
        model_base,
        image_paths,
        "baseline_yolo11n_pretrained_500ep",
        "Baseline 500ep",
        background_images,
    )
    del model_base
    torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 60)
    print("SHAP XAI Comparison Summary")
    print("=" * 60)
    print(f"  Incremental (YOLO11n-P R5):  mean SHAP HFS = {inc_result['mean_hfs']:.4f}")
    print(f"  Baseline (YOLO11n-P 500ep):  mean SHAP HFS = {base_result['mean_hfs']:.4f}")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
