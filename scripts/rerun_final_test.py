"""Re-run final test evaluation using best-round checkpoints with fixed test set.

This script applies the corrected evaluation logic:
1. Uses val_fixed (2,026 images) as the test set instead of training images
2. Uses the round-2 checkpoint (best val mAP50) instead of the last round's

Run from the project root:
    python scripts/rerun_final_test.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.evaluator import Metrics_Collector


MODELS = {
    "yolov8n": {
        "best_round": 2,
        "rounds_dir": "runs/yolov8n",
        "output_dir": "outputs/yolov8n",
        "total_training_time": 2355.78,
    },
    "yolo11n": {
        "best_round": 2,
        "rounds_dir": "runs/yolo11n",
        "output_dir": "outputs/yolo11n",
        "total_training_time": 2218.52,
    },
    "yolo12n": {
        "best_round": 2,
        "rounds_dir": "runs/yolo12n",
        "output_dir": "outputs/yolo12n",
        "total_training_time": 2417.80,
    },
}

SEED = 42


def rebuild_data_yaml_with_val_as_test(model_name: str, output_dir: str) -> str:
    """Create a data yaml where the test split points to val_fixed images."""
    import yaml

    base = Path("Weapon_Detection-1")
    val_image_dir = base / "valid" / "images"

    if not val_image_dir.exists():
        raise FileNotFoundError(f"Validation image dir not found: {val_image_dir}")

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    val_images = []
    for ext in image_extensions:
        val_images.extend([p.name for p in val_image_dir.glob(f"*{ext}")])
        val_images.extend([p.name for p in val_image_dir.glob(f"*{ext.upper()}")])
    val_images = sorted(set(val_images))

    out_dir = Path(output_dir) / "round_splits" / "reeval_fixed"
    out_dir.mkdir(parents=True, exist_ok=True)

    val_list = out_dir / "val.txt"
    test_list = out_dir / "test.txt"

    full_paths = [str((val_image_dir / img).absolute()) for img in val_images]
    val_list.write_text("\n".join(full_paths))
    test_list.write_text("\n".join(full_paths))

    data_yaml_path = out_dir / "data.yaml"
    data_config = {
        "path": str(base),
        "train": str(val_list.absolute()),
        "val": str(val_list.absolute()),
        "test": str(test_list.absolute()),
        "nc": 2,
        "names": ["knife", "pistol"],
        "fraction": 1.0,
    }
    with open(data_yaml_path, "w") as f:
        yaml.dump(data_config, f, default_flow_style=False)

    print(f"  Created data yaml: {data_yaml_path}")
    print(f"  Test/val set size: {len(val_images)} images")
    return str(data_yaml_path)


def main():
    collector = Metrics_Collector()

    for model_name, cfg in MODELS.items():
        print(f"\n{'=' * 60}")
        print(f"Re-evaluating {model_name} (Round {cfg['best_round']} checkpoint)")
        print(f"{'=' * 60}")

        checkpoint = (
            Path(cfg["rounds_dir"])
            / f"incremental_round_{cfg['best_round']}"
            / "weights"
            / "best.pt"
        )
        if not checkpoint.exists():
            print(f"  SKIP: checkpoint not found at {checkpoint}")
            continue

        data_yaml = rebuild_data_yaml_with_val_as_test(model_name, cfg["output_dir"])

        metrics = collector.evaluate_final_test(
            checkpoint_path=str(checkpoint),
            data_yaml=data_yaml,
            output_dir=cfg["output_dir"],
            config_name=model_name,
            random_seed=SEED,
            total_training_time=cfg["total_training_time"],
        )

        m = metrics["metrics"]
        pc = metrics.get("per_class_metrics", {})
        print(f"\n  Results ({model_name} round-2 checkpoint on val_fixed):")
        print(f"    mAP50:     {m['mAP50']:.4f}")
        print(f"    mAP50-95:  {m['mAP50-95']:.4f}")
        print(f"    Precision: {m['precision']:.4f}")
        print(f"    Recall:    {m['recall']:.4f}")
        print(f"    F1:        {m['f1_score']:.4f}")
        print(f"    Per-class mAP50: {pc.get('mAP50_per_class', [])}")
        print(f"    Per-class count: {len(pc.get('mAP50_per_class', []))}")

    print("\nDone. Updated final_test_metrics.json files written to outputs/*/")


if __name__ == "__main__":
    main()
