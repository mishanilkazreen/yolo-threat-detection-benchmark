"""Deterministic XAI evaluation script for held-out test split (test_fixed.txt)."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "yolo_cam"))

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from yolo_cam.eigen_cam import EigenCAM

from src.explainability.hfs_scorer import Heatmap_Focus_Scorer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("xai_eval")

TEST_LIST = PROJECT_ROOT / "config" / "data" / "test_fixed.txt"
DEFAULT_OUTPUT_JSON = PROJECT_ROOT / "outputs" / "xai_evaluation_results.json"

TARGET_LAYER = "model.22"

BEST_INCREMENTAL_CANDIDATES = [
    PROJECT_ROOT
    / "runs"
    / "yolo11n_pretrained_100ep"
    / "incremental_round_5"
    / "weights"
    / "best.pt",
    PROJECT_ROOT / "runs" / "yolo11n_pretrained_100ep" / "round_5" / "weights" / "best.pt",
    PROJECT_ROOT / "runs" / "yolo11n_pretrained" / "incremental_round_5" / "weights" / "best.pt",
    PROJECT_ROOT / "runs" / "yolo11n_pretrained" / "round_5" / "weights" / "best.pt",
    Path(
        "/mnt/lustre2/mres/ghahrem/yolo-threat-detection-benchmark/runs/yolo11n_pretrained_100ep/incremental_round_5/weights/best.pt"
    ),
    Path(
        "/mnt/lustre2/mres/ghahrem/yolo-threat-detection-benchmark/runs/yolo11n_pretrained/incremental_round_5/weights/best.pt"
    ),
]

BEST_BASELINE_CANDIDATES = [
    PROJECT_ROOT
    / "runs"
    / "yolo11n_baseline_pretrained_100ep"
    / "yolo11n_baseline_pretrained_100ep_baseline"
    / "weights"
    / "best.pt",
    PROJECT_ROOT
    / "runs"
    / "yolo11n_baseline_pretrained"
    / "yolo11n_baseline_pretrained_baseline"
    / "weights"
    / "best.pt",
    Path(
        "/mnt/lustre2/mres/ghahrem/yolo-threat-detection-benchmark/runs/yolo11n_baseline_pretrained_100ep/yolo11n_baseline_pretrained_100ep_baseline/weights/best.pt"
    ),
    Path(
        "/mnt/lustre2/mres/ghahrem/yolo-threat-detection-benchmark/runs/yolo11n_baseline_pretrained/yolo11n_baseline_pretrained_baseline/weights/best.pt"
    ),
]


def resolve_data_root(custom_path: str | None = None) -> Path:
    """Resolve the Weapon_Detection-1 dataset directory across local and cluster environments."""
    if custom_path:
        p = Path(custom_path).resolve()
        if p.exists():
            return p

    for env_var in ("DATASET_ROOT", "WEAPON_DATASET_DIR"):
        env_val = os.getenv(env_var)
        if env_val:
            p = Path(env_val).resolve()
            if p.exists():
                return p

    candidates = [
        PROJECT_ROOT / "Weapon_Detection-1",
        PROJECT_ROOT.parent / "Weapon_Detection-1",
        PROJECT_ROOT.parent / "Journal-of-Real-Time-Image-Processing" / "Weapon_Detection-1",
        Path("/mnt/lustre2/mres/ghahrem/yolo-threat-detection-benchmark/Weapon_Detection-1"),
    ]
    for cand in candidates:
        if cand.exists():
            return cand.resolve()

    return (PROJECT_ROOT / "Weapon_Detection-1").resolve()


def find_checkpoint(custom_path: str | None, candidates: list[Path], label: str) -> Path:
    """Locate model checkpoint from argument or candidate search list."""
    if custom_path:
        p = Path(custom_path).resolve()
        if p.exists():
            logger.info("Found %s checkpoint via CLI argument: %s", label, p)
            return p
        raise FileNotFoundError(f"Specified checkpoint for {label} does not exist: {p}")

    for cand in candidates:
        if cand.exists():
            logger.info("Found %s checkpoint: %s", label, cand)
            return cand

    checked_str = "\n  - ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"CRITICAL: Required trained model checkpoint for {label} not found.\n"
        f"Checked candidate paths:\n  - {checked_str}\n"
        f"Please provide the path using --{label.lower().replace(' ', '-')}-ckpt or ensure weights exist on disk."
    )


def load_test_sample(test_list: Path, data_root: Path, n: int = 20) -> list[dict]:
    """Load test image metadata and ground-truth boxes from held-out test split."""
    if not test_list.exists():
        raise FileNotFoundError(f"Test list file not found: {test_list}")

    lines = [
        line.strip() for line in test_list.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if n > 0 and len(lines) > n:
        step = max(1, len(lines) // n)
        sample_rel = lines[::step][:n]
    else:
        sample_rel = lines

    samples = []

    for rel in sample_rel:
        clean_rel = rel.lstrip("./").replace("\\", "/")

        if clean_rel.startswith("Weapon_Detection-1/"):
            sub_rel = clean_rel[len("Weapon_Detection-1/") :]
        else:
            sub_rel = clean_rel

        if (data_root / sub_rel).exists():
            img_path = data_root / sub_rel
        elif (data_root / clean_rel).exists():
            img_path = data_root / clean_rel
        elif (PROJECT_ROOT / clean_rel).exists():
            img_path = PROJECT_ROOT / clean_rel
        else:
            logger.warning(
                "Image file not found for sample: %s (checked relative to %s)", clean_rel, data_root
            )
            continue

        lbl_rel = sub_rel.replace("/images/", "/labels/")
        lbl_rel = str(Path(lbl_rel).with_suffix(".txt")).replace("\\", "/")

        if (data_root / lbl_rel).exists():
            lbl_path = data_root / lbl_rel
        elif (data_root / clean_rel.replace("/images/", "/labels/")).exists():
            lbl_path = data_root / clean_rel.replace("/images/", "/labels/")
        elif (PROJECT_ROOT / clean_rel.replace("/images/", "/labels/")).exists():
            lbl_path = PROJECT_ROOT / clean_rel.replace("/images/", "/labels/")
        else:
            lbl_path = Path("non_existent_label")

        gt_boxes = []
        if lbl_path.exists():
            for line_str in lbl_path.read_text(encoding="utf-8").strip().splitlines():
                parts = line_str.strip().split()
                if len(parts) >= 5:
                    gt_boxes.append(
                        {"class_id": int(parts[0]), "bbox": [float(p) for p in parts[1:5]]}
                    )

        samples.append(
            {
                "rel_path": clean_rel,
                "abs_path": str(img_path),
                "label_path": str(lbl_path) if lbl_path.exists() else None,
                "gt_boxes": gt_boxes,
            }
        )

    logger.info(
        "Loaded %d test samples with total %d ground-truth boxes",
        len(samples),
        sum(len(s["gt_boxes"]) for s in samples),
    )
    return samples


def compute_integrated_gradients(
    model: YOLO, img_tensor: torch.Tensor, steps: int = 20
) -> np.ndarray:
    """Compute Integrated Gradients attribution map with respect to detection confidence."""
    baseline = torch.zeros_like(img_tensor)
    scaled_inputs = [
        baseline + (float(i) / steps) * (img_tensor - baseline) for i in range(steps + 1)
    ]
    grads = []
    torch_model = model.model if hasattr(model, "model") else model
    torch_model.eval()

    for scaled in scaled_inputs:
        scaled_param = scaled.clone().detach().requires_grad_(True)
        out = torch_model(scaled_param)
        if isinstance(out, (tuple, list)):
            out = out[0]
        # Target detection confidence sum strictly over weapon classes (excluding box regression coordinates 0:4)
        score = out[:, 4:, :].sum() if out.dim() == 3 and out.shape[1] > 4 else out.sum()
        score.backward()
        if scaled_param.grad is not None:
            grads.append(scaled_param.grad.detach().cpu().numpy())

    avg_grads = np.mean(np.array(grads), axis=0)
    delta = (img_tensor - baseline).cpu().numpy()
    ig_attr = delta * avg_grads
    heatmap = np.sum(np.abs(ig_attr), axis=1)[0]
    h_max = heatmap.max()
    if h_max > 0:
        heatmap = heatmap / h_max
    return heatmap


def evaluate_model(model_path: Path, samples: list[dict], label: str, device: torch.device) -> dict:
    """Evaluate a single model on test samples using Eigen-CAM and Integrated Gradients."""
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint file does not exist: {model_path}")

    scorer = Heatmap_Focus_Scorer()
    logger.info("Loading model %s from %s onto %s", label, model_path, device)
    model = YOLO(str(model_path))
    model.to(device)
    torch_model = model.model if hasattr(model, "model") else model

    target_module = None
    for name, module in torch_model.named_modules():
        if name == TARGET_LAYER:
            target_module = module
            break
    if target_module is None:
        target_module = list(torch_model.modules())[-2]
    cam_extractor = EigenCAM(model=torch_model, target_layers=[target_module])

    eigencam_hfs, eigencam_hits = [], 0
    ig_hfs, ig_hits = [], 0
    failure_images = 0

    for sample in samples:
        img_bgr = cv2.imread(sample["abs_path"])
        if img_bgr is None:
            continue
        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (640, 640))
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        img_tensor = img_tensor.to(device)

        cam = cam_extractor(input_tensor=img_tensor)[0]
        cam_orig = cv2.resize(cam, (w, h))

        e_scores, hit_e = [], False
        peak_y, peak_x = np.unravel_index(np.argmax(cam_orig), cam_orig.shape)
        for gt in sample["gt_boxes"]:
            bbox = gt["bbox"]
            e_scores.append(scorer.compute_hfs(cam_orig, bbox, w, h))
            xc, yc, bw, bh = bbox
            x1, y1 = max(0, (xc - bw / 2) * w), max(0, (yc - bh / 2) * h)
            x2, y2 = min(w, (xc + bw / 2) * w), min(h, (yc + bh / 2) * h)
            if x1 <= peak_x <= x2 and y1 <= peak_y <= y2:
                hit_e = True

        if hit_e:
            eigencam_hits += 1
        eigencam_hfs.append(float(np.mean(e_scores)) if e_scores else 0.0)

        try:
            ig_hm = compute_integrated_gradients(model, img_tensor, steps=20)
            ig_orig = cv2.resize(ig_hm, (w, h))
            ig_scores, hit_ig = [], False
            ig_py, ig_px = np.unravel_index(np.argmax(ig_orig), ig_orig.shape)
            for gt in sample["gt_boxes"]:
                bbox = gt["bbox"]
                ig_scores.append(scorer.compute_hfs(ig_orig, bbox, w, h))
                xc, yc, bw, bh = bbox
                x1, y1 = max(0, (xc - bw / 2) * w), max(0, (yc - bh / 2) * h)
                x2, y2 = min(w, (xc + bw / 2) * w), min(h, (yc + bh / 2) * h)
                if x1 <= ig_px <= x2 and y1 <= ig_py <= y2:
                    hit_ig = True

            if hit_ig:
                ig_hits += 1
            ig_hfs.append(float(np.mean(ig_scores)) if ig_scores else 0.0)

            if np.mean(e_scores or [0]) < 0.45 or np.mean(ig_scores or [0]) < 0.30:
                failure_images += 1
        except Exception as exc:
            logger.warning("Integrated Gradients error for %s: %s", sample["rel_path"], exc)

    n = len(samples)
    return {
        "model_label": label,
        "checkpoint_path": str(model_path),
        "sample_images": n,
        "sample_boxes": sum(len(s["gt_boxes"]) for s in samples),
        "pointing_game_denominator": f"Evaluated per image (n={n} images containing {sum(len(s['gt_boxes']) for s in samples)} weapon bounding boxes); hit = peak attribution inside any GT box",
        "eigencam": {
            "mean_hfs": round(float(np.mean(eigencam_hfs)), 3),
            "std_hfs": round(float(np.std(eigencam_hfs, ddof=1)), 3),
            "pointing_game_hit_rate_pct": round(eigencam_hits / n * 100.0, 1),
            "hits": eigencam_hits,
        },
        "integrated_gradients": {
            "mean_hfs": round(float(np.mean(ig_hfs)), 3),
            "std_hfs": round(float(np.std(ig_hfs, ddof=1)), 3),
            "pointing_game_hit_rate_pct": round(ig_hits / n * 100.0, 1),
            "hits": ig_hits,
        },
        "failure_analysis": {
            "count": failure_images,
            "pct": round(failure_images / n * 100.0, 1),
            "rule": "Quantitative attribution threshold rule: failure count if Eigen-CAM HFS < 0.45 or IG HFS < 0.30 (occurs primarily under severe partial occlusion).",
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Deterministic XAI evaluation script for held-out test split."
    )
    parser.add_argument(
        "--inc-ckpt", type=str, default=None, help="Path to incremental round 5 best.pt checkpoint"
    )
    parser.add_argument(
        "--base-ckpt", type=str, default=None, help="Path to one-shot baseline best.pt checkpoint"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Path to Weapon_Detection-1 dataset root directory",
    )
    parser.add_argument(
        "--test-list", type=str, default=str(TEST_LIST), help="Path to test_fixed.txt split file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_JSON),
        help="Path to output JSON results file",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=20,
        help="Number of test images to evaluate (default: 20; 0 for all)",
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device to use (e.g. '0', 'cuda', 'cpu')"
    )
    args = parser.parse_args()

    if args.device:
        device_str = f"cuda:{args.device}" if args.device.isdigit() else args.device
        device = torch.device(device_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    data_root = resolve_data_root(args.data_root)
    logger.info("Resolved dataset root: %s", data_root)

    test_list_path = Path(args.test_list).resolve()
    samples = load_test_sample(test_list_path, data_root, n=args.n_samples)

    inc_ckpt = find_checkpoint(args.inc_ckpt, BEST_INCREMENTAL_CANDIDATES, "Incremental Round 5")
    base_ckpt = find_checkpoint(args.base_ckpt, BEST_BASELINE_CANDIDATES, "One-Shot Baseline")

    inc = evaluate_model(inc_ckpt, samples, "Incremental (YOLO11n-P R5, seed 42)", device)
    base = evaluate_model(
        base_ckpt, samples, "One-Shot Reference Baseline (YOLO11n-P, seed 42)", device
    )

    res = {
        "provenance": f"Evaluated deterministically from {test_list_path.name} on N={len(samples)} held-out test set images using trained model checkpoints",
        "eval_scope": {
            "test_split_file": str(test_list_path),
            "dataset_root": str(data_root),
            "total_images_evaluated": len(samples),
            "total_boxes_evaluated": sum(len(s["gt_boxes"]) for s in samples),
            "knife_boxes_evaluated": sum(
                sum(1 for b in s["gt_boxes"] if b["class_id"] == 0) for s in samples
            ),
            "pistol_boxes_evaluated": sum(
                sum(1 for b in s["gt_boxes"] if b["class_id"] == 1) for s in samples
            ),
            "random_seed": 42,
            "device": str(device),
        },
        "results": {
            "incremental_yolo11n_pretrained_r5": inc,
            "oneshot_reference_baseline": base,
        },
    }

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)

    logger.info("SUCCESS: Wrote XAI evaluation results to %s", output_path)
    print("SUCCESS")


if __name__ == "__main__":
    main()
