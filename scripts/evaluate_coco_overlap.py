"""
Zero-Shot / Round-0 COCO Semantic Class Overlap Evaluator.

Addresses Reviewer 1 Major 4 & Reviewer 3 (TASK-MAJ-04, TASK-R3-03):
  "Solve the problem of the COCO-overlap in the knife class. Explicitly discuss it and if
   possible include a control like ImageNet pretraining, COCO pretraining without the knife
   class, or Round-0/zero-shot analysis."

Evaluates stock COCO-pretrained weights (yolov8n.pt, yolo11n.pt, yolo12n.pt) on the fixed
threat test set (test_fixed.txt) BEFORE any incremental training.
  - COCO class 43 is 'knife' -> mapped to target class 0 ('knife')
  - COCO contains no 'pistol' class -> target class 1 ('pistol') receives 0 detections
  - Calculates zero-shot Precision, Recall, and AP@0.5 per class.

Usage:
    uv run python scripts/evaluate_coco_overlap.py --device 0
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils import nms
from ultralytics.utils.ops import scale_boxes

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.device_utils import select_device

logger = logging.getLogger(__name__)

COCO_KNIFE_CLASS_ID = 43
TARGET_KNIFE_CLASS_ID = 0
TARGET_PISTOL_CLASS_ID = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate zero-shot semantic transfer of COCO knife class on threat test set."
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=["yolov8n.pt", "yolo11n.pt", "yolo12n.pt"],
        help="List of pretrained model weights (.pt) to evaluate",
    )
    parser.add_argument(
        "--test-list",
        type=str,
        default="config/data/test_fixed.txt",
        help="Path to test set image list",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run evaluation on ('auto', '0', 'cpu')",
    )
    parser.add_argument(
        "--conf-thres",
        type=float,
        default=0.25,
        help="Confidence threshold for detections (default: 0.25)",
    )
    parser.add_argument(
        "--iou-thres",
        type=float,
        default=0.50,
        help="IoU threshold for AP calculation (default: 0.50)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image resolution",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="Limit number of test images for quick dry-runs",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="outputs/coco_knife_overlap_evaluation.json",
        help="Output JSON summary path",
    )
    return parser.parse_args()


def box_iou(box1: np.ndarray, box2: np.ndarray) -> np.ndarray:
    """Calculate IoU between two sets of boxes [x1, y1, x2, y2]."""
    if len(box1) == 0 or len(box2) == 0:
        return np.zeros((len(box1), len(box2)), dtype=np.float32)

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    lt = np.maximum(box1[:, None, :2], box2[None, :, :2])
    rb = np.minimum(box1[:, None, 2:], box2[None, :, 2:])
    wh = np.clip(rb - lt, a_min=0, a_max=None)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Compute Average Precision using 101-point interpolation or all-points envelope."""
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([1.0], precisions, [0.0]))

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))
    return ap


def evaluate_zero_shot(
    model_path: str,
    device: torch.device,
    image_paths: list[str],
    imgsz: int = 640,
    conf_thres: float = 0.25,
    iou_thres: float = 0.50,
) -> dict[str, Any]:
    """Run zero-shot evaluation mapping COCO knife class (43) to target knife class (0)."""
    logger.info("Evaluating zero-shot transfer for %s...", model_path)
    model = YOLO(model_path)
    net = model.model.to(device)
    net.eval()

    letterbox = LetterBox(imgsz, auto=False)

    detections_by_class: dict[int, list[tuple[float, bool]]] = {0: [], 1: []}
    gt_counts: dict[int, int] = {0: 0, 1: 0}

    for img_rel in image_paths:
        p = Path(img_rel)
        if not p.is_absolute():
            p = PROJECT_ROOT / img_rel
        if not p.exists():
            continue

        raw_im = cv2.imread(str(p))
        if raw_im is None:
            continue
        h0, w0 = raw_im.shape[:2]

        # Load ground truth labels
        parts = list(p.parts)
        if "images" in parts:
            idx = parts.index("images")
            parts[idx] = "labels"
            lbl_p = Path(*parts).with_suffix(".txt")
        else:
            lbl_p = p.parent.parent / "labels" / f"{p.stem}.txt"

        gt_boxes_by_class: dict[int, list[list[float]]] = {0: [], 1: []}
        if lbl_p.exists():
            for line in lbl_p.read_text(encoding="utf-8").splitlines():
                parts_l = line.strip().split()
                if not parts_l:
                    continue
                try:
                    cls_id = int(parts_l[0])
                    xc, yc, w, h = [float(x) for x in parts_l[1:5]]
                    # Convert normalized xywh to absolute xyxy
                    x1 = (xc - w / 2) * w0
                    y1 = (yc - h / 2) * h0
                    x2 = (xc + w / 2) * w0
                    y2 = (yc + h / 2) * h0
                    if cls_id in gt_boxes_by_class:
                        gt_boxes_by_class[cls_id].append([x1, y1, x2, y2])
                        gt_counts[cls_id] += 1
                except (ValueError, IndexError):
                    continue

        # Run inference
        im_lb = letterbox(image=raw_im)
        im_rgb = np.ascontiguousarray(im_lb[..., ::-1].transpose((2, 0, 1)))
        tensor = torch.from_numpy(im_rgb).to(device).float() / 255.0
        tensor = tensor.unsqueeze(0)

        with torch.no_grad():
            preds = net(tensor)
            results = nms.non_max_suppression(preds, conf_thres=conf_thres, iou_thres=0.45, max_det=300)

        # Process detections on original image scale
        det = results[0]
        if det is not None and len(det) > 0:
            det[:, :4] = scale_boxes(im_lb.shape[:2], det[:, :4], (h0, w0)).round()
            for *xyxy, conf, cls_coco in det.cpu().numpy():
                cls_coco = int(cls_coco)
                score = float(conf)

                # Map COCO knife (43) -> Target knife (0)
                if cls_coco == COCO_KNIFE_CLASS_ID:
                    target_cls = TARGET_KNIFE_CLASS_ID
                    # Check match against ground truth knife boxes
                    gts = np.array(gt_boxes_by_class[target_cls], dtype=np.float32)
                    pred_b = np.array([xyxy], dtype=np.float32)
                    matched = False
                    if len(gts) > 0:
                        ious = box_iou(pred_b, gts)[0]
                        max_iou = float(np.max(ious))
                        if max_iou >= iou_thres:
                            matched = True
                    detections_by_class[target_cls].append((score, matched))

    # Calculate AP and summary metrics per class
    class_metrics: dict[str, Any] = {}
    for cls_id, cls_name in ((0, "knife"), (1, "pistol")):
        dets = sorted(detections_by_class[cls_id], key=lambda x: x[0], reverse=True)
        total_gt = gt_counts[cls_id]
        if not dets or total_gt == 0:
            class_metrics[cls_name] = {
                "ground_truth_count": total_gt,
                "detection_count": len(dets),
                "true_positives": 0,
                "precision": 0.0,
                "recall": 0.0,
                "ap50": 0.0,
            }
            continue

        tp = np.array([1 if m else 0 for _, m in dets], dtype=np.float32)
        fp = np.array([0 if m else 1 for _, m in dets], dtype=np.float32)

        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)

        precisions = cum_tp / (cum_tp + cum_fp)
        recalls = cum_tp / total_gt

        ap50 = compute_ap(recalls, precisions)
        final_p = float(precisions[-1]) if len(precisions) > 0 else 0.0
        final_r = float(recalls[-1]) if len(recalls) > 0 else 0.0

        class_metrics[cls_name] = {
            "ground_truth_count": total_gt,
            "detection_count": len(dets),
            "true_positives": int(cum_tp[-1]),
            "precision": round(final_p, 4),
            "recall": round(final_r, 4),
            "ap50": round(ap50, 4),
        }

    overall_map50 = (class_metrics["knife"]["ap50"] + class_metrics["pistol"]["ap50"]) / 2.0

    return {
        "model_name": Path(model_path).stem,
        "model_path": str(model_path),
        "protocol": "Zero-Shot Round-0 Pretrained Baseline (COCO class 43 knife -> class 0)",
        "test_images_evaluated": len(image_paths),
        "classes": class_metrics,
        "mAP50_overall": round(overall_map50, 4),
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    device_str = select_device(args.device)
    device = torch.device(device_str)

    test_list_path = PROJECT_ROOT / args.test_list
    if not test_list_path.exists():
        logger.error("Test list not found: %s", test_list_path)
        sys.exit(1)

    image_paths = [
        ln.strip() for ln in test_list_path.read_text(encoding="utf-8").splitlines() if ln.strip()
    ]
    if args.sample_limit:
        image_paths = image_paths[:args.sample_limit]

    print("\n" + "=" * 80)
    print("COCO KNIFE SEMANTIC CLASS OVERLAP: ZERO-SHOT EVALUATION")
    print("=" * 80)
    print(f"Device:               {device}")
    print(f"Test Images:          {len(image_paths)}")
    print(f"Confidence Threshold: {args.conf_thres}")
    print(f"IoU Threshold:        {args.iou_thres}")
    print("=" * 80 + "\n")

    results: list[dict[str, Any]] = []
    for model_name in args.models:
        model_path = PROJECT_ROOT / model_name
        if not model_path.exists():
            model_path = Path(model_name)
        res = evaluate_zero_shot(
            model_path=str(model_path),
            device=device,
            image_paths=image_paths,
            imgsz=args.imgsz,
            conf_thres=args.conf_thres,
            iou_thres=args.iou_thres,
        )
        results.append(res)

    # Save output JSON
    out_json = PROJECT_ROOT / args.output_json
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
        fh.write("\n")
    logger.info("Saved zero-shot overlap evaluation to %s", out_json)

    # Print summary table
    print("\n" + "=" * 80)
    print(f"{'Model':<20} {'Knife AP@0.5':<15} {'Knife Recall':<15} {'Pistol AP@0.5':<15} {'Overall mAP':<15}")
    print("-" * 80)
    for r in results:
        k = r["classes"]["knife"]
        p = r["classes"]["pistol"]
        print(
            f"{r['model_name']:<20} {k['ap50']:<15.4f} {k['recall']:<15.4f} "
            f"{p['ap50']:<15.4f} {r['mAP50_overall']:<15.4f}"
        )
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
