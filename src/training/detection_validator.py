"""
Detection validator for incremental training.

Validates edge agent detections against ground truth using IoU and class matching.
"""

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class Detection_Validator:
    """
    Validates edge agent detections against ground truth.

    Key responsibilities:
    - Compare predicted class with ground-truth class
    - Compute IoU between predicted bbox and ground-truth bbox
    - Mark detection as verified if class matches AND IoU ≥ threshold
    - Use original ground-truth labels for verified samples
    - Log validation results per round
    """

    def __init__(self, iou_threshold: float = 0.5):
        """
        Initialize the Detection_Validator.

        Args:
            iou_threshold: IoU threshold for validation (default: 0.5)
        """
        self.iou_threshold = iou_threshold

    def validate_detections(
        self,
        detections: list[dict[str, Any]],
        ground_truth_dir: str,
        unlabeled_pool: list[str] | None = None,
        output_path: str | None = None,
    ) -> dict[str, Any]:
        """
        Validate edge agent detections against ground truth.

        Args:
            detections: List of detections from edge agent with format:
                {
                    'image_id': str,
                    'pred_class': int,
                    'bbox': [x_center, y_center, width, height],  # normalized
                    'confidence': float
                }
            ground_truth_dir: Directory containing ground truth annotations
            unlabeled_pool: List of all images in unlabeled pool (to track undetected)
            output_path: Path to save validation results JSON (optional)

        Returns:
            Dictionary with:
                - verified_samples: List of verified image IDs
                - verified_detections: List of verified detection details
                - rejected_detections: List of rejected detection details
                - undetected_images: List of images with no detections
                - total_detections: Total number of detections
                - verified_count: Number of verified detections
                - rejected_count: Number of rejected detections
                - undetected_count: Number of images with no detections
                - verification_rate: Percentage of verified detections

        Raises:
            FileNotFoundError: If ground truth directory not found
        """
        gt_dir_path = Path(ground_truth_dir)
        if not gt_dir_path.exists():
            raise FileNotFoundError(f"Ground truth directory not found: {gt_dir_path}")

        logger.info(f"Validating {len(detections)} detections against ground truth")

        verified_detections = []
        rejected_detections = []
        verified_image_ids = set()

        # Group detections by image
        detections_by_image: dict[str, list[dict[str, Any]]] = {}
        for det in detections:
            img_id = det["image_id"]
            if img_id not in detections_by_image:
                detections_by_image[img_id] = []
            detections_by_image[img_id].append(det)

        # Track undetected images (images in pool with no detections)
        undetected_images = []
        if unlabeled_pool:
            detected_images = set(detections_by_image.keys())
            undetected_images = [img for img in unlabeled_pool if img not in detected_images]
            logger.info(f"Found {len(undetected_images)} images with no detections")

        # Validate each image's detections
        for img_id, img_detections in detections_by_image.items():
            # Load ground truth for this image
            gt_file = gt_dir_path / Path(img_id).with_suffix(".txt").name

            if not gt_file.exists():
                logger.warning(f"Ground truth not found for {img_id}, rejecting all detections")
                for det in img_detections:
                    rejected_detections.append({**det, "rejection_reason": "no_ground_truth"})
                continue

            gt_annotations = self._load_ground_truth(gt_file)

            if not gt_annotations:
                logger.warning(f"Empty ground truth for {img_id}, rejecting all detections")
                for det in img_detections:
                    rejected_detections.append({**det, "rejection_reason": "empty_ground_truth"})
                continue

            # Validate each detection against ground truth
            for det in img_detections:
                is_verified, match_info = self._validate_single_detection(det, gt_annotations)

                if is_verified:
                    verified_detections.append(
                        {
                            **det,
                            "gt_class": match_info["gt_class"],
                            "iou": match_info["iou"],
                            "gt_bbox": match_info["gt_bbox"],
                        }
                    )
                    verified_image_ids.add(img_id)
                else:
                    rejected_detections.append(
                        {
                            **det,
                            "rejection_reason": match_info["reason"],
                            "best_iou": match_info.get("best_iou", 0.0),
                        }
                    )

        # Compute statistics
        total_detections = len(detections)
        verified_count = len(verified_detections)
        rejected_count = len(rejected_detections)
        undetected_count = len(undetected_images)
        verification_rate = verified_count / total_detections if total_detections > 0 else 0.0

        logger.info(f"Validation complete: {verified_count} verified, {rejected_count} rejected")
        logger.info(f"Verification rate: {verification_rate:.2%}")
        logger.info(f"Verified samples from {len(verified_image_ids)} unique images")
        logger.info(f"Undetected images: {undetected_count}")

        results = {
            "verified_samples": sorted(verified_image_ids),
            "verified_detections": verified_detections,
            "rejected_detections": rejected_detections,
            "undetected_images": undetected_images,
            "total_detections": total_detections,
            "verified_count": verified_count,
            "rejected_count": rejected_count,
            "undetected_count": undetected_count,
            "verification_rate": verification_rate,
            "iou_threshold": self.iou_threshold,
        }

        # Save results if output path provided
        if output_path:
            output_path_obj = Path(output_path)
            output_path_obj.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path_obj, "w") as f:
                json.dump(results, f, indent=2)

            logger.info(f"Saved validation results to {output_path_obj}")

        return results

    def _load_ground_truth(self, gt_file: Path) -> list[dict[str, Any]]:
        """
        Load ground truth annotations from YOLO format file.

        Args:
            gt_file: Path to ground truth annotation file

        Returns:
            List of ground truth annotations with format:
                {
                    'class_id': int,
                    'bbox': [x_center, y_center, width, height]  # normalized
                }
        """
        annotations = []

        with open(gt_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    bbox = [float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])]

                    annotations.append({"class_id": class_id, "bbox": bbox})

        return annotations

    def _validate_single_detection(
        self, detection: dict[str, Any], gt_annotations: list[dict[str, Any]]
    ) -> tuple[bool, dict[str, Any]]:
        """
        Validate a single detection against ground truth annotations.

        Args:
            detection: Detection dictionary
            gt_annotations: List of ground truth annotations

        Returns:
            Tuple of (is_verified, match_info)
            - is_verified: True if detection is verified
            - match_info: Dictionary with validation details
        """
        pred_class = detection["pred_class"]
        pred_bbox = detection["bbox"]

        best_iou = 0.0
        best_match = None

        # Find best matching ground truth annotation
        for gt_ann in gt_annotations:
            gt_class = gt_ann["class_id"]
            gt_bbox = gt_ann["bbox"]

            # Check class match
            if pred_class != gt_class:
                continue

            # Compute IoU
            iou = self._compute_iou(pred_bbox, gt_bbox)

            if iou > best_iou:
                best_iou = iou
                best_match = gt_ann

        # Verify if best match meets threshold
        if best_match is not None and best_iou >= self.iou_threshold:
            return True, {
                "gt_class": best_match["class_id"],
                "gt_bbox": best_match["bbox"],
                "iou": best_iou,
            }
        else:
            reason = "no_class_match" if best_match is None else "low_iou"

            return False, {"reason": reason, "best_iou": best_iou}

    def _compute_iou(self, bbox1: list[float], bbox2: list[float]) -> float:
        """
        Compute IoU between two bounding boxes in normalized xywh format.

        Args:
            bbox1: [x_center, y_center, width, height] (normalized)
            bbox2: [x_center, y_center, width, height] (normalized)

        Returns:
            IoU value in [0, 1]
        """
        # Convert from xywh to xyxy
        x1_min = bbox1[0] - bbox1[2] / 2
        y1_min = bbox1[1] - bbox1[3] / 2
        x1_max = bbox1[0] + bbox1[2] / 2
        y1_max = bbox1[1] + bbox1[3] / 2

        x2_min = bbox2[0] - bbox2[2] / 2
        y2_min = bbox2[1] - bbox2[3] / 2
        x2_max = bbox2[0] + bbox2[2] / 2
        y2_max = bbox2[1] + bbox2[3] / 2

        # Compute intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        inter_width = max(0, inter_x_max - inter_x_min)
        inter_height = max(0, inter_y_max - inter_y_min)
        inter_area = inter_width * inter_height

        # Compute union
        area1 = bbox1[2] * bbox1[3]
        area2 = bbox2[2] * bbox2[3]
        union_area = area1 + area2 - inter_area

        # Compute IoU
        if union_area == 0:
            return 0.0

        iou = inter_area / union_area
        return iou

    def get_validation_summary(self, validation_results: dict[str, Any]) -> str:
        """
        Get a human-readable summary of validation results.

        Args:
            validation_results: Results from validate_detections()

        Returns:
            Formatted summary string
        """
        summary = []
        summary.append("=" * 60)
        summary.append("Detection Validation Summary")
        summary.append("=" * 60)
        summary.append(f"Total detections: {validation_results['total_detections']}")
        summary.append(f"Verified detections: {validation_results['verified_count']}")
        summary.append(f"Rejected detections: {validation_results['rejected_count']}")
        summary.append(f"Verification rate: {validation_results['verification_rate']:.2%}")
        summary.append(
            f"Verified samples: {len(validation_results['verified_samples'])} unique images"
        )
        summary.append(f"IoU threshold: {validation_results['iou_threshold']}")
        summary.append("=" * 60)

        return "\n".join(summary)
