"""
Edge agent simulator for incremental training.

Simulates edge agents performing inference on unlabeled data pool.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any

from ultralytics import YOLO

logger = logging.getLogger(__name__)


class Edge_Agent_Simulator:
    """
    Simulates edge agents performing inference on unlabeled pool.

    Key responsibilities:
    - Perform inference on all images in unlabeled_pool after each training round
    - Extract detections in format (image_id, pred_class, bbox, confidence)
    - Execute on single GPU without distributed infrastructure
    - Log all detections to JSON file per round
    """

    def __init__(self, device: str = "cuda"):
        """
        Initialize the Edge_Agent_Simulator.

        Args:
            device: Device for inference (cuda, cpu, mps)
        """
        self.device = device

    def simulate_edge_inference(
        self,
        model_checkpoint: str,
        unlabeled_pool_path: str,
        image_dir: str,
        output_path: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45
    ) -> List[Dict[str, Any]]:
        """
        Simulate edge agents performing inference on unlabeled_pool.

        Args:
            model_checkpoint: Path to trained model checkpoint
            unlabeled_pool_path: Path to unlabeled pool file (list of images)
            image_dir: Directory containing unlabeled pool images
            output_path: Path to save detection results JSON
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS

        Returns:
            List of detections with format:
                {
                    'image_id': str,
                    'pred_class': int,
                    'bbox': [x_center, y_center, width, height],  # normalized
                    'confidence': float
                }

        Raises:
            FileNotFoundError: If model checkpoint or unlabeled pool not found
        """
        # Validate inputs
        model_checkpoint_path = Path(model_checkpoint)
        if not model_checkpoint_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_checkpoint_path}")

        unlabeled_pool_file = Path(unlabeled_pool_path)
        if not unlabeled_pool_file.exists():
            raise FileNotFoundError(f"Unlabeled pool file not found: {unlabeled_pool_file}")

        image_dir_path = Path(image_dir)
        if not image_dir_path.exists():
            raise FileNotFoundError(f"Image directory not found: {image_dir_path}")

        # Load model
        logger.info(f"Loading model from {model_checkpoint_path}")
        model = YOLO(str(model_checkpoint_path))

        # Load unlabeled pool images
        with open(unlabeled_pool_file, 'r') as f:
            unlabeled_images = [line.strip() for line in f if line.strip()]

        logger.info(f"Running inference on {len(unlabeled_images)} unlabeled images")

        # Run inference on all images
        all_detections = []

        for img_file in unlabeled_images:
            img_path = image_dir_path / img_file

            if not img_path.exists():
                logger.warning(f"Image not found: {img_path}, skipping")
                continue

            # Run inference
            results = model.predict(
                source=str(img_path),
                conf=conf_threshold,
                iou=iou_threshold,
                device=self.device,
                verbose=False
            )

            # Extract detections
            if results and len(results) > 0:
                result = results[0]

                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes

                    # Get normalized coordinates (xywhn format)
                    if hasattr(boxes, 'xywhn'):
                        xywhn_data = boxes.xywhn
                        # Handle both Tensor and ndarray types
                        if hasattr(xywhn_data, 'cpu'):
                            xywhn = xywhn_data.cpu().numpy()
                        else:
                            xywhn = xywhn_data
                    else:
                        # Fallback: convert from xyxy to xywh normalized
                        xyxy_data = boxes.xyxy
                        # Handle both Tensor and ndarray types
                        if hasattr(xyxy_data, 'cpu'):
                            xyxy = xyxy_data.cpu().numpy()
                        else:
                            xyxy = xyxy_data
                        img_h, img_w = result.orig_shape
                        xywhn = self._xyxy_to_xywhn(xyxy, img_w, img_h)

                    classes_data = boxes.cls
                    # Handle both Tensor and ndarray types
                    if hasattr(classes_data, 'cpu'):
                        classes = classes_data.cpu().numpy()
                    else:
                        classes = classes_data

                    confidences_data = boxes.conf
                    # Handle both Tensor and ndarray types
                    if hasattr(confidences_data, 'cpu'):
                        confidences = confidences_data.cpu().numpy()
                    else:
                        confidences = confidences_data

                    for i in range(len(boxes)):
                        detection = {
                            'image_id': img_file,
                            'pred_class': int(classes[i]),
                            'bbox': xywhn[i].tolist(),  # [x_center, y_center, width, height]
                            'confidence': float(confidences[i])
                        }
                        all_detections.append(detection)

        logger.info(f"Extracted {len(all_detections)} detections from {len(unlabeled_images)} images")

        # Save detections to JSON
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path_obj, 'w') as f:
            json.dump(all_detections, f, indent=2)

        logger.info(f"Saved detections to {output_path_obj}")

        return all_detections

    def _xyxy_to_xywhn(self, xyxy, img_w, img_h):
        """
        Convert xyxy format to normalized xywh format.

        Args:
            xyxy: Array of shape (N, 4) with [x1, y1, x2, y2]
            img_w: Image width
            img_h: Image height

        Returns:
            Array of shape (N, 4) with normalized [x_center, y_center, width, height]
        """
        import numpy as np

        xywhn = np.zeros_like(xyxy)
        xywhn[:, 0] = ((xyxy[:, 0] + xyxy[:, 2]) / 2) / img_w  # x_center
        xywhn[:, 1] = ((xyxy[:, 1] + xyxy[:, 3]) / 2) / img_h  # y_center
        xywhn[:, 2] = (xyxy[:, 2] - xyxy[:, 0]) / img_w  # width
        xywhn[:, 3] = (xyxy[:, 3] - xyxy[:, 1]) / img_h  # height

        return xywhn

    def get_detection_statistics(self, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics for a set of detections.

        Args:
            detections: List of detection dictionaries

        Returns:
            Dictionary with statistics:
                - total_detections: Total number of detections
                - images_with_detections: Number of images with at least one detection
                - detections_per_class: Dictionary mapping class_id to count
                - avg_confidence: Average confidence across all detections
        """
        if not detections:
            return {
                'total_detections': 0,
                'images_with_detections': 0,
                'detections_per_class': {},
                'avg_confidence': 0.0
            }

        unique_images = set(d['image_id'] for d in detections)

        detections_per_class: Dict[int, int] = {}
        total_confidence = 0.0

        for det in detections:
            class_id = det['pred_class']
            detections_per_class[class_id] = detections_per_class.get(class_id, 0) + 1
            total_confidence += det['confidence']

        return {
            'total_detections': len(detections),
            'images_with_detections': len(unique_images),
            'detections_per_class': detections_per_class,
            'avg_confidence': total_confidence / len(detections)
        }


    def simulate_inference(
        self,
        model_path: str,
        unlabeled_images: list[str],
        base_path: Path,
        output_dir: str,
        round_num: int,
        device: str = "cuda"
    ) -> list[dict[str, Any]]:
        """
        Simulate inference on unlabeled pool (simplified interface for runner).

        Args:
            model_path: Path to model checkpoint
            unlabeled_images: List of image paths (can be absolute or relative)
            base_path: Base path for dataset
            output_dir: Directory to save detection results
            round_num: Round number for logging
            device: Device for inference

        Returns:
            List of detections
        """
        from ultralytics import YOLO
        import numpy as np

        # Load model
        logger.info(f"Loading model from {model_path}")
        model = YOLO(model_path)

        # Run inference on all unlabeled images
        all_detections = []

        for img_filename in unlabeled_images:
            # Construct full path: base_path / train / images / filename
            full_path = base_path / "train" / "images" / img_filename

            if not full_path.exists():
                logger.warning(f"Image not found: {full_path}, skipping")
                continue

            # Run inference
            results = model.predict(
                source=str(full_path),
                conf=0.25,
                iou=0.45,
                device=device,
                verbose=False
            )

            # Extract detections
            if results and len(results) > 0:
                result = results[0]

                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes

                    # Get normalized coordinates
                    if hasattr(boxes, 'xywhn'):
                        xywhn_data = boxes.xywhn
                        # Handle both Tensor and ndarray types
                        if hasattr(xywhn_data, 'cpu'):
                            xywhn = xywhn_data.cpu().numpy()
                        else:
                            xywhn = xywhn_data
                    else:
                        xyxy_data = boxes.xyxy
                        # Handle both Tensor and ndarray types
                        if hasattr(xyxy_data, 'cpu'):
                            xyxy = xyxy_data.cpu().numpy()
                        else:
                            xyxy = xyxy_data
                        img_h, img_w = result.orig_shape
                        xywhn = self._xyxy_to_xywhn(xyxy, img_w, img_h)

                    classes_data = boxes.cls
                    # Handle both Tensor and ndarray types
                    if hasattr(classes_data, 'cpu'):
                        classes = classes_data.cpu().numpy()
                    else:
                        classes = classes_data

                    confidences_data = boxes.conf
                    # Handle both Tensor and ndarray types
                    if hasattr(confidences_data, 'cpu'):
                        confidences = confidences_data.cpu().numpy()
                    else:
                        confidences = confidences_data

                    for i in range(len(boxes)):
                        detection = {
                            'image_id': full_path.name,  # Use just the filename
                            'pred_class': int(classes[i]),
                            'bbox': xywhn[i].tolist(),
                            'confidence': float(confidences[i])
                        }
                        all_detections.append(detection)

        logger.info(f"Extracted {len(all_detections)} detections from {len(unlabeled_images)} images")

        # Save detections to JSON
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        detections_file = output_path / f"round_{round_num}_detections.json"
        with open(detections_file, 'w') as f:
            json.dump(all_detections, f, indent=2)

        logger.info(f"Saved detections to {detections_file}")

        return all_detections
