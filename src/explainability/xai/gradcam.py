"""Grad-CAM (Gradient-weighted Class Activation Mapping) implementation for YOLO models."""

import logging
from pathlib import Path
import time
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from yolo_cam.eigen_cam import EigenCAM
from yolo_cam.utils.image import show_cam_on_image

from ..hfs_scorer import Heatmap_Focus_Scorer
from .interfaces import AttributionMethod
from .output_manager import XAIOutputManager

logger = logging.getLogger(__name__)


def get_target_layer_module(model: Any, target_layer_path: str) -> torch.nn.Module:
    """Get target layer module from model using layer path.

    Args:
        model: YOLO model object
        target_layer_path: Layer path string (e.g., "model.9")

    Returns:
        Target layer module

    Raises:
        ValueError: If layer path parameter is invalid
        AttributeError: If layer path not found in model
    """
    if not isinstance(target_layer_path, str):
        raise ValueError(f"target_layer_path must be a string, got {type(target_layer_path)}")

    if not target_layer_path.strip():
        raise ValueError("target_layer_path cannot be empty")

    # For YOLO models, the actual PyTorch model is in model.model
    torch_model = model.model if hasattr(model, "model") else model

    # Try to find the target layer by iterating through named modules
    target_module: torch.nn.Module | None = None
    for name, module in torch_model.named_modules():
        if name == target_layer_path:
            target_module = module
            break

    if target_module is None:
        available_layers = [name for name, _ in torch_model.named_modules() if name]
        raise AttributeError(
            f"Target layer '{target_layer_path}' not found in model. "
            f"Available layers: {', '.join(available_layers[:10])}"
            + (" ..." if len(available_layers) > 10 else "")
        )

    return target_module


def preprocess_image_for_cam(image_np: np.ndarray, model: Any) -> np.ndarray:
    """Preprocess image for YOLO-CAM library.

    Args:
        image_np: Image as numpy array (H, W, 3) in RGB format
        model: YOLO model object

    Returns:
        Resized RGB image as numpy array (640, 640, 3)

    Raises:
        ValueError: If image array is not in expected format
    """
    if not isinstance(image_np, np.ndarray):
        raise ValueError(f"Expected numpy array, got {type(image_np)}")

    if image_np.ndim != 3 or image_np.shape[2] != 3:
        raise ValueError(f"Expected image shape (H, W, 3), got {image_np.shape}")

    target_size = model.imgsz if hasattr(model, "imgsz") else 640
    if isinstance(target_size, (list, tuple)):
        target_size = target_size[0]

    return cv2.resize(image_np, (target_size, target_size))


def generate_gradcam_attribution(
    model: Any,
    image_path: str,
    detections: dict[str, Any],
    gt_boxes: list[tuple[float, float, float, float]],
    _target_layer: str,
    _device: str = "cuda",
    _class_idx: int | None = None,
    output_dir: str | None = None,
    output_manager: XAIOutputManager | None = None,
    target_class_names: list[str] | None = None,
) -> tuple[np.ndarray, float | None, str]:
    """
    Generate Grad-CAM attribution map for YOLO model prediction using EigenCAM.

    Args:
        model: Trained YOLO model
        image_path: Path to input image
        detections: Model detections for the image
        gt_boxes: Ground-truth bounding boxes in YOLO format
        target_layer: Unused — target layer is always model.model.model[-2]
        device: Computation device
        class_idx: Unused
        output_dir: Directory to save attribution visualization (legacy)
        output_manager: Output manager instance for centralized output handling
        target_class_names: If provided, mask the CAM to only highlight detections
            for these classes (e.g. ["gun", "knife"]). None = show all classes.

    Returns:
        Tuple of (grayscale_cam, hfs_score, output_path)

    Raises:
        TypeError: If model is not a valid Ultralytics YOLO model object
        RuntimeError: If Grad-CAM computation fails
    """
    start_time = time.time()

    if not hasattr(model, "model") or not isinstance(
        getattr(model, "model", None), torch.nn.Module
    ):
        raise TypeError(
            f"Invalid model type: expected a loaded Ultralytics YOLO model object, "
            f"but got {type(model).__name__}. The model must be an Ultralytics YOLO instance "
            f"with a 'model' attribute containing a torch.nn.Module."
        )

    try:
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        image_height, image_width = image_np.shape[:2]

        model_names = getattr(model, "names", {})

        # Check for target-class detections before running EigenCAM
        has_weapon_boxes = _has_target_class_detections(detections, model_names, target_class_names)

        if not has_weapon_boxes:
            logger.debug(
                f"No target-class detections in {Path(image_path).name}, "
                f"saving original image only."
            )
            output_path = ""
            if output_dir or output_manager:
                if output_manager is not None:
                    out_dir = str(output_manager.get_method_output_dir(AttributionMethod.GRADCAM))
                else:
                    out_dir = output_dir or ""
                output_path = _save_original_only(image_np, image_path, out_dir)
            blank_cam = np.zeros((image_height, image_width), dtype=np.float32)
            return blank_cam, None, output_path

        target_size = model.imgsz if hasattr(model, "imgsz") else 640
        if isinstance(target_size, (list, tuple)):
            target_size = target_size[0]

        # Resize to model input size
        rgb_img = cv2.resize(image_np, (target_size, target_size))

        target_layers = [model.model.model[-2]]
        cam = EigenCAM(model, target_layers, task="od")
        grayscale_cam = cam(rgb_img)[0, :, :]

        if np.isnan(grayscale_cam).any():
            logger.warning(f"CAM contains NaN values for {image_path}, replacing with zeros")
            grayscale_cam = np.zeros_like(grayscale_cam)

        # Resize CAM to original image dimensions
        if grayscale_cam.shape != (image_height, image_width):
            grayscale_cam = cv2.resize(
                grayscale_cam, (image_width, image_height), interpolation=cv2.INTER_LINEAR
            )

        # Mask to target classes
        if target_class_names:
            mask = _build_class_mask(
                detections,
                model_names,
                target_class_names,
                image_height,
                image_width,
            )
            if mask.max() > 0:
                grayscale_cam = grayscale_cam * mask

        # Build overlay
        img_float = image_np.astype(np.float32) / 255.0
        cam_image = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)

        # Compute HFS score
        hfs_score = None
        if gt_boxes:
            try:
                hfs_scorer = Heatmap_Focus_Scorer()
                hfs_score = hfs_scorer.compute_hfs(
                    grayscale_cam, gt_boxes[0], image_width, image_height
                )
            except Exception as hfs_err:
                logger.warning(
                    f"HFS computation failed for {image_path}, returning attribution with hfs_score=None: {hfs_err}"
                )

        # Save 3-panel matplotlib figure
        output_path = ""
        if output_dir or output_manager:
            if output_manager:
                output_path = output_manager.save_attribution_outputs(
                    attribution_map=grayscale_cam,
                    original_image=image_np,
                    image_path=image_path,
                    method=AttributionMethod.GRADCAM,
                    detections=detections,
                    model_names=model_names,
                )
            elif output_dir is not None:
                output_path = _save_gradcam_visualization(
                    image_np, grayscale_cam, cam_image, image_path, output_dir
                )

        processing_time = time.time() - start_time
        hfs_str = f"{hfs_score:.4f}" if hfs_score is not None else "N/A"
        logger.info(
            f"Generated Grad-CAM attribution for {Path(image_path).name} "
            f"(HFS: {hfs_str}, time: {processing_time:.2f}s)"
        )

        return grayscale_cam, hfs_score, output_path

    except Exception as e:
        logger.error(f"Failed to generate Grad-CAM attribution for {image_path}: {e}")
        raise RuntimeError(f"Grad-CAM computation failed: {e}") from e


def _has_target_class_detections(
    detections: dict[str, Any],
    model_names: dict[int, str],
    target_class_names: list[str] | None,
) -> bool:
    """Return True if detections contains at least one box for a target class."""
    if not target_class_names:
        boxes = detections.get("boxes", [])
        return len(boxes) > 0

    target_ids = {
        idx
        for idx, name in model_names.items()
        if name.lower() in {c.lower() for c in target_class_names}
    }
    classes = detections.get("classes", [])
    return any(int(cls) in target_ids for cls in classes)


def _save_original_only(image_np: np.ndarray, image_path: str, output_dir: str) -> str:
    """Save just the original image when no weapon boxes were detected."""
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    image_name = Path(image_path).stem
    output_path = output_dir_path / f"{image_name}_gradcam.png"
    Image.fromarray(image_np.astype(np.uint8)).save(str(output_path))
    return str(output_path)


def _build_class_mask(
    detections: dict[str, Any],
    model_names: dict[int, str],
    target_class_names: list[str],
    h: int,
    w: int,
) -> np.ndarray:
    """Build a soft spatial mask covering only target-class bounding boxes.

    Returns a float32 array in [0, 1] of shape (h, w).
    """
    target_ids = {
        idx
        for idx, name in model_names.items()
        if name.lower() in {c.lower() for c in target_class_names}
    }

    boxes = detections.get("boxes", [])
    classes = detections.get("classes", [])

    mask = np.zeros((h, w), dtype=np.float32)
    for box, cls in zip(boxes, classes, strict=False):
        if int(cls) in target_ids:
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = 1.0

    if mask.max() > 0:
        blur_k = max(31, (min(h, w) // 20) | 1)
        mask = cv2.GaussianBlur(mask, (blur_k, blur_k), 0).astype(np.float32)
        mask = mask / mask.max()

    return mask


def _save_gradcam_visualization(
    image_np: np.ndarray,
    grayscale_cam: np.ndarray,
    cam_image: np.ndarray,
    image_path: str,
    output_dir: str,
    detections: dict[str, Any] | None = None,
    model_names: dict[int, str] | None = None,
) -> str:
    """Save Grad-CAM as a 3-panel matplotlib figure."""
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    img_rgb = image_np.astype(np.uint8)

    cam_norm = grayscale_cam.astype(np.float32)
    cam_max = cam_norm.max()
    if cam_max > 0:
        cam_norm = cam_norm / cam_max

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(img_rgb)
    axes[0].set_title("Original", fontsize=13)
    axes[0].axis("off")

    im = axes[1].imshow(cam_norm, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM Heatmap", fontsize=13)
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    cam_image_out = cam_image.copy()
    if detections and model_names:
        for box, cls, score in zip(
            detections.get("boxes", []),
            detections.get("classes", []),
            detections.get("scores", []),
            strict=False,
        ):
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            label = f"{model_names.get(int(cls), str(int(cls)))}: {score:.2f}"
            cv2.rectangle(cam_image_out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                cam_image_out,
                label,
                (x1, max(y1 - 5, 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

    axes[2].imshow(cam_image_out)
    axes[2].set_title("Overlay", fontsize=13)
    axes[2].axis("off")

    plt.tight_layout()

    image_name = Path(image_path).stem
    output_path = output_dir_path / f"{image_name}_gradcam.png"
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)

    return str(output_path)
