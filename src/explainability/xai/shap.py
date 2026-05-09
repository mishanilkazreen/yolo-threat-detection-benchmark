"""SHAP (SHapley Additive exPlanations) implementation for YOLO models."""

import logging
from pathlib import Path
import time
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")  # non-interactive backend — avoids tkinter/main-thread errors
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

from ..hfs_scorer import Heatmap_Focus_Scorer
from .interfaces import AttributionMethod
from .output_manager import XAIOutputManager

logger = logging.getLogger(__name__)


def generate_shap_attribution(
    model: Any,
    image_path: str,
    detections: dict[str, Any],  # noqa: ARG001
    gt_boxes: list[tuple[float, float, float, float]],
    background_set: list[np.ndarray],  # noqa: ARG001
    device: str = "cuda",
    output_dir: str | None = None,
    output_manager: XAIOutputManager | None = None,
) -> tuple[np.ndarray, float | None, str]:
    """
    Generate SHAP attribution map for YOLO model prediction.

    Args:
        model: Trained YOLO model
        image_path: Path to input image
        detections: Model detections for the image
        gt_boxes: Ground-truth bounding boxes in YOLO format
        background_set: Background images for SHAP baseline
        device: Computation device
        output_dir: Directory to save attribution visualization (legacy)
        output_manager: Output manager instance for centralized output handling

    Returns:
        Tuple of (attribution_map, hfs_score, output_path)
        Note: HFS score computation is supported for SHAP attribution maps

    Raises:
        ImportError: If shap library is not available
        RuntimeError: If SHAP computation fails
    """
    start_time = time.time()

    try:
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        image_height, image_width = image_np.shape[:2]

        # Preprocess image for model input
        target_size = model.imgsz if hasattr(model, "imgsz") else 640
        input_tensor = _preprocess_image_shap(image_np, target_size)
        input_tensor = input_tensor.to(device)

        # Wrap the inner PyTorch model (model.model), NOT the full YOLO wrapper.
        # The full YOLO wrapper's __call__ goes through the predictor pipeline and
        # returns Results objects, not tensors.
        inner_model = model.model if hasattr(model, "model") else model
        inner_model.eval()

        # Enable gradients for the model (required for SHAP)
        inner_model.requires_grad_(True)

        # Purge any stale backward hooks left by a prior zennit LRP pass on the same
        # model object.  Stale module hooks cause "BackwardHookFunctionBackward is a
        # view and is being modified inplace" on YOLO's C2f/DFL view tensors.
        for mod in inner_model.modules():
            mod._backward_hooks.clear()
            if hasattr(mod, "_full_backward_hooks"):
                mod._full_backward_hooks.clear()

        def _forward(x: torch.Tensor) -> torch.Tensor:
            out = inner_model(x)
            return _extract_shap_target(out)

        # Manual integrated gradients.
        # The entire loop — tensor creation, forward pass, AND backward — must be
        # inside inference_mode(False).  YOLO's @smart_inference_mode decorator causes
        # the Detect head to cache its anchor/stride tensors as inference tensors on
        # the first predict() call.  Those cached tensors flow into the autograd graph
        # and PyTorch refuses to save them for backward — even with .clone().
        # Resetting detect_head.shape = None forces the Detect head to recreate those
        # tensors fresh inside this non-inference context.
        n_steps = 20
        with torch.inference_mode(False):
            # Flush cached inference tensors from the Detect head
            _detect = getattr(inner_model, "model", [None])[-1]
            if hasattr(_detect, "shape"):
                _detect.shape = None

            input_normal = input_tensor.detach().clone()
            baseline = torch.zeros_like(input_normal)
            integrated_grads = torch.zeros_like(input_normal)

            for step in range(n_steps):
                alpha = (step + 1) / n_steps
                interp = (
                    (baseline + alpha * (input_normal - baseline)).detach().requires_grad_(True)
                )
                grad_holder: list[torch.Tensor | None] = [None]

                def hook_fn(g, holder=grad_holder):  # pylint: disable=dangerous-default-value
                    holder[0] = g.detach().clone()

                hook = interp.register_hook(hook_fn)
                try:
                    out = _forward(interp)
                    out.sum().backward()
                finally:
                    hook.remove()
                if grad_holder[0] is not None:
                    integrated_grads = integrated_grads + grad_holder[0]

        # Scale by (input - baseline) and average over steps
        attribution_tensor = (input_normal - baseline).detach() * integrated_grads / n_steps
        attribution_np = attribution_tensor.cpu().numpy()

        # Process attribution map
        attribution_np = attribution_np.squeeze()

        # Sum across channels if needed
        if attribution_np.ndim == 3:
            attribution_np = np.sum(attribution_np, axis=0)

        # Unpad letterbox borders then resize to original image size
        attribution_np = _unpad_and_resize_attribution(
            attribution_np, image_height, image_width, target_size
        )

        # Normalise to [-1, 1] via abs-max so scale is consistent across images
        abs_max = float(np.abs(attribution_np).max())
        if abs_max > 0:
            attribution_np = attribution_np / abs_max

        # Compute HFS score (bbox-weighted relevance, handles signed maps like LRP)
        hfs_score = None
        if gt_boxes:
            try:
                hfs_scorer = Heatmap_Focus_Scorer()
                hfs_score = hfs_scorer.compute_bbox_weighted_relevance(
                    attribution_np, gt_boxes[0], image_width, image_height
                )
            except Exception as e:
                logger.warning(f"Failed to compute SHAP HFS score: {e}")

        # Save visualization
        output_path = ""
        if output_dir or output_manager:
            if output_manager:
                # Use provided output manager
                output_path = output_manager.save_attribution_outputs(
                    attribution_map=attribution_np,
                    original_image=image_np,
                    image_path=image_path,
                    method=AttributionMethod.SHAP,
                )
            elif output_dir is not None:
                # Fallback to legacy method
                output_path = _save_shap_visualization(
                    image_np, attribution_np, image_path, output_dir
                )

        processing_time = time.time() - start_time

        hfs_str = f"{hfs_score:.4f}" if hfs_score is not None else "N/A"
        logger.info(
            f"Generated SHAP attribution for {Path(image_path).name} "
            f"(HFS: {hfs_str}, time: {processing_time:.2f}s, shape: {attribution_np.shape})"
        )

        return attribution_np, hfs_score, output_path

    except BaseException as e:
        logger.error(f"Failed to generate SHAP attribution for {image_path}: {e}")
        # Return a zero attribution map so the pipeline continues rather than crashing
        try:
            image = Image.open(image_path).convert("RGB")
            h, w = np.array(image).shape[:2]
        except Exception:
            h, w = 640, 640
        return np.zeros((h, w), dtype=np.float32), None, ""


def create_background_set(
    validation_images: list[str],
    background_set_size: int = 75,
    seed: int = 42,
) -> list[np.ndarray]:
    """
    Create deterministic background set for SHAP explanations.

    Args:
        validation_images: List of validation image paths
        background_set_size: Number of images to include in background set
        seed: Random seed for deterministic sampling

    Returns:
        List of background images as numpy arrays

    Raises:
        ValueError: If not enough validation images available
    """
    if len(validation_images) < background_set_size:
        logger.warning(
            f"Only {len(validation_images)} validation images available, "
            f"requested {background_set_size}. Using all available images."
        )
        background_set_size = len(validation_images)

    # Deterministic sampling
    np.random.seed(seed)
    selected_indices = np.random.choice(
        len(validation_images), size=background_set_size, replace=False
    )

    background_images = []
    for idx in selected_indices:
        try:
            image_path = validation_images[idx]
            image = Image.open(image_path).convert("RGB")
            image_np = np.array(image)
            background_images.append(image_np)
        except Exception as e:
            logger.warning(f"Failed to load background image {validation_images[idx]}: {e}")
            continue

    logger.info(f"Created SHAP background set with {len(background_images)} images (seed: {seed})")

    return background_images


def _unpad_and_resize_attribution(
    attribution: np.ndarray,
    orig_h: int,
    orig_w: int,
    target_size: int = 640,
) -> np.ndarray:
    """Remove letterbox padding from an attribution map, then resize to original dims.

    The preprocessing pipeline pads images to a square (letterboxing).  The
    attribution map is therefore in that padded space — a naive resize to the
    original image dimensions maps padding pixels into image space and shifts
    the heatmap spatially.  Crop the padding first, then resize.
    """
    scale = target_size / max(orig_h, orig_w)
    new_h = int(orig_h * scale)
    new_w = int(orig_w * scale)
    top = (target_size - new_h) // 2
    left = (target_size - new_w) // 2
    attribution_cropped = attribution[top : top + new_h, left : left + new_w]
    return cv2.resize(attribution_cropped, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)


def _preprocess_image_shap(image_np: np.ndarray, target_size: int = 640) -> torch.Tensor:
    """Preprocess image for YOLO model input (same as other methods)."""
    # Resize image while maintaining aspect ratio
    h, w = image_np.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)

    resized = cv2.resize(image_np, (new_w, new_h))

    # Pad to square
    pad_h = target_size - new_h
    pad_w = target_size - new_w
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2

    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )

    # Convert to tensor and normalize
    tensor = torch.from_numpy(padded).permute(2, 0, 1).float() / 255.0
    return tensor.unsqueeze(0)


def _prepare_background_set(
    background_images: list[np.ndarray], device: str, target_size: int = 640
) -> torch.Tensor:
    """Prepare background set as tensor for SHAP explainer."""
    background_tensors = []

    for image_np in background_images:
        tensor = _preprocess_image_shap(image_np, target_size)
        background_tensors.append(tensor)

    if not background_tensors:
        raise ValueError(
            "Background set is empty — no images could be loaded. "
            "Check that validation_images contains valid file paths."
        )

    # Stack into single tensor
    background_tensor = torch.cat(background_tensors, dim=0).to(device)

    return background_tensor


def _extract_shap_target(outputs: Any) -> torch.Tensor:
    """Extract target value for SHAP computation."""
    # Handle different YOLO output formats
    if isinstance(outputs, dict):
        # YOLO train mode returns dict - extract prediction tensor
        # YOLO26 NMS-free head returns {'one2many', 'one2one'}
        if "pred" in outputs:
            pred = outputs["pred"]
        elif "one" in outputs:
            pred = outputs["one"]
        elif "one2one" in outputs:
            # YOLO26 dual-head: use one2one predictions
            pred = outputs["one2one"]
            logger.debug("SHAP: Using 'one2one' predictions from YOLO26 dual-head")
        elif "one2many" in outputs:
            # Fallback to one2many if one2one not available
            pred = outputs["one2many"]
            logger.debug("SHAP: Using 'one2many' predictions from YOLO26 dual-head")
        else:
            # Try to find any tensor in the dict
            pred = None
            for key, value in outputs.items():
                if isinstance(value, torch.Tensor):
                    pred = value
                    logger.debug(f"SHAP: Using dict key '{key}' for target extraction")
                    break
            if pred is None:
                logger.warning(f"SHAP: No tensor found in dict output, keys: {outputs.keys()}")
                # Fallback to ones
                return torch.ones(1, 1, device="cpu")
    elif isinstance(outputs, (list, tuple)):
        pred = outputs[0] if len(outputs) > 0 else outputs
    else:
        pred = outputs

    if isinstance(pred, torch.Tensor):
        if pred.dim() == 3:  # YOLO eval mode: [batch, 4+nc, N_anchors]
            # Class scores are at pred[:, 4:, :]; sum over all classes and anchor
            # positions so the output scalar is sensitive to every detected object.
            class_scores = pred[:, 4:, :]  # [B, nc, N_anchors]
            return class_scores.sum(dim=(1, 2)).unsqueeze(1)  # [B, 1]
        elif pred.dim() == 2:  # [batch, features]
            # Return maximum value per batch as 2D tensor
            max_vals = torch.max(pred, dim=1)[0]
            return max_vals.unsqueeze(1)
        else:
            # Return as 2D tensor
            return pred.view(pred.shape[0], -1)
    else:
        # Fallback - infer device from pred to avoid device mismatch errors
        batch_size = len(outputs) if isinstance(outputs, (list, tuple)) else 1
        # Extract device from pred (handle list/tuple by using first element)
        inferred_device = "cpu"
        if isinstance(pred, (list, tuple)):
            first = pred[0] if len(pred) > 0 else None
            if first is not None and hasattr(first, "device"):
                inferred_device = first.device
        elif pred is not None and hasattr(pred, "device"):
            inferred_device = pred.device
        # Return 2D tensor: [batch_size, 1]
        return torch.ones(batch_size, 1, device=inferred_device)


def _save_shap_visualization(
    image_np: np.ndarray,
    attribution_np: np.ndarray,
    image_path: str,
    output_dir: str,
) -> str:
    """Save SHAP visualization as a 4-panel matplotlib figure.

    Panels:
      1. Original image
      2. Signed attribution map (RdBu_r, red = supports detection, blue = suppresses)
      3. Positive-only contributions (Reds — what the model focuses on)
      4. Overlay: positive attribution heatmap blended onto original image
    """
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # Collapse channel dimension if still 3-D (C, H, W or H, W, C)
    if attribution_np.ndim == 3:
        if attribution_np.shape[0] in (1, 3):  # (C, H, W)
            attr_2d = attribution_np.sum(axis=0).astype(np.float32)
        else:  # (H, W, C)
            attr_2d = attribution_np.sum(axis=-1).astype(np.float32)
    else:
        attr_2d = attribution_np.astype(np.float32)

    # Resize to match original image if needed
    h, w = image_np.shape[:2]
    if attr_2d.shape != (h, w):
        attr_2d = cv2.resize(attr_2d, (w, h), interpolation=cv2.INTER_LINEAR)

    # Symmetric colour limits for the diverging map (centred at 0)
    abs_max = float(np.abs(attr_2d).max())
    if abs_max == 0:
        abs_max = 1.0  # avoid degenerate colourbar
    vlim = abs_max

    # Positive-only map for panels 3 and 4
    pos_attr = np.maximum(attr_2d, 0)
    pos_max = float(pos_attr.max()) or 1.0

    # Ensure image is RGB uint8
    if image_np.ndim == 2:
        img_rgb = np.stack([image_np] * 3, axis=-1).astype(np.uint8)
    elif image_np.shape[-1] == 1:
        img_rgb = np.concatenate([image_np] * 3, axis=-1).astype(np.uint8)
    else:
        img_rgb = image_np.astype(np.uint8)

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))

    # Panel 1: original image
    axes[0].imshow(img_rgb)
    axes[0].set_title("Original", fontsize=13)
    axes[0].axis("off")

    # Panel 2: signed attribution — RdBu_r diverging around 0
    im2 = axes[1].imshow(attr_2d, cmap="RdBu_r", vmin=-vlim, vmax=vlim)
    axes[1].set_title("Signed Attribution\n(red = supports, blue = suppresses)", fontsize=11)
    axes[1].axis("off")
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    # Panel 3: positive-only contributions
    im3 = axes[2].imshow(pos_attr, cmap="Reds", vmin=0, vmax=pos_max)
    axes[2].set_title("Positive Contributions", fontsize=11)
    axes[2].axis("off")
    fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

    # Panel 4: overlay — positive attribution as warm heatmap on original
    axes[3].imshow(img_rgb)
    axes[3].imshow(pos_attr, cmap="Reds", alpha=0.55, vmin=0, vmax=pos_max)
    axes[3].set_title("Overlay\n(positive attribution on image)", fontsize=11)
    axes[3].axis("off")

    plt.tight_layout()

    image_name = Path(image_path).stem
    output_path = output_dir_path / f"{image_name}_shap.png"
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)

    return str(output_path)
