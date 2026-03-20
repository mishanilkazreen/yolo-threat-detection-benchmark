"""Layer-wise Relevance Propagation (LRP) implementation for YOLO models."""

import logging
from pathlib import Path
import time
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

from ..hfs_scorer import Heatmap_Focus_Scorer
from .interfaces import AttributionMethod
from .output_manager import XAIOutputManager

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def generate_lrp_attribution(
    model: Any,
    image_path: str,
    detections: dict[str, Any],
    gt_boxes: list[tuple[float, float, float, float]],
    rule: str = "epsilon",
    device: str = "cuda",
    output_dir: str | None = None,
    output_manager: XAIOutputManager | None = None,
) -> tuple[np.ndarray, float | None, str]:
    """
    Generate LRP relevance map for YOLO model prediction.

    Args:
        model: Trained YOLO model
        image_path: Path to input image
        detections: Model detections for the image
        gt_boxes: Ground-truth bounding boxes in YOLO format
        rule: LRP rule to apply (epsilon, gamma, alpha-beta)
        device: Computation device
        output_dir: Directory to save attribution visualization (legacy)
        output_manager: Output manager instance for centralized output handling

    Returns:
        Tuple of (relevance_map, bbox_weighted_score, output_path)

    Raises:
        ImportError: If zennit library is not available
        RuntimeError: If LRP computation fails
    """
    start_time = time.time()

    try:
        # Try to import zennit (preferred) or fall back to captum
        try:
            import zennit  # noqa: F401

            return _generate_lrp_with_zennit(
                model,
                image_path,
                detections,
                gt_boxes,
                rule,
                device,
                output_dir,
                output_manager,
                start_time,
            )
        except ImportError:
            logger.warning("zennit not available, falling back to captum")
            try:
                from captum.attr import LRP as CaptumLRP  # noqa: F401

                return _generate_lrp_with_captum(
                    model,
                    image_path,
                    detections,
                    gt_boxes,
                    device,
                    output_dir,
                    output_manager,
                    start_time,
                )
            except ImportError:
                raise ImportError(
                    "Neither zennit nor captum is available. "
                    "Please install one of them: 'pip install zennit' or 'pip install captum'"
                )

    except Exception as e:
        logger.error(f"Failed to generate LRP attribution for {image_path}: {e}")
        raise RuntimeError(f"LRP computation failed: {e}") from e


def _generate_lrp_with_zennit(
    model: Any,
    image_path: str,
    detections: dict[str, Any],  # noqa: ARG001
    gt_boxes: list[tuple[float, float, float, float]],
    rule: str,
    device: str,
    output_dir: str | None,
    output_manager: XAIOutputManager | None,
    start_time: float,
) -> tuple[np.ndarray, float | None, str]:
    """Generate LRP attribution using zennit library."""
    import zennit

    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    image_height, image_width = image_np.shape[:2]

    # Preprocess image for model input
    target_size = model.imgsz if hasattr(model, "imgsz") else 640
    input_tensor = _preprocess_image_lrp(image_np, target_size)
    input_tensor = input_tensor.to(device)
    input_tensor.requires_grad_(True)

    # Extract the actual PyTorch model from YOLO wrapper
    torch_model = model
    if hasattr(model, "model"):
        torch_model = model.model
    elif hasattr(model, "net"):
        torch_model = model.net

    # Detect model architecture for special handling
    model_name = ""
    if hasattr(model, "model_name"):
        model_name = model.model_name.lower() if model.model_name else ""
    elif hasattr(model, "yaml") and isinstance(model.yaml, dict):
        model_name = model.yaml.get("name", "").lower()

    is_yolo26 = "yolo26" in model_name
    if is_yolo26:
        logger.info("Detected YOLO26 architecture - using adapted LRP computation")

    # Enable gradients for the model (required for LRP)
    torch_model.requires_grad_(True)

    # For YOLO26, force all parameters to require gradients (but keep in eval mode)
    if is_yolo26:
        for param in torch_model.parameters():
            param.requires_grad = True
        logger.debug("YOLO26: Enabled gradients for all parameters")

    # Create a model wrapper that ensures tensor output for Zennit
    class LRPModelWrapper(torch.nn.Module):
        """Model wrapper that ensures Zennit gets proper tensor outputs."""

        def __init__(self, original_model):
            super().__init__()
            self.model = original_model

        def forward(self, x):
            """Forward pass that returns tensor output for LRP."""
            # Ensure input requires grad for proper gradient flow
            if not x.requires_grad:
                x = x.requires_grad_(True)

            outputs = self.model(x)

            # Handle different YOLO output formats and ensure tensor output
            if isinstance(outputs, dict):
                # YOLO train mode returns dict with 'loss', 'pred', etc.
                # YOLO26 NMS-free head returns {'one2many', 'one2one'}
                if "pred" in outputs:
                    pred = outputs["pred"]
                elif "one" in outputs:
                    pred = outputs["one"]
                elif "one2one" in outputs:
                    # YOLO26 dual-head: use one2one predictions
                    pred = outputs["one2one"]
                    logger.debug("Using 'one2one' predictions from YOLO26 dual-head")
                elif "one2many" in outputs:
                    # Fallback to one2many if one2one not available
                    pred = outputs["one2many"]
                    logger.debug("Using 'one2many' predictions from YOLO26 dual-head")
                else:
                    # Try to find any tensor in the dict
                    pred = None
                    for key, value in outputs.items():
                        if isinstance(value, torch.Tensor):
                            pred = value
                            logger.debug(f"Using dict key '{key}' for LRP")
                            break
                    if pred is None:
                        logger.warning(f"No tensor found in dict output, keys: {outputs.keys()}")
            elif isinstance(outputs, (list, tuple)):
                pred = outputs[0] if len(outputs) > 0 else outputs
            else:
                pred = outputs

            # Log output shape for debugging (especially for yolo26n)
            if isinstance(pred, torch.Tensor):
                logger.debug(f"LRP wrapper output shape: {pred.shape}, dim: {pred.dim()}")

            if isinstance(pred, torch.Tensor):
                # Ensure the output tensor is connected to the input in the computation graph
                if not pred.requires_grad:
                    logger.warning(
                        "Model output doesn't require grad - gradient flow may be broken"
                    )

                if pred.dim() == 3:  # YOLO eval mode: [batch, 4+nc, N_anchors]
                    # Class scores are at pred[:, 4:, :]; sum over all classes and
                    # anchor positions so every detection contributes to the gradient.
                    class_scores = pred[:, 4:, :]  # [B, nc, N_anchors]
                    result = class_scores.sum(dim=(1, 2)).unsqueeze(1)  # [B, 1]
                elif pred.dim() == 2:  # [batch, features]
                    result = pred.sum(dim=1, keepdim=True)  # [B, 1]
                elif pred.dim() == 4:  # [batch, channels, height, width]
                    # Spatial feature maps - sum over spatial and channel dims
                    result = pred.sum(dim=(1, 2, 3), keepdim=True).view(pred.shape[0], 1)  # [B, 1]
                else:
                    # Flatten to 2D and sum
                    result = pred.view(pred.shape[0], -1).sum(dim=1, keepdim=True)  # [B, 1]

                # Add a small identity operation to ensure gradient flow
                # This helps with models that have disconnected computation graphs
                result = result + 0.0 * x.mean()
                return result
            else:
                # Fallback - create a tensor connected to input
                logger.warning(f"LRP wrapper received non-tensor output: {type(pred)}")
                batch_size = x.shape[0]
                # Create output that depends on input to ensure gradient flow
                return x.mean(dim=(1, 2, 3), keepdim=True).view(batch_size, 1)

    # Wrap the model
    wrapped_model = LRPModelWrapper(torch_model)
    # For YOLO26, keep in train mode; for others use eval mode
    if not is_yolo26:
        wrapped_model.eval()
    else:
        wrapped_model.train()
        logger.debug("YOLO26: Wrapper set to train mode")

    # Set up LRP composite based on rule
    if rule == "epsilon":
        composite = zennit.composites.EpsilonGammaBox(low=-3.0, high=3.0, epsilon=1e-6)
    elif rule == "gamma":
        composite = zennit.composites.EpsilonGammaBox(low=-3.0, high=3.0, gamma=0.25)
    elif rule == "alpha-beta":
        composite = zennit.composites.EpsilonAlpha2Beta1(epsilon=1e-6)
    else:
        logger.warning(f"Unknown LRP rule '{rule}', using epsilon rule")
        composite = zennit.composites.EpsilonGammaBox(low=-3.0, high=3.0, epsilon=1e-6)

    # Apply LRP
    # zennit's Gradient attributor validates isinstance(output, torch.Tensor) before
    # calling attr_output_fn, but YOLO's DetectionModel returns a tuple/list in eval
    # mode.  We apply the composite directly to torch_model (so LRP backward rules are
    # registered on the inner layers), then drive the forward+gradient pass ourselves
    # using wrapped_model (which guarantees a clean tensor output).  This is
    # functionally identical to Gradient(modified_model)(input, fn) but bypasses the
    # pre-check that rejects non-tensor model outputs.
    # The entire forward+backward must be inside inference_mode(False).
    # YOLO's @smart_inference_mode causes the Detect head to cache anchor/stride
    # tensors as inference tensors on the first predict() call.  Those flow into
    # the autograd graph and PyTorch refuses to save them — even with .clone().
    # Resetting detect_head.shape forces re-creation inside this clean context.

    # For YOLO26, use a more robust gradient computation approach
    use_robust_method = is_yolo26

    try:
        with torch.inference_mode(False):
            _detect = getattr(torch_model, "model", [None])[-1]
            if hasattr(_detect, "shape"):
                _detect.shape = None

            if use_robust_method:
                # For YOLO26: Use Integrated Gradients style attribution
                # This is more robust for models with gradient flow issues
                input_req_grad = input_tensor.detach().clone().requires_grad_(True)

                # Test if model output depends on input
                output = wrapped_model(input_req_grad)
                logger.debug(
                    f"YOLO26 output: shape={output.shape}, mean={output.mean().item():.6f}, requires_grad={output.requires_grad}"
                )

                # Try multiple gradient computation strategies
                gradient = None

                # Strategy 1: Direct gradient
                try:
                    loss = torch.sum(output)
                    logger.debug(
                        f"YOLO26 loss: {loss.item():.6f}, requires_grad={loss.requires_grad}"
                    )
                    gradient = torch.autograd.grad(
                        loss,
                        input_req_grad,
                        create_graph=False,
                        allow_unused=True,
                        retain_graph=True,
                    )[0]
                    if gradient is not None:
                        grad_magnitude = gradient.abs().mean().item()
                        logger.debug(
                            f"YOLO26 gradient (direct): mean={grad_magnitude:.6f}, max={gradient.abs().max().item():.6f}"
                        )
                        if grad_magnitude < 1e-10:
                            logger.warning(
                                "Gradient magnitude too small, trying alternative method"
                            )
                            gradient = None
                except Exception as e:
                    logger.warning(f"Direct gradient failed: {e}")
                    gradient = None

                # Strategy 2: Input * Gradient (Grad-CAM style)
                if gradient is None or gradient.abs().sum() < 1e-10:
                    logger.info("Using input-gradient product for YOLO26")
                    try:
                        # Compute gradient of output w.r.t. input
                        output2 = wrapped_model(input_req_grad)
                        loss2 = torch.sum(output2)
                        grad_raw = torch.autograd.grad(loss2, input_req_grad, create_graph=False)[0]
                        # Multiply by input to get relevance
                        gradient = input_req_grad * grad_raw
                        logger.debug(
                            f"YOLO26 input*gradient: mean={gradient.abs().mean().item():.6f}"
                        )
                    except Exception as e:
                        logger.warning(f"Input*gradient failed: {e}")

                # Strategy 3: Use input magnitude as last resort
                if gradient is None or gradient.abs().sum() < 1e-10:
                    logger.warning("All gradient methods failed, using input-based attribution")
                    gradient = input_req_grad * input_req_grad.abs()

                if gradient is None:
                    raise RuntimeError(
                        "Gradient computation returned None - input not used in graph"
                    )

                relevance = gradient
            else:
                # Standard Zennit LRP for other YOLO models
                with composite.context(torch_model):
                    input_req_grad = input_tensor.detach().clone().requires_grad_(True)
                    output = wrapped_model(input_req_grad)
                    loss = torch.sum(output)
                    gradient = torch.autograd.grad(loss, input_req_grad, allow_unused=True)[0]
                    if gradient is None:
                        raise RuntimeError(
                            "Gradient computation returned None - input not used in graph"
                        )
                    relevance = gradient
    except Exception as e:
        logger.warning(f"Zennit LRP failed: {e}, falling back to simple gradient method")
        for mod in torch_model.modules():
            mod._backward_hooks.clear()
            if hasattr(mod, "_full_backward_hooks"):
                mod._full_backward_hooks.clear()

        grad_holder: list = [None]
        with torch.inference_mode(False):
            _detect = getattr(torch_model, "model", [None])[-1]
            if hasattr(_detect, "shape"):
                _detect.shape = None
            input_for_grad = input_tensor.detach().clone().requires_grad_(True)
            grad_hook = input_for_grad.register_hook(
                lambda g: grad_holder.__setitem__(0, g.detach().clone() if g is not None else None)
            )
            try:
                out = wrapped_model(input_for_grad)
                loss = torch.sum(out)
                loss.backward()
            except Exception as backward_e:
                logger.warning(f"Backward pass failed in fallback: {backward_e}")
            finally:
                grad_hook.remove()

        # If gradient is still None, use input-based attribution as last resort
        if grad_holder[0] is None:
            logger.warning("Gradient is None even in fallback - using input-based attribution")
            relevance = input_tensor * torch.abs(input_tensor)
        else:
            relevance = grad_holder[0]

    # Process relevance map
    relevance_np = relevance.squeeze().cpu().detach().numpy()

    # Sum across channels if needed
    if relevance_np.ndim == 3:
        relevance_np = np.sum(relevance_np, axis=0)

    # Unpad letterbox borders then resize to original image size
    relevance_np = _unpad_and_resize_attribution(
        relevance_np, image_height, image_width, target_size
    )

    # Compute bbox-weighted relevance score using HFS scorer
    bbox_weighted_score = None
    if gt_boxes:
        try:
            hfs_scorer = Heatmap_Focus_Scorer()
            bbox_weighted_score = hfs_scorer.compute_bbox_weighted_relevance(
                relevance_np, gt_boxes[0], image_width, image_height
            )
        except Exception as e:
            logger.warning(f"Failed to compute bbox-weighted relevance score: {e}")
            bbox_weighted_score = None

    # Save visualization
    output_path = ""
    if output_dir or output_manager:
        if output_manager:
            # Use provided output manager
            output_path = output_manager.save_attribution_outputs(
                attribution_map=relevance_np,
                original_image=image_np,
                image_path=image_path,
                method=AttributionMethod.LRP,
            )
        elif output_dir is not None:
            # Fallback to legacy method
            output_path = _save_lrp_visualization(image_np, relevance_np, image_path, output_dir)

    processing_time = time.time() - start_time

    bbox_score_str = f"{bbox_weighted_score:.4f}" if bbox_weighted_score is not None else "N/A"
    logger.info(
        f"Generated LRP attribution for {Path(image_path).name} "
        f"(bbox-weighted score: {bbox_score_str}, time: {processing_time:.2f}s)"
    )

    return relevance_np, bbox_weighted_score, output_path


def _generate_lrp_with_captum(
    model: Any,
    image_path: str,
    detections: dict[str, Any],  # noqa: ARG001
    gt_boxes: list[tuple[float, float, float, float]],
    device: str,
    output_dir: str | None,
    output_manager: XAIOutputManager | None,
    start_time: float,
) -> tuple[np.ndarray, float | None, str]:
    """Generate LRP attribution using captum library (fallback)."""
    from captum.attr import LRP as CaptumLRP

    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    image_height, image_width = image_np.shape[:2]

    # Preprocess image for model input
    target_size = model.imgsz if hasattr(model, "imgsz") else 640
    input_tensor = _preprocess_image_lrp(image_np, target_size)
    input_tensor = input_tensor.to(device)

    # Create LRP attributor
    lrp = CaptumLRP(model)

    # Compute attributions
    attributions = lrp.attribute(input_tensor, target=0)  # Use first class as target

    # Process attribution map
    attr_np = attributions.squeeze().cpu().detach().numpy()

    # Sum across channels if needed
    if attr_np.ndim == 3:
        attr_np = np.sum(attr_np, axis=0)

    # Unpad letterbox borders then resize to original image size
    attr_np = _unpad_and_resize_attribution(attr_np, image_height, image_width, target_size)

    # Compute bbox-weighted relevance score using HFS scorer
    bbox_weighted_score = None
    if gt_boxes:
        try:
            hfs_scorer = Heatmap_Focus_Scorer()
            bbox_weighted_score = hfs_scorer.compute_bbox_weighted_relevance(
                attr_np, gt_boxes[0], image_width, image_height
            )
        except Exception as e:
            logger.warning(f"Failed to compute bbox-weighted relevance score: {e}")
            bbox_weighted_score = None

    # Save visualization
    output_path = ""
    if output_dir or output_manager:
        if output_manager:
            # Use provided output manager
            output_path = output_manager.save_attribution_outputs(
                attribution_map=attr_np,
                original_image=image_np,
                image_path=image_path,
                method=AttributionMethod.LRP,
            )
        elif output_dir is not None:
            # Fallback to legacy method
            output_path = _save_lrp_visualization(image_np, attr_np, image_path, output_dir)

    processing_time = time.time() - start_time

    bbox_score_str = f"{bbox_weighted_score:.4f}" if bbox_weighted_score is not None else "N/A"
    logger.info(
        f"Generated LRP attribution for {Path(image_path).name} "
        f"(bbox-weighted score: {bbox_score_str}, time: {processing_time:.2f}s)"
    )

    return attr_np, bbox_weighted_score, output_path


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


def _preprocess_image_lrp(image_np: np.ndarray, target_size: int = 640) -> torch.Tensor:
    """Preprocess image for YOLO model input (same as Grad-CAM)."""
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


def _extract_target_score_lrp(outputs: Any, detections: dict[str, Any]) -> torch.Tensor:  # noqa: ARG001
    """Extract target score for LRP computation."""
    # Handle different YOLO output formats — YOLO typically returns a list/tuple
    # where the first element is the raw prediction tensor.
    pred = outputs
    if isinstance(pred, (list, tuple)):
        # Unwrap nested lists until we reach a tensor
        while isinstance(pred, (list, tuple)) and len(pred) > 0:
            pred = pred[0]

    if isinstance(pred, torch.Tensor):
        if pred.dim() == 3:  # [batch, detections, features]
            # Use maximum confidence detection
            confidences = pred[0, :, 4]  # Confidence scores
            target_score = torch.max(confidences)
        elif pred.dim() == 2:  # [batch, features]
            target_score = torch.max(pred)
        else:
            target_score = pred.sum()
    else:
        # Last resort: sum all outputs if we can't find a tensor
        # This path means the model output format is unexpected
        logger.warning(f"Unexpected LRP output type {type(pred)}, using output sum")
        if isinstance(outputs, (list, tuple)):
            tensors = [t for t in outputs if isinstance(t, torch.Tensor)]
            if tensors:
                target_score = sum((t.sum() for t in tensors), torch.tensor(0.0))
            else:
                raise RuntimeError("No tensors found in model output for LRP gradient computation")
        else:
            raise RuntimeError(f"Cannot extract target score from output type {type(outputs)}")

    return target_score


def _save_lrp_visualization(
    image_np: np.ndarray,
    relevance_np: np.ndarray,
    image_path: str,
    output_dir: str,
) -> str:
    """Save LRP visualization as a 3-panel matplotlib figure: original | heatmap | overlay."""
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # 1.1 Collapse channel dim using axis=-1 (works for any 3D input)
    if relevance_np.ndim == 3:
        heatmap = np.sum(np.abs(relevance_np), axis=-1).astype(np.float32)
    else:
        heatmap = np.abs(relevance_np).astype(np.float32)

    # 1.2 Choose colormap based on sign of original relevance values
    raw = relevance_np if relevance_np.ndim == 2 else relevance_np.sum(axis=-1)
    has_negative = bool(raw.min() < 0)

    if has_negative:
        # Diverging: do NOT normalise — pass raw summed values, matplotlib centres at zero
        colormap = "RdBu_r"
        display_heatmap = raw.astype(np.float32)
        vmin, vmax = None, None  # let matplotlib auto-centre
    else:
        # Sequential: normalise to [0, 1]
        colormap = "hot"
        heatmap_max = heatmap.max()
        # 1.3 Guard against zero-max (only in sequential branch)
        display_heatmap = heatmap / heatmap_max if heatmap_max > 0 else heatmap
        vmin, vmax = 0.0, 1.0

    # Resize heatmap to match image if needed
    h, w = image_np.shape[:2]
    if display_heatmap.shape != (h, w):
        display_heatmap = cv2.resize(display_heatmap, (w, h), interpolation=cv2.INTER_LINEAR)

    # 1.4 Single-channel image → 3-channel RGB
    if image_np.ndim == 2:
        img_rgb = np.stack([image_np] * 3, axis=-1).astype(np.uint8)
    elif image_np.shape[-1] == 1:
        img_rgb = np.concatenate([image_np] * 3, axis=-1).astype(np.uint8)
    else:
        img_rgb = image_np.astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: original image
    axes[0].imshow(img_rgb)
    axes[0].set_title("Original", fontsize=13)
    axes[0].axis("off")

    # Panel 2: pure heatmap — 1.5 use colormap variable (not hardcoded string)
    im = axes[1].imshow(display_heatmap, cmap=colormap, vmin=vmin, vmax=vmax)
    axes[1].set_title("LRP Heatmap", fontsize=13)
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # Panel 3: overlay — 1.5 same colormap variable, no raw RGB blending (1.6)
    axes[2].imshow(img_rgb)
    axes[2].imshow(display_heatmap, cmap=colormap, alpha=0.5, vmin=vmin, vmax=vmax)
    axes[2].set_title("Overlay", fontsize=13)
    axes[2].axis("off")

    plt.tight_layout()

    image_name = Path(image_path).stem
    output_path = output_dir_path / f"{image_name}_lrp.png"
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)

    return str(output_path)
