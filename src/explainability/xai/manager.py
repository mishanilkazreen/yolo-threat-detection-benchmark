"""XAI Manager for orchestrating explainability methods."""

import logging
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from src.utils.output_manager import Output_Manager

from .config import XAIConfig
from .gradcam import generate_gradcam_attribution
from .interfaces import AttributionMethod, XAIBatchResult, XAIResult
from .lrp import generate_lrp_attribution
from .output_manager import XAIOutputManager
from .shap import create_background_set, generate_shap_attribution

logger = logging.getLogger(__name__)


class XAIManager:
    """Manager for coordinating XAI methods and processing batches of images."""

    def __init__(self, config: XAIConfig, output_manager: Output_Manager | None = None):
        """
        Initialize XAI Manager with configuration.

        Args:
            config: XAI configuration object
            output_manager: Optional Output_Manager instance for path construction.
                          If None, creates a new instance with default base directory.
        """
        self.config = config
        self.background_set: list[np.ndarray] | None = None
        self.output_manager: XAIOutputManager | None = None
        self.path_manager = output_manager if output_manager is not None else Output_Manager()
        self._setup_logging()

    def _setup_logging(self):
        """Set up logging for XAI processing."""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

    def is_enabled(self) -> bool:
        """Check if XAI processing is enabled."""
        return self.config.enabled

    def process_batch(
        self,
        model: Any,
        image_paths: list[str],
        detections: dict[str, Any],
        gt_boxes: dict[str, list[tuple[float, float, float, float]]],
        round_num: int,
        output_dir: str,
        validation_images: list[str] | None = None,
        model_name: str | None = None,
        round_name: str | None = None,
    ) -> XAIBatchResult | None:
        """
        Process a batch of images through enabled XAI methods.

        Args:
            model: Trained YOLO model
            image_paths: List of image paths to process
            detections: Dictionary mapping image paths to detection results
            gt_boxes: Dictionary mapping image paths to ground truth boxes
            round_num: Training round number
            output_dir: Base output directory for results (deprecated, use model_name/round_name)
            validation_images: List of validation images for SHAP background set
            model_name: Model identifier for Output_Manager path construction
            round_name: Round identifier for Output_Manager path construction

        Returns:
            XAIBatchResult with processing results, or None if XAI is disabled
        """
        if not self.is_enabled():
            return None

        start_time = time.time()
        enabled_methods = self.config.get_enabled_methods()

        if not enabled_methods:
            self.logger.info("No XAI methods enabled, skipping processing")
            return None

        self.logger.info(
            f"Processing {len(image_paths)} images with XAI methods: {', '.join(enabled_methods)}"
        )

        # Apply sample limit if configured
        if self.config.sample_limit is not None:
            image_paths = image_paths[: self.config.sample_limit]
            self.logger.info(f"Limited processing to {len(image_paths)} images")

        # Initialize background set for SHAP if needed
        if "shap" in enabled_methods and validation_images:
            self._initialize_background_set(validation_images)

        # Determine base output directory using Output_Manager if model_name is provided
        if model_name is not None:
            # Use Output_Manager for standardized path construction
            base_output_dir = str(
                self.path_manager.get_explainability_output_path(
                    model_name=model_name,
                    round_name=round_name,
                    method=None,  # Base directory without method subdirectory
                    create=True,
                )
            )
        else:
            # Fall back to legacy output_dir parameter
            base_output_dir = output_dir

        # Initialize output manager for this round
        if self.config.output_overlays or self.config.save_raw_attributions:
            self.output_manager = XAIOutputManager(
                base_output_dir=base_output_dir,
                save_overlays=self.config.output_overlays,
                save_raw=self.config.save_raw_attributions,
                round_num=round_num,
            )

        # Process each image
        results = []
        failed_images = []

        for image_path in image_paths:
            image_id = Path(image_path).stem

            try:
                # Get detections and ground truth for this image
                image_detections = detections.get(image_path, {})
                image_gt_boxes = gt_boxes.get(image_path, [])

                # Process with each enabled method
                for method in enabled_methods:
                    try:
                        result = self._process_single_image(
                            model,
                            image_path,
                            image_detections,
                            image_gt_boxes,
                            method,
                            base_output_dir,
                            image_id,
                        )
                        if result:
                            results.append(result)

                    except BaseException as e:
                        self.logger.error(f"Failed to process {image_path} with {method}: {e}")
                        import traceback

                        self.logger.debug(
                            f"Full traceback for {method} failure:\n{traceback.format_exc()}"
                        )
                        continue

            except Exception as e:
                self.logger.error(f"Failed to process image {image_path}: {e}")
                failed_images.append(image_id)
                continue
        aggregate_hfs = self._compute_aggregate_hfs(results)
        method_counts = self._compute_method_counts(results)
        total_processing_time = time.time() - start_time

        batch_result = XAIBatchResult(
            round_num=round_num,
            results=results,
            aggregate_hfs=aggregate_hfs,
            total_processing_time=total_processing_time,
            num_images_processed=len(image_paths) - len(failed_images),
            failed_images=failed_images,
        )

        # Save CSV logs if output manager is available
        if self.output_manager and results:
            try:
                # Extract config name from output_dir path
                config_name = Path(base_output_dir).name if base_output_dir else "unknown"

                # Extract model name for CSV logging
                model_name_for_csv = getattr(model, "model_name", None)
                if (
                    model_name_for_csv is None
                    and hasattr(model, "yaml")
                    and isinstance(model.yaml, dict)
                ):
                    model_name_for_csv = model.yaml.get("name", "")
                model_name_for_csv = model_name_for_csv or ""

                # Save individual HFS scores
                self.output_manager.save_hfs_metrics_csv(
                    results=results,
                    round_num=round_num,
                    config_name=config_name,
                    output_dir=base_output_dir,
                    model_name=model_name_for_csv,
                )

                # Prepare round data for cumulative CSV
                round_data = {
                    "round_num": round_num,
                    "aggregate_hfs": aggregate_hfs,
                    "method_counts": method_counts,
                    "num_images_processed": len(image_paths) - len(failed_images),
                    "total_processing_time": total_processing_time,
                }

                # Append to cumulative CSV
                self.output_manager.append_to_cumulative_hfs_csv(
                    round_data=round_data,
                    config_name=config_name,
                    output_dir=base_output_dir,
                )

            except Exception as e:
                self.logger.warning(f"Failed to save CSV logs: {e}")

        self.logger.info(
            f"Completed XAI processing: {len(results)} attributions generated, "
            f"{len(failed_images)} failures, {total_processing_time:.2f}s total"
        )

        return batch_result

    def _initialize_background_set(self, validation_images: list[str]):
        """Initialize SHAP background set if not already created."""
        if self.background_set is None:
            self.logger.info("Creating SHAP background set...")
            self.background_set = create_background_set(
                validation_images,
                self.config.background_set_size,
                seed=self.config.background_set_seed,
            )

    def _process_single_image(
        self,
        model: Any,
        image_path: str,
        detections: dict[str, Any],
        gt_boxes: list[tuple[float, float, float, float]],
        method: str,
        output_dir: str,
        image_id: str,
    ) -> XAIResult | None:
        """Process a single image with a specific XAI method."""
        start_time = time.time()

        # Determine device
        device = getattr(self.config, "device", None) or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        try:
            if method == "gradcam":
                target_layer = self._get_target_layer(model)
                attribution_map, hfs_score, output_path = generate_gradcam_attribution(
                    model=model,
                    image_path=image_path,
                    detections=detections,
                    gt_boxes=gt_boxes,
                    _target_layer=target_layer,
                    output_manager=self.output_manager,
                    _device=device,
                    target_class_names=getattr(self.config, "gradcam_target_classes", None),
                )
                attribution_method = AttributionMethod.GRADCAM

            elif method == "lrp":
                attribution_map, hfs_score, output_path = generate_lrp_attribution(
                    model=model,
                    image_path=image_path,
                    detections=detections,
                    gt_boxes=gt_boxes,
                    rule=self.config.lrp_rule,
                    output_manager=self.output_manager,
                    device=device,
                )
                attribution_method = AttributionMethod.LRP

            elif method == "shap":
                if self.background_set is None:
                    self.logger.warning("SHAP background set not initialized, skipping")
                    return None

                attribution_map, hfs_score, output_path = generate_shap_attribution(
                    model=model,
                    image_path=image_path,
                    detections=detections,
                    gt_boxes=gt_boxes,
                    background_set=self.background_set,
                    output_manager=self.output_manager,
                    device=device,
                )
                attribution_method = AttributionMethod.SHAP

            else:
                self.logger.error(f"Unknown XAI method: {method}")
                return None

        except Exception as e:
            self.logger.error(f"Failed to process {image_path} with {method}: {e}")
            return None

        processing_time = time.time() - start_time

        try:
            return XAIResult(
                method=attribution_method,
                attribution_map=attribution_map,
                hfs_score=hfs_score,
                output_path=output_path,
                processing_time=processing_time,
                image_id=image_id,
            )
        except Exception as e:
            self.logger.error(f"Failed to construct XAIResult for {image_path} with {method}: {e}")
            return None

    def _get_target_layer(self, model: Any) -> str:
        """Get target layer for the model architecture."""
        model_name = getattr(model, "model_name", None)
        if model_name is None and hasattr(model, "yaml") and isinstance(model.yaml, dict):
            model_name = model.yaml.get("name")

        if model_name is None:
            model_name = "yolov11"

        return self.config.get_target_layer(model_name, strict=True)

    def _compute_aggregate_hfs(self, results: list[XAIResult]) -> dict[str, float]:
        """Compute aggregate HFS scores by method."""
        method_hfs = {}

        for method in AttributionMethod:
            method_results = [r for r in results if r.method == method and r.hfs_score is not None]
            if method_results:
                hfs_scores = [r.hfs_score for r in method_results if r.hfs_score is not None]
                method_hfs[method.value] = float(np.mean(hfs_scores))

        return method_hfs

    def _compute_method_counts(self, results: list[XAIResult]) -> dict[str, int]:
        """Compute count of processed images by method."""
        method_counts = {}

        for method in AttributionMethod:
            method_results = [r for r in results if r.method == method and r.hfs_score is not None]
            method_counts[method.value] = len(method_results)

        return method_counts

    def generate_final_aggregate_report(
        self,
        all_round_data: list[dict],
        config_name: str,
        output_dir: str,
    ) -> str:
        """
        Generate final aggregate HFS report across all rounds.

        Args:
            all_round_data: List of round data dictionaries
            config_name: Configuration name for file naming
            output_dir: Output directory for the report

        Returns:
            Path to saved aggregate CSV file
        """
        if not self.output_manager:
            self.logger.warning("Output manager not initialized, cannot generate aggregate report")
            return ""

        try:
            return self.output_manager.save_aggregate_hfs_csv(
                aggregate_data=all_round_data,
                config_name=config_name,
                output_dir=output_dir,
            )
        except Exception as e:
            self.logger.error(f"Failed to generate final aggregate report: {e}")
            return ""
