"""Metrics collection and evaluation for trained models."""

import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from ultralytics import YOLO
from .checkpoint import Checkpoint_Selector

logger = logging.getLogger(__name__)


class Metrics_Collector:
    """Collects and saves model performance metrics."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.checkpoint_selector = Checkpoint_Selector()
    
    def evaluate_and_save(
        self,
        project_dir: str,
        config_name: str,
        data_yaml: str,
        output_dir: str,
        run_id: Optional[int] = None,
        random_seed: Optional[int] = None,
        training_time: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Evaluate model and save metrics.
        
        Args:
            project_dir: YOLO training project directory
            config_name: Configuration name
            data_yaml: Path to data.yaml
            output_dir: Directory to save evaluation results
            run_id: Run identifier for multi-run experiments
            random_seed: Random seed used for training
            training_time: Training time in seconds
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.logger.info(f"Evaluating model from {project_dir}")
        
        # Select best checkpoint
        try:
            best_checkpoint = self.checkpoint_selector.select_best_checkpoint(project_dir)
        except FileNotFoundError as e:
            self.logger.error(f"Checkpoint selection failed: {e}")
            raise
        
        # Load model
        model = YOLO(best_checkpoint)
        
        # Run validation
        self.logger.info("Running validation...")
        val_start = time.time()
        results = model.val(data=data_yaml, verbose=False)
        val_time = time.time() - val_start
        
        # Extract metrics
        metrics = self._extract_metrics(
            results=results,
            model=model,
            config_name=config_name,
            best_checkpoint=best_checkpoint,
            run_id=run_id,
            random_seed=random_seed,
            training_time=training_time,
            val_time=val_time
        )
        
        # Save metrics
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if run_id is not None:
            metrics_file = output_path / f"run_{run_id}" / "evaluation_results.json"
            metrics_file.parent.mkdir(parents=True, exist_ok=True)
        else:
            metrics_file = output_path / "evaluation_results.json"
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        self.logger.info(f"Metrics saved to {metrics_file}")
        
        return metrics
    
    def _extract_metrics(
        self,
        results: Any,
        model: YOLO,
        config_name: str,
        best_checkpoint: str,
        run_id: Optional[int],
        random_seed: Optional[int],
        training_time: Optional[float],
        val_time: float
    ) -> Dict[str, Any]:
        """Extract metrics from validation results."""
        
        # Get model info
        model_info = self._get_model_info(model)
        
        # Build metrics dictionary
        metrics = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "config_name": config_name,
            "best_checkpoint": best_checkpoint,
            "model_info": model_info,
            "metrics": {
                "mAP50": float(results.box.map50) if hasattr(results.box, 'map50') else 0.0,
                "mAP50-95": float(results.box.map) if hasattr(results.box, 'map') else 0.0,
                "precision": float(results.box.mp) if hasattr(results.box, 'mp') else 0.0,
                "recall": float(results.box.mr) if hasattr(results.box, 'mr') else 0.0,
                "fitness": float(results.fitness) if hasattr(results, 'fitness') else 0.0,
            }
        }
        
        # Add per-class metrics if available
        if hasattr(results.box, 'maps'):
            metrics["per_class_metrics"] = {
                "mAP50_per_class": [float(x) for x in results.box.ap50],
                "mAP50-95_per_class": [float(x) for x in results.box.ap]
            }
        
        # Add run information
        if run_id is not None:
            metrics["run_id"] = run_id
        
        if random_seed is not None:
            metrics["random_seed"] = random_seed
        
        if training_time is not None:
            metrics["training_time_seconds"] = training_time
            metrics["training_time_minutes"] = training_time / 60.0
        
        metrics["validation_time_seconds"] = val_time
        
        return metrics
    
    def _get_model_info(self, model: YOLO) -> Dict[str, Any]:
        """Get model architecture information."""
        info = {}
        
        try:
            # Get model parameters
            total_params = sum(p.numel() for p in model.model.parameters())
            info["parameters"] = int(total_params)
            
            # Get model name
            if hasattr(model, 'ckpt_path'):
                info["model_name"] = Path(model.ckpt_path).stem
            
            # Try to get GFLOPs (may not be available)
            if hasattr(model.model, 'info'):
                model_info = model.model.info(verbose=False)
                if isinstance(model_info, dict) and 'GFLOPs' in model_info:
                    info["GFLOPs"] = float(model_info['GFLOPs'])
            
            # Get inference time (approximate)
            # This will be measured during actual inference
            info["inference_time_ms"] = None  # To be filled during inference
            
        except Exception as e:
            self.logger.warning(f"Could not extract all model info: {e}")
        
        return info
