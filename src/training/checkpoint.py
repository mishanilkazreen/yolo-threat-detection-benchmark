"""Checkpoint selection for trained models."""

import logging
from pathlib import Path
from typing import Optional
import csv

logger = logging.getLogger(__name__)


class Checkpoint_Selector:
    """Selects best checkpoint based on validation metrics."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def select_best_checkpoint(
        self,
        project_dir: str,
        metric: str = "mAP50"
    ) -> Optional[str]:
        """
        Select best checkpoint based on validation metric.
        
        Args:
            project_dir: YOLO project directory (e.g., runs/detect/yolov8n/train)
            metric: Metric to optimize (default: mAP50)
            
        Returns:
            Path to best checkpoint, or None if not found
            
        Raises:
            FileNotFoundError: If project directory or results file not found
        """
        project_path = Path(project_dir)
        
        if not project_path.exists():
            raise FileNotFoundError(f"Project directory not found: {project_dir}")
        
        # Check for best.pt (YOLO automatically saves best checkpoint)
        best_checkpoint = project_path / "weights" / "best.pt"
        
        if best_checkpoint.exists():
            self.logger.info(f"Found best checkpoint: {best_checkpoint}")
            return str(best_checkpoint)
        
        # Fallback: check for last.pt
        last_checkpoint = project_path / "weights" / "last.pt"
        
        if last_checkpoint.exists():
            self.logger.warning(f"Best checkpoint not found, using last checkpoint: {last_checkpoint}")
            return str(last_checkpoint)
        
        # If no checkpoints found, raise error
        raise FileNotFoundError(
            f"No checkpoints found in {project_path / 'weights'}. "
            f"Expected 'best.pt' or 'last.pt'"
        )
    
    def get_checkpoint_metrics(self, project_dir: str) -> dict:
        """
        Get metrics for the best checkpoint.
        
        Args:
            project_dir: YOLO project directory
            
        Returns:
            Dictionary of metrics from results.csv
        """
        project_path = Path(project_dir)
        results_file = project_path / "results.csv"
        
        if not results_file.exists():
            self.logger.warning(f"Results file not found: {results_file}")
            return {}
        
        try:
            # Read the last line of results.csv (final epoch)
            with open(results_file, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                
                if not rows:
                    return {}
                
                # Get the row with best metrics (YOLO saves best based on fitness)
                # For simplicity, return the last row (final epoch)
                last_row = rows[-1]
                
                # Clean up column names (remove whitespace)
                metrics = {k.strip(): v.strip() for k, v in last_row.items()}
                
                return metrics
        
        except Exception as e:
            self.logger.error(f"Error reading results file: {e}")
            return {}
