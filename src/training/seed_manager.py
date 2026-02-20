"""Seed management for reproducible experiments."""

import random
import numpy as np
import torch
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


class Seed_Manager:
    """Manages random seeds for reproducible multi-run experiments."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def set_seed(self, seed: int) -> None:
        """
        Set random seed for reproducibility across all libraries.
        
        Args:
            seed: Random seed value
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # Make CUDA operations deterministic
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        self.logger.info(f"Random seed set to: {seed}")
    
    def get_run_seed(self, run_id: int, seeds: Optional[List[int]] = None, base_seed: int = 42) -> int:
        """
        Get seed for a specific run.
        
        Args:
            run_id: Run identifier (0-indexed)
            seeds: Optional list of fixed seeds for each run
            base_seed: Base seed to use if seeds list not provided
            
        Returns:
            Seed value for the specified run
        """
        if seeds is not None and len(seeds) > run_id:
            seed = seeds[run_id]
            self.logger.debug(f"Using configured seed for run {run_id}: {seed}")
            return seed
        else:
            # Generate seed based on base_seed and run_id
            seed = base_seed + run_id * 1000
            self.logger.debug(f"Using generated seed for run {run_id}: {seed}")
            return seed
