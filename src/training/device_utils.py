"""
Device selection utilities for training.
"""

from collections.abc import Callable
import logging

import torch

logger = logging.getLogger(__name__)


def create_step_decay_callback(lr0: float, lrf: float, step_interval: int = 5) -> Callable:
    """Return a YOLO on_train_epoch_start callback that applies step-decay LR.

    Formula: LR[epoch] = lr0 * (lrf ** (epoch // step_interval))
    """

    def on_train_epoch_start(trainer):
        epoch = trainer.epoch
        expected_lr = lr0 * (lrf ** (epoch // step_interval))
        for param_group in trainer.optimizer.param_groups:
            param_group["lr"] = expected_lr
        if epoch % step_interval == 0 and epoch > 0:
            logger.info(
                "Step decay: LR *= %s at epoch %d (new LR: %.6f)",
                lrf,
                epoch,
                expected_lr,
            )

    return on_train_epoch_start


def get_model_gflops(model_name: str) -> float:
    """Return theoretical forward GFLOPs at 640x640 for the architecture."""
    name = model_name.lower()
    if "yolov8n" in name:
        return 8.7
    if "yolo11n" in name:
        return 6.5
    if "yolo12n" in name:
        return 6.5
    return 8.7


def select_device(device_config: str = "auto") -> str:
    """
    Select the appropriate device for training.

    Args:
        device_config: Device configuration string
            - "auto": Automatically select CUDA if available, otherwise CPU
            - "cuda": Force CUDA (will fail if not available)
            - "cpu": Force CPU
            - "0", "1", etc.: Specific CUDA device

    Returns:
        Device string to use for training
    """
    if device_config == "auto":
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            # Test if CUDA actually works by trying a simple operation
            try:
                test_tensor = torch.zeros(1).cuda()
                del test_tensor
                device = "cuda:0"
                print(f"CUDA available: Using {torch.cuda.get_device_name(0)}")
            except RuntimeError as e:
                print(f"CUDA detected but not functional: {e}")
                print("Falling back to CPU")
                print("\nTo fix CUDA issues:")
                print("1. Check your CUDA version: nvidia-smi")
                print("2. Install matching PyTorch:")
                print(
                    "   For CUDA 13.1: pip3 install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu130"
                )
                print(
                    "   For CUDA 12.1: pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121"
                )
                print(
                    "   For CUDA 11.8: pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118"
                )
                device = "cpu"
        else:
            device = "cpu"
            print("CUDA not available: Using CPU")
        return device

    if device_config.isdigit():
        if torch.cuda.is_available() and int(device_config) < torch.cuda.device_count():
            return f"cuda:{device_config}"
        else:
            logger.warning(
                "CUDA device %s requested but not available. Falling back to CPU.", device_config
            )
            return "cpu"

    if device_config.startswith("cuda"):
        if torch.cuda.is_available():
            return device_config
        else:
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            return "cpu"

    return device_config
