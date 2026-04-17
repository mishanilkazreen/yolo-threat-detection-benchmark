"""Pytest configuration and fixtures."""

from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Add yolo_cam to path (cloned from YOLO-26-CAM repository)
yolo_cam_path = project_root / "yolo_cam"
if yolo_cam_path.exists():
    sys.path.insert(0, str(yolo_cam_path))
