"""Annotation serialization module for YOLO format annotations."""

import logging
from pathlib import Path

from .annotation_parser import Annotation

logger = logging.getLogger(__name__)


class Annotation_Serializer:
    """Serializes Annotation objects to YOLO annotation format."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def serialize_annotations(self, annotations: list[Annotation]) -> str:
        """
        Serialize Annotation objects to YOLO format string.

        Args:
            annotations: List of Annotation objects

        Returns:
            YOLO format annotation string (one line per annotation)
        """
        lines = []

        for annotation in annotations:
            line = self._serialize_annotation(annotation)
            lines.append(line)

        return "\n".join(lines)

    def _serialize_annotation(self, annotation: Annotation) -> str:
        """
        Serialize a single Annotation to YOLO format.

        YOLO format: class_id center_x center_y width height

        Args:
            annotation: Annotation object

        Returns:
            YOLO format annotation line
        """
        return (
            f"{annotation.class_id} "
            f"{annotation.bbox.center_x:.6f} "
            f"{annotation.bbox.center_y:.6f} "
            f"{annotation.bbox.width:.6f} "
            f"{annotation.bbox.height:.6f}"
        )

    def save_annotations(self, annotations: list[Annotation], output_path: str) -> None:
        """
        Save Annotation objects to a YOLO annotation file.

        Args:
            annotations: List of Annotation objects
            output_path: Path to output annotation file (.txt)
        """
        annotation_str = self.serialize_annotations(annotations)

        # Create parent directory if needed
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Write to file
        with open(path, "w", encoding="utf-8") as f:
            f.write(annotation_str)

        self.logger.debug(f"Saved {len(annotations)} annotations to {output_path}")
