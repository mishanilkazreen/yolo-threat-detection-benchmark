"""Annotation parsing module for YOLO format annotations."""

import logging
from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BoundingBox:
    """Bounding box in YOLO format (normalized coordinates)."""
    center_x: float  # Normalized [0, 1]
    center_y: float  # Normalized [0, 1]
    width: float     # Normalized [0, 1]
    height: float    # Normalized [0, 1]
    
    def __post_init__(self):
        """Validate bounding box coordinates."""
        if not (0 <= self.center_x <= 1):
            raise ValueError(f"center_x must be in [0, 1], got {self.center_x}")
        if not (0 <= self.center_y <= 1):
            raise ValueError(f"center_y must be in [0, 1], got {self.center_y}")
        if not (0 <= self.width <= 1):
            raise ValueError(f"width must be in [0, 1], got {self.width}")
        if not (0 <= self.height <= 1):
            raise ValueError(f"height must be in [0, 1], got {self.height}")


@dataclass
class Annotation:
    """Single annotation in YOLO format."""
    class_id: int
    bbox: BoundingBox


class Annotation_Parser:
    """Parses YOLO annotation files to Annotation objects."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def parse_annotation_file(self, annotation_path: str) -> Tuple[List[Annotation], List[str]]:
        """
        Parse a YOLO annotation file.
        
        Args:
            annotation_path: Path to YOLO annotation file (.txt)
            
        Returns:
            Tuple of (list of Annotation objects, list of error messages)
        """
        annotations = []
        errors = []
        
        path = Path(annotation_path)
        if not path.exists():
            errors.append(f"Annotation file not found: {annotation_path}")
            return annotations, errors
        
        # Read file
        try:
            with open(path, 'r') as f:
                lines = f.readlines()
        except Exception as e:
            errors.append(f"Failed to read file: {e}")
            return annotations, errors
        
        # Parse each line
        for line_num, line in enumerate(lines, start=1):
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Parse annotation
            try:
                annotation = self._parse_annotation_line(line, line_num)
                annotations.append(annotation)
            except ValueError as e:
                errors.append(f"Line {line_num}: {e}")
        
        return annotations, errors
    
    def _parse_annotation_line(self, line: str, line_num: int) -> Annotation:
        """
        Parse a single annotation line.
        
        YOLO format: class_id center_x center_y width height
        
        Args:
            line: Annotation line
            line_num: Line number for error reporting
            
        Returns:
            Annotation object
            
        Raises:
            ValueError: If line format is invalid
        """
        parts = line.split()
        
        if len(parts) != 5:
            raise ValueError(f"Expected 5 values (class_id center_x center_y width height), got {len(parts)}")
        
        try:
            class_id = int(parts[0])
            center_x = float(parts[1])
            center_y = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
        except ValueError as e:
            raise ValueError(f"Invalid numeric value: {e}")
        
        # Validate class_id
        if class_id < 0:
            raise ValueError(f"class_id must be non-negative, got {class_id}")
        
        # Create bounding box (validation happens in __post_init__)
        try:
            bbox = BoundingBox(
                center_x=center_x,
                center_y=center_y,
                width=width,
                height=height
            )
        except ValueError as e:
            raise ValueError(f"Invalid bounding box: {e}")
        
        return Annotation(class_id=class_id, bbox=bbox)
    
    def parse_annotation_string(self, annotation_str: str) -> Tuple[List[Annotation], List[str]]:
        """
        Parse annotation content from a string.
        
        Args:
            annotation_str: Annotation content as string
            
        Returns:
            Tuple of (list of Annotation objects, list of error messages)
        """
        annotations = []
        errors = []
        
        lines = annotation_str.strip().split('\n')
        
        for line_num, line in enumerate(lines, start=1):
            line = line.strip()
            
            if not line:
                continue
            
            try:
                annotation = self._parse_annotation_line(line, line_num)
                annotations.append(annotation)
            except ValueError as e:
                errors.append(f"Line {line_num}: {e}")
        
        return annotations, errors
