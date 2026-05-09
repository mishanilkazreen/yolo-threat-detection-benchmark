"""Unit tests for Annotation_Serializer."""

from pathlib import Path
import tempfile

import pytest

from src.data.annotation_parser import Annotation, BoundingBox
from src.data.annotation_serializer import Annotation_Serializer


class TestAnnotationSerializer:
    """Tests for Annotation_Serializer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.serializer = Annotation_Serializer()

    def test_serialize_single_annotation(self):
        """Test serializing a single annotation."""
        annotation = Annotation(
            class_id=0, bbox=BoundingBox(center_x=0.5, center_y=0.5, width=0.3, height=0.4)
        )

        result = self.serializer.serialize_annotations([annotation])

        # Parse the result
        parts = result.split()
        assert len(parts) == 5
        assert int(parts[0]) == 0
        assert float(parts[1]) == pytest.approx(0.5, abs=1e-6)
        assert float(parts[2]) == pytest.approx(0.5, abs=1e-6)
        assert float(parts[3]) == pytest.approx(0.3, abs=1e-6)
        assert float(parts[4]) == pytest.approx(0.4, abs=1e-6)

    def test_serialize_multiple_annotations(self):
        """Test serializing multiple annotations."""
        annotations = [
            Annotation(
                class_id=0, bbox=BoundingBox(center_x=0.5, center_y=0.5, width=0.3, height=0.4)
            ),
            Annotation(
                class_id=1, bbox=BoundingBox(center_x=0.2, center_y=0.3, width=0.1, height=0.2)
            ),
            Annotation(
                class_id=2, bbox=BoundingBox(center_x=0.8, center_y=0.7, width=0.15, height=0.25)
            ),
        ]

        result = self.serializer.serialize_annotations(annotations)

        lines = result.split("\n")
        assert len(lines) == 3

        # Check first line
        parts = lines[0].split()
        assert int(parts[0]) == 0

        # Check second line
        parts = lines[1].split()
        assert int(parts[0]) == 1

        # Check third line
        parts = lines[2].split()
        assert int(parts[0]) == 2

    def test_serialize_empty_list(self):
        """Test serializing empty annotation list."""
        result = self.serializer.serialize_annotations([])
        assert result == ""

    def test_serialize_precision(self):
        """Test that serialization maintains 6 decimal places."""
        annotation = Annotation(
            class_id=0,
            bbox=BoundingBox(
                center_x=0.123456789, center_y=0.987654321, width=0.111111111, height=0.999999999
            ),
        )

        result = self.serializer.serialize_annotations([annotation])
        parts = result.split()

        # Check precision (6 decimal places)
        assert parts[1] == "0.123457"  # Rounded
        assert parts[2] == "0.987654"  # Rounded
        assert parts[3] == "0.111111"  # Rounded
        assert parts[4] == "1.000000"  # Rounded

    def test_save_annotations_to_file(self):
        """Test saving annotations to file."""
        annotations = [
            Annotation(
                class_id=0, bbox=BoundingBox(center_x=0.5, center_y=0.5, width=0.3, height=0.4)
            )
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_annotation.txt"
            self.serializer.save_annotations(annotations, str(output_path))

            # Verify file was created
            assert output_path.exists()

            # Verify content
            with open(output_path, encoding="utf-8") as f:
                content = f.read()

            parts = content.split()
            assert len(parts) == 5
            assert int(parts[0]) == 0

    def test_save_annotations_creates_parent_directory(self):
        """Test that save_annotations creates parent directories."""
        annotations = [
            Annotation(
                class_id=0, bbox=BoundingBox(center_x=0.5, center_y=0.5, width=0.3, height=0.4)
            )
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "subdir" / "nested" / "test_annotation.txt"
            self.serializer.save_annotations(annotations, str(output_path))

            # Verify file was created
            assert output_path.exists()
            assert output_path.parent.exists()
