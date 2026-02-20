"""Unit tests for Annotation_Parser."""

import os
import tempfile

import pytest

from src.data.annotation_parser import Annotation_Parser, BoundingBox


class TestBoundingBox:
    """Tests for BoundingBox validation."""

    def test_valid_bounding_box(self):
        """Test creating a valid bounding box."""
        bbox = BoundingBox(center_x=0.5, center_y=0.5, width=0.3, height=0.4)
        assert bbox.center_x == 0.5
        assert bbox.center_y == 0.5
        assert bbox.width == 0.3
        assert bbox.height == 0.4

    def test_boundary_values(self):
        """Test bounding box with boundary values (0 and 1)."""
        bbox = BoundingBox(center_x=0.0, center_y=1.0, width=0.0, height=1.0)
        assert bbox.center_x == 0.0
        assert bbox.center_y == 1.0

    def test_invalid_center_x_negative(self):
        """Test that negative center_x raises ValueError."""
        with pytest.raises(ValueError, match="center_x must be in"):
            BoundingBox(center_x=-0.1, center_y=0.5, width=0.3, height=0.4)

    def test_invalid_center_x_too_large(self):
        """Test that center_x > 1 raises ValueError."""
        with pytest.raises(ValueError, match="center_x must be in"):
            BoundingBox(center_x=1.1, center_y=0.5, width=0.3, height=0.4)

    def test_invalid_width_negative(self):
        """Test that negative width raises ValueError."""
        with pytest.raises(ValueError, match="width must be in"):
            BoundingBox(center_x=0.5, center_y=0.5, width=-0.1, height=0.4)


class TestAnnotationParser:
    """Tests for Annotation_Parser."""

    def setup_method(self):
        """Set up test fixtures."""
        self.parser = Annotation_Parser()

    def test_parse_valid_single_annotation(self):
        """Test parsing a single valid annotation."""
        annotation_str = "0 0.5 0.5 0.3 0.4"
        annotations, errors = self.parser.parse_annotation_string(annotation_str)

        assert len(annotations) == 1
        assert len(errors) == 0
        assert annotations[0].class_id == 0
        assert annotations[0].bbox.center_x == 0.5
        assert annotations[0].bbox.center_y == 0.5
        assert annotations[0].bbox.width == 0.3
        assert annotations[0].bbox.height == 0.4

    def test_parse_multiple_annotations(self):
        """Test parsing multiple annotations."""
        annotation_str = """0 0.5 0.5 0.3 0.4
1 0.2 0.3 0.1 0.2
2 0.8 0.7 0.15 0.25"""
        annotations, errors = self.parser.parse_annotation_string(annotation_str)

        assert len(annotations) == 3
        assert len(errors) == 0
        assert annotations[0].class_id == 0
        assert annotations[1].class_id == 1
        assert annotations[2].class_id == 2

    def test_parse_empty_string(self):
        """Test parsing empty string returns empty list."""
        annotations, errors = self.parser.parse_annotation_string("")
        assert len(annotations) == 0
        assert len(errors) == 0

    def test_parse_with_empty_lines(self):
        """Test parsing with empty lines (should be skipped)."""
        annotation_str = """0 0.5 0.5 0.3 0.4

1 0.2 0.3 0.1 0.2

"""
        annotations, errors = self.parser.parse_annotation_string(annotation_str)
        assert len(annotations) == 2
        assert len(errors) == 0

    def test_parse_invalid_format_too_few_values(self):
        """Test parsing line with too few values."""
        annotation_str = "0 0.5 0.5 0.3"
        annotations, errors = self.parser.parse_annotation_string(annotation_str)

        assert len(annotations) == 0
        assert len(errors) == 1
        assert "Line 1" in errors[0]
        assert "Expected 5 values" in errors[0]

    def test_parse_invalid_format_too_many_values(self):
        """Test parsing line with too many values."""
        annotation_str = "0 0.5 0.5 0.3 0.4 0.5"
        annotations, errors = self.parser.parse_annotation_string(annotation_str)

        assert len(annotations) == 0
        assert len(errors) == 1
        assert "Expected 5 values" in errors[0]

    def test_parse_invalid_class_id_non_numeric(self):
        """Test parsing with non-numeric class_id."""
        annotation_str = "abc 0.5 0.5 0.3 0.4"
        annotations, errors = self.parser.parse_annotation_string(annotation_str)

        assert len(annotations) == 0
        assert len(errors) == 1
        assert "Invalid numeric value" in errors[0]

    def test_parse_invalid_class_id_negative(self):
        """Test parsing with negative class_id."""
        annotation_str = "-1 0.5 0.5 0.3 0.4"
        annotations, errors = self.parser.parse_annotation_string(annotation_str)

        assert len(annotations) == 0
        assert len(errors) == 1
        assert "class_id must be non-negative" in errors[0]

    def test_parse_invalid_coordinates_out_of_range(self):
        """Test parsing with coordinates out of [0, 1] range."""
        annotation_str = "0 1.5 0.5 0.3 0.4"
        annotations, errors = self.parser.parse_annotation_string(annotation_str)

        assert len(annotations) == 0
        assert len(errors) == 1
        assert "Invalid bounding box" in errors[0]

    def test_parse_invalid_coordinates_non_numeric(self):
        """Test parsing with non-numeric coordinates."""
        annotation_str = "0 abc 0.5 0.3 0.4"
        annotations, errors = self.parser.parse_annotation_string(annotation_str)

        assert len(annotations) == 0
        assert len(errors) == 1
        assert "Invalid numeric value" in errors[0]

    def test_parse_mixed_valid_invalid(self):
        """Test parsing with mix of valid and invalid annotations."""
        annotation_str = """0 0.5 0.5 0.3 0.4
invalid line
1 0.2 0.3 0.1 0.2"""
        annotations, errors = self.parser.parse_annotation_string(annotation_str)

        assert len(annotations) == 2
        assert len(errors) == 1
        assert "Line 2" in errors[0]

    def test_parse_file_not_found(self):
        """Test parsing non-existent file."""
        annotations, errors = self.parser.parse_annotation_file("nonexistent.txt")

        assert len(annotations) == 0
        assert len(errors) == 1
        assert "not found" in errors[0]

    def test_parse_file_valid(self):
        """Test parsing valid annotation file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("0 0.5 0.5 0.3 0.4\n")
            f.write("1 0.2 0.3 0.1 0.2\n")
            temp_path = f.name

        try:
            annotations, errors = self.parser.parse_annotation_file(temp_path)

            assert len(annotations) == 2
            assert len(errors) == 0
        finally:
            os.unlink(temp_path)

    def test_error_reporting_includes_line_number(self):
        """Test that error messages include line numbers."""
        annotation_str = """0 0.5 0.5 0.3 0.4
1 0.2 0.3 0.1
2 0.8 0.7 0.15 0.25"""
        annotations, errors = self.parser.parse_annotation_string(annotation_str)

        assert len(errors) == 1
        assert "Line 2" in errors[0]
