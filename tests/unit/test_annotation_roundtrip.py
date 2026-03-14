"""Property tests for annotation parsing/serialization round-trip."""

from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st
import pytest

from src.data.annotation_parser import Annotation, Annotation_Parser, BoundingBox
from src.data.annotation_serializer import Annotation_Serializer


# Strategy for generating valid bounding boxes
@st.composite
def bounding_box_strategy(draw):
    """Generate valid BoundingBox objects."""
    return BoundingBox(
        center_x=draw(st.floats(min_value=0.0, max_value=1.0)),
        center_y=draw(st.floats(min_value=0.0, max_value=1.0)),
        width=draw(st.floats(min_value=0.0, max_value=1.0)),
        height=draw(st.floats(min_value=0.0, max_value=1.0)),
    )


# Strategy for generating valid annotations
@st.composite
def annotation_strategy(draw):
    """Generate valid Annotation objects."""
    return Annotation(
        class_id=draw(st.integers(min_value=0, max_value=100)), bbox=draw(bounding_box_strategy())
    )


class TestAnnotationRoundTrip:
    """Property-based tests for annotation round-trip conversion.

    Validates: Property 53 - Annotation Round-Trip
    Requirements: 21.4
    """

    def setup_method(self):
        """Set up test fixtures."""
        self.parser = Annotation_Parser()
        self.serializer = Annotation_Serializer()

    @given(annotation=annotation_strategy())
    @settings(suppress_health_check=[HealthCheck.too_slow])
    def test_single_annotation_roundtrip(self, annotation):
        """
        Property 53: Annotation Round-Trip

        For any valid Annotation object, serializing then parsing should
        produce an equivalent object.
        """
        # Serialize
        serialized = self.serializer.serialize_annotations([annotation])

        # Parse
        parsed_annotations, errors = self.parser.parse_annotation_string(serialized)

        # Verify no errors
        assert len(errors) == 0, f"Parsing errors: {errors}"
        assert len(parsed_annotations) == 1

        parsed = parsed_annotations[0]

        # Verify equivalence (with floating point tolerance)
        assert parsed.class_id == annotation.class_id
        assert parsed.bbox.center_x == pytest.approx(annotation.bbox.center_x, abs=1e-5)
        assert parsed.bbox.center_y == pytest.approx(annotation.bbox.center_y, abs=1e-5)
        assert parsed.bbox.width == pytest.approx(annotation.bbox.width, abs=1e-5)
        assert parsed.bbox.height == pytest.approx(annotation.bbox.height, abs=1e-5)

    @given(annotations=st.lists(annotation_strategy(), min_size=0, max_size=20))
    def test_multiple_annotations_roundtrip(self, annotations):
        """
        Property 53: Annotation Round-Trip (multiple annotations)

        For any list of valid Annotation objects, serializing then parsing
        should produce equivalent objects.
        """
        # Serialize
        serialized = self.serializer.serialize_annotations(annotations)

        # Parse
        parsed_annotations, errors = self.parser.parse_annotation_string(serialized)

        # Verify no errors
        assert len(errors) == 0, f"Parsing errors: {errors}"
        assert len(parsed_annotations) == len(annotations)

        # Verify each annotation
        for original, parsed in zip(annotations, parsed_annotations):
            assert parsed.class_id == original.class_id
            assert parsed.bbox.center_x == pytest.approx(original.bbox.center_x, abs=1e-5)
            assert parsed.bbox.center_y == pytest.approx(original.bbox.center_y, abs=1e-5)
            assert parsed.bbox.width == pytest.approx(original.bbox.width, abs=1e-5)
            assert parsed.bbox.height == pytest.approx(original.bbox.height, abs=1e-5)

    def test_empty_list_roundtrip(self):
        """Test round-trip with empty annotation list."""
        annotations = []

        # Serialize
        serialized = self.serializer.serialize_annotations(annotations)

        # Parse
        parsed_annotations, errors = self.parser.parse_annotation_string(serialized)

        assert len(errors) == 0
        assert len(parsed_annotations) == 0
