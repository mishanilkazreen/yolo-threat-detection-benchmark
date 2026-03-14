"""Tests for Bug 7 — Correct label path resolution in HFS scorer.

Validates: Requirements 2.5, 3.5
"""

import pytest

from src.explainability.hfs_scorer import Heatmap_Focus_Scorer


@pytest.fixture
def scorer():
    return Heatmap_Focus_Scorer()


@pytest.fixture
def dataset_dir(tmp_path):
    """Create a minimal YOLO-style dataset layout with a label file."""
    images_dir = tmp_path / "valid" / "images"
    labels_dir = tmp_path / "valid" / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    # Create a dummy image file (content doesn't matter for path resolution)
    img_file = images_dir / "img.jpg"
    img_file.write_bytes(b"")

    # Create the label file at the correct YOLO path
    label_file = labels_dir / "img.txt"
    label_file.write_text("0 0.3 0.4 0.2 0.1\n")

    return tmp_path


# ---------------------------------------------------------------------------
# 5.2 Exploratory test — fails on unfixed code (fallback used), passes after fix
# ---------------------------------------------------------------------------


def test_exploratory_label_path_not_fallback(scorer, dataset_dir):
    """Bug 7 exploratory: bbox loaded from labels/ dir, not the fallback centroid.

    On unfixed code Path(image_path).with_suffix('.txt') resolves to
    valid/images/img.txt which does not exist, so the fallback (0.5,0.5,0.5,0.5)
    is returned.  After the fix the path is valid/labels/img.txt which exists.
    """
    image_path = str(dataset_dir / "valid" / "images" / "img.jpg")
    bboxes = scorer.load_ground_truth_bboxes([image_path], dataset_dir)

    assert len(bboxes) == 1
    assert bboxes[0] != (0.5, 0.5, 0.5, 0.5), (
        "Expected real bbox from labels file, got fallback — label path not resolved correctly"
    )


# ---------------------------------------------------------------------------
# 5.3 Preservation test — correct values are returned when label file exists
# ---------------------------------------------------------------------------


def test_preservation_correct_bbox_values_returned(scorer, dataset_dir):
    """Bug 7 preservation: when label file exists at correct YOLO path, exact bbox is returned."""
    image_path = str(dataset_dir / "valid" / "images" / "img.jpg")
    bboxes = scorer.load_ground_truth_bboxes([image_path], dataset_dir)

    assert len(bboxes) == 1
    x_center, y_center, width, height = bboxes[0]
    assert pytest.approx(x_center, abs=1e-6) == 0.3
    assert pytest.approx(y_center, abs=1e-6) == 0.4
    assert pytest.approx(width, abs=1e-6) == 0.2
    assert pytest.approx(height, abs=1e-6) == 0.1
