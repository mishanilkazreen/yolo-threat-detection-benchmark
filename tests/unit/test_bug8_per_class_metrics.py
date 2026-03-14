"""
Tests for Bug 8 — Per-class precision, recall, and F1 in _extract_metrics().

6.2 Exploratory test: _extract_metrics() with results.box.p and results.box.r present
    → per_class_metrics must contain precision_per_class, recall_per_class, f1_score_per_class
    (fails on unfixed code: keys absent)

6.3 Preservation test: mAP50_per_class and mAP50-95_per_class keys are still present and
    unchanged after the fix.
"""

from unittest.mock import MagicMock
import pytest

from src.training.evaluator import Metrics_Collector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_results(num_classes: int = 3):
    """Build a minimal mock YOLO validation result with per-class arrays."""
    box = MagicMock()
    box.map50 = 0.75
    box.map = 0.55
    box.mp = 0.80
    box.mr = 0.70

    # Per-class arrays
    box.maps = [0.70, 0.75, 0.80]          # triggers the if-block
    box.ap50 = [0.70, 0.75, 0.80]
    box.ap   = [0.50, 0.55, 0.60]
    box.p    = [0.78, 0.82, 0.80]
    box.r    = [0.65, 0.72, 0.73]

    results = MagicMock()
    results.box = box
    results.fitness = 0.60
    return results


def _make_mock_model():
    model = MagicMock()
    model.model.parameters.return_value = []
    return model


def _call_extract_metrics(results):
    collector = Metrics_Collector()
    return collector._extract_metrics(
        results=results,
        model=_make_mock_model(),
        config_name="test_config",
        best_checkpoint="best.pt",
        run_id=1,
        random_seed=42,
        training_time=10.0,
        val_time=1.0,
    )


# ---------------------------------------------------------------------------
# 6.2 Exploratory test — new per-class keys must be present
# ---------------------------------------------------------------------------

class TestBug8Exploratory:
    def test_precision_per_class_present(self):
        """precision_per_class must be in per_class_metrics (was absent before fix)."""
        metrics = _call_extract_metrics(_make_mock_results())
        assert "per_class_metrics" in metrics
        assert "precision_per_class" in metrics["per_class_metrics"], (
            "precision_per_class missing from per_class_metrics"
        )

    def test_recall_per_class_present(self):
        """recall_per_class must be in per_class_metrics (was absent before fix)."""
        metrics = _call_extract_metrics(_make_mock_results())
        assert "recall_per_class" in metrics["per_class_metrics"], (
            "recall_per_class missing from per_class_metrics"
        )

    def test_f1_score_per_class_present(self):
        """f1_score_per_class must be in per_class_metrics (was absent before fix)."""
        metrics = _call_extract_metrics(_make_mock_results())
        assert "f1_score_per_class" in metrics["per_class_metrics"], (
            "f1_score_per_class missing from per_class_metrics"
        )

    def test_per_class_values_are_correct(self):
        """Values must match results.box.p / results.box.r and computed F1."""
        results = _make_mock_results()
        metrics = _call_extract_metrics(results)
        pcm = metrics["per_class_metrics"]

        assert pcm["precision_per_class"] == [float(x) for x in results.box.p]
        assert pcm["recall_per_class"] == [float(x) for x in results.box.r]

        collector = Metrics_Collector()
        expected_f1 = [
            collector._compute_f1(float(p), float(r))
            for p, r in zip(results.box.p, results.box.r)
        ]
        assert pcm["f1_score_per_class"] == expected_f1

    def test_all_three_new_keys_present_together(self):
        """All three new keys must appear together in a single call."""
        metrics = _call_extract_metrics(_make_mock_results())
        pcm = metrics["per_class_metrics"]
        for key in ("precision_per_class", "recall_per_class", "f1_score_per_class"):
            assert key in pcm, f"Key '{key}' missing from per_class_metrics"


# ---------------------------------------------------------------------------
# 6.3 Preservation test — existing keys unchanged
# ---------------------------------------------------------------------------

class TestBug8Preservation:
    def test_map50_per_class_still_present(self):
        """mAP50_per_class must still be present after the fix."""
        results = _make_mock_results()
        metrics = _call_extract_metrics(results)
        assert "mAP50_per_class" in metrics["per_class_metrics"]

    def test_map50_95_per_class_still_present(self):
        """mAP50-95_per_class must still be present after the fix."""
        results = _make_mock_results()
        metrics = _call_extract_metrics(results)
        assert "mAP50-95_per_class" in metrics["per_class_metrics"]

    def test_map50_per_class_values_unchanged(self):
        """mAP50_per_class values must equal results.box.ap50."""
        results = _make_mock_results()
        metrics = _call_extract_metrics(results)
        assert metrics["per_class_metrics"]["mAP50_per_class"] == [
            float(x) for x in results.box.ap50
        ]

    def test_map50_95_per_class_values_unchanged(self):
        """mAP50-95_per_class values must equal results.box.ap."""
        results = _make_mock_results()
        metrics = _call_extract_metrics(results)
        assert metrics["per_class_metrics"]["mAP50-95_per_class"] == [
            float(x) for x in results.box.ap
        ]

    def test_no_per_class_block_when_maps_absent(self):
        """When results.box has no 'maps' attribute, per_class_metrics must be absent."""
        results = _make_mock_results()
        del results.box.maps          # remove the attribute
        # hasattr will now return False
        results.box.configure_mock(**{"maps": MagicMock()})
        # Rebuild without maps by using spec
        box = MagicMock(spec=["map50", "map", "mp", "mr", "ap50", "ap", "p", "r", "fitness"])
        box.map50 = 0.75
        box.map = 0.55
        box.mp = 0.80
        box.mr = 0.70
        box.ap50 = [0.70, 0.75, 0.80]
        box.ap   = [0.50, 0.55, 0.60]
        box.p    = [0.78, 0.82, 0.80]
        box.r    = [0.65, 0.72, 0.73]
        results2 = MagicMock()
        results2.box = box
        results2.fitness = 0.60

        metrics = _call_extract_metrics(results2)
        assert "per_class_metrics" not in metrics
