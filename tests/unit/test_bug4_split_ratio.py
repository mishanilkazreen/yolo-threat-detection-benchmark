"""
Tests for Bug 4 fix: dataset split ratio verification.

Task 2.3 — Exploratory test: _verify_split_ratio() logs a warning for a 60/24/16 split
           (fails on unfixed code: method does not exist).
Task 2.4 — Preservation test: _verify_split_ratio() with a 70/20/10 split logs no warning
           and returns without error.
"""

import logging
import pytest

from src.training.runner import Experiment_Runner


# ---------------------------------------------------------------------------
# Task 2.3 — Exploratory test (fails on unfixed code, passes after fix)
# ---------------------------------------------------------------------------

def test_verify_split_ratio_warns_on_deviant_split(caplog):
    """
    _verify_split_ratio() with a 60/24/16 split (deviates from 70/20/10) must
    log a warning.

    Validates: Requirements 2.2 (Bug 4 fix)
    """
    runner = Experiment_Runner()

    # 60/24/16 split: 3000 train, 1200 valid, 800 test → total 5000
    with caplog.at_level(logging.WARNING, logger=runner.logger.name):
        runner._verify_split_ratio(train_count=3000, val_count=1200, test_count=800)

    assert len(caplog.records) > 0, "Expected a warning to be logged for a deviant split"
    assert any("deviate" in record.message.lower() or "split" in record.message.lower()
               for record in caplog.records), (
        f"Warning message did not mention split deviation. Got: {[r.message for r in caplog.records]}"
    )


# ---------------------------------------------------------------------------
# Task 2.4 — Preservation test: no warning for a 70/20/10 split
# ---------------------------------------------------------------------------

def test_verify_split_ratio_no_warning_on_valid_split(caplog):
    """
    _verify_split_ratio() with a 70/20/10 split must not log any warning and
    must return without error.

    Validates: Requirements 3.2 (preservation)
    """
    runner = Experiment_Runner()

    # Exact 70/20/10 split: 3543 train, 1012 valid, 506 test → total 5061
    with caplog.at_level(logging.WARNING, logger=runner.logger.name):
        result = runner._verify_split_ratio(train_count=3543, val_count=1012, test_count=506)

    assert result is None, "Method should return None"
    assert len(caplog.records) == 0, (
        f"Expected no warnings for a valid 70/20/10 split, but got: "
        f"{[r.message for r in caplog.records]}"
    )
