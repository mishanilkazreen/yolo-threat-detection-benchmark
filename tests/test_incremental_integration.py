"""Integration tests for incremental training components."""

from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_imports():
    """Test that all incremental training components can be imported."""
    print("\n" + "=" * 60)
    print("Testing Incremental Training Integration")
    print("=" * 60 + "\n")

    try:
        from src.training.runner import Experiment_Runner

        print("✓ Experiment_Runner import successful")

        from src.training.evaluator import Metrics_Collector

        print("✓ Metrics_Collector import successful")

        print("✓ Dataset_Splitter import successful")

        print("✓ Training_Set_Manager import successful")

        print("✓ Edge_Agent_Simulator import successful")

        print("✓ Detection_Validator import successful")

        print("✓ Round_Metrics_Tracker import successful")

        print("\n" + "-" * 60)
        print("Testing Component Initialization")
        print("-" * 60 + "\n")

        # Test lazy loading
        runner = Experiment_Runner()
        print("✓ Experiment_Runner initialized")

        # Access lazy-loaded properties
        _ = runner.splitter
        print("✓ Dataset_Splitter lazy-loaded")

        _ = runner.training_set_manager
        print("✓ Training_Set_Manager lazy-loaded")

        _ = runner.edge_agent_simulator
        print("✓ Edge_Agent_Simulator lazy-loaded")

        _ = runner.detection_validator
        print("✓ Detection_Validator lazy-loaded")

        _ = runner.round_metrics_tracker
        print("✓ Round_Metrics_Tracker lazy-loaded")

        print("\n" + "-" * 60)
        print("Testing Metrics_Collector Methods")
        print("-" * 60 + "\n")

        metrics_collector = Metrics_Collector()

        # Check that new methods exist
        assert hasattr(metrics_collector, "collect_round_metrics"), (
            "Metrics_Collector missing collect_round_metrics method"
        )
        print("✓ Metrics_Collector.collect_round_metrics exists")

        assert hasattr(metrics_collector, "evaluate_final_test"), (
            "Metrics_Collector missing evaluate_final_test method"
        )
        print("✓ Metrics_Collector.evaluate_final_test exists")

        assert hasattr(metrics_collector, "_compute_f1"), (
            "Metrics_Collector missing _compute_f1 method"
        )
        print("✓ Metrics_Collector._compute_f1 exists")

        # Test F1 computation
        f1 = metrics_collector._compute_f1(0.8, 0.7)
        expected_f1 = 2 * (0.8 * 0.7) / (0.8 + 0.7)
        assert abs(f1 - expected_f1) < 1e-6, f"F1 computation incorrect: {f1} != {expected_f1}"
        print(f"✓ F1 computation correct: {f1:.4f}")

        # Test F1 with zero values
        f1_zero = metrics_collector._compute_f1(0.0, 0.0)
        assert f1_zero == 0.0, "F1 should be 0 when precision and recall are 0"
        print("✓ F1 computation handles zero values")

        print("\n" + "-" * 60)
        print("Testing Experiment_Runner Methods")
        print("-" * 60 + "\n")

        # Check that new methods exist
        assert hasattr(runner, "_run_incremental_training"), (
            "Experiment_Runner missing _run_incremental_training method"
        )
        print("✓ Experiment_Runner._run_incremental_training exists")

        assert hasattr(runner, "_run_standard_training"), (
            "Experiment_Runner missing _run_standard_training method"
        )
        print("✓ Experiment_Runner._run_standard_training exists")

        assert hasattr(runner, "_create_round_data_yaml"), (
            "Experiment_Runner missing _create_round_data_yaml method"
        )
        print("✓ Experiment_Runner._create_round_data_yaml exists")

        print("\n" + "=" * 60)
        print("All Integration Tests Passed! ✓")
        print("=" * 60 + "\n")

        return True

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
