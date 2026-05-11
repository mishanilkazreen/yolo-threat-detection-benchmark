"""
Report results from completed nano/small model experiments.

Reads all round_N_metrics.json and final_test_metrics.json files from
outputs/ and prints a comparison table matching the MDPI paper metrics:
  - Per-round: F1, Precision, mAP@0.5 (overall + per-class knife/pistol)
  - Final test: same metrics
  - Early stopping runs: actual stopped epoch per round

Usage:
    uv run python scripts/report_results.py
    uv run python scripts/report_results.py --phase nano
    uv run python scripts/report_results.py --phase small
    uv run python scripts/report_results.py --configs yolov8n_random yolov8n_pretrained
"""

import argparse
import json
import logging
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.WARNING)

# ---------------------------------------------------------------------------
# Config groups
# ---------------------------------------------------------------------------

NANO_CONFIGS = [
    "yolov8n_random",
    "yolov8n_pretrained",
    "yolo12n_random",
    "yolo12n_pretrained",
    "yolov8n_random_100ep",
    "yolov8n_pretrained_100ep",
    "yolo12n_random_100ep",
    "yolo12n_pretrained_100ep",
]

SMALL_CONFIGS = [
    "yolov8s_random",
    "yolov8s_pretrained",
    "yolo12s_random",
    "yolo12s_pretrained",
    "yolov8s_random_100ep",
    "yolov8s_pretrained_100ep",
    "yolo12s_random_100ep",
    "yolo12s_pretrained_100ep",
]

# One-shot baselines (no incremental rounds). These anchor every other
# experiment: each "round" config is compared to its matching baseline on
# metrics (mAP, F1, precision, per-class) and wall-clock time. See issue #13.
BASELINE_CONFIGS = [
    "yolov8n_baseline_pretrained",
    "yolov8n_baseline_random",
    "yolov8n_baseline_pretrained_100ep",
    "yolov8n_baseline_random_100ep",
    "yolov8s_baseline_pretrained",
    "yolov8s_baseline_random",
    "yolov8s_baseline_pretrained_100ep",
    "yolov8s_baseline_random_100ep",
    "yolo12n_baseline_pretrained",
    "yolo12n_baseline_random",
    "yolo12n_baseline_pretrained_100ep",
    "yolo12n_baseline_random_100ep",
    "yolo12s_baseline_pretrained",
    "yolo12s_baseline_random",
    "yolo12s_baseline_pretrained_100ep",
    "yolo12s_baseline_random_100ep",
]

OUTPUTS_DIR = project_root / "outputs"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> dict:
    """Load a JSON file, returning empty dict on failure."""
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        print(f"  [warn] Could not read {path}: {exc}")
        return {}


def _fmt(value, decimals: int = 4) -> str:
    """Format a float or return '—' for missing values."""
    if value is None or value == "":
        return "—"
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return str(value)


def _class_names(data_dir: Path) -> list[str]:
    """Try to read class names from the outputs split_metadata or return defaults."""
    meta = _load_json(data_dir / "split_metadata.json")
    names = meta.get("class_names") or meta.get("names")
    if names:
        return list(names)
    return ["knife", "pistol"]


# ---------------------------------------------------------------------------
# Per-round table
# ---------------------------------------------------------------------------


def _build_round_row(
    rnd: int,
    data: dict,
    class_names: list[str],
) -> str:
    """Build a single formatted row for the per-round table."""
    m = data.get("metrics", {})
    pc = data.get("per_class_metrics", {})

    f1 = m.get("f1_score")
    prec = m.get("precision")
    map50 = m.get("mAP50")
    stopped = data.get("actual_stopped_epoch")
    best_ep = data.get("best_epoch")

    f1_pc = pc.get("f1_score_per_class", [])
    prec_pc = pc.get("precision_per_class", [])
    map50_pc = pc.get("mAP50_per_class", [])

    row = [
        f"{'R' + str(rnd):>5}",
        f"{_fmt(stopped, 0):>8}",
        f"{_fmt(best_ep, 0):>7}",
        f"{_fmt(f1):>7}",
        f"{_fmt(prec):>7}",
        f"{_fmt(map50):>7}",
    ]
    for idx in range(len(class_names)):
        row += [
            f"{_fmt(f1_pc[idx] if idx < len(f1_pc) else None):>10}",
            f"{_fmt(prec_pc[idx] if idx < len(prec_pc) else None):>9}",
            f"{_fmt(map50_pc[idx] if idx < len(map50_pc) else None):>10}",
        ]
    return "  " + "  ".join(row)


def print_round_table(config_name: str, data_dir: Path, class_names: list[str]) -> None:
    """Print per-round metrics table for one config."""
    print(f"\n{'=' * 80}")
    print(f"  {config_name}  —  per-round metrics")
    print(f"{'=' * 80}")

    header_parts = [
        f"{'Round':>5}",
        f"{'Stopped':>8}",
        f"{'BestEp':>7}",
        f"{'F1':>7}",
        f"{'Prec':>7}",
        f"{'mAP50':>7}",
    ]
    for cls in class_names:
        header_parts += [
            f"{cls[:6] + '-F1':>10}",
            f"{cls[:6] + '-P':>9}",
            f"{cls[:6] + '-mAP':>10}",
        ]

    print("  " + "  ".join(header_parts))
    print("  " + "-" * (len("  ".join(header_parts)) + 2))

    for rnd in range(1, 6):
        rnd_file = data_dir / f"round_{rnd}_metrics.json"
        if not rnd_file.exists():
            print(f"  {'R' + str(rnd):>5}  (not found)")
            continue

        data = _load_json(rnd_file)
        print(_build_round_row(rnd, data, class_names))


# ---------------------------------------------------------------------------
# Final test table
# ---------------------------------------------------------------------------


def print_final_test_row(config_name: str, data_dir: Path, class_names: list[str]) -> None:
    """Print final test metrics for one config (single row)."""
    test_file = data_dir / "final_test_metrics.json"
    if not test_file.exists():
        print(f"  {config_name:<40}  (final_test_metrics.json not found)")
        return

    data = _load_json(test_file)
    m = data.get("metrics", {})
    pc = data.get("per_class_metrics", {})

    f1 = m.get("f1_score")
    prec = m.get("precision")
    map50 = m.get("mAP50")

    f1_pc = pc.get("f1_score_per_class", [])
    prec_pc = pc.get("precision_per_class", [])
    map50_pc = pc.get("mAP50_per_class", [])

    parts = [
        f"{config_name:<42}",
        f"{_fmt(f1):>7}",
        f"{_fmt(prec):>7}",
        f"{_fmt(map50):>7}",
    ]
    for idx in range(len(class_names)):
        parts += [
            f"{_fmt(f1_pc[idx] if idx < len(f1_pc) else None):>10}",
            f"{_fmt(prec_pc[idx] if idx < len(prec_pc) else None):>9}",
            f"{_fmt(map50_pc[idx] if idx < len(map50_pc) else None):>10}",
        ]

    print("  " + "  ".join(parts))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Report experiment results")
    parser.add_argument(
        "--phase",
        choices=["nano", "small", "baseline", "all"],
        default="all",
        help="Which phase to report (default: all)",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        help="Specific config names to report (overrides --phase)",
    )
    args = parser.parse_args()

    if args.configs:
        configs = args.configs
    elif args.phase == "nano":
        configs = NANO_CONFIGS
    elif args.phase == "small":
        configs = SMALL_CONFIGS
    elif args.phase == "baseline":
        configs = BASELINE_CONFIGS
    else:
        configs = NANO_CONFIGS + SMALL_CONFIGS + BASELINE_CONFIGS

    # -----------------------------------------------------------------------
    # Per-round tables
    # -----------------------------------------------------------------------
    print("\n" + "#" * 80)
    print("# PER-ROUND METRICS")
    print("#" * 80)

    for cfg in configs:
        data_dir = OUTPUTS_DIR / cfg
        if not data_dir.exists():
            print(f"\n  [{cfg}]  output directory not found — run not completed yet")
            continue
        class_names = _class_names(data_dir)
        print_round_table(cfg, data_dir, class_names)

    # -----------------------------------------------------------------------
    # Final test summary table
    # -----------------------------------------------------------------------
    print("\n\n" + "#" * 80)
    print("# FINAL TEST SET METRICS  (best checkpoint across all rounds)")
    print("#" * 80)

    # Determine class names from first available config
    sample_class_names = ["knife", "pistol"]
    for cfg in configs:
        data_dir = OUTPUTS_DIR / cfg
        if data_dir.exists():
            sample_class_names = _class_names(data_dir)
            break

    header_parts = [
        f"{'Config':<42}",
        f"{'F1':>7}",
        f"{'Prec':>7}",
        f"{'mAP50':>7}",
    ]
    for cls in sample_class_names:
        header_parts += [
            f"{cls[:6] + '-F1':>10}",
            f"{cls[:6] + '-P':>9}",
            f"{cls[:6] + '-mAP':>10}",
        ]

    print("\n  " + "  ".join(header_parts))
    print("  " + "-" * (len("  ".join(header_parts)) + 2))

    for cfg in configs:
        data_dir = OUTPUTS_DIR / cfg
        if not data_dir.exists():
            print(f"  {cfg:<42}  (not run yet)")
            continue
        class_names = _class_names(data_dir)
        print_final_test_row(cfg, data_dir, class_names)

    print()


if __name__ == "__main__":
    main()
