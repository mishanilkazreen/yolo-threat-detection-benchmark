"""Benchmark inference speed, GFLOPs, and parameter count for all trained configs.

Iterates over every ``runs/<config>/.../weights/best.pt`` produced by the
training pipeline, loads each model on CUDA at imgsz=640, captures GFLOPs and
parameter counts from Ultralytics' built-in summary, then runs a timed
inference loop over a small image batch to estimate ms/image and FPS.

Output: ``outputs/inference_speed_benchmark.csv`` with one row per checkpoint.

Usage::

    uv run python scripts/benchmark_inference.py

The script is idempotent: it overwrites the CSV on each run.
"""

from __future__ import annotations

import csv
from pathlib import Path
import sys
import time

import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).parent.parent
RUNS_DIR = PROJECT_ROOT / "runs"
OUTPUT_CSV = PROJECT_ROOT / "outputs" / "inference_speed_benchmark.csv"

WARMUP_ITERS = 5
TIMED_ITERS = 50
IMG_SIZE = 640


def _arch_from_path(rel_path: Path) -> str:
    parts = rel_path.parts
    # First segment of run dir, e.g. "yolov8n_pretrained" -> "yolov8n".
    top = parts[0]
    for token in ("yolov8n", "yolo11n", "yolo12n"):
        if top.startswith(token):
            return token
    return top.split("_")[0]


def _config_label(rel_path: Path) -> str:
    """Build a short label like ``yolov8n_pretrained/incremental_round_5``."""
    parts = list(rel_path.parts)
    if parts[-2] == "weights":
        parts = parts[:-2]
    return "/".join(parts)


def _collect_weights() -> list[Path]:
    weights: list[Path] = []
    for pt in RUNS_DIR.rglob("best.pt"):
        rel = pt.relative_to(RUNS_DIR)
        # Skip stray ad-hoc runs in detect/.
        if rel.parts[0] == "detect":
            continue
        weights.append(pt)
    return sorted(weights)


def _benchmark_one(weight_path: Path, device: torch.device) -> dict[str, float | int | str]:
    model = YOLO(str(weight_path))
    model.to(device)

    # Ultralytics ``model.info`` returns ``(layers, params, gradients, gflops)``
    # but only when ``verbose=True``; with ``verbose=False`` it returns ``None``.
    info = model.info(detailed=False, verbose=True)
    layers, params, _gradients, gflops = info

    dummy = torch.rand(1, 3, IMG_SIZE, IMG_SIZE, device=device)

    for _ in range(WARMUP_ITERS):
        _ = model.predict(dummy, imgsz=IMG_SIZE, device=device, verbose=False)
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(TIMED_ITERS):
        _ = model.predict(dummy, imgsz=IMG_SIZE, device=device, verbose=False)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    ms_per_image = (elapsed / TIMED_ITERS) * 1000.0
    fps = 1000.0 / ms_per_image if ms_per_image > 0 else float("nan")

    return {
        "layers": int(layers),
        "params": int(params),
        "gflops": round(float(gflops), 2),
        "ms_per_image": round(ms_per_image, 3),
        "fps": round(fps, 2),
    }


def main() -> int:
    if not RUNS_DIR.exists():
        print(f"runs directory not found: {RUNS_DIR}", file=sys.stderr)
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    weights = _collect_weights()
    if not weights:
        print("No best.pt files found under runs/", file=sys.stderr)
        return 1
    print(f"Found {len(weights)} checkpoints to benchmark")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "config",
        "architecture",
        "layers",
        "params",
        "gflops",
        "ms_per_image",
        "fps",
        "device",
        "warmup_iters",
        "timed_iters",
        "img_size",
    ]

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()

        for idx, pt in enumerate(weights, start=1):
            rel = pt.relative_to(RUNS_DIR)
            label = _config_label(rel)
            arch = _arch_from_path(rel)
            print(f"[{idx:3d}/{len(weights)}] {label}", flush=True)
            try:
                metrics = _benchmark_one(pt, device)
            except Exception as exc:  # pylint: disable=broad-except
                print(f"  ! benchmark failed: {exc}", file=sys.stderr)
                continue

            writer.writerow(
                {
                    "config": label,
                    "architecture": arch,
                    "layers": metrics["layers"],
                    "params": metrics["params"],
                    "gflops": metrics["gflops"],
                    "ms_per_image": metrics["ms_per_image"],
                    "fps": metrics["fps"],
                    "device": str(device),
                    "warmup_iters": WARMUP_ITERS,
                    "timed_iters": TIMED_ITERS,
                    "img_size": IMG_SIZE,
                }
            )
            fh.flush()

    print(f"\nWrote {OUTPUT_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
