"""
Decomposed Real-Time Latency Benchmark & Profiler.

Evaluates end-to-end inference latency with microsecond-level stage decomposition:
  1. Preprocessing (letterbox resize, BGR-to-RGB, normalization, tensor transfer)
  2. Model Forward Pass (GPU inference with synchronization)
  3. Postprocessing (Non-Maximum Suppression and box decoding)
  4. Total End-to-End Latency (sum of stages 1-3)

Addresses peer review mandates (Reviewer 1 Major 1, Major 10; Reviewer 3):
  - Warmup >= 100 iterations
  - Measurement >= 500 iterations over real test images
  - CUDA event synchronization (torch.cuda.synchronize)
  - Full latency distributions: Mean, SD, P50 (median), P95, P99, Min, Max
  - Both FP32 and FP16 (half-precision) modes
  - Concrete application deadline compliance (30 FPS: 33.33ms, 60 FPS: 16.67ms, 120 FPS: 8.33ms)
  - Throughput strictly defined as 1000 / mean_total_ms

Usage:
    uv run python scripts/benchmark_inference.py --device 0 --warmup 100 --iterations 500
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils import nms

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.device_utils import get_model_gflops, select_device

logger = logging.getLogger(__name__)

# Standard application deadlines in milliseconds
APPLICATION_DEADLINES_MS = {
    "Surveillance_30FPS": 33.333,
    "HighSpeed_60FPS": 16.667,
    "EdgeUltraFast_120FPS": 8.333,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decomposed real-time latency profiler with statistical percentiles and deadline compliance."
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="List of model checkpoint paths (.pt) to evaluate. If omitted, evaluates stock v8n/11n/12n and runs/* checkpoints.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Target device: 'auto', '0', 'cuda', 'cpu'",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=100,
        help="Number of warmup iterations (default: 100)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=500,
        help="Number of timed benchmark iterations (default: 500)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image resolution (default: 640)",
    )
    parser.add_argument(
        "--image-list",
        type=str,
        default="config/data/test_fixed.txt",
        help="Path to text file containing test image paths",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="outputs/decomposed_latency_benchmark.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="outputs/decomposed_latency_benchmark.json",
        help="Output JSON path",
    )
    return parser.parse_args()


def load_test_images(image_list_path: Path, limit: int = 50) -> list[np.ndarray]:
    """Load and cache a set of real test images in memory."""
    if not image_list_path.exists():
        logger.warning("Image list %s not found. Using synthetic random images.", image_list_path)
        return [np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8) for _ in range(10)]

    lines = [
        ln.strip() for ln in image_list_path.read_text(encoding="utf-8").splitlines() if ln.strip()
    ]
    images: list[np.ndarray] = []
    for rel_p in lines[:limit]:
        p = Path(rel_p)
        if not p.is_absolute():
            p = PROJECT_ROOT / rel_p
        if p.exists():
            im = cv2.imread(str(p))
            if im is not None:
                images.append(im)

    if not images:
        logger.warning(
            "Could not read real images from %s. Using synthetic images.", image_list_path
        )
        images = [np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8) for _ in range(10)]

    logger.info("Loaded %d real test images for profiling cache", len(images))
    return images


def compute_distribution(samples: list[float]) -> dict[str, float]:
    """Calculate descriptive statistics and percentiles for latency measurements."""
    arr = np.array(samples, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def benchmark_single_model(
    model_path: str,
    device: torch.device,
    images: list[np.ndarray],
    imgsz: int = 640,
    warmup: int = 100,
    iterations: int = 500,
    half_precision: bool = False,
) -> dict[str, Any]:
    """Execute decomposed latency benchmarking for a single model and precision mode."""
    mode_name = "FP16" if half_precision else "FP32"
    logger.info(
        "Profiling %s [%s] on %s (%s)...", Path(model_path).name, mode_name, device, device.type
    )

    yolo_model = YOLO(model_path)
    net = yolo_model.model.to(device)
    net.eval()

    if half_precision:
        net.half()

    # Get model summary parameters and GFLOPs
    params = sum(p.numel() for p in net.parameters())
    gflops = get_model_gflops(Path(model_path).stem)

    letterbox = LetterBox(imgsz, auto=False)
    is_cuda = device.type == "cuda"

    # Pre-warmup phase
    num_images = len(images)
    for i in range(warmup):
        raw_img = images[i % num_images]
        im_lb = letterbox(image=raw_img)
        im_rgb = np.ascontiguousarray(im_lb[..., ::-1].transpose((2, 0, 1)))
        tensor = torch.from_numpy(im_rgb).to(device)
        tensor = tensor.half() if half_precision else tensor.float()
        tensor /= 255.0
        tensor = tensor.unsqueeze(0)

        with torch.no_grad():
            preds = net(tensor)
            _ = nms.non_max_suppression(preds, conf_thres=0.25, iou_thres=0.45, max_det=300)

        if is_cuda:
            torch.cuda.synchronize(device)

    # Measurement phase
    preprocess_times_ms: list[float] = []
    forward_times_ms: list[float] = []
    postprocess_times_ms: list[float] = []
    total_times_ms: list[float] = []

    for i in range(iterations):
        raw_img = images[i % num_images]

        # Stage 1: Preprocessing
        if is_cuda:
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()

        im_lb = letterbox(image=raw_img)
        im_rgb = np.ascontiguousarray(im_lb[..., ::-1].transpose((2, 0, 1)))
        tensor = torch.from_numpy(im_rgb).to(device)
        tensor = tensor.half() if half_precision else tensor.float()
        tensor /= 255.0
        tensor = tensor.unsqueeze(0)

        if is_cuda:
            torch.cuda.synchronize(device)
        t1 = time.perf_counter()

        # Stage 2: Forward pass
        with torch.no_grad():
            preds = net(tensor)

        if is_cuda:
            torch.cuda.synchronize(device)
        t2 = time.perf_counter()

        # Stage 3: Postprocessing (NMS)
        _ = nms.non_max_suppression(preds, conf_thres=0.25, iou_thres=0.45, max_det=300)

        if is_cuda:
            torch.cuda.synchronize(device)
        t3 = time.perf_counter()

        t_prep = (t1 - t0) * 1000.0
        t_fwd = (t2 - t1) * 1000.0
        t_post = (t3 - t2) * 1000.0
        t_tot = (t3 - t0) * 1000.0

        preprocess_times_ms.append(t_prep)
        forward_times_ms.append(t_fwd)
        postprocess_times_ms.append(t_post)
        total_times_ms.append(t_tot)

    dist_prep = compute_distribution(preprocess_times_ms)
    dist_fwd = compute_distribution(forward_times_ms)
    dist_post = compute_distribution(postprocess_times_ms)
    dist_tot = compute_distribution(total_times_ms)

    fps_mean = 1000.0 / dist_tot["mean"] if dist_tot["mean"] > 0 else 0.0
    fps_p50 = 1000.0 / dist_tot["p50"] if dist_tot["p50"] > 0 else 0.0

    # Application deadline analysis
    deadline_compliance: dict[str, dict[str, Any]] = {}
    for deadline_name, deadline_ms in APPLICATION_DEADLINES_MS.items():
        slack_ms = deadline_ms - dist_tot["p99"]
        compliant = bool(slack_ms >= 0)
        deadline_compliance[deadline_name] = {
            "deadline_ms": deadline_ms,
            "slack_margin_p99_ms": round(slack_ms, 3),
            "meets_deadline_p99": compliant,
        }

    return {
        "model_name": Path(model_path).stem,
        "model_path": str(model_path),
        "precision": mode_name,
        "parameters": params,
        "gflops": gflops,
        "batch_size": 1,
        "imgsz": imgsz,
        "iterations": iterations,
        "warmup": warmup,
        "device_name": torch.cuda.get_device_name(device) if is_cuda else "CPU",
        "preprocess_ms": dist_prep,
        "forward_ms": dist_fwd,
        "postprocess_ms": dist_post,
        "end_to_end_ms": dist_tot,
        "fps_mean": round(fps_mean, 2),
        "fps_median": round(fps_p50, 2),
        "deadline_compliance": deadline_compliance,
    }


def collect_target_models(cli_models: list[str] | None) -> list[str]:
    """Identify checkpoints or default models to benchmark."""
    if cli_models:
        return cli_models

    targets: list[str] = []
    # Check if checkpoints exist under runs/
    runs_dir = PROJECT_ROOT / "runs"
    if runs_dir.exists():
        for pt in sorted(runs_dir.rglob("best.pt")):
            if "detect" not in pt.parts:
                targets.append(str(pt))

    # Also include stock YOLO models for cross-generation baseline comparison
    for stock in ("yolov8n.pt", "yolo11n.pt", "yolo12n.pt"):
        targets.append(stock)

    return targets


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    device_str = select_device(args.device)
    device = torch.device(device_str)
    print("\n" + "=" * 80)
    print("DECOMPOSED REAL-TIME INFERENCE BENCHMARK")
    print("=" * 80)
    print(
        f"Device:               {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})"
    )
    print(f"Warmup Cycles:        {args.warmup}")
    print(f"Measurement Cycles:   {args.iterations}")
    print(f"Input Resolution:     {args.imgsz}x{args.imgsz} (Batch size = 1)")
    print("=" * 80 + "\n")

    images = load_test_images(PROJECT_ROOT / args.image_list, limit=50)
    models = collect_target_models(args.models)

    results: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []

    for model_target in models:
        # Benchmark FP32
        res_fp32 = benchmark_single_model(
            model_path=model_target,
            device=device,
            images=images,
            imgsz=args.imgsz,
            warmup=args.warmup,
            iterations=args.iterations,
            half_precision=False,
        )
        results.append(res_fp32)

        # Benchmark FP16 (if CUDA available)
        if device.type == "cuda":
            res_fp16 = benchmark_single_model(
                model_path=model_target,
                device=device,
                images=images,
                imgsz=args.imgsz,
                warmup=args.warmup,
                iterations=args.iterations,
                half_precision=True,
            )
            results.append(res_fp16)

    # Flatten for CSV export
    for r in results:
        e2e = r["end_to_end_ms"]
        fwd = r["forward_ms"]
        prep = r["preprocess_ms"]
        post = r["postprocess_ms"]
        dl = r["deadline_compliance"]
        csv_rows.append(
            {
                "model_name": r["model_name"],
                "precision": r["precision"],
                "device": r["device_name"],
                "params": r["parameters"],
                "gflops": r["gflops"],
                "prep_mean_ms": round(prep["mean"], 3),
                "fwd_mean_ms": round(fwd["mean"], 3),
                "fwd_p50_ms": round(fwd["p50"], 3),
                "fwd_p99_ms": round(fwd["p99"], 3),
                "post_mean_ms": round(post["mean"], 3),
                "total_mean_ms": round(e2e["mean"], 3),
                "total_p50_ms": round(e2e["p50"], 3),
                "total_p95_ms": round(e2e["p95"], 3),
                "total_p99_ms": round(e2e["p99"], 3),
                "fps_mean": r["fps_mean"],
                "fps_median": r["fps_median"],
                "slack_30fps_p99_ms": dl["Surveillance_30FPS"]["slack_margin_p99_ms"],
                "meets_30fps_p99": dl["Surveillance_30FPS"]["meets_deadline_p99"],
                "slack_60fps_p99_ms": dl["HighSpeed_60FPS"]["slack_margin_p99_ms"],
                "meets_60fps_p99": dl["HighSpeed_60FPS"]["meets_deadline_p99"],
            }
        )

    # Save outputs
    out_csv = PROJECT_ROOT / args.output_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if csv_rows:
        with open(out_csv, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)
        logger.info("Saved CSV results to %s", out_csv)

    out_json = PROJECT_ROOT / args.output_json
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
        fh.write("\n")
    logger.info("Saved JSON results to %s", out_json)

    # Print summary table
    print("\n" + "=" * 110)
    print(
        f"{'Model':<24} {'Mode':<6} {'Params':<10} {'Fwd P50':<10} {'Tot P50':<10} {'Tot P99':<10} {'FPS':<8} {'30FPS Slack':<12} {'60FPS Slack':<12}"
    )
    print("-" * 110)
    for row in csv_rows:
        s30 = f"{row['slack_30fps_p99_ms']:+.2f}ms"
        s60 = f"{row['slack_60fps_p99_ms']:+.2f}ms"
        print(
            f"{row['model_name']:<24} {row['precision']:<6} {row['params']:<10,d} "
            f"{row['fwd_p50_ms']:<10.2f} {row['total_p50_ms']:<10.2f} {row['total_p99_ms']:<10.2f} "
            f"{row['fps_median']:<8.1f} {s30:<12} {s60:<12}"
        )
    print("=" * 110 + "\n")


if __name__ == "__main__":
    main()
