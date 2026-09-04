#!/usr/bin/env python3
"""
SCIAMA Real-Time Cluster Monitor & Dynamic ETA Dashboard.

Queries SCIAMA SLURM queue and Lustre outputs to report:
  - Running, pending, and completed jobs
  - Exact epochs completed for active jobs
  - Real-time completion progress (% of all 68 jobs)
  - Dynamically estimated time of completion (ETA)

Usage:
    python3 scripts/monitor_sciama_progress.py
    python3 scripts/monitor_sciama_progress.py --watch 30
"""

import argparse
from datetime import datetime, timedelta
import json
import subprocess
import sys
import time

TOTAL_EXPECTED_JOBS = 68  # 1 coco + 1 latency + 6 baselines + 60 multiseed runs
MAX_CONCURRENT_GPUS = 2   # SCIAMA QOSMaxJobsPerUserLimit for gpu.q


def run_ssh(cmd: str) -> str:
    res = subprocess.run(
        ["ssh", "-o", "ConnectTimeout=10", "sciama", cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return res.stdout.strip()


def get_queue_info() -> tuple[list[dict], list[dict]]:
    raw_squeue = run_ssh('squeue -u ghahrem -o "%i|%j|%T|%M|%R"')
    running = []
    pending = []
    for line in raw_squeue.splitlines():
        parts = line.strip().split("|")
        if len(parts) >= 5 and parts[0] != "JOBID":
            job = {
                "id": parts[0],
                "name": parts[1],
                "state": parts[2],
                "time": parts[3],
                "node_or_reason": parts[4],
            }
            if job["state"] == "RUNNING":
                running.append(job)
            else:
                pending.append(job)
    return running, pending


def get_completed_counts() -> tuple[int, int, int]:
    cmd = (
        "python3 -c 'import glob, json, os; "
        "new_m = sum(1 for p in glob.glob(\"/mnt/lustre2/mres/ghahrem/yolo-threat-detection-benchmark/outputs/**/final_test_metrics.json\", recursive=True) if json.load(open(p)).get(\"cumulative_optimizer_steps\", 0) > 0); "
        "base_m = sum(1 for p in glob.glob(\"/mnt/lustre2/mres/ghahrem/yolo-threat-detection-benchmark/outputs/**/baseline_summary.json\", recursive=True) if \"100ep\" not in p and json.load(open(p)).get(\"total_optimizer_steps\", 0) > 0); "
        "fast_m = sum(1 for p in [\"/mnt/lustre2/mres/ghahrem/yolo-threat-detection-benchmark/outputs/coco_knife_overlap_evaluation.json\", \"/mnt/lustre2/mres/ghahrem/yolo-threat-detection-benchmark/outputs/decomposed_latency_benchmark.json\"] if os.path.exists(p)); "
        "print(new_m, base_m, fast_m)'"
    )
    raw = run_ssh(cmd)
    try:
        parts = raw.split()
        multiseed_done = int(parts[0])
        baselines_done = int(parts[1])
        fast_done = int(parts[2])
        return multiseed_done, baselines_done, fast_done
    except Exception:
        return 0, 0, 0


def get_active_job_progress() -> list[str]:
    cmd = (
        "tail -n 1 /mnt/lustre2/mres/ghahrem/yolo-threat-detection-benchmark/runs/slurm_logs/*.out 2>/dev/null | "
        "grep -E '==>|[0-9]+/500|[0-9]+/100|Epochs completed|Baseline Training Complete' || true"
    )
    raw = run_ssh(cmd)
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    return lines[-10:] if lines else []


def display_dashboard():
    now = datetime.now()
    running, pending = get_queue_info()
    multiseed_done, baselines_done, fast_done = get_completed_counts()
    total_done = multiseed_done + baselines_done + fast_done
    remaining_multiseed = max(0, 60 - multiseed_done)

    # Realistic ETA Calculation based on actual benchmarks on NVIDIA A100/L40:
    # 50-epoch runs: ~10 minutes/run
    # 100-epoch runs: ~75 minutes/run (with patience=10 early stopping)
    # Remaining 49 runs are a mix of ~20 50-epoch and ~29 100-epoch runs
    # Total remaining GPU time ≈ (20 * 10 + 29 * 75) = 2,375 GPU-minutes
    # SCIAMA concurrency: 2 GPU workers (QOSMaxJobsPerUserLimit)
    est_remaining_minutes = (remaining_multiseed * 48) / MAX_CONCURRENT_GPUS
    eta_time = now + timedelta(minutes=est_remaining_minutes)

    print("\033[2J\033[H", end="")  # Clear screen
    print("=" * 80)
    print(f"SCIAMA HPC LIVE DASHBOARD — {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print(f"Overall Progress:       {total_done}/{TOTAL_EXPECTED_JOBS} jobs completed ({total_done / TOTAL_EXPECTED_JOBS * 100:.1f}%)")
    print(f"  • Fast Benchmarks:    {fast_done}/2 completed")
    print(f"  • 500-Ep Baselines:   {baselines_done}/6 completed")
    print(f"  • Multi-Seed Sweeps:  {multiseed_done}/60 completed")
    print(f"Active GPU Workers:     {len(running)} running (Max concurrency: {MAX_CONCURRENT_GPUS})")
    print(f"Pending SLURM Queue:    {len(pending)} jobs queued")
    print("-" * 80)
    print(f"ESTIMATED COMPLETION:   {eta_time.strftime('%A, %d %b %Y at %H:%M:%S')} (~{est_remaining_minutes / 60:.1f} hours remaining)")
    print("=" * 80)

    if running:
        print("\nCurrently Executing on Cluster:")
        print(f"{'Job ID':<15} {'Name':<20} {'Time':<12} {'Node':<15}")
        print("-" * 62)
        for r in running:
            print(f"{r['id']:<15} {r['name']:<20} {r['time']:<12} {r['node_or_reason']:<15}")

    print("\nRecent GPU Log Output:")
    print("-" * 80)
    active_logs = get_active_job_progress()
    if active_logs:
        for l in active_logs:
            print(f"  {l[:78]}")
    else:
        print("  Waiting for active logs...")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Live monitor for SCIAMA JRTIP jobs")
    parser.add_argument("--watch", type=int, default=0, help="Refresh interval in seconds (0 for single run)")
    args = parser.parse_args()

    if args.watch > 0:
        try:
            while True:
                display_dashboard()
                time.sleep(args.watch)
        except KeyboardInterrupt:
            print("\nMonitor stopped.")
    else:
        display_dashboard()


if __name__ == "__main__":
    main()
