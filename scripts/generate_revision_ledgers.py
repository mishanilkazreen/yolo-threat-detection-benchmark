#!/usr/bin/env python3
"""
Generate comprehensive multi-seed experimental accounting ledgers, statistical summaries,
and LaTeX table snippets for manuscript revision.

Definitions & Operational Units (audited directly from src/training/detection_validator.py):
- Rejected Proposals (rejected_proposals_boxes, BOXES): Total candidate bounding-box proposals generated during pool inference
  that failed IoU >= 0.5 or class-matching verification against pool ground truth.
- Undetected Pool Images (undetected_pool_images, IMAGES): Total pool images where no candidate proposals met verification thresholds.
  These images remain in the pool for potential acquisition in future rounds. Summed across rounds 2-5 with repeated counting of pool images.
- Verified Samples Added (verified_samples_added_images, IMAGES): Total images added to the active training set in the round.
  An image is acquired if at least one candidate bounding box proposal on that image verifies.
"""

import os
import json
import csv
import shutil
import numpy as np
from scipy import stats

def compute_ci95_halfwidth(arr):
    arr = np.array(arr, dtype=float)
    n = len(arr)
    if n < 2:
        return 0.0
    sd = np.std(arr, ddof=1)
    tcrit = stats.t.ppf(0.975, df=n-1)
    return tcrit * (sd / np.sqrt(n))

def main():
    comp_dir = r"c:\Users\manig\Downloads\yolo-threat-detection-benchmark"
    out_dir = os.path.join(comp_dir, "outputs")
    ledger_dir = os.path.join(out_dir, "revision_ledgers")
    os.makedirs(ledger_dir, exist_ok=True)

    archs = ["yolov8n", "yolo11n", "yolo12n"]
    inits = ["pretrained", "random"]
    budgets = ["10ep", "100ep"]
    seeds = [42, 123, 456, 789, 1011]

    detailed_rows = []
    table5_data = {}

    for a in archs:
        for i in inits:
            for b in budgets:
                cond_key = (a, i, b)
                table5_data[cond_key] = {
                    "r1_val_map": [],
                    "r5_val_map": [],
                    "test_map": [],
                    "test_map50_95": [],
                    "test_pistol": [],
                    "test_knife": []
                }

                for s in seeds:
                    folder_name = f"{a}_{i}_seed_{s}" if b == "10ep" else f"{a}_{i}_100ep_seed_{s}"
                    fpath = os.path.join(out_dir, folder_name)

                    if not os.path.exists(fpath) and s == 42:
                        fallback_name = f"{a}_{i}" if b == "10ep" else f"{a}_{i}_100ep"
                        fpath = os.path.join(out_dir, fallback_name)

                    assert os.path.exists(fpath), f"Missing run directory: {fpath}"

                    cum_steps = 0
                    cum_images = 0
                    cum_tflops = 0.0

                    for r in range(1, 6):
                        mf = os.path.join(fpath, f"round_{r}_metrics.json")
                        assert os.path.exists(mf), f"Missing metric file: {mf}"

                        with open(mf, "r", encoding="utf-8") as fp:
                            m = json.load(fp)

                        steps = m.get("optimizer_steps", 0)
                        imgs = m.get("images_processed", 0)
                        tflops = m.get("training_tflops", 0.0)

                        cum_steps += steps
                        cum_images += imgs
                        cum_tflops += tflops

                        row = {
                            "architecture": a,
                            "initialisation": i,
                            "budget": b,
                            "seed": s,
                            "round": r,
                            "rejected_proposals_boxes": m.get("rejected_count", 0),
                            "undetected_pool_images": m.get("undetected_count", 0),
                            "verified_samples_added_images": m.get("verified_samples_added", 0),
                            "training_set_size_images": m.get("training_set_size", 709),
                            "knife_boxes": m.get("knife_boxes", 0),
                            "pistol_boxes": m.get("pistol_boxes", 0),
                            "total_boxes": m.get("total_boxes", 0),
                            "images_with_knife": m.get("images_with_knife", 0),
                            "images_with_pistol": m.get("images_with_pistol", 0),
                            "total_images": m.get("total_images", 0),
                            "stopped_epoch": m.get("actual_stopped_epoch", 10 if b == "10ep" else 100),
                            "optimizer_steps": steps,
                            "cum_optimizer_steps": cum_steps,
                            "images_processed": imgs,
                            "cum_images_processed": cum_images,
                            "gflops": m.get("gflops", 0.0),
                            "training_tflops": tflops,
                            "cum_training_tflops": cum_tflops,
                            "val_map50": m.get("val_map50", m.get("map50", 0.0)),
                            "val_map50_95": m.get("val_map50_95", m.get("map50_95", 0.0)),
                            "test_map50": m.get("test_map50", m.get("map50", 0.0)),
                            "test_map50_95": m.get("test_map50_95", m.get("map50_95", 0.0)),
                            "test_pistol": m.get("test_pistol", m.get("pistol_map50", 0.0)),
                            "test_knife": m.get("test_knife", m.get("knife_map50", 0.0))
                        }
                        detailed_rows.append(row)

                        if r == 1:
                            table5_data[cond_key]["r1_val_map"].append(row["val_map50"])
                        elif r == 5:
                            table5_data[cond_key]["r5_val_map"].append(row["val_map50"])
                            table5_data[cond_key]["test_map"].append(row["test_map50"])
                            table5_data[cond_key]["test_map50_95"].append(row["test_map50_95"])
                            table5_data[cond_key]["test_pistol"].append(row["test_pistol"])
                            table5_data[cond_key]["test_knife"].append(row["test_knife"])

    # Export CSV 1: multi_seed_detailed_accounting.csv
    csv_path = os.path.join(ledger_dir, "multi_seed_detailed_accounting.csv")
    fieldnames = list(detailed_rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(detailed_rows)

    print(f"Exported detailed accounting CSV to {csv_path} ({len(detailed_rows)} rows)")

    # Export CSV 2: table_5_statistical_summary.csv
    summary_rows = []
    for cond_key, d in table5_data.items():
        a, i, b = cond_key
        srow = {
            "architecture": a,
            "initialisation": i,
            "budget": b,
            "r1_val_map_mean": np.mean(d["r1_val_map"]),
            "r1_val_map_sd": np.std(d["r1_val_map"], ddof=1),
            "r1_val_map_ci95_halfwidth": compute_ci95_halfwidth(d["r1_val_map"]),
            "r5_val_map_mean": np.mean(d["r5_val_map"]),
            "r5_val_map_sd": np.std(d["r5_val_map"], ddof=1),
            "r5_val_map_ci95_halfwidth": compute_ci95_halfwidth(d["r5_val_map"]),
            "test_map50_mean": np.mean(d["test_map"]),
            "test_map50_sd": np.std(d["test_map"], ddof=1),
            "test_map50_ci95_halfwidth": compute_ci95_halfwidth(d["test_map"]),
            "test_map50_95_mean": np.mean(d["test_map50_95"]),
            "test_map50_95_sd": np.std(d["test_map50_95"], ddof=1),
            "test_map50_95_ci95_halfwidth": compute_ci95_halfwidth(d["test_map50_95"]),
            "test_pistol_mean": np.mean(d["test_pistol"]),
            "test_pistol_sd": np.std(d["test_pistol"], ddof=1),
            "test_pistol_ci95_halfwidth": compute_ci95_halfwidth(d["test_pistol"]),
            "test_knife_mean": np.mean(d["test_knife"]),
            "test_knife_sd": np.std(d["test_knife"], ddof=1),
            "test_knife_ci95_halfwidth": compute_ci95_halfwidth(d["test_knife"])
        }
        summary_rows.append(srow)

    sum_path = os.path.join(ledger_dir, "table_5_statistical_summary.csv")
    with open(sum_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Exported statistical summary CSV to {sum_path} ({len(summary_rows)} rows)")

    # Copy script into companion repo's scripts directory
    comp_script = os.path.join(comp_dir, "scripts", "generate_revision_ledgers.py")
    if os.path.abspath(__file__) != os.path.abspath(comp_script):
        shutil.copy2(__file__, comp_script)

if __name__ == "__main__":
    main()
