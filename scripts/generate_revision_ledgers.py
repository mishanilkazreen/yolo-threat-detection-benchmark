#!/usr/bin/env python3
"""
Generate comprehensive multi-seed experimental accounting ledgers and statistical summary tables for manuscript revision.

Definitions & Operational Units (audited directly from src/training/detection_validator.py):
- Rejected Proposals (rejected_proposals_boxes, BOXES): Total candidate bounding-box proposals generated during pool inference
  that failed IoU >= 0.5 or class-matching verification against pool ground truth.
- Undetected Pool Images (undetected_pool_images, IMAGES): Total pool images where no candidate proposals met verification thresholds.
  These images remain in the pool for potential acquisition in future rounds.
- Verified Samples Added (verified_samples_added_images, IMAGES): Total images added to the active training set in the round.
  An image is acquired if at least one candidate bounding box proposal on that image verifies.
"""

import os
import json
import csv
import numpy as np
from scipy import stats

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(base_dir, "outputs")
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
                b_str = "" if b == "10ep" else "_100ep"
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
                    p = os.path.join(out_dir, folder_name)

                    if not os.path.isdir(p):
                        raise RuntimeError(f"Required run directory missing: {p}")

                    test_map50 = None
                    test_map50_95 = None
                    test_pistol = None
                    test_knife = None
                    
                    ft_path = os.path.join(p, "final_test_metrics.json")
                    if os.path.exists(ft_path):
                        with open(ft_path, "r", encoding="utf-8") as f:
                            ft = json.load(f)
                        met_test = ft.get("metrics", {})
                        test_map50 = met_test.get("mAP50", None)
                        test_map50_95 = met_test.get("mAP50_95", None)
                        test_pistol = met_test.get("pistol_mAP50", None)
                        test_knife = met_test.get("knife_mAP50", None)

                    prev_set_size = 709
                    for r in range(1, 6):
                        m_path = os.path.join(p, f"round_{r}_metrics.json")
                        if not os.path.exists(m_path):
                            raise RuntimeError(f"Required round metrics file missing: {m_path}")
                        
                        with open(m_path, "r", encoding="utf-8") as f:
                            m = json.load(f)

                        cd = m.get("class_distribution", {})
                        met = m.get("metrics", {})

                        current_size = m["training_set_size"]
                        added = m["verified_samples_added"]
                        if current_size != prev_set_size + added:
                            raise ValueError(f"Consistency failure in {folder_name} R{r}: {current_size} != {prev_set_size} + {added}")
                        prev_set_size = current_size

                        if r >= 2:
                            if "rejected_count" not in m or "undetected_count" not in m:
                                raise KeyError(f"Missing rejected_count or undetected_count in {folder_name} R{r}")
                        rej_boxes = m.get("rejected_count", 0)
                        undet_img = m.get("undetected_count", 0)

                        row = {
                            "architecture": a,
                            "initialisation": i,
                            "budget_epochs": 10 if b == "10ep" else 100,
                            "seed": s,
                            "round": r,
                            "training_set_size": current_size,
                            "verified_samples_added_images": added,
                            "rejected_proposals_boxes": rej_boxes,
                            "undetected_pool_images": undet_img,
                            "knife_boxes": cd.get("knife_boxes", 0),
                            "pistol_boxes": cd.get("pistol_boxes", 0),
                            "total_boxes": cd.get("total_boxes", 0),
                            "images_with_knife": cd.get("images_with_knife", 0),
                            "images_with_pistol": cd.get("images_with_pistol", 0),
                            "total_images": cd.get("total_images", 0),
                            "actual_stopped_epoch": m.get("actual_stopped_epoch", 0),
                            "optimizer_steps": m.get("optimizer_steps", 0),
                            "cumulative_optimizer_steps": m.get("cumulative_optimizer_steps", 0),
                            "images_processed": m.get("images_processed", 0),
                            "cumulative_images_processed": m.get("cumulative_images_processed", 0),
                            "gflops": m.get("gflops", 0.0),
                            "training_tflops": m.get("training_tflops", 0.0),
                            "cumulative_training_tflops": m.get("cumulative_training_tflops", 0.0),
                            "val_mAP50": met.get("mAP50", 0.0),
                            "val_precision": met.get("precision", 0.0),
                            "val_recall": met.get("recall", 0.0),
                            "final_held_out_test_mAP50": test_map50 if r == 5 else None
                        }
                        detailed_rows.append(row)

                        if r == 1:
                            table5_data[cond_key]["r1_val_map"].append(met.get("mAP50", 0.0))
                        elif r == 5:
                            table5_data[cond_key]["r5_val_map"].append(met.get("mAP50", 0.0))
                            if test_map50 is not None:
                                table5_data[cond_key]["test_map"].append(test_map50)
                            if test_map50_95 is not None:
                                table5_data[cond_key]["test_map50_95"].append(test_map50_95)
                            if test_pistol is not None:
                                table5_data[cond_key]["test_pistol"].append(test_pistol)
                            if test_knife is not None:
                                table5_data[cond_key]["test_knife"].append(test_knife)

    assert len(detailed_rows) == 300, f"Expected 300 detailed rows, got {len(detailed_rows)}"

    fieldnames = list(detailed_rows[0].keys())
    with open(os.path.join(ledger_dir, "multi_seed_detailed_accounting.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(detailed_rows)

    def get_stats(arr):
        n = len(arr)
        if n == 0:
            return 0.0, 0.0, 0.0
        m = float(np.mean(arr))
        sd = float(np.std(arr, ddof=1)) if n > 1 else 0.0
        t_crit = float(stats.t.ppf(0.975, df=n-1)) if n > 1 else 0.0
        ci_halfwidth = float(t_crit * (sd / np.sqrt(n))) if n > 1 else 0.0
        return m, sd, ci_halfwidth

    stat_rows = []
    for (a, i, b), data in table5_data.items():
        r1_m, r1_sd, r1_ci = get_stats(data["r1_val_map"])
        r5_m, r5_sd, r5_ci = get_stats(data["r5_val_map"])
        t_m, t_sd, t_ci = get_stats(data["test_map"])
        t5095_m, t5095_sd, t5095_ci = get_stats(data["test_map50_95"])
        tpis_m, tpis_sd, tpis_ci = get_stats(data["test_pistol"])
        tkni_m, tkni_sd, tkni_ci = get_stats(data["test_knife"])

        stat_rows.append({
            "architecture": a,
            "initialisation": i,
            "budget_epochs": 10 if b == "10ep" else 100,
            "r1_val_mAP_mean": r1_m,
            "r1_val_mAP_sd": r1_sd,
            "r1_val_mAP_ci95_halfwidth": r1_ci,
            "r5_val_mAP_mean": r5_m,
            "r5_val_mAP_sd": r5_sd,
            "r5_val_mAP_ci95_halfwidth": r5_ci,
            "held_out_test_mAP50_mean": t_m,
            "held_out_test_mAP50_sd": t_sd,
            "held_out_test_mAP50_ci95_halfwidth": t_ci,
            "held_out_test_mAP50_95_mean": t5095_m,
            "held_out_test_mAP50_95_sd": t5095_sd,
            "held_out_test_mAP50_95_ci95_halfwidth": t_ci,
            "held_out_test_pistol_mAP50_mean": tpis_m,
            "held_out_test_pistol_mAP50_sd": tpis_sd,
            "held_out_test_pistol_mAP50_ci95_halfwidth": tpis_ci,
            "held_out_test_knife_mAP50_mean": tkni_m,
            "held_out_test_knife_mAP50_sd": tkni_sd,
            "held_out_test_knife_mAP50_ci95_halfwidth": tkni_ci
        })

    assert len(stat_rows) == 12, f"Expected 12 statistical summary rows, got {len(stat_rows)}"

    with open(os.path.join(ledger_dir, "table_5_statistical_summary.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(stat_rows[0].keys()))
        writer.writeheader()
        writer.writerows(stat_rows)

    print("Successfully generated detailed accounting ledgers and statistical summaries in:", ledger_dir)

if __name__ == "__main__":
    main()
