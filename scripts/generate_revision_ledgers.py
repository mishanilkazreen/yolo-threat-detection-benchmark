#!/usr/bin/env python3
"""
Generate comprehensive multi-seed experimental accounting ledgers, statistical summaries,
and LaTeX table snippets for manuscript revision.

Definitions & Operational Units (audited directly from src/training/detection_validator.py):
- Rejected Proposals (rejected_proposals_boxes, BOXES): Total candidate bounding-box proposals generated during pool inference
  that failed IoU >= 0.5 or class-matching verification against pool ground truth.
- Undetected Pool Images (undetected_pool_images, IMAGES): Pool images for which the model produced no candidate detections
  during round acquisition. Because unacquired images remain in the pool for potential acquisition in future rounds,
  these counts represent the cumulative sum of unacquired pool images across Rounds 2-5, incorporating repeated counting of images
  that remain unacquired across multiple rounds (cumulative totals up to 8,884 against the fixed 2,836-image unlabelled pool).
- Verified Samples Added (verified_samples_added_images, IMAGES): Total images added to the active training set in the round.
  An image is acquired if at least one candidate bounding box proposal on that image verifies.
"""

import os
import json
import csv
from pathlib import Path
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
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    
    # Locate outputs directory from companion repo or repo root
    comp_out = Path(r"c:\Users\manig\Downloads\yolo-threat-detection-benchmark\outputs")
    if comp_out.exists():
        out_dir = comp_out
    else:
        out_dir = repo_root / "outputs"

    ledger_dir = out_dir / "revision_ledgers"
    ledger_dir.mkdir(parents=True, exist_ok=True)

    # Determine submission directory for LaTeX snippets
    manuscript_sub_dir = repo_root / "submission"

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
                    fpath = out_dir / folder_name

                    if not fpath.exists() and s == 42:
                        fallback_name = f"{a}_{i}" if b == "10ep" else f"{a}_{i}_100ep"
                        fpath = out_dir / fallback_name

                    assert fpath.exists(), f"Missing run directory: {fpath}"

                    # Load final_test_metrics.json for test evaluation
                    tfpath = fpath / "final_test_metrics.json"
                    assert tfpath.exists(), f"Missing final test metric file: {tfpath}"
                    with open(tfpath, "r", encoding="utf-8") as fp:
                        tm = json.load(fp)

                    test_m50_val = tm["metrics"]["mAP50"]
                    test_m50_95_val = tm["metrics"]["mAP50-95"]
                    # Index 0 is knife, index 1 is pistol
                    test_knife_val = tm["per_class_metrics"]["mAP50_per_class"][0]
                    test_pistol_val = tm["per_class_metrics"]["mAP50_per_class"][1]

                    cum_steps = 0
                    cum_images = 0
                    cum_tflops = 0.0
                    prev_train_size = 709

                    for r in range(1, 6):
                        mf = fpath / f"round_{r}_metrics.json"
                        assert mf.exists(), f"Missing metric file: {mf}"

                        with open(mf, "r", encoding="utf-8") as fp:
                            m = json.load(fp)

                        # Fail-loud key access for audit integrity
                        steps = m["optimizer_steps"]
                        imgs = m["images_processed"]
                        gflops = 8.7 if a == "yolov8n" else 6.5
                        tflops = round(3.0 * gflops * imgs / 1000.0, 1)
                        train_size = m["training_set_size"]
                        verified_added = m["verified_samples_added"] if r > 1 else 0
                        rej_count = m["rejected_count"] if r > 1 else 0
                        und_count = m["undetected_count"] if r > 1 else 0

                        # Size-consistency verification
                        if r == 1:
                            assert train_size == 709, f"Round 1 train size {train_size} != 709 in {mf}"
                        else:
                            assert train_size == prev_train_size + verified_added, \
                                f"Inconsistency in {mf}: train_size ({train_size}) != prev ({prev_train_size}) + verified ({verified_added})"
                        prev_train_size = train_size

                        cum_steps += steps
                        cum_images += imgs
                        cum_tflops = round(3.0 * gflops * cum_images / 1000.0, 1)

                        val_m50 = m["metrics"]["mAP50"]
                        val_m50_95 = m["metrics"]["mAP50-95"]

                        class_dist = m["class_distribution"]

                        row = {
                            "architecture": a,
                            "initialisation": i,
                            "budget": b,
                            "seed": s,
                            "round": r,
                            "rejected_proposals_boxes": rej_count,
                            "undetected_pool_images": und_count,
                            "verified_samples_added_images": verified_added,
                            "training_set_size_images": train_size,
                            "knife_boxes": class_dist["knife_boxes"],
                            "pistol_boxes": class_dist["pistol_boxes"],
                            "total_boxes": class_dist["total_boxes"],
                            "images_with_knife": class_dist["images_with_knife"],
                            "images_with_pistol": class_dist["images_with_pistol"],
                            "total_images": class_dist["total_images"],
                            "stopped_epoch": m["actual_stopped_epoch"],
                            "optimizer_steps": steps,
                            "cum_optimizer_steps": cum_steps,
                            "images_processed": imgs,
                            "cum_images_processed": cum_images,
                            "gflops": gflops,
                            "training_tflops": tflops,
                            "cum_training_tflops": cum_tflops,
                            "val_map50": val_m50,
                            "val_map50_95": val_m50_95,
                            "test_map50": test_m50_val,
                            "test_map50_95": test_m50_95_val,
                            "test_pistol": test_pistol_val,
                            "test_knife": test_knife_val
                        }
                        detailed_rows.append(row)

                        if r == 1:
                            table5_data[cond_key]["r1_val_map"].append(val_m50)
                        elif r == 5:
                            table5_data[cond_key]["r5_val_map"].append(val_m50)
                            table5_data[cond_key]["test_map"].append(test_m50_val)
                            table5_data[cond_key]["test_map50_95"].append(test_m50_95_val)
                            table5_data[cond_key]["test_pistol"].append(test_pistol_val)
                            table5_data[cond_key]["test_knife"].append(test_knife_val)

    # Strict 300-row assertion for completeness (12 conditions * 5 seeds * 5 rounds)
    assert len(detailed_rows) == 300, f"Expected 300 rows, got {len(detailed_rows)}"

    # Export CSV 1: multi_seed_detailed_accounting.csv
    csv_path = ledger_dir / "multi_seed_detailed_accounting.csv"
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

    sum_path = ledger_dir / "table_5_statistical_summary.csv"
    with open(sum_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Exported statistical summary CSV to {sum_path} ({len(summary_rows)} rows)")

    # -------------------------------------------------------------------------
    # Generate LaTeX Table Snippets directly from audited CSV data
    # -------------------------------------------------------------------------

    # 1. supp_table3.tex (Supplementary Table 3)
    # Computed pooled across 3 architectures x 5 seeds = 15 runs per condition group
    t3_groups = {
        ("pretrained", "10ep"): {"cum": {r: [] for r in range(1, 6)}, "new": {r: [] for r in range(2, 6)}},
        ("random", "10ep"): {"cum": {r: [] for r in range(1, 6)}, "new": {r: [] for r in range(2, 6)}},
        ("pretrained", "100ep"): {"cum": {r: [] for r in range(1, 6)}, "new": {r: [] for r in range(2, 6)}},
        ("random", "100ep"): {"cum": {r: [] for r in range(1, 6)}, "new": {r: [] for r in range(2, 6)}},
    }

    for row in detailed_rows:
        grp = (row["initialisation"], row["budget"])
        r = row["round"]
        t3_groups[grp]["cum"][r].append(row["training_set_size_images"])
        if r > 1:
            t3_groups[grp]["new"][r].append(row["verified_samples_added_images"])

    def fmt_t3_cum(grp_key):
        res = ["709"]
        for r in range(2, 6):
            arr = t3_groups[grp_key]["cum"][r]
            m_val = int(round(np.mean(arr)))
            sd_val = int(round(np.std(arr, ddof=1)))
            res.append(f"{m_val:,} $\\pm$ {sd_val}")
        return " & ".join(res)

    def fmt_t3_new(grp_key):
        res = ["{--}"]
        for r in range(2, 6):
            arr = t3_groups[grp_key]["new"][r]
            m_val = int(round(np.mean(arr)))
            sd_val = int(round(np.std(arr, ddof=1)))
            res.append(f"$+${m_val:,} $\\pm$ {sd_val}")
        return " & ".join(res)

    t3_tex = f"""\\begin{{table*}}[hbt!]
\\caption{{Actual cumulative training set sizes across incremental rounds (mean $\\pm$ std over $N=15$ runs per condition group, pooled across three architectures and 5 seeds) and ground-truth split accounting for both 10-epoch and 100-epoch budgets. Round~1 is fixed at 709 images for all conditions. Bounding-box instances exceed unique image counts because 296 images across the dataset contain multiple instances of the same class (38 in train\\_init, 165 in pool, 63 in val, and 30 in test); zero images contain both classes simultaneously.}}%
\\label{{tab:supp_dataset}}
\\centering
\\footnotesize
\\setlength{{\\tabcolsep}}{{4pt}}
\\begin{{tabular}}{{l c c c c c}}
\\toprule
\\textbf{{Condition}} & \\textbf{{Round 1}} & \\textbf{{Round 2}} & \\textbf{{Round 3}} & \\textbf{{Round 4}} & \\textbf{{Round 5}} \\\\
\\midrule
\\multicolumn{{6}}{{l}}{{\\textit{{10 Epochs / Round Budget}}}} \\\\
COCO-Pretrained (10 ep) & {fmt_t3_cum(('pretrained', '10ep'))} \\\\
(new per round) & {fmt_t3_new(('pretrained', '10ep'))} \\\\
Random Init (10 ep) & {fmt_t3_cum(('random', '10ep'))} \\\\
(new per round) & {fmt_t3_new(('random', '10ep'))} \\\\
\\midrule
\\multicolumn{{6}}{{l}}{{\\textit{{100 Epochs / Round Budget}}}} \\\\
COCO-Pretrained (100 ep) & {fmt_t3_cum(('pretrained', '100ep'))} \\\\
(new per round) & {fmt_t3_new(('pretrained', '100ep'))} \\\\
Random Init (100 ep) & {fmt_t3_cum(('random', '100ep'))} \\\\
(new per round) & {fmt_t3_new(('random', '100ep'))} \\\\
\\midrule
\\multicolumn{{6}}{{l}}{{\\textit{{Fixed Ground-Truth Dataset Partition Accounting}}}} \\\\
Train Initial ($\\mathcal{{D}}_{{\\text{{init}}}}$) & \\multicolumn{{5}}{{l}}{{709 images (291 knife, 418 pistol), 784 boxes (306 knife [39.0\\%], 478 pistol [61.0\\%])}} \\\\
Unlabelled Pool ($\\mathcal{{U}}$) & \\multicolumn{{5}}{{l}}{{2,836 images (1,164 knife, 1,672 pistol), 3,133 boxes (1,207 knife [38.5\\%], 1,926 pistol [61.5\\%])}} \\\\
Validation ($\\mathcal{{D}}_{{\\text{{val}}}}$) & \\multicolumn{{5}}{{l}}{{1,013 images (416 knife, 597 pistol), 1,130 boxes (426 knife [37.7\\%], 704 pistol [62.3\\%])}} \\\\
Test Held-Out ($\\mathcal{{D}}_{{\\text{{test}}}}$) & \\multicolumn{{5}}{{l}}{{506 images (207 knife, 299 pistol), 556 boxes (216 knife [38.8\\%], 340 pistol [61.2\\%])}} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table*}}
"""

    # 2. supp_table4.tex (Supplementary Table 4)
    # Computed per condition across N=5 seeds
    t4_data = {}
    for a in archs:
        for i in inits:
            for b in budgets:
                t4_data[(a, i, b)] = {
                    "ver": [], "rej": [], "undet": [], "steps": [], "tflops": []
                }

    # Aggregate per run (5 seeds)
    run_aggs = {}
    for row in detailed_rows:
        run_key = (row["architecture"], row["initialisation"], row["budget"], row["seed"])
        if run_key not in run_aggs:
            run_aggs[run_key] = {"ver": 0, "rej": 0, "undet": 0, "steps": 0, "tflops": 0.0}
        if row["round"] > 1:
            run_aggs[run_key]["ver"] += row["verified_samples_added_images"]
            run_aggs[run_key]["rej"] += row["rejected_proposals_boxes"]
            run_aggs[run_key]["undet"] += row["undetected_pool_images"]
        run_aggs[run_key]["steps"] += row["optimizer_steps"]
        if row["round"] == 5:
            run_aggs[run_key]["tflops"] = row["cum_training_tflops"]

    for (a, i, b, s), rdict in run_aggs.items():
        t4_data[(a, i, b)]["ver"].append(rdict["ver"])
        t4_data[(a, i, b)]["rej"].append(rdict["rej"])
        t4_data[(a, i, b)]["undet"].append(rdict["undet"])
        t4_data[(a, i, b)]["steps"].append(rdict["steps"])
        t4_data[(a, i, b)]["tflops"].append(rdict["tflops"])

    t4_rows_tex = []
    for a in archs:
        for i in inits:
            for b in budgets:
                d = t4_data[(a, i, b)]
                v_m, v_sd = np.mean(d["ver"]), np.std(d["ver"], ddof=1)
                r_m, r_sd = np.mean(d["rej"]), np.std(d["rej"], ddof=1)
                u_m, u_sd = np.mean(d["undet"]), np.std(d["undet"], ddof=1)
                s_m, s_sd = np.mean(d["steps"]), np.std(d["steps"], ddof=1)
                t_m, t_sd = np.mean(d["tflops"]), np.std(d["tflops"], ddof=1)

                arch_disp = {"yolov8n": "YOLOv8n", "yolo11n": "YOLO11n", "yolo12n": "YOLO12n"}[a]
                init_disp = {"pretrained": "COCO", "random": "Rand."}[i]
                b_disp = "10 ep" if b == "10ep" else "100 ep"

                line = f"{arch_disp} & {init_disp} & {b_disp} & {v_m:,.0f} $\\pm$ {v_sd:.0f} & {r_m:,.0f} $\\pm$ {r_sd:.0f} & {u_m:,.0f} $\\pm$ {u_sd:.0f} & {s_m:,.0f} $\\pm$ {s_sd:,.0f} & {t_m:,.1f} $\\pm$ {t_sd:,.1f} \\\\"
                t4_rows_tex.append(line)
        if a != archs[-1]:
            t4_rows_tex.append("\\midrule")

    t4_body = "\n".join(t4_rows_tex)

    t4_tex = f"""\\begin{{table*}}[hbt!]
\\caption{{Comprehensive active acquisition proposal accounting and training compute metrics (mean $\\pm$ std over $N=5$ seeds per condition). Verified (Img), Rej.\\ Boxes, and Undet.\\ Pool (Img) cover Rounds 2--5; Cumul.\\ Steps and Cumul.\\ TFLOPs cover Rounds 1--5. Undet.\\ Pool (Img) counts unlabelled pool images for which the model produced no candidate detections during round acquisition (summed across Rounds 2--5 with repeated counting of unacquired pool images across rounds).}}%
\\label{{tab:supp_compute_accounting}}
\\centering
\\scriptsize
\\setlength{{\\tabcolsep}}{{1.2pt}}
\\begin{{tabular}}{{llcccccc}}
\\toprule
\\textbf{{Model}} & \\textbf{{Init.}} & \\textbf{{Budget}} & \\textbf{{Verified (Img)}} & \\textbf{{Rej.\\ Boxes}} & \\textbf{{Undet.\\ Pool (Img)}} & \\textbf{{Cumul.\\ Steps}} & \\textbf{{Cumul.\\ TFLOPs}} \\\\
\\midrule
{t4_body}
\\bottomrule
\\end{{tabular}}
\\end{{table*}}
"""

    # 3. supp_table_ci.tex (Supplementary Table of exact 95% Confidence Intervals for Table 5)
    ci_rows_tex = []
    for srow in summary_rows:
        a, i, b = srow["architecture"], srow["initialisation"], srow["budget"]
        arch_disp = {"yolov8n": "YOLOv8n", "yolo11n": "YOLO11n", "yolo12n": "YOLO12n"}[a]
        init_disp = {"pretrained": "COCO", "random": "Rand."}[i]
        b_disp = "10 ep" if b == "10ep" else "100 ep"

        r1_ci = f"$\\pm${srow['r1_val_map_ci95_halfwidth']:.3f}"
        r5_ci = f"$\\pm${srow['r5_val_map_ci95_halfwidth']:.3f}"
        test_ci = f"$\\pm${srow['test_map50_ci95_halfwidth']:.3f}"
        m50_95_ci = f"$\\pm${srow['test_map50_95_ci95_halfwidth']:.3f}"
        pistol_ci = f"$\\pm${srow['test_pistol_ci95_halfwidth']:.3f}"
        knife_ci = f"$\\pm${srow['test_knife_ci95_halfwidth']:.3f}"

        line = f"{arch_disp} & {init_disp} & {b_disp} & {r1_ci} & {r5_ci} & {test_ci} & {m50_95_ci} & {pistol_ci} & {knife_ci} \\\\"
        ci_rows_tex.append(line)
        if b == "100ep" and not (a == "yolo12n" and i == "random"):
            ci_rows_tex.append("\\midrule")

    ci_body = "\n".join(ci_rows_tex)

    ci_tex = f"""\\begin{{table*}}[hbt!]
\\caption{{Exact 95\\% Student-$t$ confidence interval half-widths ($\\pm$, $N=5$ independent seeds per condition, $t_{{0.975,4}} = 2.7764$) corresponding to the multi-seed threat detection benchmark in Table~5. All values computed directly from \\texttt{{table\\_5\\_statistical\\_summary.csv}}.}}%
\\label{{tab:supp_ci95}}
\\centering
\\scriptsize
\\setlength{{\\tabcolsep}}{{3.5pt}}
\\begin{{tabular}}{{lllcccccc}}
\\toprule
\\textbf{{Model}} & \\textbf{{Init.}} & \\textbf{{Budget}} & \\textbf{{Val.\\ R1 95\\% CI}} & \\textbf{{Val.\\ R5 95\\% CI}} & \\textbf{{Held-Out Test 95\\% CI}} & \\textbf{{mAP}}$_{{50:95}}$ \\textbf{{95\\% CI}} & \\textbf{{Pistol 95\\% CI}} & \\textbf{{Knife 95\\% CI}} \\\\
\\midrule
{ci_body}
\\bottomrule
\\end{{tabular}}
\\end{{table*}}
"""

    # 4. supp_table_per_round.tex (Supplementary Table of compact per-round condition-level metrics)
    pr_rows_tex = []
    for a in archs:
        for i in inits:
            for b in budgets:
                for r in range(1, 6):
                    sub_rows = [row for row in detailed_rows if row["architecture"] == a and row["initialisation"] == i and row["budget"] == b and row["round"] == r]
                    if r == 1:
                        ver_str = "{--}"
                        rej_str = "{--}"
                        und_str = "{--}"
                    else:
                        ver_m, ver_sd = np.mean([row["verified_samples_added_images"] for row in sub_rows]), np.std([row["verified_samples_added_images"] for row in sub_rows], ddof=1)
                        rej_m, rej_sd = np.mean([row["rejected_proposals_boxes"] for row in sub_rows]), np.std([row["rejected_proposals_boxes"] for row in sub_rows], ddof=1)
                        und_m, und_sd = np.mean([row["undetected_pool_images"] for row in sub_rows]), np.std([row["undetected_pool_images"] for row in sub_rows], ddof=1)
                        ver_str = f"{ver_m:,.0f} $\\pm$ {ver_sd:.0f}"
                        rej_str = f"{rej_m:,.0f} $\\pm$ {rej_sd:.0f}"
                        und_str = f"{und_m:,.0f} $\\pm$ {und_sd:.0f}"

                    k_m, k_sd = np.mean([row["knife_boxes"] for row in sub_rows]), np.std([row["knife_boxes"] for row in sub_rows], ddof=1)
                    p_m, p_sd = np.mean([row["pistol_boxes"] for row in sub_rows]), np.std([row["pistol_boxes"] for row in sub_rows], ddof=1)
                    steps_m, steps_sd = np.mean([row["optimizer_steps"] for row in sub_rows]), np.std([row["optimizer_steps"] for row in sub_rows], ddof=1)
                    imgs_m, imgs_sd = np.mean([row["images_processed"] for row in sub_rows]), np.std([row["images_processed"] for row in sub_rows], ddof=1)
                    ep_m, ep_sd = np.mean([row["stopped_epoch"] for row in sub_rows]), np.std([row["stopped_epoch"] for row in sub_rows], ddof=1)

                    arch_disp = {"yolov8n": "YOLOv8n", "yolo11n": "YOLO11n", "yolo12n": "YOLO12n"}[a]
                    init_disp = {"pretrained": "COCO", "random": "Rand."}[i]
                    b_disp = "10 ep" if b == "10ep" else "100 ep"

                    line = f"{arch_disp} & {init_disp} & {b_disp} & R{r} & {ver_str} & {rej_str} & {und_str} & {k_m:,.0f} $\\pm$ {k_sd:.0f} & {p_m:,.0f} $\\pm$ {p_sd:.0f} & {steps_m:,.0f} $\\pm$ {steps_sd:.0f} & {imgs_m:,.0f} $\\pm$ {imgs_sd:.0f} & {ep_m:.1f} $\\pm$ {ep_sd:.1f} \\\\"
                    pr_rows_tex.append(line)
                pr_rows_tex.append("\\midrule")

    if pr_rows_tex and pr_rows_tex[-1] == "\\midrule":
        pr_rows_tex.pop()

    pr_body = "\n".join(pr_rows_tex)

    pr_tex = f"""\\begin{{table*}}[hbt!]
\\caption{{Granular per-round active data acquisition, class distribution, and training execution metrics across all 12 experimental conditions (mean $\\pm$ std over $N=5$ seeds per condition). Verified (Img), Rej.\\ Boxes, and Undet.\\ Pool (Img) report per-round active acquisition; Cum.\\ Knife Boxes and Cum.\\ Pistol Boxes report cumulative training set ground-truth box counts; Steps and Images report per-round executed optimizer steps and images processed; Stop Ep reports mean $\\pm$ std executed epochs per round.}}%
\\label{{tab:supp_per_round}}
\\centering
\\tiny
\\setlength{{\\tabcolsep}}{{1.8pt}}
\\begin{{tabular}}{{lllccccccccc}}
\\toprule
\\textbf{{Model}} & \\textbf{{Init.}} & \\textbf{{Budget}} & \\textbf{{Rnd}} & \\textbf{{Verified (Img)}} & \\textbf{{Rej.\\ Boxes}} & \\textbf{{Undet.\\ Pool (Img)}} & \\textbf{{Cum.\\ Knife}} & \\textbf{{Cum.\\ Pistol}} & \\textbf{{Steps}} & \\textbf{{Images}} & \\textbf{{Stop Ep}} \\\\
\\midrule
{pr_body}
\\bottomrule
\\end{{tabular}}
\\end{{table*}}
"""

    # Write LaTeX snippet files to ledger_dir AND manuscript_sub_dir if available
    snippets = {
        "supp_table3.tex": t3_tex,
        "supp_table4.tex": t4_tex,
        "supp_table_ci.tex": ci_tex,
        "supp_table_per_round.tex": pr_tex
    }

    for name, content in snippets.items():
        target_path = ledger_dir / name
        with open(target_path, "w", encoding="utf-8") as fp:
            fp.write(content)
        print(f"Generated LaTeX snippet {target_path}")

        if manuscript_sub_dir.exists():
            man_target = manuscript_sub_dir / name
            with open(man_target, "w", encoding="utf-8") as fp:
                fp.write(content)
            print(f"Copied LaTeX snippet to {man_target}")

if __name__ == "__main__":
    main()
