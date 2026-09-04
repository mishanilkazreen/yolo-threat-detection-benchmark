"""Regenerate outputs/full_evaluation_results.csv from all completed experiments."""

import csv
import json
from pathlib import Path

outputs = Path("outputs")
rows: list[dict] = []


def _pc(pc: dict, key: str, idx: int) -> float:
    vals = pc.get(key, [0, 0])
    return round(vals[idx], 4) if idx < len(vals) else 0.0


# --- Incremental results ---
configs_incremental = sorted(
    d.name for d in outputs.iterdir() if d.is_dir() and (d / "round_1_metrics.json").exists()
)

for cfg in configs_incremental:
    cfg_dir = outputs / cfg
    for r in range(1, 6):
        rf = cfg_dir / f"round_{r}_metrics.json"
        if not rf.exists():
            continue
        data = json.loads(rf.read_text(encoding="utf-8"))
        m = data.get("metrics", {})
        pc = data.get("per_class_metrics", {})
        rows.append(
            {
                "config": cfg,
                "phase": "round",
                "round": r,
                "training_set_size": data.get("training_set_size", ""),
                "verified_samples_added": data.get("verified_samples_added", ""),
                "training_time_seconds": round(data.get("training_time_seconds", 0), 2),
                "validation_time_seconds": round(data.get("validation_time_seconds", 0), 2),
                "actual_stopped_epoch": data.get("actual_stopped_epoch", ""),
                "best_epoch": data.get("best_epoch", ""),
                "mAP50": round(m.get("mAP50", 0), 4),
                "mAP50_95": round(m.get("mAP50-95", 0), 4),
                "precision": round(m.get("precision", 0), 4),
                "recall": round(m.get("recall", 0), 4),
                "f1_score": round(m.get("f1_score", 0), 4),
                "knife_mAP50": _pc(pc, "mAP50_per_class", 0),
                "pistol_mAP50": _pc(pc, "mAP50_per_class", 1),
                "knife_precision": _pc(pc, "precision_per_class", 0),
                "pistol_precision": _pc(pc, "precision_per_class", 1),
                "knife_recall": _pc(pc, "recall_per_class", 0),
                "pistol_recall": _pc(pc, "recall_per_class", 1),
                "knife_f1": _pc(pc, "f1_score_per_class", 0),
                "pistol_f1": _pc(pc, "f1_score_per_class", 1),
            }
        )

    # Final test for incremental
    ftf = cfg_dir / "final_test_metrics.json"
    if ftf.exists():
        data = json.loads(ftf.read_text(encoding="utf-8"))
        m = data.get("metrics", {})
        pc = data.get("per_class_metrics", {})
        rows.append(
            {
                "config": cfg,
                "phase": "final_test",
                "round": "",
                "training_set_size": "",
                "verified_samples_added": "",
                "training_time_seconds": round(data.get("total_training_time_seconds", 0), 2),
                "validation_time_seconds": round(data.get("test_time_seconds", 0), 2),
                "actual_stopped_epoch": "",
                "best_epoch": "",
                "mAP50": round(m.get("mAP50", 0), 4),
                "mAP50_95": round(m.get("mAP50-95", 0), 4),
                "precision": round(m.get("precision", 0), 4),
                "recall": round(m.get("recall", 0), 4),
                "f1_score": round(m.get("f1_score", 0), 4),
                "knife_mAP50": _pc(pc, "mAP50_per_class", 0),
                "pistol_mAP50": _pc(pc, "mAP50_per_class", 1),
                "knife_precision": _pc(pc, "precision_per_class", 0),
                "pistol_precision": _pc(pc, "precision_per_class", 1),
                "knife_recall": _pc(pc, "recall_per_class", 0),
                "pistol_recall": _pc(pc, "recall_per_class", 1),
                "knife_f1": _pc(pc, "f1_score_per_class", 0),
                "pistol_f1": _pc(pc, "f1_score_per_class", 1),
            }
        )

# --- Baseline results (in {config}_baseline/train/ directories) ---
baseline_dirs = sorted(
    d
    for d in outputs.iterdir()
    if d.is_dir()
    and d.name.endswith("_baseline")
    and (d / "train" / "final_test_metrics.json").exists()
)

for bd in baseline_dirs:
    cfg_name = bd.name.replace("_baseline", "")
    ftf = bd / "train" / "final_test_metrics.json"
    data = json.loads(ftf.read_text(encoding="utf-8"))
    test_m = data.get("test_metrics", {})
    # Per-class may be nested inside test_metrics or at top level
    test_pc = data.get("per_class_test_metrics", test_m.get("per_class_metrics", {}))
    train_time = data.get("training_time_seconds", 0)
    epochs = data.get("epochs", "")

    rows.append(
        {
            "config": cfg_name,
            "phase": "baseline",
            "round": "",
            "training_set_size": data.get("training_set_size", ""),
            "verified_samples_added": "",
            "training_time_seconds": round(train_time, 2),
            "validation_time_seconds": "",
            "actual_stopped_epoch": epochs,
            "best_epoch": "",
            "mAP50": round(test_m.get("mAP50", 0), 4),
            "mAP50_95": round(test_m.get("mAP50-95", 0), 4),
            "precision": round(test_m.get("precision", 0), 4),
            "recall": round(test_m.get("recall", 0), 4),
            "f1_score": round(test_m.get("f1_score", 0), 4),
            "knife_mAP50": _pc(test_pc, "mAP50_per_class", 0) if test_pc else 0,
            "pistol_mAP50": _pc(test_pc, "mAP50_per_class", 1) if test_pc else 0,
            "knife_precision": _pc(test_pc, "precision_per_class", 0) if test_pc else 0,
            "pistol_precision": _pc(test_pc, "precision_per_class", 1) if test_pc else 0,
            "knife_recall": _pc(test_pc, "recall_per_class", 0) if test_pc else 0,
            "pistol_recall": _pc(test_pc, "recall_per_class", 1) if test_pc else 0,
            "knife_f1": _pc(test_pc, "f1_score_per_class", 0) if test_pc else 0,
            "pistol_f1": _pc(test_pc, "f1_score_per_class", 1) if test_pc else 0,
        }
    )

# Write CSV
out_path = outputs / "full_evaluation_results.csv"
fieldnames = list(rows[0].keys())
with open(out_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Written {len(rows)} rows to {out_path}")
print(f"  Incremental configs: {len(configs_incremental)}")
print(f"  Baseline configs: {len(baseline_dirs)}")
print()
print("Baseline results summary:")
for r in rows:
    if r["phase"] == "baseline":
        print(
            f"  {r['config']:<45} mAP50={r['mAP50']}  F1={r['f1_score']}"
            f"  epochs={r['actual_stopped_epoch']}  time={r['training_time_seconds']}s"
        )
