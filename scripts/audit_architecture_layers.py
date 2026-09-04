"""
Architecture Layer & Freeze Parameter Auditor.

Addresses Reviewer 1 Minor 5, 6, and 7 (TASK-MIN-05, TASK-MIN-06, TASK-MIN-07):
  - Identifies generation-specific layer indices for YOLOv8n, YOLO11n, and YOLOv12n.
  - Computes exact parameter counts and percentages for:
      * Full fine-tuning (all parameters trainable)
      * Frozen backbone (layers 0..backbone_end frozen)
      * Detection head-only (backbone and neck frozen)
  - Details exact weight retention and re-initialization behavior when loading 80-class
    COCO weights for the 2-class threat detection setup.
  - Generates publication-ready LaTeX tables for manuscript / supplementary insertion.

Usage:
    uv run python scripts/audit_architecture_layers.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
from typing import Any

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

# Generation-specific layer boundaries
ARCH_BOUNDARIES = {
    "yolov8n": {
        "model_file": "yolov8n.pt",
        "backbone_indices": list(range(0, 10)),   # 0..9 (Conv, C2f, SPPF)
        "neck_indices": list(range(10, 22)),       # 10..21
        "head_indices": [22],                      # 22 (Detect)
        "freeze_backbone_val": 10,
        "freeze_headonly_val": 22,
    },
    "yolo11n": {
        "model_file": "yolo11n.pt",
        "backbone_indices": list(range(0, 11)),   # 0..10 (Conv, C3k2, SPPF, C2PSA)
        "neck_indices": list(range(11, 23)),       # 11..22
        "head_indices": [23],                      # 23 (Detect)
        "freeze_backbone_val": 11,
        "freeze_headonly_val": 23,
    },
    "yolo12n": {
        "model_file": "yolo12n.pt",
        "backbone_indices": list(range(0, 9)),    # 0..8 (Conv, C3k2, A2C2f)
        "neck_indices": list(range(9, 21)),        # 9..20
        "head_indices": [21],                      # 21 (Detect)
        "freeze_backbone_val": 9,
        "freeze_headonly_val": 21,
    },
}


def audit_single_architecture(arch_name: str, meta: dict[str, Any]) -> dict[str, Any]:
    """Inspect model layers, count parameters per component, and evaluate freeze strategies."""
    model = YOLO(meta["model_file"])
    net = model.model
    layers = list(net.model)

    total_params = sum(p.numel() for p in net.parameters())

    layer_details: list[dict[str, Any]] = []
    backbone_params = 0
    neck_params = 0
    head_params = 0

    for idx, layer in enumerate(layers):
        l_params = sum(p.numel() for p in layer.parameters())
        l_type = type(layer).__name__

        if idx in meta["backbone_indices"]:
            component = "Backbone"
            backbone_params += l_params
        elif idx in meta["neck_indices"]:
            component = "Neck"
            neck_params += l_params
        elif idx in meta["head_indices"]:
            component = "Head"
            head_params += l_params
        else:
            component = "Other"

        layer_details.append({
            "index": idx,
            "type": l_type,
            "component": component,
            "parameters": l_params,
        })

    # Strategy 1: Full fine-tuning
    full_trainable = total_params
    full_frozen = 0

    # Strategy 2: Frozen Backbone
    # When freeze=K is passed to Ultralytics, layers 0..(K-1) have requires_grad=False
    k_bb = meta["freeze_backbone_val"]
    bb_frozen_params = sum(
        sum(p.numel() for p in layers[i].parameters()) for i in range(k_bb)
    )
    bb_trainable_params = total_params - bb_frozen_params

    # Strategy 3: Head-only
    k_ho = meta["freeze_headonly_val"]
    ho_frozen_params = sum(
        sum(p.numel() for p in layers[i].parameters()) for i in range(k_ho)
    )
    ho_trainable_params = total_params - ho_frozen_params

    # Inspect 2-class head adaptation details (TASK-MIN-05)
    detect_module = layers[meta["head_indices"][0]]
    coco_nc = getattr(detect_module, "nc", 80)

    return {
        "architecture": arch_name,
        "total_layers": len(layers),
        "total_parameters": total_params,
        "components": {
            "backbone": {
                "layer_indices": f"{meta['backbone_indices'][0]}–{meta['backbone_indices'][-1]}",
                "parameters": backbone_params,
                "percentage": round((backbone_params / total_params) * 100, 2),
            },
            "neck": {
                "layer_indices": f"{meta['neck_indices'][0]}–{meta['neck_indices'][-1]}",
                "parameters": neck_params,
                "percentage": round((neck_params / total_params) * 100, 2),
            },
            "head": {
                "layer_indices": f"{meta['head_indices'][0]}",
                "parameters": head_params,
                "percentage": round((head_params / total_params) * 100, 2),
            },
        },
        "freeze_strategies": {
            "full_finetuning": {
                "freeze_arg": 0,
                "trainable_parameters": full_trainable,
                "trainable_percentage": 100.0,
                "frozen_parameters": full_frozen,
                "frozen_percentage": 0.0,
            },
            "frozen_backbone": {
                "freeze_arg": k_bb,
                "frozen_layers": f"0–{k_bb - 1}",
                "trainable_parameters": bb_trainable_params,
                "trainable_percentage": round((bb_trainable_params / total_params) * 100, 2),
                "frozen_parameters": bb_frozen_params,
                "frozen_percentage": round((bb_frozen_params / total_params) * 100, 2),
            },
            "head_only": {
                "freeze_arg": k_ho,
                "frozen_layers": f"0–{k_ho - 1}",
                "trainable_parameters": ho_trainable_params,
                "trainable_percentage": round((ho_trainable_params / total_params) * 100, 2),
                "frozen_parameters": ho_frozen_params,
                "frozen_percentage": round((ho_frozen_params / total_params) * 100, 2),
            },
        },
        "detection_head_weight_handling": {
            "source_classes": coco_nc,
            "target_classes": 2,
            "box_regression_cv2": "Transferred verbatim (class-agnostic coordinates and DFL representation)",
            "classification_cv3": "Final projection layers (cv3.2) resized from 80 to 2 outputs; weights re-initialized randomly via Ultralytics intersect_dicts",
            "backbone_and_neck": "100% pretrained weights preserved and loaded directly",
        },
        "layer_breakdown": layer_details,
    }


def generate_latex_table(audits: list[dict[str, Any]]) -> str:
    """Generate a clean Springer SVJour3 compatible LaTeX table."""
    lines = [
        r"\begin{table*}[t]",
        r"\caption{Layer indexing, parameter breakdown, and freeze strategy accounting across YOLO generations.}",
        r"\label{tab:architecture_layer_freeze}",
        r"\centering",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llrrrrr@{}}",
        r"\hline",
        r"Architecture & Component / Strategy & Layers & Total Params & Trainable Params & Frozen Params & \% Trainable \\",
        r"\hline",
    ]

    for a in audits:
        arch = a["architecture"].upper()
        tot = a["total_parameters"]
        c = a["components"]
        f = a["freeze_strategies"]

        lines.append(f"{arch} & Total Architecture & 0--{a['total_layers']-1} & {tot:,} & {tot:,} & 0 & 100.0\\% \\\\")
        lines.append(f" & Backbone & {c['backbone']['layer_indices']} & {c['backbone']['parameters']:,} & -- & -- & {c['backbone']['percentage']}\\% \\\\")
        lines.append(f" & Neck & {c['neck']['layer_indices']} & {c['neck']['parameters']:,} & -- & -- & {c['neck']['percentage']}\\% \\\\")
        lines.append(f" & Head & {c['head']['layer_indices']} & {c['head']['parameters']:,} & -- & -- & {c['head']['percentage']}\\% \\\\")
        lines.append(f" & Strategy: Frozen Backbone & Freeze={f['frozen_backbone']['freeze_arg']} & {tot:,} & {f['frozen_backbone']['trainable_parameters']:,} & {f['frozen_backbone']['frozen_parameters']:,} & {f['frozen_backbone']['trainable_percentage']}\\% \\\\")
        lines.append(f" & Strategy: Head-Only & Freeze={f['head_only']['freeze_arg']} & {tot:,} & {f['head_only']['trainable_parameters']:,} & {f['head_only']['frozen_parameters']:,} & {f['head_only']['trainable_percentage']}\\% \\\\")
        lines.append(r"\hline")

    lines.extend([
        r"\end{tabular*}",
        r"\end{table*}",
    ])
    return "\n".join(lines)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("\n" + "=" * 90)
    print("YOLO ARCHITECTURE LAYER & FREEZE PARAMETER AUDIT")
    print("=" * 90 + "\n")

    audits = []
    for arch_name, meta in ARCH_BOUNDARIES.items():
        res = audit_single_architecture(arch_name, meta)
        audits.append(res)

        print(f"--- {arch_name.upper()} ---")
        print(f"Total Layers:     {res['total_layers']}")
        print(f"Total Parameters: {res['total_parameters']:,}")
        print(f"  Backbone (layers {res['components']['backbone']['layer_indices']}): {res['components']['backbone']['parameters']:,} ({res['components']['backbone']['percentage']}%)")
        print(f"  Neck     (layers {res['components']['neck']['layer_indices']}): {res['components']['neck']['parameters']:,} ({res['components']['neck']['percentage']}%)")
        print(f"  Head     (layers {res['components']['head']['layer_indices']}): {res['components']['head']['parameters']:,} ({res['components']['head']['percentage']}%)")
        fb = res["freeze_strategies"]["frozen_backbone"]
        print(f"  [Strategy Frozen Backbone (freeze={fb['freeze_arg']})]: Trainable: {fb['trainable_parameters']:,} ({fb['trainable_percentage']}%), Frozen: {fb['frozen_parameters']:,} ({fb['frozen_percentage']}%)")
        ho = res["freeze_strategies"]["head_only"]
        print(f"  [Strategy Head-Only (freeze={ho['freeze_arg']})]: Trainable: {ho['trainable_parameters']:,} ({ho['trainable_percentage']}%), Frozen: {ho['frozen_parameters']:,} ({ho['frozen_percentage']}%)")
        print()

    # Save output JSON
    out_dir = PROJECT_ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "architecture_layer_audit.json"
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(audits, fh, indent=2)
        fh.write("\n")
    logger.info("Saved audit JSON to %s", out_json)

    # Save LaTeX table
    latex_table = generate_latex_table(audits)
    out_tex = out_dir / "table_architecture_freeze.tex"
    out_tex.write_text(latex_table, encoding="utf-8")
    logger.info("Saved LaTeX table to %s", out_tex)

    print("=" * 90)
    print("GENERATED LATEX TABLE SNIPPET:")
    print("=" * 90)
    print(latex_table)
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
