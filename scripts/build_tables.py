"""
Build LaTeX tables for the paper from outputs/full_evaluation_results.csv.

Produces five tables:
  A  Baseline reference          -- one-shot results for all 12 baseline configs
  B  Per-round trajectory        -- R1-R5 + final test for 16 incremental configs
  C  Per-class breakdown at R1 and R5
  D  Extended-training comparison (100-ep vs 10-ep)
  E  Wall-clock time comparison  -- incremental vs one-shot

Usage:
    uv run python scripts/build_tables.py              # print all tables
    uv run python scripts/build_tables.py --table A    # print one table
    uv run python scripts/build_tables.py --out paper_figures/tables.tex
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
CSV_PATH = PROJECT_ROOT / "outputs" / "full_evaluation_results.csv"

# ---------------------------------------------------------------------------
# Display labels
# ---------------------------------------------------------------------------
CONFIG_LABEL: dict[str, str] = {
    # baselines
    "yolov8n_pretrained": r"v8n-P",
    "yolov8n_random": r"v8n-R",
    "yolo11n_pretrained": r"v11n-P",
    "yolo11n_random": r"v11n-R",
    "yolo12n_pretrained": r"v12n-P",
    "yolo12n_random": r"v12n-R",
    "yolov8n_pretrained_100ep": r"v8n-P-100",
    "yolov8n_random_100ep": r"v8n-R-100",
    "yolo11n_pretrained_100ep": r"v11n-P-100",
    "yolo11n_random_100ep": r"v11n-R-100",
    "yolo12n_pretrained_100ep": r"v12n-P-100",
    "yolo12n_random_100ep": r"v12n-R-100",
    # incremental (round-based)
    "yolov8n_pretrained_frozen_backbone": r"v8n-P-frozen",
    "yolov8n_pretrained_headonly": r"v8n-P-head",
    "yolo12n_pretrained_frozen_backbone": r"v12n-P-frozen",
    "yolo12n_pretrained_headonly": r"v12n-P-head",
}

BASELINE_CONFIGS = [
    "yolov8n_random",
    "yolov8n_pretrained",
    "yolo11n_random",
    "yolo11n_pretrained",
    "yolo12n_random",
    "yolo12n_pretrained",
    "yolov8n_random_100ep",
    "yolov8n_pretrained_100ep",
    "yolo11n_random_100ep",
    "yolo11n_pretrained_100ep",
    "yolo12n_random_100ep",
    "yolo12n_pretrained_100ep",
]

# All 16 incremental (round-based) experiment configs
INCREMENTAL_CONFIGS = [
    "yolov8n_random",
    "yolov8n_pretrained",
    "yolov8n_pretrained_frozen_backbone",
    "yolov8n_pretrained_headonly",
    "yolo11n_random",
    "yolo11n_pretrained",
    "yolo12n_random",
    "yolo12n_pretrained",
    "yolov8n_random_100ep",
    "yolov8n_pretrained_100ep",
    "yolo11n_random_100ep",
    "yolo11n_pretrained_100ep",
    "yolo12n_random_100ep",
    "yolo12n_pretrained_100ep",
    "yolo12n_pretrained_frozen_backbone",
    "yolo12n_pretrained_headonly",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_csv() -> list[dict]:
    with open(CSV_PATH, newline="") as f:
        return list(csv.DictReader(f))


def get(row: dict, key: str, default: float = float("nan")) -> float:
    v = row.get(key, "")
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


def label(config: str) -> str:
    return CONFIG_LABEL.get(config, config)


def fmt(v: float, decimals: int = 4) -> str:
    if v != v:  # nan  # noqa: PLR0124
        return "--"
    return f"{v:.{decimals}f}"


def bold(s: str) -> str:
    return r"\textbf{" + s + "}"


def best_col(values: list[float]) -> int:
    """Return index of the max non-nan value."""
    valid = [(v, i) for i, v in enumerate(values) if v == v]  # noqa: PLR0124
    if not valid:
        return -1
    return max(valid)[1]


# ---------------------------------------------------------------------------
# LaTeX helpers
# ---------------------------------------------------------------------------

BOOKTABS_HEADER = r"""\usepackage{booktabs}  % ensure this is in your preamble
"""


def table_wrap(body: str, caption: str, label_str: str, wide: bool = False) -> str:
    resize = r"\resizebox{\columnwidth}{!}{%" + "\n" if wide else ""
    resize_end = r"}" + "\n" if wide else ""
    return (
        r"\begin{table}[t]" + "\n"
        r"\caption{" + caption + "}\n"
        r"\label{" + label_str + "}\n"
        r"\centering" + "\n" + resize + body + resize_end + r"\end{table}" + "\n"
    )


# ---------------------------------------------------------------------------
# Table A — Baseline reference
# ---------------------------------------------------------------------------


def table_a(rows: list[dict]) -> str:
    baseline_rows = {r["config"]: r for r in rows if r["phase"] == "baseline"}

    col_headers = [
        r"\textbf{Config}",
        r"\textbf{Init}",
        r"\textbf{Ep}",
        r"\textbf{mAP@.5}",
        r"\textbf{mAP@.5:.95}",
        r"\textbf{F1}",
        r"\textbf{Prec.}",
        r"\textbf{Recall}",
        r"\textbf{Knife mAP}",
        r"\textbf{Pistol mAP}",
        r"\textbf{Time (min)}",
    ]

    data: list[tuple] = []
    for cfg in BASELINE_CONFIGS:
        if cfg not in baseline_rows:
            continue
        r = baseline_rows[cfg]
        init = "Pretrained" if "pretrained" in cfg else "Random"
        ep = "100" if "100ep" in cfg else "50"
        data.append(
            (
                label(cfg),
                init,
                ep,
                get(r, "mAP50"),
                get(r, "mAP50_95"),
                get(r, "f1_score"),
                get(r, "precision"),
                get(r, "recall"),
                get(r, "knife_mAP50"),
                get(r, "pistol_mAP50"),
                get(r, "training_time_seconds") / 60,
            )
        )

    # Find best per numeric column (cols 3-10 = indices 3..10)
    NUMERIC_START = 3
    col_count = len(data[0]) if data else 0
    best_indices = []
    for col_idx in range(NUMERIC_START, col_count):
        vals = [row[col_idx] for row in data]
        best_indices.append(best_col(vals))

    lines = []
    lines.append(r"\begin{tabular}{l l r r r r r r r r r}")
    lines.append(r"\toprule")
    lines.append(" & ".join(col_headers) + r" \\")
    lines.append(r"\midrule")

    # Group by epoch budget
    prev_ep = None
    for row_i, row in enumerate(data):
        ep = row[2]
        if prev_ep and ep != prev_ep:
            lines.append(r"\midrule")
        prev_ep = ep

        cells = [str(row[0]), row[1], row[2]]
        for col_offset, col_idx in enumerate(range(NUMERIC_START, col_count)):
            v = row[col_idx]
            decimals = 1 if col_idx == col_count - 1 else 4  # time in 1 decimal
            s = fmt(v, decimals)
            if best_indices[col_offset] == row_i:
                s = bold(s)
            cells.append(s)
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    body = "\n".join(lines) + "\n"

    return table_wrap(
        body,
        caption=(
            "One-shot baseline results for all 12 nano configurations. "
            "Bold = best per column. "
            r"Init: Pretrained = COCO weights; Random = random initialisation. "
            r"Ep = epoch budget (early stopping patience\,=\,10 for 100-ep runs)."
        ),
        label_str="tab:baselines",
        wide=True,
    )


# ---------------------------------------------------------------------------
# Table B — Per-round trajectory (mAP@0.5 only, keep it readable)
# ---------------------------------------------------------------------------


def table_b(rows: list[dict]) -> str:
    # Round rows: round 1-5 per config
    round_rows: dict[str, dict[int, dict]] = {}
    for r in rows:
        if r["phase"] != "round":
            continue
        cfg = r["config"]
        rnd = int(r["round"])
        round_rows.setdefault(cfg, {})[rnd] = r

    # Final test rows
    final_rows = {r["config"]: r for r in rows if r["phase"] == "final_test"}

    col_headers = [
        r"\textbf{Config}",
        r"\textbf{Init}",
        r"\textbf{Ep/rnd}",
        r"\textbf{R1}",
        r"\textbf{R2}",
        r"\textbf{R3}",
        r"\textbf{R4}",
        r"\textbf{R5}",
        r"\textbf{Final}",
    ]

    lines = []
    lines.append(r"\begin{tabular}{l l r r r r r r r}")
    lines.append(r"\toprule")
    lines.append(" & ".join(col_headers) + r" \\")
    lines.append(r"\midrule")
    # MDPI reference row
    lines.append(
        r"Kutlu \& Emiroglu~\cite{Kutlu2025} & Rand & 10 & "
        r"0.518 & 0.767 & 0.838 & 0.861 & 0.886 & -- \\"
    )
    lines.append(r"\midrule")

    prev_ep = None
    for cfg in INCREMENTAL_CONFIGS:
        rnd_data = round_rows.get(cfg, {})
        final = final_rows.get(cfg, {})

        init = "Pretrained" if "pretrained" in cfg else "Random"
        ep = "100" if "100ep" in cfg else "10"

        if prev_ep and ep != prev_ep:
            lines.append(r"\midrule")
        prev_ep = ep

        cells = [label(cfg), init, ep]
        for rnd in range(1, 6):
            r = rnd_data.get(rnd, {})
            v = get(r, "mAP50") if r else float("nan")
            cells.append(fmt(v))
        cells.append(fmt(get(final, "mAP50")))
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    body = "\n".join(lines) + "\n"

    return table_wrap(
        body,
        caption=(
            r"mAP@0.5 trajectory across all five incremental rounds and final test evaluation. "
            r"The MDPI reference row (Kutlu \& Emiroglu) uses the same dataset and protocol."
        ),
        label_str="tab:trajectory",
        wide=True,
    )


# ---------------------------------------------------------------------------
# Table C — Per-class breakdown at R1 and R5
# ---------------------------------------------------------------------------


def table_c(rows: list[dict]) -> str:
    round_rows: dict[str, dict[int, dict]] = {}
    for r in rows:
        if r["phase"] != "round":
            continue
        round_rows.setdefault(r["config"], {})[int(r["round"])] = r

    col_headers = [
        r"\textbf{Config}",
        r"\textbf{Init}",
        r"\multicolumn{2}{c}{\textbf{Round 1}}",
        r"\multicolumn{2}{c}{\textbf{Round 5}}",
    ]
    sub_headers = [
        "",
        "",
        r"\textbf{Knife}",
        r"\textbf{Pistol}",
        r"\textbf{Knife}",
        r"\textbf{Pistol}",
    ]

    lines = []
    lines.append(r"\begin{tabular}{l l r r r r}")
    lines.append(r"\toprule")
    lines.append(" & ".join(col_headers) + r" \\")
    lines.append(r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}")
    lines.append(" & ".join(sub_headers) + r" \\")
    lines.append(r"\midrule")

    prev_ep = None
    for cfg in INCREMENTAL_CONFIGS:
        rnd_data = round_rows.get(cfg, {})
        r1 = rnd_data.get(1, {})
        r5 = rnd_data.get(5, {})
        init = "Pretrained" if "pretrained" in cfg else "Random"
        ep = "100" if "100ep" in cfg else "10"
        if prev_ep and ep != prev_ep:
            lines.append(r"\midrule")
        prev_ep = ep

        cells = [
            label(cfg),
            init,
            fmt(get(r1, "knife_mAP50")),
            fmt(get(r1, "pistol_mAP50")),
            fmt(get(r5, "knife_mAP50")),
            fmt(get(r5, "pistol_mAP50")),
        ]
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    body = "\n".join(lines) + "\n"

    return table_wrap(
        body,
        caption=(r"Per-class mAP@0.5 at Round~1 and Round~5 for all incremental configurations."),
        label_str="tab:per_class",
        wide=True,
    )


# ---------------------------------------------------------------------------
# Table D — Extended training comparison (100-ep vs 10-ep)
# ---------------------------------------------------------------------------


def table_d(rows: list[dict]) -> str:
    final_rows = {r["config"]: r for r in rows if r["phase"] == "final_test"}
    baseline_rows = {r["config"]: r for r in rows if r["phase"] == "baseline"}

    pairs = [
        # (10-ep incremental, 100-ep incremental, 10-ep baseline, 100-ep baseline)
        ("yolov8n_random", "yolov8n_random_100ep", "yolov8n_random", "yolov8n_random_100ep"),
        (
            "yolov8n_pretrained",
            "yolov8n_pretrained_100ep",
            "yolov8n_pretrained",
            "yolov8n_pretrained_100ep",
        ),
        ("yolo11n_random", "yolo11n_random_100ep", "yolo11n_random", "yolo11n_random_100ep"),
        (
            "yolo11n_pretrained",
            "yolo11n_pretrained_100ep",
            "yolo11n_pretrained",
            "yolo11n_pretrained_100ep",
        ),
        ("yolo12n_random", "yolo12n_random_100ep", "yolo12n_random", "yolo12n_random_100ep"),
        (
            "yolo12n_pretrained",
            "yolo12n_pretrained_100ep",
            "yolo12n_pretrained",
            "yolo12n_pretrained_100ep",
        ),
    ]

    col_headers = [
        r"\textbf{Arch}",
        r"\textbf{Init}",
        r"\multicolumn{2}{c}{\textbf{Incremental mAP@.5}}",
        r"\multicolumn{2}{c}{\textbf{Baseline mAP@.5}}",
    ]
    sub_headers = [
        "",
        "",
        r"\textbf{10 ep/rnd}",
        r"\textbf{100 ep/rnd}",
        r"\textbf{50 ep}",
        r"\textbf{100 ep}",
    ]

    lines = []
    lines.append(r"\begin{tabular}{l l r r r r}")
    lines.append(r"\toprule")
    lines.append(" & ".join(col_headers) + r" \\")
    lines.append(r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}")
    lines.append(" & ".join(sub_headers) + r" \\")
    lines.append(r"\midrule")

    arch_map = {"yolov8n": "v8n", "yolo11n": "v11n", "yolo12n": "v12n"}
    prev_arch = None
    for c10, c100, b10, b100 in pairs:
        arch_key = c10.split("_")[0]
        arch = arch_map.get(arch_key, arch_key)
        init = "Pretrained" if "pretrained" in c10 else "Random"

        if prev_arch and arch != prev_arch:
            lines.append(r"\midrule")
        prev_arch = arch

        v_inc_10 = get(final_rows.get(c10, {}), "mAP50")
        v_inc_100 = get(final_rows.get(c100, {}), "mAP50")
        v_b10 = get(baseline_rows.get(b10, {}), "mAP50")
        v_b100 = get(baseline_rows.get(b100, {}), "mAP50")

        cells = [arch, init, fmt(v_inc_10), fmt(v_inc_100), fmt(v_b10), fmt(v_b100)]
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    body = "\n".join(lines) + "\n"

    return table_wrap(
        body,
        caption=(
            r"Effect of epoch budget: incremental (5\,\texttimes\,10 or 5\,\texttimes\,100 ep/round) "
            r"vs one-shot baseline (50 or 100 ep). All values are final test mAP@0.5."
        ),
        label_str="tab:epoch_budget",
        wide=False,
    )


# ---------------------------------------------------------------------------
# Table E — Wall-clock time comparison
# ---------------------------------------------------------------------------


def table_e(rows: list[dict]) -> str:
    final_rows = {r["config"]: r for r in rows if r["phase"] == "final_test"}
    baseline_rows = {r["config"]: r for r in rows if r["phase"] == "baseline"}

    configs_10 = [
        ("yolov8n_random", "yolov8n_random"),
        ("yolov8n_pretrained", "yolov8n_pretrained"),
        ("yolo11n_random", "yolo11n_random"),
        ("yolo11n_pretrained", "yolo11n_pretrained"),
        ("yolo12n_random", "yolo12n_random"),
        ("yolo12n_pretrained", "yolo12n_pretrained"),
    ]
    configs_100 = [
        ("yolov8n_random_100ep", "yolov8n_random_100ep"),
        ("yolov8n_pretrained_100ep", "yolov8n_pretrained_100ep"),
        ("yolo11n_random_100ep", "yolo11n_random_100ep"),
        ("yolo11n_pretrained_100ep", "yolo11n_pretrained_100ep"),
        ("yolo12n_random_100ep", "yolo12n_random_100ep"),
        ("yolo12n_pretrained_100ep", "yolo12n_pretrained_100ep"),
    ]

    col_headers = [
        r"\textbf{Config}",
        r"\textbf{Init}",
        r"\textbf{Ep budget}",
        r"\textbf{Incr. time (min)}",
        r"\textbf{Baseline time (min)}",
        r"\textbf{Incr. mAP@.5}",
        r"\textbf{Baseline mAP@.5}",
    ]

    lines = []
    lines.append(r"\begin{tabular}{l l r r r r r}")
    lines.append(r"\toprule")
    lines.append(" & ".join(col_headers) + r" \\")
    lines.append(r"\midrule")

    arch_map = {"yolov8n": "v8n", "yolo11n": "v11n", "yolo12n": "v12n"}
    prev_ep = None
    for all_pairs in [configs_10, configs_100]:
        ep_label = "10 ep/rnd" if all_pairs is configs_10 else "100 ep/rnd"
        if prev_ep:
            lines.append(r"\midrule")
        prev_ep = ep_label
        for inc_cfg, base_cfg in all_pairs:
            arch_key = inc_cfg.split("_")[0]
            arch = arch_map.get(arch_key, arch_key)
            init = "Pretrained" if "pretrained" in inc_cfg else "Random"

            inc_row = final_rows.get(inc_cfg, {})
            base_row = baseline_rows.get(base_cfg, {})

            inc_time = get(inc_row, "training_time_seconds") / 60
            base_time = get(base_row, "training_time_seconds") / 60
            inc_map = get(inc_row, "mAP50")
            base_map = get(base_row, "mAP50")

            cells = [
                arch,
                init,
                ep_label,
                fmt(inc_time, 1),
                fmt(base_time, 1),
                fmt(inc_map),
                fmt(base_map),
            ]
            lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    body = "\n".join(lines) + "\n"

    return table_wrap(
        body,
        caption=(
            r"Wall-clock training time (minutes) and final mAP@0.5 for incremental "
            r"vs one-shot baseline at matched epoch budgets."
        ),
        label_str="tab:wallclock",
        wide=True,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

TABLE_BUILDERS = {
    "A": ("Baseline reference table", table_a),
    "B": ("Per-round trajectory table", table_b),
    "C": ("Per-class breakdown at R1 and R5", table_c),
    "D": ("Extended-training comparison", table_d),
    "E": ("Wall-clock time comparison", table_e),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--table",
        choices=list(TABLE_BUILDERS.keys()),
        help="Which table to generate (default: all)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Write output to this .tex file instead of stdout",
    )
    args = parser.parse_args()

    rows = load_csv()

    tables_to_run = [args.table] if args.table else list(TABLE_BUILDERS.keys())

    output_parts = []
    for key in tables_to_run:
        title, builder = TABLE_BUILDERS[key]
        output_parts.append(f"% === Table {key}: {title} ===\n")
        output_parts.append(builder(rows))
        output_parts.append("\n")

    output = "\n".join(output_parts)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output, encoding="utf-8")
        print(f"Wrote {args.out}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
