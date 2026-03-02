#!/usr/bin/env python3
"""Compile a comprehensive WLST E2E report from GCS-downloaded stage outputs.

Usage:
    .venv/bin/python scripts/compile_wlst_report.py \
        --stage1-dir cloud_outputs/stage1/outputs/ \
        --stage2-dir cloud_outputs/stage2/outputs/ \
        --output wlst_e2e_report.md

Each stage directory should contain the outputs/ tree uploaded by cloud_train.py:
    outputs/
        run_info.json
        wlst/{stage}/
            cohort_summary.md
            graph_analysis.md
            patient_ids.json
            feature_stats.json
            hetero_meta.json
            *_evaluation.md
            comparison.md
            gnn_experiments/*/metrics.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _read_text(path: Path) -> str | None:
    """Read a text file, returning None if missing."""
    if path.exists():
        return path.read_text()
    return None


def _read_json(path: Path) -> dict | list | None:
    """Read a JSON file, returning None if missing."""
    text = _read_text(path)
    if text is None:
        return None
    return json.loads(text)


def _find_stage_subdir(base: Path) -> Path | None:
    """Find the wlst/{stage}/ subdirectory inside an outputs tree."""
    wlst_dir = base / "wlst"
    if not wlst_dir.exists():
        return None
    for child in sorted(wlst_dir.iterdir()):
        if child.is_dir() and child.name.startswith("stage"):
            return child
    return None


def _collect_evaluation_reports(stage_dir: Path) -> dict[str, str]:
    """Collect all *_evaluation.md files in a stage directory."""
    reports = {}
    for path in sorted(stage_dir.glob("*_evaluation.md")):
        model_name = path.stem.replace("_evaluation", "")
        reports[model_name] = path.read_text()
    return reports


def _collect_gnn_metrics(stage_dir: Path) -> dict[str, dict]:
    """Collect GNN experiment metrics.json files."""
    results = {}
    gnn_dir = stage_dir / "gnn_experiments"
    if not gnn_dir.exists():
        return results
    for metrics_path in sorted(gnn_dir.glob("*/metrics.json")):
        exp_name = metrics_path.parent.name
        results[exp_name] = json.loads(metrics_path.read_text())
    return results


def _format_timing_table(
    stage1_info: dict | None, stage2_info: dict | None,
) -> str:
    """Format a timing comparison table from run_info step_timings_seconds."""
    s1_timings = (stage1_info or {}).get("step_timings_seconds", {})
    s2_timings = (stage2_info or {}).get("step_timings_seconds", {})
    all_steps = sorted(set(list(s1_timings.keys()) + list(s2_timings.keys())))

    if not all_steps:
        return "_No step timing data available._\n"

    lines = [
        "| Step | Stage 1 (min) | Stage 2 (min) |",
        "|------|---------------|---------------|",
    ]
    for step in all_steps:
        s1 = f"{s1_timings[step] / 60:.1f}" if step in s1_timings else "—"
        s2 = f"{s2_timings[step] / 60:.1f}" if step in s2_timings else "—"
        lines.append(f"| {step} | {s1} | {s2} |")

    # Totals
    s1_total = (stage1_info or {}).get("elapsed_seconds")
    s2_total = (stage2_info or {}).get("elapsed_seconds")
    s1_str = f"{s1_total / 60:.1f}" if s1_total else "—"
    s2_str = f"{s2_total / 60:.1f}" if s2_total else "—"
    lines.append(f"| **Total** | **{s1_str}** | **{s2_str}** |")

    return "\n".join(lines) + "\n"


def _format_feature_stats_table(stats: dict) -> str:
    """Format a summary of feature statistics (from DataFrame.describe().to_json())."""
    if not stats:
        return "_No feature statistics available._\n"

    # stats is {stat_name: {feature: value}} (orient="columns" default from to_json)
    # We want a subset: count, mean, std, min, max
    desired = ["count", "mean", "std", "min", "max"]
    features = list(next(iter(stats.values())).keys()) if stats else []

    if not features:
        return "_No features found._\n"

    lines = [
        f"| Feature | Count | Mean | Std | Min | Max |",
        f"|---------|-------|------|-----|-----|-----|",
    ]
    for feat in features[:30]:  # Cap at 30 to keep report readable
        row = [feat]
        for stat in desired:
            val = stats.get(stat, {}).get(feat)
            if val is None:
                row.append("—")
            elif isinstance(val, float):
                row.append(f"{val:.3f}")
            else:
                row.append(str(val))
        lines.append("| " + " | ".join(row) + " |")

    if len(features) > 30:
        lines.append(f"\n_...and {len(features) - 30} more features (truncated)._")

    return "\n".join(lines) + "\n"


def _format_hetero_meta(meta: dict) -> str:
    """Format HeteroData metadata."""
    if not meta:
        return "_No HeteroData metadata available._\n"

    lines = []
    if "node_types" in meta:
        lines.append("**Node types:**\n")
        lines.append("| Type | Count |")
        lines.append("|------|-------|")
        for nt in meta["node_types"]:
            name = nt.get("name", "?")
            count = nt.get("num_nodes", "?")
            lines.append(f"| {name} | {count} |")
        lines.append("")

    if "edge_types" in meta:
        lines.append("**Edge types:**\n")
        lines.append("| Source | Relation | Target | Count |")
        lines.append("|--------|----------|--------|-------|")
        for et in meta["edge_types"]:
            src = et.get("src", "?")
            rel = et.get("rel", "?")
            dst = et.get("dst", "?")
            count = et.get("num_edges", "?")
            lines.append(f"| {src} | {rel} | {dst} | {count} |")
        lines.append("")

    if "label_distribution" in meta:
        dist = meta["label_distribution"]
        lines.append(f"**Label distribution:** positive={dist.get('positive', '?')}, "
                      f"negative={dist.get('negative', '?')}, "
                      f"total={dist.get('total', '?')}")
        lines.append("")

    return "\n".join(lines) if lines else "_No structured metadata found._\n"


def _format_baseline_summary_table(eval_reports: dict[str, str]) -> str:
    """Extract key metrics from evaluation markdown reports into a summary table."""
    lines = [
        "| Model | AUROC | AUPRC | Brier | Sensitivity | Specificity |",
        "|-------|-------|-------|-------|-------------|-------------|",
    ]

    for model_name, report_text in sorted(eval_reports.items()):
        # Parse metrics from the markdown table
        metrics = {}
        for line in report_text.splitlines():
            line = line.strip()
            if line.startswith("| **") and "|" in line:
                parts = [p.strip().strip("*") for p in line.split("|") if p.strip()]
                if len(parts) >= 2:
                    key = parts[0].lower()
                    val = parts[1]
                    metrics[key] = val

        auroc = metrics.get("auroc", "—")
        auprc = metrics.get("auprc", "—")
        brier = metrics.get("brier score", "—")
        sens = metrics.get("sensitivity", "—")
        spec = metrics.get("specificity", "—")
        lines.append(f"| {model_name} | {auroc} | {auprc} | {brier} | {sens} | {spec} |")

    return "\n".join(lines) + "\n"


def _format_gnn_summary_table(gnn_metrics: dict[str, dict]) -> str:
    """Format GNN experiment results into a summary table."""
    if not gnn_metrics:
        return "_No GNN experiment results found._\n"

    lines = [
        "| Experiment | AUROC | AUPRC | Brier |",
        "|------------|-------|-------|-------|",
    ]

    for exp_name, metrics in sorted(gnn_metrics.items()):
        auroc = metrics.get("auroc", metrics.get("test_auroc", "—"))
        auprc = metrics.get("auprc", metrics.get("test_auprc", "—"))
        brier = metrics.get("brier_score", metrics.get("test_brier", "—"))
        if isinstance(auroc, float):
            auroc = f"{auroc:.4f}"
        if isinstance(auprc, float):
            auprc = f"{auprc:.4f}"
        if isinstance(brier, float):
            brier = f"{brier:.4f}"
        lines.append(f"| {exp_name} | {auroc} | {auprc} | {brier} |")

    return "\n".join(lines) + "\n"


def _build_stage_section(
    stage_name: str,
    stage_dir: Path | None,
    run_info: dict | None,
) -> str:
    """Build the report section for a single stage."""
    if stage_dir is None:
        return f"## {stage_name}\n\n_Stage directory not found._\n\n"

    sections = [f"## {stage_name}\n"]

    # Graph analysis
    graph_md = _read_text(stage_dir / "graph_analysis.md")
    if graph_md:
        # Strip the top-level heading (we provide our own)
        graph_lines = graph_md.splitlines()
        if graph_lines and graph_lines[0].startswith("# "):
            graph_lines = graph_lines[1:]
        sections.append("### Graph Structure\n")
        sections.append("\n".join(graph_lines).strip())
        sections.append("")
    else:
        sections.append("### Graph Structure\n\n_No graph analysis available._\n")

    # HeteroData
    hetero_meta = _read_json(stage_dir / "hetero_meta.json")
    sections.append("### HeteroData Summary\n")
    sections.append(_format_hetero_meta(hetero_meta if isinstance(hetero_meta, dict) else {}))

    # Features
    feature_stats = _read_json(stage_dir / "feature_stats.json")
    sections.append("### Feature Matrix\n")
    sections.append(_format_feature_stats_table(feature_stats if isinstance(feature_stats, dict) else {}))

    # Classical baselines
    eval_reports = _collect_evaluation_reports(stage_dir)
    if eval_reports:
        sections.append("### Classical Baselines\n")
        sections.append(_format_baseline_summary_table(eval_reports))
    else:
        sections.append("### Classical Baselines\n\n_No baseline evaluation reports found._\n")

    # GNN experiments
    gnn_metrics = _collect_gnn_metrics(stage_dir)
    comparison_md = _read_text(stage_dir / "comparison.md")
    if gnn_metrics or comparison_md:
        sections.append("### GNN Experiments\n")
        sections.append(_format_gnn_summary_table(gnn_metrics))
        if comparison_md:
            sections.append("#### Ablation Comparison\n")
            sections.append(comparison_md.strip())
            sections.append("")
    else:
        sections.append("### GNN Experiments\n\n_No GNN results found._\n")

    return "\n".join(sections) + "\n"


def _try_cross_stage_comparison(
    stage1_dir: Path | None, stage2_dir: Path | None,
) -> str:
    """Attempt cross-stage comparison using evaluation report metrics."""
    if stage1_dir is None or stage2_dir is None:
        return "_Both stages required for cross-stage comparison._\n"

    s1_evals = _collect_evaluation_reports(stage1_dir)
    s2_evals = _collect_evaluation_reports(stage2_dir)
    common_models = sorted(set(s1_evals.keys()) & set(s2_evals.keys()))

    if not common_models:
        return "_No common models found between stages._\n"

    def _extract_metric(report_text: str, metric_name: str) -> float | None:
        for line in report_text.splitlines():
            if metric_name.lower() in line.lower() and "|" in line:
                parts = [p.strip().strip("*") for p in line.split("|") if p.strip()]
                if len(parts) >= 2:
                    try:
                        return float(parts[1].split("(")[0].strip())
                    except ValueError:
                        pass
        return None

    lines = [
        "| Model | S1 AUROC | S2 AUROC | Delta | S1 AUPRC | S2 AUPRC | Delta |",
        "|-------|----------|----------|-------|----------|----------|-------|",
    ]

    for model in common_models:
        s1_auroc = _extract_metric(s1_evals[model], "auroc")
        s2_auroc = _extract_metric(s2_evals[model], "auroc")
        s1_auprc = _extract_metric(s1_evals[model], "auprc")
        s2_auprc = _extract_metric(s2_evals[model], "auprc")

        auroc_delta = ""
        if s1_auroc is not None and s2_auroc is not None:
            auroc_delta = f"{s2_auroc - s1_auroc:+.4f}"
        auprc_delta = ""
        if s1_auprc is not None and s2_auprc is not None:
            auprc_delta = f"{s2_auprc - s1_auprc:+.4f}"

        s1a = f"{s1_auroc:.4f}" if s1_auroc is not None else "—"
        s2a = f"{s2_auroc:.4f}" if s2_auroc is not None else "—"
        s1p = f"{s1_auprc:.4f}" if s1_auprc is not None else "—"
        s2p = f"{s2_auprc:.4f}" if s2_auprc is not None else "—"
        lines.append(f"| {model} | {s1a} | {s2a} | {auroc_delta} | {s1p} | {s2p} | {auprc_delta} |")

    return "\n".join(lines) + "\n"


def compile_report(
    stage1_dir: Path | None,
    stage2_dir: Path | None,
) -> str:
    """Compile a comprehensive E2E report from downloaded stage outputs."""
    sections = ["# WLST E2E Pipeline Report\n"]

    # --- Run Metadata ---
    s1_info = _read_json(stage1_dir / "run_info.json") if stage1_dir else None
    s2_info = _read_json(stage2_dir / "run_info.json") if stage2_dir else None
    if not isinstance(s1_info, dict):
        s1_info = None
    if not isinstance(s2_info, dict):
        s2_info = None

    sections.append("## Run Metadata\n")
    meta_lines = []
    if s1_info:
        meta_lines.append(f"- **Stage 1 Run ID:** {s1_info.get('run_id', '?')}")
    if s2_info:
        meta_lines.append(f"- **Stage 2 Run ID:** {s2_info.get('run_id', '?')}")
    info = s1_info or s2_info
    if info:
        meta_lines.append(f"- **GCP Project:** {info.get('gcp_project', '?')}")
        meta_lines.append(f"- **Patients Limit:** {info.get('patients_limit', '?')}")
        meta_lines.append(f"- **Seed:** {info.get('seed', '?')}")
        meta_lines.append(f"- **Run All:** {info.get('run_all', '?')}")
    if s1_info:
        elapsed = s1_info.get("elapsed_seconds", 0)
        meta_lines.append(f"- **Stage 1 Total Runtime:** {elapsed / 60:.1f} min")
    if s2_info:
        elapsed = s2_info.get("elapsed_seconds", 0)
        meta_lines.append(f"- **Stage 2 Total Runtime:** {elapsed / 60:.1f} min")
    sections.append("\n".join(meta_lines))
    sections.append("")

    # --- Cohort ---
    s1_stage_dir = _find_stage_subdir(stage1_dir) if stage1_dir else None
    s2_stage_dir = _find_stage_subdir(stage2_dir) if stage2_dir else None

    sections.append("## Cohort\n")
    cohort_md = None
    cohort_dir = s1_stage_dir or s2_stage_dir
    if cohort_dir:
        cohort_md = _read_text(cohort_dir / "cohort_summary.md")
    if cohort_md:
        sections.append(cohort_md.strip())
    else:
        sections.append("_No cohort summary available._")
    sections.append("")

    # Patient IDs
    patient_ids = None
    if s1_stage_dir:
        patient_ids = _read_json(s1_stage_dir / "patient_ids.json")
    elif s2_stage_dir:
        patient_ids = _read_json(s2_stage_dir / "patient_ids.json")
    if isinstance(patient_ids, list):
        sections.append(f"**Patient IDs ({len(patient_ids)} patients):** {patient_ids}\n")

    # --- Stage 1 ---
    sections.append(_build_stage_section("Stage 1: Clinical Trajectory", s1_stage_dir, s1_info))

    # --- Stage 2 ---
    sections.append(_build_stage_section("Stage 2: Non-Clinical Confounders", s2_stage_dir, s2_info))

    # --- Cross-stage comparison ---
    sections.append("## Stage 1 vs Stage 2 Comparison\n")
    sections.append(_try_cross_stage_comparison(s1_stage_dir, s2_stage_dir))

    # --- Timing ---
    sections.append("## Timing Breakdown\n")
    sections.append(_format_timing_table(s1_info, s2_info))

    return "\n".join(sections)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile WLST E2E report from downloaded GCS outputs.",
    )
    parser.add_argument(
        "--stage1-dir", type=Path, default=None,
        help="Path to downloaded Stage 1 outputs/ directory.",
    )
    parser.add_argument(
        "--stage2-dir", type=Path, default=None,
        help="Path to downloaded Stage 2 outputs/ directory.",
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=Path("wlst_e2e_report.md"),
        help="Output path for the compiled report (default: wlst_e2e_report.md).",
    )
    args = parser.parse_args()

    if args.stage1_dir is None and args.stage2_dir is None:
        parser.error("At least one of --stage1-dir or --stage2-dir is required.")

    # Validate directories
    for label, d in [("stage1", args.stage1_dir), ("stage2", args.stage2_dir)]:
        if d is not None and not d.exists():
            print(f"Warning: {label} directory does not exist: {d}", file=sys.stderr)

    report = compile_report(args.stage1_dir, args.stage2_dir)
    args.output.write_text(report)
    print(f"Report written to {args.output} ({len(report)} chars)")


if __name__ == "__main__":
    main()
