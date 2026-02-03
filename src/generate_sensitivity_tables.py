#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate LaTeX tables and figures for sensitivity analysis results.

Converts CSV results from:
- sensitivity_dedup_summary.csv (DedupShift h_thresh analysis)
- distilbert_multiseed.csv (DistilBERT seed aggregation)

Into publication-ready LaTeX tables and visualization PNG files.
"""
import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def generate_sensitivity_dedup_table(in_csv: Path, out_tex: Path) -> None:
    """Convert sensitivity_dedup_summary.csv to LaTeX table."""
    if not in_csv.exists():
        print(f"⚠️  {in_csv} not found; skipping")
        return
    
    df = pd.read_csv(in_csv)
    
    # Create LaTeX table
    tex_lines = [
        r"\begin{table}[!h]",
        r"\centering",
        r"\caption{DedupShift Sensitivity: Impact of SimHash Hamming Threshold on Deduplication Rate and Model F1.}",
        r"\label{tab:sensitivity_dedup_threshold}",
        r"\small",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"\textbf{Dataset} & \textbf{$h_{\text{thresh}}$} & \textbf{Input} & \textbf{Removed} & \textbf{Output} & \textbf{Dedup \%} & \textbf{TF-IDF F1} \\",
        r"\midrule",
    ]
    
    for _, row in df.iterrows():
        dataset = row["dataset"][:10]  # Abbreviate
        h_thresh = int(row["h_thresh"])
        n_in = int(row["n_input"])
        n_removed = int(row["n_exact_removed"]) + int(row["n_near_removed"])
        n_out = int(row["n_output"])
        dedup_pct = row["dedup_rate_%"]
        f1 = row["f1_score"]
        
        tex_lines.append(
            f"{dataset} & {h_thresh} & {n_in} & {n_removed} & {n_out} & {dedup_pct:.1f}\\% & {f1:.4f} \\\\"
        )
    
    tex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text("\n".join(tex_lines))
    print(f"✅ Generated {out_tex}")


def plot_sensitivity_dedup(in_csv: Path, out_png: Path) -> None:
    """Plot F1 vs h_thresh for DedupShift sensitivity."""
    if not in_csv.exists():
        print(f"⚠️  {in_csv} not found; skipping plot")
        return
    
    df = pd.read_csv(in_csv)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot 1: Dedup rate vs h_thresh
    for dataset in df["dataset"].unique():
        data = df[df["dataset"] == dataset]
        data_sorted = data.sort_values("h_thresh")
        axes[0].plot(data_sorted["h_thresh"], data_sorted["dedup_rate_%"], marker="o", label=dataset)
    
    axes[0].set_xlabel("Hamming Threshold ($h_{\\text{thresh}}$)")
    axes[0].set_ylabel("Deduplication Rate (%)")
    axes[0].set_title("Dedup Rate vs. Threshold")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: F1 score vs h_thresh
    for dataset in df["dataset"].unique():
        data = df[df["dataset"] == dataset]
        data_sorted = data.sort_values("h_thresh")
        axes[1].plot(data_sorted["h_thresh"], data_sorted["f1_score"], marker="s", label=dataset)
    
    axes[1].set_xlabel("Hamming Threshold ($h_{\\text{thresh}}$)")
    axes[1].set_ylabel("F1 Score")
    axes[1].set_title("Model Performance vs. Threshold")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Generated plot {out_png}")


def generate_distilbert_multiseed_table(in_csv: Path, out_tex: Path) -> None:
    """Convert distilbert_multiseed.csv to LaTeX table with mean±std."""
    if not in_csv.exists():
        print(f"⚠️  {in_csv} not found; skipping")
        return
    
    df = pd.read_csv(in_csv)
    
    # Create LaTeX table
    tex_lines = [
        r"\begin{table}[!h]",
        r"\centering",
        r"\caption{DistilBERT Multi-Seed Training: Aggregated F1 Scores (Mean $\pm$ Std) Over Seeds 0, 1, 2.}",
        r"\label{tab:distilbert_multiseed}",
        r"\small",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"\textbf{Train Domain} & \textbf{Test Domain} & \textbf{Split} & \textbf{Model} & \textbf{F1 (Mean $\pm$ Std)} \\",
        r"\midrule",
    ]
    
    for _, row in df.iterrows():
        train = str(row.get("train_domain", "")).strip()
        test = str(row.get("test_domain", "")).strip()
        split = str(row.get("split", "")).strip()
        model = str(row.get("model", "")).strip()
        f1_mean = row.get("f1_mean")
        f1_std = row.get("f1_std")
        
        if pd.notna(f1_mean) and pd.notna(f1_std):
            f1_str = f"${f1_mean:.4f} \\pm {f1_std:.4f}$"
        else:
            f1_str = str(row.get("f1", "N/A"))
        
        tex_lines.append(
            f"{train} & {test} & {split} & {model} & {f1_str} \\\\"
        )
    
    tex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text("\n".join(tex_lines))
    print(f"✅ Generated {out_tex}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sensitivity_csv", default=str(ROOT / "results" / "sensitivity_dedup_summary.csv"))
    ap.add_argument("--distilbert_csv", default=str(ROOT / "results" / "distilbert_multiseed.csv"))
    ap.add_argument("--out_dir", default=str(ROOT / "paper" / "tables"))
    ap.add_argument("--fig_dir", default=str(ROOT / "paper" / "figs"))
    args = ap.parse_args()
    
    out_dir = Path(args.out_dir)
    fig_dir = Path(args.fig_dir)
    
    # Generate DedupShift sensitivity table and plot
    print("🔄 Generating DedupShift sensitivity analysis...")
    generate_sensitivity_dedup_table(
        Path(args.sensitivity_csv),
        out_dir / "sensitivity_dedup_threshold.tex"
    )
    plot_sensitivity_dedup(
        Path(args.sensitivity_csv),
        fig_dir / "fig_sensitivity_dedup_threshold.png"
    )
    
    # Generate DistilBERT multi-seed table
    print("🔄 Generating DistilBERT multi-seed results table...")
    generate_distilbert_multiseed_table(
        Path(args.distilbert_csv),
        out_dir / "distilbert_multiseed.tex"
    )
    
    print("✨ Sensitivity tables generated successfully!")


if __name__ == "__main__":
    main()
