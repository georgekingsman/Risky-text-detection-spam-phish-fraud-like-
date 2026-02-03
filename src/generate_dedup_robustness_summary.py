#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate DedupShift Robustness Delta Summary

Analyzes how deduplication affects robustness estimates by comparing
delta_f1 before and after DedupShift across different attack types.

Key insight: Leakage control changes robustness estimates, showing that
data contamination can mask or exaggerate adversarial vulnerabilities.

Outputs:
    results/dedup_robustness_summary.csv - Statistical summary table
"""
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def generate_robustness_summary():
    """Generate robustness delta comparison summary."""
    
    dedup_effect_path = ROOT / "results" / "dedup_effect.csv"
    
    if not dedup_effect_path.exists():
        print("⚠️  dedup_effect.csv not found. Run 'python -m src.compare_robustness_dedup' first.")
        return
    
    df = pd.read_csv(dedup_effect_path)
    
    # Filter out clean (no attack) rows
    df_attacks = df[df["attack"] != "clean"].copy()
    
    # Group by attack type and defense
    summary_rows = []
    
    for attack in df_attacks["attack"].unique():
        for defense in df_attacks["defense"].unique():
            subset = df_attacks[
                (df_attacks["attack"] == attack) & 
                (df_attacks["defense"] == defense)
            ]
            
            if len(subset) == 0:
                continue
            
            # Calculate statistics
            orig_mean = subset["delta_f1_orig"].mean()
            orig_std = subset["delta_f1_orig"].std()
            dedup_mean = subset["delta_f1_dedup"].mean()
            dedup_std = subset["delta_f1_dedup"].std()
            change_mean = subset["delta_f1_change"].mean()
            change_std = subset["delta_f1_change"].std()
            
            # Count models/datasets
            n_models = len(subset)
            
            summary_rows.append({
                "attack": attack,
                "defense": defense,
                "n_models": n_models,
                "orig_delta_mean": orig_mean,
                "orig_delta_std": orig_std,
                "dedup_delta_mean": dedup_mean,
                "dedup_delta_std": dedup_std,
                "change_mean": change_mean,
                "change_std": change_std,
                "change_abs_mean": abs(change_mean),
            })
    
    df_summary = pd.DataFrame(summary_rows)
    
    # Sort by absolute change magnitude
    df_summary = df_summary.sort_values("change_abs_mean", ascending=False)
    
    # Save
    out_path = ROOT / "results" / "dedup_robustness_summary.csv"
    df_summary.to_csv(out_path, index=False)
    print(f"✅ Robustness summary saved to {out_path}")
    
    # Print formatted summary
    print("\n📊 DedupShift Effect on Robustness Delta:")
    print("=" * 100)
    print(f"{'Attack':<20} {'Defense':<12} {'N':<4} {'Orig Δ':<12} {'Dedup Δ':<12} {'Change':<12} {'Direction':<10}")
    print("=" * 100)
    
    for _, row in df_summary.iterrows():
        direction = "↑ Less robust" if row["change_mean"] < 0 else "↓ More robust"
        print(
            f"{row['attack']:<20} "
            f"{row['defense']:<12} "
            f"{row['n_models']:<4} "
            f"{row['orig_delta_mean']:>6.3f}±{row['orig_delta_std']:.3f}  "
            f"{row['dedup_delta_mean']:>6.3f}±{row['dedup_delta_std']:.3f}  "
            f"{row['change_mean']:>+6.3f}±{row['change_std']:.3f}  "
            f"{direction}"
        )
    
    print("=" * 100)
    print("\n💡 Key Insight:")
    print("   Leakage control reveals true adversarial vulnerability.")
    print("   Non-zero changes indicate that data contamination was masking/exaggerating robustness.")
    
    # Overall statistics
    print(f"\n📈 Overall Statistics:")
    print(f"   Mean absolute change: {df_summary['change_abs_mean'].mean():.4f}")
    print(f"   Max absolute change: {df_summary['change_abs_mean'].max():.4f}")
    print(f"   Attack types analyzed: {df_attacks['attack'].nunique()}")
    print(f"   Total model×attack combinations: {len(df_attacks)}")
    
    return df_summary


if __name__ == "__main__":
    generate_robustness_summary()
