#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate DedupShift Leakage Statistics Table

Extracts key metrics from dedup reports to show quantitative evidence:
- Train/test overlap before and after deduplication
- Sample retention rate
- Near-duplicate removal statistics

Outputs:
    results/dedup_leakage_stats.csv - Summary table for paper
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def generate_leakage_stats():
    """Generate leakage statistics comparison table."""
    
    # Read dedup reports
    sms_report = ROOT / "results" / "dedup_report_sms.csv"
    spam_report = ROOT / "results" / "dedup_report_spamassassin.csv"
    
    if not sms_report.exists() or not spam_report.exists():
        print("⚠️  Dedup reports not found. Run 'make dedup' first.")
        return
    
    df_sms = pd.read_csv(sms_report)
    df_spam = pd.read_csv(spam_report)
    
    # Extract last row (most recent run)
    sms = df_sms.iloc[-1]
    spam = df_spam.iloc[-1]
    
    # Build summary table
    rows = []
    
    # SMS statistics
    rows.append({
        "dataset": "SMS (UCI)",
        "n_original": int(sms["n_in"]),
        "n_exact_dup": int(sms["n_exact_removed"]),
        "n_near_dup": int(sms["n_near_removed"]),
        "n_total_removed": int(sms["n_exact_removed"]) + int(sms["n_near_removed"]),
        "n_deduplicated": int(sms["n_out"]),
        "retention_rate_%": round(int(sms["n_out"]) / int(sms["n_in"]) * 100, 1),
        "orig_train_test_overlap": int(sms["orig_overlap_train_test"]),
        "orig_train_val_overlap": int(sms["orig_overlap_train_val"]),
        "orig_val_test_overlap": int(sms["orig_overlap_val_test"]),
        "dedup_train_test_overlap": int(sms["dedup_overlap_train_test"]),
        "dedup_train_val_overlap": int(sms["dedup_overlap_train_val"]),
        "dedup_val_test_overlap": int(sms["dedup_overlap_val_test"]),
        "total_orig_overlap": int(sms["orig_overlap_train_test"]) + int(sms["orig_overlap_train_val"]) + int(sms["orig_overlap_val_test"]),
        "total_dedup_overlap": int(sms["dedup_overlap_train_test"]) + int(sms["dedup_overlap_train_val"]) + int(sms["dedup_overlap_val_test"]),
    })
    
    # SpamAssassin statistics
    rows.append({
        "dataset": "SpamAssassin",
        "n_original": int(spam["n_in"]),
        "n_exact_dup": int(spam["n_exact_removed"]),
        "n_near_dup": int(spam["n_near_removed"]),
        "n_total_removed": int(spam["n_exact_removed"]) + int(spam["n_near_removed"]),
        "n_deduplicated": int(spam["n_out"]),
        "retention_rate_%": round(int(spam["n_out"]) / int(spam["n_in"]) * 100, 1),
        "orig_train_test_overlap": int(spam["orig_overlap_train_test"]),
        "orig_train_val_overlap": int(spam["orig_overlap_train_val"]),
        "orig_val_test_overlap": int(spam["orig_overlap_val_test"]),
        "dedup_train_test_overlap": int(spam["dedup_overlap_train_test"]),
        "dedup_train_val_overlap": int(spam["dedup_overlap_train_val"]),
        "dedup_val_test_overlap": int(spam["dedup_overlap_val_test"]),
        "total_orig_overlap": int(spam["orig_overlap_train_test"]) + int(spam["orig_overlap_train_val"]) + int(spam["orig_overlap_val_test"]),
        "total_dedup_overlap": int(spam["dedup_overlap_train_test"]) + int(spam["dedup_overlap_train_val"]) + int(spam["dedup_overlap_val_test"]),
    })
    
    df_stats = pd.DataFrame(rows)
    
    # Calculate aggregates
    df_stats["overlap_reduction"] = df_stats["total_orig_overlap"] - df_stats["total_dedup_overlap"]
    df_stats["overlap_reduction_%"] = round(
        (df_stats["total_orig_overlap"] - df_stats["total_dedup_overlap"]) / df_stats["total_orig_overlap"] * 100, 1
    )
    
    # Save
    out_path = ROOT / "results" / "dedup_leakage_stats.csv"
    df_stats.to_csv(out_path, index=False)
    print(f"✅ Leakage statistics saved to {out_path}")
    
    # Print summary
    print("\n📊 DedupShift Leakage Statistics Summary:")
    print("=" * 80)
    for _, row in df_stats.iterrows():
        print(f"\n{row['dataset']}:")
        print(f"  Samples: {row['n_original']:,} → {row['n_deduplicated']:,} (retention: {row['retention_rate_%']:.1f}%)")
        print(f"  Removed: {row['n_total_removed']:,} ({row['n_exact_dup']} exact + {row['n_near_dup']} near)")
        print(f"  Train/Test overlap: {row['orig_train_test_overlap']} → {row['dedup_train_test_overlap']} (✓ eliminated)")
        print(f"  Train/Val overlap: {row['orig_train_val_overlap']} → {row['dedup_train_val_overlap']} (✓ eliminated)")
        print(f"  Val/Test overlap: {row['orig_val_test_overlap']} → {row['dedup_val_test_overlap']} (✓ eliminated)")
        print(f"  Total overlap reduction: {row['total_orig_overlap']} → {row['total_dedup_overlap']} (-{row['overlap_reduction']}, -{row['overlap_reduction_%']:.1f}%)")
    print("=" * 80)
    
    return df_stats


if __name__ == "__main__":
    generate_leakage_stats()
