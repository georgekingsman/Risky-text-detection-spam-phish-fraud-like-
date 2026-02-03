#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DistilBERT Multi-Seed Training and Aggregation

Trains DistilBERT with multiple seeds (0, 1, 2) and reports mean±std metrics.

Usage:
    python src/train_distilbert_multiseed.py \
        --train_csv dataset/sms_uci/dedup/processed/data.csv \
        --train_domain sms \
        --eval_csvs dataset/sms_uci/dedup/processed/data.csv dataset/spamassassin/dedup/processed/data.csv \
        --eval_domains sms spamassassin \
        --out_dir models/distilbert_sms_dedup_multiseed \
        --results_csv results/distilbert_multiseed.csv

Outputs:
    results/distilbert_multiseed.csv - aggregated results with mean±std
    results/distilbert_multiseed_seeds.csv - per-seed raw results
"""
import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def run_single_seed(
    train_csv: str,
    train_domain: str,
    eval_csvs: List[str],
    eval_domains: List[str],
    out_dir: str,
    seed: int,
    temp_results_csv: str,
    epochs: int = 2,
    batch: int = 8,
    max_len: int = 128,
    device: str = None,
) -> bool:
    """Run DistilBERT training for a single seed."""
    cmd = [
        "python", "-m", "src.nn_distilbert_ft",
        "--train_csv", train_csv,
        "--train_domain", train_domain,
        "--eval_csvs"] + eval_csvs + ["--eval_domains"] + eval_domains + [
        "--out_dir", f"{out_dir}_seed{seed}",
        "--results_csv", temp_results_csv,
        "--seed", str(seed),
        "--epochs", str(epochs),
        "--batch", str(batch),
        "--max_len", str(max_len),
    ]
    
    if device:
        cmd.extend(["--device", device])
    
    print(f"\n🚀 Running seed {seed}...")
    result = subprocess.run(cmd, cwd=str(ROOT))
    return result.returncode == 0


def aggregate_results(
    seed_results_list: List[Dict],
    out_csv: str,
    raw_out_csv: str,
) -> None:
    """Aggregate per-seed results into mean±std format."""
    
    # First, save raw per-seed results
    if seed_results_list:
        df_raw = pd.DataFrame(seed_results_list)
        df_raw.to_csv(raw_out_csv, index=False)
        print(f"\n📝 Raw per-seed results saved to {raw_out_csv}")
    
    # Group by (train_domain, test_domain, model, split) and compute mean±std
    df = pd.DataFrame(seed_results_list)
    
    # Identify numeric columns for aggregation
    numeric_cols = ["f1", "precision", "recall", "roc_auc"]
    for col in numeric_cols:
        if col not in df.columns:
            numeric_cols.remove(col)
    
    group_cols = ["train_domain", "test_domain", "model", "split"]
    agg_spec = {col: ["mean", "std"] for col in numeric_cols}
    
    df_agg = df.groupby(group_cols).agg(agg_spec)
    df_agg = df_agg.reset_index()
    
    # Flatten multi-level columns
    df_agg.columns = ["_".join(col).rstrip("_") if col[1] else col[0] for col in df_agg.columns.values]
    
    # Format as metric_mean and metric_std
    final_rows = []
    for _, row in df_agg.iterrows():
        final_row = {col: row[col] for col in group_cols}
        for col in numeric_cols:
            mean_val = row.get(f"{col}_mean", "")
            std_val = row.get(f"{col}_std", "")
            if pd.notna(mean_val) and pd.notna(std_val):
                final_row[f"{col}_mean"] = round(mean_val, 4)
                final_row[f"{col}_std"] = round(std_val, 4)
                final_row[f"{col}"] = f"{round(mean_val, 4)}±{round(std_val, 4)}"
        final_rows.append(final_row)
    
    df_final = pd.DataFrame(final_rows)
    df_final.to_csv(out_csv, index=False)
    print(f"✅ Aggregated results (mean±std) saved to {out_csv}")
    print(df_final.to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description="Train DistilBERT with multiple seeds and aggregate.")
    ap.add_argument("--train_csv", required=True, help="Training CSV path")
    ap.add_argument("--train_domain", required=True, help="Training domain name")
    ap.add_argument("--eval_csvs", nargs="+", required=True, help="Evaluation CSV paths")
    ap.add_argument("--eval_domains", nargs="+", required=True, help="Evaluation domain names")
    ap.add_argument("--out_dir", required=True, help="Output directory for models")
    ap.add_argument("--results_csv", required=True, help="Output aggregated results CSV")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2], help="Seeds to run (default: 0 1 2)")
    ap.add_argument("--epochs", type=int, default=2, help="Training epochs")
    ap.add_argument("--batch", type=int, default=8, help="Batch size")
    ap.add_argument("--max_len", type=int, default=128, help="Max token length")
    ap.add_argument("--device", default=None, help="Device: cpu | mps | cuda")
    args = ap.parse_args()
    
    if len(args.eval_csvs) != len(args.eval_domains):
        raise ValueError("--eval_csvs and --eval_domains must have same length")
    
    seeds = args.seeds
    print(f"🔄 Training DistilBERT with seeds: {seeds}")
    
    # Collect all results from all seeds
    all_results = []
    
    for seed in seeds:
        # Create temp CSV for this seed's results
        temp_csv = ROOT / "results" / f"distilbert_temp_seed{seed}.csv"
        temp_csv.parent.mkdir(parents=True, exist_ok=True)
        
        # Run training
        success = run_single_seed(
            args.train_csv,
            args.train_domain,
            args.eval_csvs,
            args.eval_domains,
            args.out_dir,
            seed,
            str(temp_csv),
            epochs=args.epochs,
            batch=args.batch,
            max_len=args.max_len,
            device=args.device,
        )
        
        if not success:
            print(f"⚠️  Seed {seed} training failed; skipping")
            continue
        
        # Read results for this seed
        if temp_csv.exists():
            df_seed = pd.read_csv(temp_csv)
            df_seed["seed"] = seed
            all_results.extend(df_seed.to_dict("records"))
            temp_csv.unlink()  # Clean up temp file
        else:
            print(f"⚠️  No results CSV generated for seed {seed}")
    
    if not all_results:
        print("❌ No results collected. Check training logs.")
        sys.exit(1)
    
    # Aggregate and save
    out_csv = ROOT / args.results_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    raw_csv = out_csv.parent / f"{out_csv.stem}_seeds{out_csv.suffix}"
    
    aggregate_results(all_results, str(out_csv), str(raw_csv))
    print(f"\n✨ Multi-seed training complete!")


if __name__ == "__main__":
    main()
