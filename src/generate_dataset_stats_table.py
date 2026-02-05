#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Dataset Statistics Table for Paper

Creates a unified table with:
- Dataset name and source
- #Samples (total)
- Spam% (spam ratio)
- Avg tokens (mean token count)
- Dedup retention (% retained after deduplication)
- Split leakage before/after deduplication

Output: results/dataset_stats_table.csv + paper/tables/dataset_stats_table.tex
"""
import argparse
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def compute_dataset_stats_from_splits(split_files: list, dedup_files: list, name: str, source: str) -> dict:
    """Compute statistics from multiple split files."""
    stats = {
        "Dataset": name,
        "Source": source,
        "#Samples": 0,
        "Spam%": 0.0,
        "Avg tokens": 0.0,
        "Dedup retention": 0.0,
    }
    
    # Load and combine all split files
    dfs = []
    for f in split_files:
        if os.path.exists(f):
            try:
                dfs.append(pd.read_csv(f))
            except Exception as e:
                print(f"  Warning: Error loading {f}: {e}")
    
    if not dfs:
        print(f"  Warning: No data files found for {name}")
        return stats
    
    df_raw = pd.concat(dfs, ignore_index=True)
    
    # Detect text and label columns
    text_col = "text" if "text" in df_raw.columns else df_raw.columns[0]
    label_col = "label" if "label" in df_raw.columns else df_raw.columns[1]
    
    # Basic stats from raw
    n_raw = len(df_raw)
    
    # Map labels to 0/1 for spam ratio
    if df_raw[label_col].dtype == object:
        label_map = {"spam": 1, "ham": 0, "1": 1, "0": 0}
        labels = df_raw[label_col].astype(str).str.lower().map(label_map).fillna(0)
    else:
        labels = df_raw[label_col].astype(int)
    
    spam_ratio = labels.mean() * 100
    
    # Token count (simple whitespace split)
    avg_tokens = df_raw[text_col].astype(str).apply(lambda x: len(x.split())).mean()
    
    # Dedup stats - combine dedup files
    dedup_dfs = []
    for f in dedup_files:
        if os.path.exists(f):
            try:
                dedup_dfs.append(pd.read_csv(f))
            except:
                pass
    
    n_dedup = sum(len(df) for df in dedup_dfs) if dedup_dfs else n_raw
    dedup_retention = (n_dedup / n_raw * 100) if n_raw > 0 else 0
    
    stats.update({
        "#Samples": n_raw,
        "Spam%": round(spam_ratio, 1),
        "Avg tokens": round(avg_tokens, 1),
        "Dedup retention": round(dedup_retention, 1),
    })
    
    return stats


def compute_dataset_stats(raw_csv: str, dedup_csv: str, name: str, source: str) -> dict:
    """Compute statistics for a single dataset (legacy function)."""
    stats = {
        "Dataset": name,
        "Source": source,
        "#Samples": 0,
        "Spam%": 0.0,
        "Avg tokens": 0.0,
        "Dedup retention": 0.0,
    }
    
    # Check if files exist
    if not os.path.exists(raw_csv):
        print(f"  Warning: {raw_csv} not found")
        return stats
    
    # Load raw data
    try:
        df_raw = pd.read_csv(raw_csv)
    except Exception as e:
        print(f"  Error loading {raw_csv}: {e}")
        return stats
    
    # Detect text and label columns
    text_col = "text" if "text" in df_raw.columns else df_raw.columns[0]
    label_col = "label" if "label" in df_raw.columns else df_raw.columns[1]
    
    # Basic stats from raw
    n_raw = len(df_raw)
    
    # Map labels to 0/1 for spam ratio
    if df_raw[label_col].dtype == object:
        label_map = {"spam": 1, "ham": 0, "1": 1, "0": 0}
        labels = df_raw[label_col].astype(str).str.lower().map(label_map).fillna(0)
    else:
        labels = df_raw[label_col].astype(int)
    
    spam_ratio = labels.mean() * 100
    
    # Token count (simple whitespace split)
    avg_tokens = df_raw[text_col].astype(str).apply(lambda x: len(x.split())).mean()
    
    # Dedup stats
    n_dedup = n_raw
    if os.path.exists(dedup_csv):
        try:
            df_dedup = pd.read_csv(dedup_csv)
            n_dedup = len(df_dedup)
        except:
            pass
    
    dedup_retention = (n_dedup / n_raw * 100) if n_raw > 0 else 0
    
    stats.update({
        "#Samples": n_raw,
        "Spam%": round(spam_ratio, 1),
        "Avg tokens": round(avg_tokens, 1),
        "Dedup retention": round(dedup_retention, 1),
    })
    
    return stats


def compute_leakage(raw_csv: str, dedup_csv: str) -> tuple:
    """Compute split leakage before and after deduplication."""
    leakage_before = 0.0
    leakage_after = 0.0
    
    # We compute leakage as % of test samples that have near-duplicates in train
    # This is a simplified version - actual leakage detection uses SimHash
    
    def _compute_text_overlap(df):
        """Compute exact text overlap between train and test."""
        if "split" not in df.columns:
            return 0.0
        text_col = "text" if "text" in df.columns else df.columns[0]
        train_texts = set(df[df["split"] == "train"][text_col].astype(str).str.lower())
        test_texts = df[df["split"] == "test"][text_col].astype(str).str.lower()
        if len(test_texts) == 0:
            return 0.0
        overlap = sum(1 for t in test_texts if t in train_texts)
        return overlap / len(test_texts) * 100
    
    # Before dedup
    if os.path.exists(raw_csv):
        try:
            df_raw = pd.read_csv(raw_csv)
            leakage_before = _compute_text_overlap(df_raw)
        except:
            pass
    
    # After dedup
    if os.path.exists(dedup_csv):
        try:
            df_dedup = pd.read_csv(dedup_csv)
            leakage_after = _compute_text_overlap(df_dedup)
        except:
            pass
    
    return round(leakage_before, 2), round(leakage_after, 2)


def compute_leakage_from_files(train_raw: str, test_raw: str, train_dedup: str, test_dedup: str) -> tuple:
    """Compute split leakage from separate train/test files."""
    leakage_before = 0.0
    leakage_after = 0.0
    
    def _compute_overlap(train_file: str, test_file: str) -> float:
        if not os.path.exists(train_file) or not os.path.exists(test_file):
            return 0.0
        try:
            df_train = pd.read_csv(train_file)
            df_test = pd.read_csv(test_file)
            text_col = "text" if "text" in df_train.columns else df_train.columns[0]
            train_texts = set(df_train[text_col].astype(str).str.lower())
            test_texts = df_test[text_col].astype(str).str.lower()
            if len(test_texts) == 0:
                return 0.0
            overlap = sum(1 for t in test_texts if t in train_texts)
            return overlap / len(test_texts) * 100
        except:
            return 0.0
    
    leakage_before = _compute_overlap(train_raw, test_raw)
    leakage_after = _compute_overlap(train_dedup, test_dedup)
    
    return round(leakage_before, 2), round(leakage_after, 2)


def generate_latex_table(df: pd.DataFrame, out_path: str):
    """Generate LaTeX table from dataframe."""
    latex = r"""\begin{table}[t]
\centering
\caption{Dataset statistics and DedupShift retention. Dedup retention shows the percentage of samples retained after near-duplicate removal. Split leakage measures exact text overlap between train and test splits before/after deduplication.}
\label{tab:dataset_stats}
\begin{tabular}{lllrrrrr}
\toprule
Dataset & Source & \#Samples & Spam\% & Avg tokens & Dedup ret. & Leakage (before/after) \\
\midrule
"""
    for _, row in df.iterrows():
        latex += f"{row['Dataset']} & {row['Source']} & {row['#Samples']:,} & {row['Spam%']:.1f}\\% & {row['Avg tokens']:.1f} & {row['Dedup retention']:.1f}\\% & {row['Leakage (before)']}\\% / {row['Leakage (after)']}\\% \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    with open(out_path, "w") as f:
        f.write(latex)
    print(f"✅ LaTeX table saved to {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_csv", default="results/dataset_stats_table.csv")
    ap.add_argument("--out_tex", default="paper/tables/dataset_stats_table.tex")
    args = ap.parse_args()
    
    # Define datasets configuration
    # Note: SMS uses dataset/processed/, SpamAssassin uses dataset/spamassassin/processed/
    datasets = [
        {
            "name": "SMS (UCI)",
            "source": "SMSSpamCollection",
            "raw_csv": "dataset/processed/train.csv",  # Use train.csv as raw proxy
            "dedup_csv": "dataset/dedup/processed/train.csv",
            "all_splits": ["dataset/processed/train.csv", "dataset/processed/val.csv", "dataset/processed/test.csv"],
            "dedup_splits": ["dataset/dedup/processed/train.csv", "dataset/dedup/processed/val.csv", "dataset/dedup/processed/test.csv"],
        },
        {
            "name": "SpamAssassin",
            "source": "Apache SpamAssassin",
            "raw_csv": "dataset/spamassassin/processed/train.csv",
            "dedup_csv": "dataset/spamassassin/dedup/processed/train.csv",
            "all_splits": ["dataset/spamassassin/processed/train.csv", "dataset/spamassassin/processed/val.csv", "dataset/spamassassin/processed/test.csv"],
            "dedup_splits": ["dataset/spamassassin/dedup/processed/train.csv", "dataset/spamassassin/dedup/processed/val.csv", "dataset/spamassassin/dedup/processed/test.csv"],
        },
        {
            "name": "Telegram",
            "source": "Kaggle (2024)",
            "raw_csv": "dataset/telegram_spam_ham/processed/train.csv",
            "dedup_csv": "dataset/telegram_spam_ham/dedup/processed/train.csv",
            "all_splits": ["dataset/telegram_spam_ham/processed/train.csv", "dataset/telegram_spam_ham/processed/val.csv", "dataset/telegram_spam_ham/processed/test.csv"],
            "dedup_splits": ["dataset/telegram_spam_ham/dedup/processed/train.csv", "dataset/telegram_spam_ham/dedup/processed/val.csv", "dataset/telegram_spam_ham/dedup/processed/test.csv"],
        },
    ]
    
    rows = []
    for ds in datasets:
        print(f"Processing {ds['name']}...")
        
        # Use new function with split files if available
        if "all_splits" in ds:
            all_splits = [str(ROOT / f) for f in ds["all_splits"]]
            dedup_splits = [str(ROOT / f) for f in ds["dedup_splits"]]
            stats = compute_dataset_stats_from_splits(all_splits, dedup_splits, ds["name"], ds["source"])
        else:
            raw_path = str(ROOT / ds["raw_csv"])
            dedup_path = str(ROOT / ds["dedup_csv"])
            stats = compute_dataset_stats(raw_path, dedup_path, ds["name"], ds["source"])
        
        # Compute leakage from train/test files
        train_raw = str(ROOT / ds.get("all_splits", [ds["raw_csv"]])[0])
        test_raw = str(ROOT / ds.get("all_splits", [ds["raw_csv"]])[-1]) if "all_splits" in ds else train_raw
        train_dedup = str(ROOT / ds.get("dedup_splits", [ds["dedup_csv"]])[0])
        test_dedup = str(ROOT / ds.get("dedup_splits", [ds["dedup_csv"]])[-1]) if "dedup_splits" in ds else train_dedup
        
        leakage_before, leakage_after = compute_leakage_from_files(train_raw, test_raw, train_dedup, test_dedup)
        stats["Leakage (before)"] = leakage_before
        stats["Leakage (after)"] = leakage_after
        rows.append(stats)
    
    df = pd.DataFrame(rows)
    
    # Save CSV
    out_csv = ROOT / args.out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"✅ CSV saved to {out_csv}")
    
    # Save LaTeX
    out_tex = ROOT / args.out_tex
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    generate_latex_table(df, str(out_tex))
    
    # Print summary
    print("\n📊 Dataset Statistics Summary:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
