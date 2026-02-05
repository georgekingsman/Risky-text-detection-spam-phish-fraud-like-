#!/usr/bin/env python
"""
Telegram Spam or Ham dataset preparation.

Downloads and standardizes the Telegram Spam or Ham dataset from Kaggle
into the unified format used by this project.

Usage:
    # Option A: Download from Kaggle API (requires kaggle CLI configured)
    kaggle datasets download -d mexwell/telegram-spam-or-ham \
        -p dataset/telegram_spam_ham/raw --unzip

    # Option B: Manual download from Kaggle, then run:
    python -m src.prepare_telegram \
        --in_csv dataset/telegram_spam_ham/raw/telegram_spam_or_ham.csv \
        --out_csv dataset/telegram_spam_ham/processed/data.csv

The output CSV will have columns: text, label (0=ham, 1=spam)
"""
import argparse
import re
from pathlib import Path

import pandas as pd


def infer_columns(df: pd.DataFrame, text_col: str | None, label_col: str | None):
    """Auto-detect text and label columns if not specified."""
    if text_col is None:
        for cand in ["text", "message", "content", "body", "Message", "Text", "Content"]:
            if cand in df.columns:
                text_col = cand
                break
    
    if label_col is None:
        for cand in ["label", "class", "spam", "target", "y", "Label", "Class", "Spam", "category"]:
            if cand in df.columns:
                label_col = cand
                break
    
    return text_col, label_col


def standardize_label(val) -> str | None:
    """Standardize label to 0 (ham) or 1 (spam)."""
    s = str(val).lower().strip()
    
    # Handle common label formats
    mapping = {
        "spam": "1",
        "ham": "0",
        "1": "1",
        "0": "0",
        "1.0": "1",
        "0.0": "0",
        "true": "1",
        "false": "0",
        "yes": "1",
        "no": "0",
        "positive": "1",
        "negative": "0",
    }
    
    return mapping.get(s, None)


def clean_text(text: str) -> str:
    """Basic text cleaning."""
    if pd.isna(text):
        return ""
    
    text = str(text).strip()
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove null characters
    text = text.replace('\x00', '')
    
    return text


def main():
    ap = argparse.ArgumentParser(
        description="Prepare Telegram Spam or Ham dataset for unified processing"
    )
    ap.add_argument(
        "--in_csv", required=True,
        help="Input CSV from Kaggle raw download"
    )
    ap.add_argument(
        "--out_csv", required=True,
        help="Output standardized CSV path"
    )
    ap.add_argument(
        "--out_split_dir", default=None,
        help="If set, also create train/val/test.csv splits here (80/10/10)"
    )
    ap.add_argument(
        "--text_col", default=None,
        help="Text column name (auto-detect if None)"
    )
    ap.add_argument(
        "--label_col", default=None,
        help="Label column name (auto-detect if None)"
    )
    ap.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for splitting"
    )
    args = ap.parse_args()
    
    # Read raw CSV
    in_path = Path(args.in_csv)
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")
    
    print(f"[INFO] Reading {in_path}")
    df = pd.read_csv(in_path)
    print(f"[INFO] Raw shape: {df.shape}")
    print(f"[INFO] Columns: {list(df.columns)}")
    
    # Auto-detect columns
    text_col, label_col = infer_columns(df, args.text_col, args.label_col)
    
    if text_col is None:
        raise ValueError(f"Cannot infer text column. Available: {list(df.columns)}")
    if label_col is None:
        raise ValueError(f"Cannot infer label column. Available: {list(df.columns)}")
    
    print(f"[INFO] Using text_col={text_col}, label_col={label_col}")
    
    # Extract and rename columns
    out = df[[text_col, label_col]].copy()
    out.columns = ["text", "label"]
    
    # Standardize labels
    out["label"] = out["label"].apply(standardize_label)
    
    # Check for unmapped labels
    unmapped = out["label"].isna()
    if unmapped.any():
        original_vals = df.loc[unmapped.values, label_col].unique()
        print(f"[WARN] {unmapped.sum()} rows with unmapped labels: {original_vals}")
        out = out[~unmapped]
    
    # Clean text
    out["text"] = out["text"].apply(clean_text)
    
    # Remove empty texts
    empty_mask = out["text"].str.len() == 0
    if empty_mask.any():
        print(f"[WARN] Removing {empty_mask.sum()} rows with empty text")
        out = out[~empty_mask]
    
    # Convert label to int
    out["label"] = out["label"].astype(int)
    
    # Print class distribution
    print(f"\n[INFO] Class distribution:")
    print(f"  ham (0):  {(out['label'] == 0).sum()}")
    print(f"  spam (1): {(out['label'] == 1).sum()}")
    print(f"  total:    {len(out)}")
    
    # Write output
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"\n[OK] Wrote {out_path} (rows={len(out)})")
    
    # Always create train/val/test splits for dedup_split.py compatibility
    # Use same directory as output or specified split dir
    split_dir = Path(args.out_split_dir) if args.out_split_dir else out_path.parent
    from sklearn.model_selection import train_test_split
    
    split_dir.mkdir(parents=True, exist_ok=True)
    
    # Stratified 80/10/10 split
    train_df, temp_df = train_test_split(
        out, test_size=0.2, random_state=args.seed,
        stratify=out["label"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=args.seed,
        stratify=temp_df["label"]
    )
    
    train_df.to_csv(split_dir / "train.csv", index=False)
    val_df.to_csv(split_dir / "val.csv", index=False)
    test_df.to_csv(split_dir / "test.csv", index=False)
    
    print(f"\n[OK] Wrote splits to {split_dir}:")
    print(f"  train.csv: {len(train_df)} rows")
    print(f"  val.csv:   {len(val_df)} rows")
    print(f"  test.csv:  {len(test_df)} rows")


if __name__ == "__main__":
    main()
