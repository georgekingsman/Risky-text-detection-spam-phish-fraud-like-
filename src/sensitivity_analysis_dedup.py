#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DedupShift Hyperparameter Sensitivity Analysis

Analyzes the impact of SimHash Hamming threshold (h_thresh) on:
- Deduplication rate (% of texts removed)
- In-domain F1 scores across different models
- Trade-off between dedup aggressiveness and model performance

Thresholds tested: 2, 3 (default), 4

Usage:
    python -m src.sensitivity_analysis_dedup

Outputs:
    results/sensitivity_dedup_summary.csv - aggregated sensitivity table
"""
from pathlib import Path
from typing import List, Tuple
import tempfile
import subprocess
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]


def dedup_and_train_eval(
    input_csv: Path,
    h_thresh: int,
    dataset_name: str,
) -> dict:
    """
    Apply dedup with specific h_thresh and evaluate TF-IDF LR on resulting data.
    
    Returns a dict with sensitivity metrics.
    """
    # Read input data
    df = pd.read_csv(input_csv)
    texts = df["text"].astype(str).tolist()
    labels = [1 if x == 1 else 0 for x in df["label"].tolist()]
    n_in = len(texts)
    
    # Create temp dedup directory and run dedup_split.py
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        temp_report = temp_dir_path / "dedup_report.csv"
        temp_output = temp_dir_path / "dedup_output"
        temp_output.mkdir(exist_ok=True)
        
        # Run dedup_split to generate deduplicated data
        cmd = [
            "python", "-m", "src.dedup_split",
            "--data-dir", str(input_csv.parent),
            "--out-dir", str(temp_output),
            "--report", str(temp_report),
            "--near",
            f"--h-thresh={h_thresh}",
            "--seed=0"
        ]
        
        result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        if result.returncode != 0:
            print(f"⚠️  Dedup failed for {dataset_name} h_thresh={h_thresh}: {result.stderr}")
            return None
        
        # Read deduplicated data
        dedup_test_csv = temp_output / "test.csv"
        if not dedup_test_csv.exists():
            print(f"⚠️  No dedup output for {dataset_name}")
            return None
        
        df_dedup = pd.read_csv(dedup_test_csv)
        texts_dedup = df_dedup["text"].astype(str).tolist()
        labels_dedup = [1 if x == 1 else 0 for x in df_dedup["label"].tolist()]
        n_out = len(texts_dedup)
        
        # Read dedup report to get exact/near removed counts
        n_exact = 0
        n_near = 0
        if temp_report.exists():
            df_report = pd.read_csv(temp_report)
            if len(df_report) > 0:
                n_exact = int(df_report["n_exact_removed"].iloc[0])
                n_near = int(df_report["n_near_removed"].iloc[0])
        
        dedup_rate = (n_in - n_out) / n_in * 100 if n_in > 0 else 0
        
        # Train TF-IDF + LR on deduplicated training data
        if len(texts_dedup) < 10:
            print(f"⚠️  Too few samples after dedup ({n_out})")
            return None
        
        # Simple split
        X_train, X_test, y_train, y_test = train_test_split(
            texts_dedup, labels_dedup, test_size=0.2, random_state=0, stratify=labels_dedup if len(set(labels_dedup)) > 1 else None
        )
        
        try:
            vec = TfidfVectorizer(analyzer="word", max_features=10000, ngram_range=(1, 1), token_pattern=r"\b\w+\b")
            X_train_vec = vec.fit_transform(X_train)
            X_test_vec = vec.transform(X_test)
            
            clf = LogisticRegression(max_iter=1000, random_state=0)
            clf.fit(X_train_vec, y_train)
            preds = clf.predict(X_test_vec)
            f1 = f1_score(y_test, preds)
        except Exception as e:
            print(f"⚠️  Training failed: {e}")
            return None
    
    return {
        "dataset": dataset_name,
        "h_thresh": h_thresh,
        "n_input": n_in,
        "n_exact_removed": n_exact,
        "n_near_removed": n_near,
        "n_output": n_out,
        "dedup_rate_%": round(dedup_rate, 2),
        "model": "tfidf_word_lr",
        "f1_score": round(f1, 4),
    }


def main():
    """Run sensitivity analysis for different h_thresh values."""
    h_thresholds = [2, 3, 4]
    
    # Input CSVs
    sms_input = ROOT / "dataset" / "processed" / "train.csv"
    spam_input = ROOT / "dataset" / "spamassassin" / "processed" / "train.csv"
    
    datasets = [
        ("SMS (UCI)", sms_input),
        ("SpamAssassin", spam_input),
    ]
    
    results = []
    
    for dataset_name, input_csv in datasets:
        if not input_csv.exists():
            print(f"⚠️  Skipping {dataset_name}: input file not found at {input_csv}")
            continue
        
        print(f"\n📊 Analyzing {dataset_name}...")
        
        for h_thresh in h_thresholds:
            print(f"  h_thresh={h_thresh}...", end=" ", flush=True)
            result = dedup_and_train_eval(input_csv, h_thresh, dataset_name)
            if result:
                results.append(result)
                print(f"✓ F1={result['f1_score']:.4f} (dedup {result['dedup_rate_%']:.1f}%)")
            else:
                print("✗ failed")
    
    # Save results
    if results:
        out_path = ROOT / "results" / "sensitivity_dedup_summary.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_results = pd.DataFrame(results)
        df_results.to_csv(out_path, index=False)
        print(f"\n✅ Sensitivity analysis saved to {out_path}")
        print(df_results.to_string(index=False))
    else:
        print("⚠️  No results generated.")


if __name__ == "__main__":
    main()
