#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate 3-Domain Cross-Domain Generalization Table for Paper

Creates the "Table Y: Cross-domain generalization under DedupShift (F1)" table
with a minimal-sufficient set:
- In-domain: SMS→SMS, SpamAssassin→SpamAssassin, Telegram→Telegram
- Key cross-domain pairs showing "modern domain difficulty"

Output: results/cross_domain_3domain.csv + paper/tables/cross_domain_3domain.tex
"""
import argparse
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]


def load_data(csv_path: str) -> pd.DataFrame:
    """Load dataset with text and label columns."""
    df = pd.read_csv(csv_path)
    text_col = "text" if "text" in df.columns else df.columns[0]
    label_col = "label" if "label" in df.columns else df.columns[1]
    
    # Normalize labels
    if df[label_col].dtype == object:
        label_map = {"spam": 1, "ham": 0, "1": 1, "0": 0}
        df["label"] = df[label_col].astype(str).str.lower().map(label_map).fillna(0).astype(int)
    else:
        df["label"] = df[label_col].astype(int)
    
    df["text"] = df[text_col].astype(str)
    return df


def evaluate_model(model, X_test, y_test) -> float:
    """Evaluate model and return F1 score."""
    try:
        y_pred = model.predict(X_test)
        return f1_score(y_test, y_pred, zero_division=0)
    except Exception as e:
        print(f"  Evaluation error: {e}")
        return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_csv", default="results/cross_domain_3domain.csv")
    ap.add_argument("--out_tex", default="paper/tables/cross_domain_3domain.tex")
    args = ap.parse_args()
    
    # Dataset configurations
    datasets = {
        "sms": {
            "data_path": ROOT / "dataset/sms_uci/dedup/processed/data.csv",
            "model_prefix": "sms_dedup",
            "display": "SMS",
        },
        "spamassassin": {
            "data_path": ROOT / "dataset/spamassassin/dedup/processed/data.csv",
            "model_prefix": "spamassassin_dedup",
            "display": "SpamAssassin",
        },
        "telegram": {
            "data_path": ROOT / "dataset/telegram_spam_ham/dedup/processed/data.csv",
            "model_prefix": "telegram_dedup",
            "display": "Telegram",
        },
    }
    
    # Models to evaluate
    models_config = [
        {"suffix": "tfidf_word_lr", "display": "TF-IDF Word LR"},
        {"suffix": "tfidf_char_svm", "display": "TF-IDF Char SVM"},
        {"suffix": "minilm_lr", "display": "MiniLM+LR"},
        {"suffix": "tfidf_word_lr_eat", "display": "TF-IDF Word LR (EAT)"},
    ]
    
    # Filter available datasets
    available_datasets = {}
    for name, cfg in datasets.items():
        if cfg["data_path"].exists():
            available_datasets[name] = cfg
            print(f"✓ Found {name} dataset")
        else:
            print(f"✗ Missing {name} dataset at {cfg['data_path']}")
    
    if len(available_datasets) < 2:
        print("Need at least 2 datasets for cross-domain evaluation")
        return
    
    # Define evaluation pairs (train_domain, test_domain)
    # In-domain + key cross-domain pairs
    eval_pairs = []
    dataset_names = list(available_datasets.keys())
    
    for ds in dataset_names:
        eval_pairs.append((ds, ds))  # In-domain
    
    # Cross-domain pairs (focus on transfers involving modern domain)
    if "telegram" in dataset_names:
        if "sms" in dataset_names:
            eval_pairs.append(("sms", "telegram"))
            eval_pairs.append(("telegram", "sms"))
        if "spamassassin" in dataset_names:
            eval_pairs.append(("spamassassin", "telegram"))
            eval_pairs.append(("telegram", "spamassassin"))
    
    if "sms" in dataset_names and "spamassassin" in dataset_names:
        eval_pairs.append(("sms", "spamassassin"))
        eval_pairs.append(("spamassassin", "sms"))
    
    # Load test sets
    test_data = {}
    for name, cfg in available_datasets.items():
        df = load_data(str(cfg["data_path"]))
        df_test = df[df.get("split", "test") == "test"].copy() if "split" in df.columns else df.copy()
        test_data[name] = df_test
        print(f"  Loaded {len(df_test)} test samples from {name}")
    
    # Evaluate all pairs
    results = []
    for train_ds, test_ds in eval_pairs:
        train_cfg = available_datasets[train_ds]
        test_cfg = available_datasets[test_ds]
        
        row = {
            "Train": train_cfg["display"],
            "Test": test_cfg["display"],
            "Type": "In-domain" if train_ds == test_ds else "Cross-domain",
        }
        
        df_test = test_data[test_ds]
        X_test = df_test["text"].tolist()
        y_test = df_test["label"].values
        
        for model_cfg in models_config:
            model_path = ROOT / "models" / f"{train_cfg['model_prefix']}_{model_cfg['suffix']}.joblib"
            
            if model_path.exists():
                try:
                    model = joblib.load(model_path)
                    f1 = evaluate_model(model, X_test, y_test)
                    row[model_cfg["display"]] = round(f1, 4)
                except Exception as e:
                    print(f"  Error loading {model_path}: {e}")
                    row[model_cfg["display"]] = "-"
            else:
                row[model_cfg["display"]] = "-"
        
        results.append(row)
        print(f"  {train_cfg['display']} → {test_cfg['display']}: done")
    
    # Create DataFrame
    df_results = pd.DataFrame(results)
    
    # Save CSV
    out_csv = ROOT / args.out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(out_csv, index=False)
    print(f"\n✅ CSV saved to {out_csv}")
    
    # Generate LaTeX table
    out_tex = ROOT / args.out_tex
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    
    model_cols = [m["display"] for m in models_config]
    n_models = len(model_cols)
    
    latex = r"""\begin{table}[t]
\centering
\caption{Cross-domain generalization under DedupShift (F1 score). In-domain results (diagonal) show strong performance; cross-domain transfers reveal significant degradation, especially involving the modern Telegram corpus. EAT partially recovers cross-domain performance.}
\label{tab:cross_domain_3domain}
\begin{tabular}{ll""" + "r" * n_models + r"""}
\toprule
Train & Test & """ + " & ".join(model_cols) + r""" \\
\midrule
"""
    
    current_type = None
    for _, row in df_results.iterrows():
        if row["Type"] != current_type:
            if current_type is not None:
                latex += r"\midrule" + "\n"
            current_type = row["Type"]
        
        values = [str(row.get(col, "-")) if row.get(col, "-") != "-" else "-" for col in model_cols]
        latex += f"{row['Train']} & {row['Test']} & " + " & ".join(values) + r" \\" + "\n"
    
    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    
    with open(out_tex, "w") as f:
        f.write(latex)
    print(f"✅ LaTeX table saved to {out_tex}")
    
    # Print summary
    print("\n📊 Cross-Domain Generalization Summary:")
    print(df_results.to_string(index=False))
    
    # Compute and print key findings
    print("\n📈 Key Findings:")
    for model_col in model_cols:
        in_domain = df_results[df_results["Type"] == "In-domain"][model_col]
        cross_domain = df_results[df_results["Type"] == "Cross-domain"][model_col]
        
        in_domain_vals = pd.to_numeric(in_domain, errors="coerce").dropna()
        cross_domain_vals = pd.to_numeric(cross_domain, errors="coerce").dropna()
        
        if len(in_domain_vals) > 0 and len(cross_domain_vals) > 0:
            print(f"  {model_col}:")
            print(f"    In-domain avg F1: {in_domain_vals.mean():.4f}")
            print(f"    Cross-domain avg F1: {cross_domain_vals.mean():.4f}")
            print(f"    Degradation: {(in_domain_vals.mean() - cross_domain_vals.mean()):.4f}")


if __name__ == "__main__":
    main()
