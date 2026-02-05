#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Robustness Summary Table for Paper

Creates a compact summary table showing F1 degradation under attacks.

Output: paper/tables/robustness_summary.tex
"""
import argparse
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_tex", default="paper/tables/robustness_summary.tex")
    args = ap.parse_args()
    
    # Load robustness results from all datasets
    results_files = [
        ("SMS", ROOT / "results/robustness_dedup_sms.csv"),
        ("SpamAssassin", ROOT / "results/robustness_dedup_spamassassin.csv"),
        ("Telegram", ROOT / "results/robustness_dedup_telegram.csv"),
    ]
    
    all_data = []
    for dataset_name, path in results_files:
        if path.exists():
            df = pd.read_csv(path)
            df["dataset_display"] = dataset_name
            all_data.append(df)
    
    if not all_data:
        print("No robustness results found")
        return
    
    df_all = pd.concat(all_data, ignore_index=True)
    
    # Focus on key models and attacks
    key_models = ["tfidf_word_lr", "tfidf_char_svm", "minilm_lr", "tfidf_word_lr_eat"]
    attacks = ["clean", "obfuscate", "paraphrase_like", "prompt_injection"]
    
    # Filter to no-defense results for cleaner comparison
    df_filtered = df_all[df_all["defense"] == "none"].copy()
    
    # Create pivot for each dataset
    summary_rows = []
    
    for _, row in df_filtered.iterrows():
        model_name = row["model"]
        # Extract model type from filename
        for key in key_models:
            if key in model_name:
                summary_rows.append({
                    "Dataset": row["dataset_display"],
                    "Model": key.replace("_", " ").title().replace("Tfidf", "TF-IDF").replace("Minilm", "MiniLM").replace("Eat", "(EAT)"),
                    "Attack": row["attack"],
                    "F1": row.get("f1_attacked", row.get("f1_clean", 0)),
                })
                break
    
    df_summary = pd.DataFrame(summary_rows)
    
    # Pivot to wide format
    df_pivot = df_summary.pivot_table(
        index=["Dataset", "Model"],
        columns="Attack",
        values="F1",
        aggfunc="mean"
    ).reset_index()
    
    # Generate LaTeX
    latex = r"""\begin{table}[t]
\centering
\caption{Robustness summary: F1 scores under different attacks (no defense). Lower F1 under attacks indicates higher vulnerability. EAT models show improved robustness compared to baseline.}
\label{tab:robustness_summary}
\begin{tabular}{llrrrr}
\toprule
Dataset & Model & Clean & Obfuscate & Paraphrase & Prompt Inj. \\
\midrule
"""
    
    current_dataset = None
    for _, row in df_pivot.iterrows():
        dataset = row["Dataset"]
        if dataset != current_dataset:
            if current_dataset is not None:
                latex += r"\midrule" + "\n"
            current_dataset = dataset
        
        clean = row.get("clean", 0)
        obfuscate = row.get("obfuscate", 0)
        paraphrase = row.get("paraphrase_like", 0)
        prompt_inj = row.get("prompt_injection", 0)
        
        latex += f"{dataset} & {row['Model']} & {clean:.3f} & {obfuscate:.3f} & {paraphrase:.3f} & {prompt_inj:.3f} \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    
    out_path = ROOT / args.out_tex
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(latex)
    
    print(f"✅ LaTeX table saved to {out_path}")
    print(df_pivot.to_string(index=False))


if __name__ == "__main__":
    main()
