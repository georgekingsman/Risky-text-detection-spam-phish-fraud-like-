#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cost-Throughput Analysis: Robustness vs. Inference Cost Trade-off

Measures inference latency and throughput for different models, then
generates a trade-off figure showing:
- x-axis: inference latency (ms/msg) or throughput (msg/s)
- y-axis: robust F1 (under obfuscation attack)

Output: 
- results/cost_throughput.csv
- paper/figs/fig_cost_throughput.png

Usage:
    python src/cost_throughput_analysis.py --dataset sms --n_samples 200
"""
import argparse
import os
import time
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]


def measure_inference_time(model, texts: List[str], n_runs: int = 3) -> Dict:
    """Measure inference latency and throughput for a model."""
    # Warm-up run
    try:
        _ = model.predict(texts[:10])
    except Exception:
        pass
    
    latencies = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = model.predict(texts)
        end = time.perf_counter()
        latencies.append(end - start)
    
    avg_time = np.mean(latencies)
    std_time = np.std(latencies)
    n_samples = len(texts)
    
    return {
        "total_time_s": avg_time,
        "total_time_std": std_time,
        "latency_ms_per_msg": (avg_time / n_samples) * 1000,
        "throughput_msg_per_s": n_samples / avg_time if avg_time > 0 else 0,
    }


def load_robustness_results(results_path: str) -> pd.DataFrame:
    """Load robustness results for F1 under attacks."""
    if not os.path.exists(results_path):
        return pd.DataFrame()
    return pd.read_csv(results_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sms", choices=["sms", "spamassassin", "telegram"])
    ap.add_argument("--n_samples", type=int, default=200, help="Number of samples for timing")
    ap.add_argument("--n_runs", type=int, default=3, help="Number of timing runs")
    ap.add_argument("--out_csv", default="results/cost_throughput.csv")
    ap.add_argument("--out_fig", default="paper/figs/fig_cost_throughput.png")
    args = ap.parse_args()
    
    # Dataset paths
    dataset_paths = {
        "sms": ROOT / "dataset/sms_uci/dedup/processed/data.csv",
        "spamassassin": ROOT / "dataset/spamassassin/dedup/processed/data.csv",
        "telegram": ROOT / "dataset/telegram_spam_ham/dedup/processed/data.csv",
    }
    
    # Model configurations
    models_config = {
        "sms": [
            ("TF-IDF Word LR", "models/sms_dedup_tfidf_word_lr.joblib"),
            ("TF-IDF Char SVM", "models/sms_dedup_tfidf_char_svm.joblib"),
            ("MiniLM+LR", "models/sms_dedup_minilm_lr.joblib"),
            ("TF-IDF Word LR (EAT)", "models/sms_dedup_tfidf_word_lr_eat.joblib"),
        ],
        "spamassassin": [
            ("TF-IDF Word LR", "models/spamassassin_dedup_tfidf_word_lr.joblib"),
            ("TF-IDF Char SVM", "models/spamassassin_dedup_tfidf_char_svm.joblib"),
            ("MiniLM+LR", "models/spamassassin_dedup_minilm_lr.joblib"),
            ("TF-IDF Word LR (EAT)", "models/spamassassin_dedup_tfidf_word_lr_eat.joblib"),
        ],
        "telegram": [
            ("TF-IDF Word LR", "models/telegram_dedup_tfidf_word_lr.joblib"),
            ("TF-IDF Char SVM", "models/telegram_dedup_tfidf_char_svm.joblib"),
            ("MiniLM+LR", "models/telegram_dedup_minilm_lr.joblib"),
            ("TF-IDF Word LR (EAT)", "models/telegram_dedup_tfidf_word_lr_eat.joblib"),
        ],
    }
    
    # Robustness results paths
    robustness_paths = {
        "sms": ROOT / "results/robustness_dedup_sms.csv",
        "spamassassin": ROOT / "results/robustness_dedup_spamassassin.csv",
        "telegram": ROOT / "results/robustness_dedup_telegram.csv",
    }
    
    # Load test data
    data_path = dataset_paths.get(args.dataset)
    if not data_path or not data_path.exists():
        print(f"Dataset not found: {data_path}")
        return
    
    df = pd.read_csv(data_path)
    text_col = "text" if "text" in df.columns else df.columns[0]
    
    # Get test split if available
    if "split" in df.columns:
        df_test = df[df["split"] == "test"].copy()
    else:
        df_test = df.sample(n=min(args.n_samples, len(df)), random_state=42)
    
    texts = df_test[text_col].astype(str).tolist()[:args.n_samples]
    print(f"Using {len(texts)} samples from {args.dataset} for timing")
    
    # Load robustness results
    robustness_df = load_robustness_results(str(robustness_paths.get(args.dataset, "")))
    
    # Measure each model
    results = []
    for model_name, model_path in models_config.get(args.dataset, []):
        full_path = ROOT / model_path
        if not full_path.exists():
            print(f"  Skipping {model_name}: model not found at {full_path}")
            continue
        
        print(f"  Measuring {model_name}...")
        
        try:
            model = joblib.load(full_path)
            timing = measure_inference_time(model, texts, args.n_runs)
            
            # Get robustness F1 (obfuscate attack, no defense)
            robust_f1 = None
            if not robustness_df.empty:
                model_file = os.path.basename(model_path)
                mask = (
                    (robustness_df["attack"] == "obfuscate") & 
                    (robustness_df["model"] == model_file) &
                    (robustness_df["defense"] == "none")
                )
                if mask.any():
                    robust_f1 = robustness_df.loc[mask, "f1_attacked"].values[0]
            
            # Get clean F1
            clean_f1 = None
            if not robustness_df.empty:
                mask_clean = (
                    (robustness_df["attack"] == "clean") & 
                    (robustness_df["model"] == model_file) &
                    (robustness_df["defense"] == "none")
                )
                if mask_clean.any():
                    clean_f1 = robustness_df.loc[mask_clean, "f1_attacked"].values[0]
            
            results.append({
                "model": model_name,
                "model_file": os.path.basename(model_path),
                "dataset": args.dataset,
                "latency_ms": round(timing["latency_ms_per_msg"], 3),
                "throughput_msg_s": round(timing["throughput_msg_per_s"], 1),
                "clean_f1": round(clean_f1, 4) if clean_f1 is not None else None,
                "robust_f1_obfuscate": round(robust_f1, 4) if robust_f1 is not None else None,
            })
            
            print(f"    Latency: {timing['latency_ms_per_msg']:.3f} ms/msg")
            print(f"    Throughput: {timing['throughput_msg_per_s']:.1f} msg/s")
            
        except Exception as e:
            print(f"  Error measuring {model_name}: {e}")
    
    if not results:
        print("No models measured successfully")
        return
    
    # Save results
    df_results = pd.DataFrame(results)
    out_csv = ROOT / args.out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(out_csv, index=False)
    print(f"\n✅ Results saved to {out_csv}")
    
    # Generate figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Filter models with robustness data
        df_plot = df_results.dropna(subset=["robust_f1_obfuscate"])
        
        if len(df_plot) > 0:
            # Plot
            colors = plt.cm.Set2(np.linspace(0, 1, len(df_plot)))
            markers = ['o', 's', '^', 'D', 'v', 'p']
            
            for idx, (_, row) in enumerate(df_plot.iterrows()):
                ax.scatter(
                    row["latency_ms"], 
                    row["robust_f1_obfuscate"],
                    s=150,
                    c=[colors[idx]],
                    marker=markers[idx % len(markers)],
                    label=row["model"],
                    edgecolors='black',
                    linewidths=1,
                    alpha=0.8
                )
            
            ax.set_xlabel("Inference Latency (ms/message)", fontsize=12)
            ax.set_ylabel("Robust F1 (under obfuscation)", fontsize=12)
            ax.set_title(f"Robustness vs. Cost Trade-off ({args.dataset.upper()})", fontsize=14)
            ax.legend(loc="best", fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Add annotation for best trade-off
            best_idx = (df_plot["robust_f1_obfuscate"] / df_plot["latency_ms"]).idxmax()
            best_row = df_plot.loc[best_idx]
            ax.annotate(
                "Best trade-off",
                xy=(best_row["latency_ms"], best_row["robust_f1_obfuscate"]),
                xytext=(10, -20),
                textcoords="offset points",
                fontsize=9,
                arrowprops=dict(arrowstyle="->", color="red", lw=1),
                color="red"
            )
            
            plt.tight_layout()
            
            out_fig = ROOT / args.out_fig
            out_fig.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_fig, dpi=150, bbox_inches="tight")
            print(f"✅ Figure saved to {out_fig}")
        else:
            print("No data with robustness results for plotting")
            
    except ImportError:
        print("matplotlib not available, skipping figure generation")
    
    # Print summary
    print("\n📊 Cost-Throughput Summary:")
    print(df_results.to_string(index=False))


if __name__ == "__main__":
    main()
