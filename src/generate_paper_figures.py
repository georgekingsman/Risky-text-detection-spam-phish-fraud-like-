#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate All Paper Figures for CCF-B Submission.

Outputs:
- paper/figs/fig_cross_domain_heatmap.png - Cross-domain F1 heatmap
- paper/figs/fig_eat_gain.png - EAT defense gain across domains
- paper/figs/fig_robustness_comparison.png - Robustness under attacks
- paper/figs/fig_cost_vs_robustness.png - Green AI trade-off
"""
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 150


def fig_cross_domain_heatmap():
    """Figure 1: Cross-domain generalization heatmap."""
    csv_path = ROOT / "results/cross_domain_3domain.csv"
    if not csv_path.exists():
        print(f"⚠️  {csv_path} not found, skipping")
        return
    
    df = pd.read_csv(csv_path)
    
    # Create figure with subplots for each model
    models = ["TF-IDF Word LR", "TF-IDF Char SVM", "MiniLM+LR"]
    domains = ["SMS", "SpamAssassin", "Telegram"]
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    for idx, model in enumerate(models):
        if model not in df.columns:
            continue
        
        # Build matrix
        matrix = np.zeros((3, 3))
        for i, train in enumerate(["SMS", "SpamAssassin", "Telegram"]):
            for j, test in enumerate(["SMS", "SpamAssassin", "Telegram"]):
                row = df[(df["Train"] == train) & (df["Test"] == test)]
                if not row.empty:
                    matrix[i, j] = row[model].values[0]
        
        ax = axes[idx]
        sns.heatmap(matrix, ax=ax, annot=True, fmt=".2f", cmap="RdYlGn",
                    vmin=0.0, vmax=1.0, xticklabels=domains, yticklabels=domains,
                    cbar_kws={'shrink': 0.8})
        ax.set_title(model)
        ax.set_xlabel("Test Domain")
        ax.set_ylabel("Train Domain")
    
    plt.suptitle("Cross-Domain Generalization (F1 Score)", fontsize=14, y=1.02)
    plt.tight_layout()
    
    out_path = ROOT / "paper/figs/fig_cross_domain_heatmap.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved {out_path}")


def fig_eat_gain():
    """Figure 2: EAT defense gain across cross-domain transfers."""
    csv_path = ROOT / "results/eat_cross_domain_gain.csv"
    if not csv_path.exists():
        print(f"⚠️  {csv_path} not found, skipping")
        return
    
    df = pd.read_csv(csv_path)
    
    # CSV has columns: scenario, model, attack, f1_clean_train, f1_eat_train, gain
    # Filter to cross-domain only (exclude in-domain)
    df = df[df["scenario"].str.contains("→")].copy()
    
    # Convert gain to percentage
    df["gain_pct"] = df["gain"] * 100
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    attacks = ["clean", "obfuscate"]
    models = ["tfidf_word_lr", "tfidf_char_svm"]
    
    transfers = df["scenario"].unique()
    x = np.arange(len(transfers))
    width = 0.2
    
    colors = {"tfidf_word_lr": "#2ecc71", "tfidf_char_svm": "#3498db"}
    hatches = {"clean": "", "obfuscate": "//"}
    
    for i, model in enumerate(models):
        for j, attack in enumerate(attacks):
            mask = (df["model"] == model) & (df["attack"] == attack)
            gains = []
            for t in transfers:
                val = df[(df["scenario"] == t) & mask]["gain_pct"]
                gains.append(val.values[0] if len(val) > 0 else 0)
            
            offset = (i * len(attacks) + j - 1.5) * width
            bars = ax.bar(x + offset, gains, width, label=f"{model} ({attack})",
                         color=colors[model], alpha=0.8 if attack == "clean" else 0.5,
                         hatch=hatches[attack], edgecolor='black', linewidth=0.5)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel("Cross-Domain Transfer")
    ax.set_ylabel("EAT Gain (%)")
    ax.set_title("EAT Defense Effectiveness Across Domain Transfers")
    ax.set_xticks(x)
    ax.set_xticklabels(transfers, rotation=45, ha='right')
    ax.legend(loc='upper left', ncol=2)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    out_path = ROOT / "paper/figs/fig_eat_gain.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved {out_path}")


def fig_robustness_comparison():
    """Figure 3: Robustness comparison across 3 domains and attacks."""
    results = []
    for dataset, path in [
        ("SMS", ROOT / "results/robustness_dedup_sms.csv"),
        ("SpamAssassin", ROOT / "results/robustness_dedup_spamassassin.csv"),
        ("Telegram", ROOT / "results/robustness_dedup_telegram.csv"),
    ]:
        if path.exists():
            df = pd.read_csv(path)
            df["Dataset"] = dataset
            results.append(df)
    
    if not results:
        print("⚠️  No robustness results found, skipping")
        return
    
    df = pd.concat(results)
    
    # Filter to baseline models (no defense)
    df = df[df["defense"] == "none"].copy()
    
    # Simplify model names
    df["model_short"] = df["model"].str.replace("_dedup", "").str.replace(".joblib", "")
    df["model_short"] = df["model_short"].str.replace("sms_", "").str.replace("spamassassin_", "").str.replace("telegram_", "")
    
    # Create grouped bar chart
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
    
    attacks = ["clean", "obfuscate", "paraphrase_like", "prompt_injection"]
    models = df["model_short"].unique()[:4]  # Limit to 4 models
    
    for idx, dataset in enumerate(["SMS", "SpamAssassin", "Telegram"]):
        ax = axes[idx]
        subset = df[df["Dataset"] == dataset]
        
        x = np.arange(len(attacks))
        width = 0.2
        
        colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]
        
        for i, model in enumerate(models):
            model_data = subset[subset["model_short"] == model]
            f1s = [model_data[model_data["attack"] == atk]["f1_attacked"].values[0] 
                   if len(model_data[model_data["attack"] == atk]) > 0 else 0 
                   for atk in attacks]
            ax.bar(x + i * width, f1s, width, label=model if idx == 0 else "", color=colors[i % len(colors)])
        
        ax.set_xlabel("Attack Type")
        ax.set_ylabel("F1 Score" if idx == 0 else "")
        ax.set_title(dataset)
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(["Clean", "Obfuscate", "Paraphrase", "Prompt Inj."], rotation=30, ha='right')
        ax.set_ylim(0, 1.0)
    
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.08), ncol=4)
    plt.suptitle("Model Robustness Under Adversarial Attacks", fontsize=14, y=1.12)
    plt.tight_layout()
    
    out_path = ROOT / "paper/figs/fig_robustness_comparison.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved {out_path}")


def fig_cost_vs_robustness():
    """Figure 4: Cost-throughput vs robustness trade-off (Green AI)."""
    csv_path = ROOT / "results/cost_throughput.csv"
    if not csv_path.exists():
        print(f"⚠️  {csv_path} not found, skipping")
        return
    
    df = pd.read_csv(csv_path)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Color by model type
    colors = {
        "TF-IDF Word LR": "#2ecc71",
        "TF-IDF Char SVM": "#3498db", 
        "MiniLM+LR": "#e74c3c",
        "TF-IDF Word LR (EAT)": "#27ae60",
    }
    
    for _, row in df.iterrows():
        model = row["model"]
        x = row["latency_ms"]
        y = row.get("robust_f1_obfuscate", row.get("clean_f1", 0))
        
        if pd.isna(y):
            y = row.get("clean_f1", 0)
        
        ax.scatter(x, y, s=200, c=colors.get(model, "#95a5a6"), 
                   label=model, alpha=0.8, edgecolors='black', linewidths=1)
        
        # Add annotation
        ax.annotate(model, (x, y), xytext=(5, 5), textcoords='offset points',
                    fontsize=9, alpha=0.8)
    
    ax.set_xscale('log')
    ax.set_xlabel("Inference Latency (ms/message, log scale)")
    ax.set_ylabel("Robust F1 (under obfuscation attack)")
    ax.set_title("Cost-Robustness Trade-off: Green AI Perspective")
    ax.grid(True, alpha=0.3)
    
    # Add efficiency frontier annotation
    ax.annotate("← Lower latency\n(more efficient)", xy=(0.01, 0.7), fontsize=10, color='green')
    ax.annotate("Higher robustness →", xy=(1, 0.95), fontsize=10, color='blue')
    
    plt.tight_layout()
    out_path = ROOT / "paper/figs/fig_cost_vs_robustness.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved {out_path}")


def fig_jsd_bar():
    """Figure 5: JSD domain shift visualization."""
    csv_path = ROOT / "results/domain_shift_js_3domains.csv"
    if not csv_path.exists():
        print(f"⚠️  {csv_path} not found, skipping")
        return
    
    df = pd.read_csv(csv_path)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    pairs = [f"{row['domain_a'].upper()} ↔ {row['domain_b'].upper()}" 
             for _, row in df.iterrows()]
    jsds = df["jsd_char_3to5"].values
    
    colors = ["#e74c3c", "#3498db", "#2ecc71"]
    bars = ax.barh(pairs, jsds, color=colors, edgecolor='black', linewidth=1)
    
    ax.set_xlabel("Jensen-Shannon Divergence (JSD)")
    ax.set_title("Domain Shift: Character N-gram Distribution Divergence")
    ax.set_xlim(0, 0.3)
    
    # Add value labels
    for bar, jsd in zip(bars, jsds):
        ax.text(jsd + 0.01, bar.get_y() + bar.get_height()/2, 
                f"{jsd:.3f}", va='center', fontsize=11)
    
    plt.tight_layout()
    out_path = ROOT / "paper/figs/fig_jsd_domain_shift.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved {out_path}")


def main():
    print("=" * 60)
    print("Generating Paper Figures")
    print("=" * 60)
    
    os.makedirs(ROOT / "paper/figs", exist_ok=True)
    
    fig_cross_domain_heatmap()
    fig_eat_gain()
    fig_robustness_comparison()
    fig_cost_vs_robustness()
    fig_jsd_bar()
    
    print("\n✅ All figures generated!")


if __name__ == "__main__":
    main()
