#!/usr/bin/env python
"""
Generate EAT (Evasion-Aware Training) summary tables and visualizations.
"""
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_all_eat_results():
    """Load all EAT results from results folder."""
    results = []
    for f in Path("results").glob("eat_results_*.csv"):
        df = pd.read_csv(f)
        results.append(df)
    if results:
        return pd.concat(results, ignore_index=True)
    return pd.DataFrame()


def load_all_gains():
    """Load all gain tables."""
    gains = []
    for f in Path("results").glob("eat_gain_*.csv"):
        df = pd.read_csv(f)
        gains.append(df)
    if gains:
        return pd.concat(gains, ignore_index=True)
    return pd.DataFrame()


def load_all_tradeoffs():
    """Load all trade-off tables."""
    tradeoffs = []
    for f in Path("results").glob("eat_tradeoff_*.csv"):
        df = pd.read_csv(f)
        tradeoffs.append(df)
    if tradeoffs:
        return pd.concat(tradeoffs, ignore_index=True)
    return pd.DataFrame()


def generate_summary_table():
    """Generate a comprehensive summary table."""
    gains = load_all_gains()
    if gains.empty:
        print("[WARN] No gain data found")
        return

    # Filter out attacks that don't work well (prompt_injection for word models)
    # Focus on obfuscate which is the main attack
    summary = gains[gains["attack"].isin(["clean", "obfuscate", "paraphrase_like"])]

    # Pivot table
    pivot = summary.pivot_table(
        index="model",
        columns="attack",
        values=["f1_clean_train", "f1_eat_train", "gain"],
        aggfunc="first"
    )

    # Save
    pivot.to_csv("results/eat_summary_pivot.csv")
    print("[OK] Saved results/eat_summary_pivot.csv")

    # Create markdown summary
    md_lines = [
        "# EAT (Evasion-Aware Training) Results Summary\n",
        "## Key Findings\n",
    ]

    # Calculate average gains
    obf_gains = gains[gains["attack"] == "obfuscate"]["gain"]
    clean_gains = gains[gains["attack"] == "clean"]["gain"]

    if len(obf_gains) > 0:
        md_lines.append(f"- **Obfuscate attack robustness gain**: {obf_gains.mean():.2%} (avg)\n")
    if len(clean_gains) > 0:
        md_lines.append(f"- **Clean performance change**: {clean_gains.mean():+.2%} (avg)\n")

    md_lines.append("\n## Detailed Results by Model\n")
    md_lines.append("| Model | Attack | F1 (Clean Train) | F1 (EAT Train) | Gain |\n")
    md_lines.append("|-------|--------|-----------------|----------------|------|\n")

    for _, row in summary.iterrows():
        md_lines.append(
            f"| {row['model']} | {row['attack']} | "
            f"{row['f1_clean_train']:.4f} | {row['f1_eat_train']:.4f} | "
            f"{row['gain']:+.4f} |\n"
        )

    with open("results/eat_summary.md", "w") as f:
        f.writelines(md_lines)
    print("[OK] Saved results/eat_summary.md")


def plot_robustness_gain():
    """Plot robustness gain comparison."""
    gains = load_all_gains()
    if gains.empty:
        print("[WARN] No gain data for plotting")
        return

    # Focus on obfuscate attack (main threat model)
    obf = gains[gains["attack"] == "obfuscate"]

    if obf.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    x = range(len(obf))
    width = 0.35

    bars1 = ax.bar([i - width/2 for i in x], obf["f1_clean_train"], width,
                   label="Clean Training", color="#3498db", alpha=0.8)
    bars2 = ax.bar([i + width/2 for i in x], obf["f1_eat_train"], width,
                   label="EAT Training", color="#e74c3c", alpha=0.8)

    ax.set_xlabel("Model")
    ax.set_ylabel("F1 Score (under obfuscate attack)")
    ax.set_title("EAT Improves Robustness Against Obfuscation Attacks")
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", "\n") for m in obf["model"]], fontsize=8)
    ax.legend()
    ax.set_ylim(0, 1)

    # Add gain annotations
    for i, (_, row) in enumerate(obf.iterrows()):
        gain = row["gain"]
        y_pos = max(row["f1_clean_train"], row["f1_eat_train"]) + 0.02
        ax.annotate(f"+{gain:.1%}" if gain > 0 else f"{gain:.1%}",
                    xy=(i, y_pos), ha="center", fontsize=9,
                    color="green" if gain > 0 else "red")

    plt.tight_layout()
    plt.savefig("results/fig_eat_robustness_gain.png", dpi=150)
    plt.savefig("paper/figs/fig_eat_robustness_gain.pdf")
    print("[OK] Saved results/fig_eat_robustness_gain.png")
    print("[OK] Saved paper/figs/fig_eat_robustness_gain.pdf")
    plt.close()


def plot_tradeoff():
    """Plot clean vs robustness trade-off."""
    tradeoffs = load_all_tradeoffs()
    gains = load_all_gains()

    if tradeoffs.empty or gains.empty:
        print("[WARN] No data for trade-off plot")
        return

    # Merge for scatter plot
    obf_gains = gains[gains["attack"] == "obfuscate"][["model", "gain"]].rename(
        columns={"gain": "obf_gain"}
    )
    merged = tradeoffs.merge(obf_gains, on="model", how="left")

    fig, ax = plt.subplots(figsize=(8, 6))

    colors = plt.cm.Set2(range(len(merged)))
    for i, (_, row) in enumerate(merged.iterrows()):
        ax.scatter(row["clean_diff"], row["obf_gain"],
                   s=150, c=[colors[i]], edgecolors="black", linewidth=0.5)
        # Get short model name for label
        short_name = row["model"].split("_")[-2] if "_" in row["model"] else row["model"]
        dataset = "SMS" if "sms" in row["model"] else "SpamA"
        ax.annotate(f"{dataset}-{short_name}",
                    xy=(row["clean_diff"], row["obf_gain"]),
                    xytext=(5, 5), textcoords="offset points", fontsize=8)

    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Clean F1 Change (EAT - Clean Training)")
    ax.set_ylabel("Obfuscate Robustness Gain")
    ax.set_title("EAT Trade-off: Clean Performance vs Robustness")

    # Quadrant annotations
    ax.fill_between([0, ax.get_xlim()[1]], 0, ax.get_ylim()[1], alpha=0.1, color="green")
    ax.text(0.95, 0.95, "Win-Win ✓", transform=ax.transAxes, fontsize=10, 
            color="green", ha="right", va="top", fontweight="bold")

    plt.tight_layout()
    plt.savefig("results/fig_eat_tradeoff.png", dpi=150)
    print("[OK] Saved results/fig_eat_tradeoff.png")
    plt.close()


def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("paper/figs", exist_ok=True)

    print("Generating EAT summary tables...")
    generate_summary_table()

    print("\nGenerating EAT visualizations...")
    plot_robustness_gain()
    plot_tradeoff()

    print("\n[DONE] All EAT summaries generated!")


if __name__ == "__main__":
    main()
