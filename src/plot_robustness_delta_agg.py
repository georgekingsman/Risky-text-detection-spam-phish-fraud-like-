import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

root = Path(__file__).resolve().parents[1]
robust_path = root / "results" / "robustness_agg.csv"
fig_path = root / "report" / "fig_robustness_delta_agg.png"

_df = pd.read_csv(robust_path)
_df = _df[_df["attack"] != "clean"].copy()

attacks = ["obfuscate", "paraphrase_like", "prompt_injection"]

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

datasets = ["sms_uci", "spamassassin"]
for ax, dataset in zip(axes, datasets):
    sub = _df[_df["dataset"] == dataset].copy()
    sub = sub[sub["defense"] == "none"]

    if dataset == "sms_uci":
        sub = sub[~sub["model"].str.startswith("spamassassin_")]
    else:
        sub = sub[sub["model"].str.startswith("spamassassin_")]

    models = sorted(sub["model"].unique().tolist())
    x = range(len(models))
    width = 0.25

    for i, attack in enumerate(attacks):
        a = sub[sub["attack"] == attack]
        mean_map = dict(zip(a["model"], a["delta_f1_mean"]))
        std_map = dict(zip(a["model"], a["delta_f1_std"]))
        means = [mean_map.get(m, 0) for m in models]
        stds = [std_map.get(m, 0) for m in models]
        ax.bar(
            [p + (i - 1) * width for p in x],
            means,
            width=width,
            yerr=stds,
            capsize=3,
            label=attack,
        )

    ax.set_title(dataset)
    ax.set_xticks(list(x))
    ax.set_xticklabels(models, rotation=70, ha="right")
    ax.set_ylabel("ΔF1 (mean ± std)")
    ax.grid(axis="y", linestyle="--", alpha=0.3)

axes[0].legend(title="attack", loc="best")
fig.tight_layout()
fig_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(fig_path, dpi=200)
print("Wrote", fig_path)
