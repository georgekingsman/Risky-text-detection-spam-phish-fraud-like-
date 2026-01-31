import csv
from pathlib import Path

import pandas as pd

root = Path(__file__).resolve().parents[1]
robust_path = root / "results" / "robustness_agg.csv"
out_path = root / "results" / "defense_tradeoff.csv"

_df = pd.read_csv(robust_path)
_df = _df[_df["attack"] == "clean"].copy()

rows = []
for dataset, model in [
    ("sms_uci", "tfidf_word_lr.joblib"),
    ("spamassassin", "spamassassin_tfidf_word_lr.joblib"),
]:
    sub = _df[(_df["dataset"] == dataset) & (_df["model"] == model)]
    for defense in ["none", "normalize"]:
        row = sub[sub["defense"] == defense]
        if row.empty:
            continue
        r = row.iloc[0]
        rows.append({
            "dataset": dataset,
            "model": model,
            "defense": defense,
            "f1_clean_mean": r["f1_clean_mean"],
            "f1_clean_std": r["f1_clean_std"],
        })

out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["dataset", "model", "defense", "f1_clean_mean", "f1_clean_std"],
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {out_path}")
