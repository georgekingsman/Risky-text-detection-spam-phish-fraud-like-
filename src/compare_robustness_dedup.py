import csv
from pathlib import Path

import pandas as pd

root = Path(__file__).resolve().parents[1]
orig_path = root / "results" / "robustness.csv"
dedup_path = root / "results" / "robustness_dedup.csv"
out_path = root / "results" / "dedup_effect.csv"

orig = pd.read_csv(orig_path)
dedup = pd.read_csv(dedup_path)

def normalize_model(name: str, dataset_base: str) -> str | None:
    base = name
    if dataset_base == "sms_uci":
        if base.startswith("spamassassin_") or base.startswith("spamassassin_dedup_"):
            return None
        if base.startswith("sms_dedup_"):
            base = base[len("sms_dedup_") :]
        return base
    if dataset_base == "spamassassin":
        if base.startswith("spamassassin_dedup_"):
            base = "spamassassin_" + base[len("spamassassin_dedup_") :]
        if not base.startswith("spamassassin_"):
            return None
        return base
    return base

def normalize_dataset(name: str) -> str:
    return name.replace("_dedup", "")

orig["dataset_base"] = orig["dataset"].map(normalize_dataset)
dedup["dataset_base"] = dedup["dataset"].map(normalize_dataset)
orig["model_base"] = orig.apply(lambda r: normalize_model(r["model"], r["dataset_base"]), axis=1)
dedup["model_base"] = dedup.apply(lambda r: normalize_model(r["model"], r["dataset_base"]), axis=1)

orig = orig.dropna(subset=["model_base"]).copy()
dedup = dedup.dropna(subset=["model_base"]).copy()

orig = orig.drop_duplicates(subset=["attack", "dataset_base", "model_base", "defense"], keep="first")
dedup = dedup.drop_duplicates(subset=["attack", "dataset_base", "model_base", "defense"], keep="first")

orig_keyed = orig.set_index(["attack", "dataset_base", "model_base", "defense"])
dedup_keyed = dedup.set_index(["attack", "dataset_base", "model_base", "defense"])

rows = []
for key, row in dedup_keyed.iterrows():
    if key not in orig_keyed.index:
        continue
    o = orig_keyed.loc[key]
    rows.append({
        "attack": key[0],
        "dataset": key[1],
        "model": key[2],
        "defense": key[3],
        "delta_f1_orig": o["delta_f1"],
        "delta_f1_dedup": row["delta_f1"],
        "delta_f1_change": row["delta_f1"] - o["delta_f1"],
    })

out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "attack",
            "dataset",
            "model",
            "defense",
            "delta_f1_orig",
            "delta_f1_dedup",
            "delta_f1_change",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {out_path}")
