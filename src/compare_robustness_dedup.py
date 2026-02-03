import csv
from pathlib import Path

import pandas as pd

root = Path(__file__).resolve().parents[1]
orig_path = root / "results" / "robustness.csv"
dedup_path = root / "results" / "robustness_dedup.csv"
out_path = root / "results" / "dedup_effect.csv"

orig = pd.read_csv(orig_path)
dedup = pd.read_csv(dedup_path)

orig_keyed = orig.set_index(["attack", "dataset", "model", "defense"])
dedup_keyed = dedup.set_index(["attack", "dataset", "model", "defense"])

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
