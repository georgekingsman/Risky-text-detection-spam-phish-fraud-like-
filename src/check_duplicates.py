import csv
import hashlib
from pathlib import Path

import pandas as pd

root = Path(__file__).resolve().parents[1]


def norm_text(t: str) -> str:
    return " ".join(str(t).strip().lower().split())


def hash_text(t: str) -> str:
    return hashlib.sha256(norm_text(t).encode("utf-8")).hexdigest()


def dup_rate(series):
    total = len(series)
    if total == 0:
        return 0.0
    unique = len(set(series))
    return 1.0 - (unique / total)


def main():
    rows = []
    for dataset, data_dir in [
        ("sms_uci", root / "dataset" / "processed"),
        ("spamassassin", root / "dataset" / "spamassassin" / "processed"),
    ]:
        splits = {}
        for split in ["train", "val", "test"]:
            df = pd.read_csv(Path(data_dir) / f"{split}.csv")
            hashes = df["text"].apply(hash_text)
            splits[split] = hashes

            rows.append({
                "dataset": dataset,
                "split": split,
                "dup_rate_in_split": dup_rate(hashes),
            })

        train_hash = set(splits["train"])
        val_hash = set(splits["val"])
        test_hash = set(splits["test"])

        rows.append({
            "dataset": dataset,
            "split": "train∩val",
            "dup_rate_in_split": len(train_hash & val_hash) / max(1, len(val_hash)),
        })
        rows.append({
            "dataset": dataset,
            "split": "train∩test",
            "dup_rate_in_split": len(train_hash & test_hash) / max(1, len(test_hash)),
        })
        rows.append({
            "dataset": dataset,
            "split": "val∩test",
            "dup_rate_in_split": len(val_hash & test_hash) / max(1, len(test_hash)),
        })

    out_path = root / "results" / "duplicate_check.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "split", "dup_rate_in_split"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
