import csv
from pathlib import Path

import pandas as pd

root = Path(__file__).resolve().parents[1]
results_path = root / "results" / "results.csv"
out_path = root / "results" / "cross_domain_table.csv"

_df = pd.read_csv(results_path)

models = ["tfidf_word_lr", "tfidf_char_svm", "minilm_lr"]

rows = []
for model in models:
    sms_in = _df[
        (_df["dataset"] == "sms_uci")
        & (_df["model"] == model)
        & (_df["notes"].str.contains("in-domain"))
        & (_df["notes"].str.contains("train=sms_uci"))
    ]
    spam_in = _df[
        (_df["dataset"] == "spamassassin")
        & (_df["model"] == model)
        & (_df["notes"].str.contains("in-domain"))
        & (_df["notes"].str.contains("train=spamassassin"))
    ]
    sms_to_spam = _df[
        (_df["dataset"] == "spamassassin")
        & (_df["model"] == model)
        & (_df["notes"].str.contains("cross-domain"))
        & (_df["notes"].str.contains("train=sms_uci"))
    ]
    spam_to_sms = _df[
        (_df["dataset"] == "sms_uci")
        & (_df["model"] == model)
        & (_df["notes"].str.contains("cross-domain"))
        & (_df["notes"].str.contains("train=spamassassin"))
    ]

    def pick(df):
        return df.iloc[0]["f1"] if not df.empty else ""

    rows.append({
        "model": model,
        "sms_in_domain_f1": pick(sms_in),
        "spam_in_domain_f1": pick(spam_in),
        "sms_to_spam_f1": pick(sms_to_spam),
        "spam_to_sms_f1": pick(spam_to_sms),
    })

out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "model",
            "sms_in_domain_f1",
            "spam_in_domain_f1",
            "sms_to_spam_f1",
            "spam_to_sms_f1",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {out_path}")
