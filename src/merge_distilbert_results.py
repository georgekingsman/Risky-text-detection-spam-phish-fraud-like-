import argparse
from pathlib import Path

import pandas as pd


def load_nn(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    if "test_domain" not in df.columns and "eval_domain" in df.columns:
        df = df.rename(columns={"eval_domain": "test_domain"})
    if "train_domain" not in df.columns and "source_domain" in df.columns:
        df = df.rename(columns={"source_domain": "train_domain"})
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sms", default="results/nn_distilbert_sms_train.csv")
    ap.add_argument("--spam", default="results/nn_distilbert_spam_train.csv")
    ap.add_argument("--results_dedup", default="results/results_dedup.csv")
    ap.add_argument("--cross_table", default="results/cross_domain_table_dedup.csv")
    args = ap.parse_args()

    sms_df = load_nn(Path(args.sms))
    spam_df = load_nn(Path(args.spam))
    nn = pd.concat([sms_df, spam_df], ignore_index=True)

    # build long-format rows for results_dedup.csv
    rows = []
    for _, r in nn.iterrows():
        train = r["train_domain"]
        test = r["test_domain"]
        rows.append({
            "dataset": f"{test}_dedup",
            "split": "test",
            "model": "distilbert_ft",
            "seed": "",
            "f1": r["f1"],
            "precision": r["precision"],
            "recall": r["recall"],
            "roc_auc": r["roc_auc"],
            "notes": f"train={train}_dedup cross-domain" if train != test else f"train={train}_dedup in-domain",
        })

    results_path = Path(args.results_dedup)
    res_df = pd.read_csv(results_path)
    res_df = res_df[res_df["model"] != "distilbert_ft"]
    res_df = pd.concat([res_df, pd.DataFrame(rows)], ignore_index=True)
    res_df.to_csv(results_path, index=False)

    # update cross-domain wide table
    def get_f1(train, test):
        match = nn[(nn["train_domain"] == train) & (nn["test_domain"] == test)]
        if match.empty:
            return ""
        return float(match.iloc[0]["f1"])

    sms_in = get_f1("sms", "sms")
    spam_in = get_f1("spamassassin", "spamassassin")
    sms2spam = get_f1("sms", "spamassassin")
    spam2sms = get_f1("spamassassin", "sms")

    cross_path = Path(args.cross_table)
    cross_df = pd.read_csv(cross_path)
    cross_df = cross_df[~cross_df["model"].isin(["distilbert_ft", "tfidf_word_lr_augtrain", "minilm_lr_coral"]) ]

    # AugTrain and CORAL rows from results_dedup
    res_df = pd.read_csv(results_path)

    def add_row_from_results(model_name: str):
        def pick(train, test):
            sub = res_df[(res_df["model"] == model_name) & (res_df["notes"].str.contains(f"train={train}")) & (res_df["dataset"].str.startswith(test))]
            if sub.empty:
                return ""
            return float(sub.iloc[0]["f1"])

        return {
            "model": model_name,
            "sms_in_domain_f1": pick("sms_uci_dedup", "sms_uci"),
            "spam_in_domain_f1": pick("spamassassin_dedup", "spamassassin"),
            "sms_to_spam_f1": pick("sms_uci_dedup", "spamassassin"),
            "spam_to_sms_f1": pick("spamassassin_dedup", "sms_uci"),
        }

    rows_extra = [
        {
            "model": "distilbert_ft",
            "sms_in_domain_f1": sms_in,
            "spam_in_domain_f1": spam_in,
            "sms_to_spam_f1": sms2spam,
            "spam_to_sms_f1": spam2sms,
        },
        add_row_from_results("tfidf_word_lr_augtrain"),
        add_row_from_results("minilm_lr_coral"),
    ]

    cross_df = pd.concat([cross_df, pd.DataFrame(rows_extra)], ignore_index=True)
    cross_df.to_csv(cross_path, index=False)

    print("Wrote", results_path)
    print("Wrote", cross_path)


if __name__ == "__main__":
    main()
