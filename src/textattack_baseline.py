import argparse
import csv
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit


def load_split(data_dir):
    df = pd.read_csv(Path(data_dir) / "test.csv")
    return df


def stratified_sample(df, n_samples, seed):
    if n_samples is None or n_samples <= 0 or n_samples >= len(df):
        return df.reset_index(drop=True)
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=n_samples, random_state=seed)
    idx, _ = next(splitter.split(df, df["label"]))
    return df.iloc[idx].reset_index(drop=True)


def normalize_preds(preds):
    if len(preds) == 0:
        return []
    if isinstance(preds[0], str):
        return [1 if p.lower().startswith("spam") else 0 for p in preds]
    return [int(p) for p in preds]


def predict_labels(model, texts):
    preds = model.predict(texts)
    return normalize_preds(preds)


def build_model_bundle(model_path):
    m = joblib.load(model_path)
    if isinstance(m, dict) and "tfidf" in m and "clf" in m:
        from sklearn.pipeline import Pipeline
        pipeline = Pipeline([("tfidf", m["tfidf"]), ("clf", m["clf"])])
        tokenizer = m["tfidf"]
        clf = m["clf"]
    elif hasattr(m, "named_steps") and "tfidf" in m.named_steps and "clf" in m.named_steps:
        pipeline = m
        tokenizer = m.named_steps["tfidf"]
        clf = m.named_steps["clf"]
    elif isinstance(m, dict) and "embed_model" in m and "clf" in m:
        return None
    else:
        return None

    if not hasattr(clf, "predict_proba"):
        return None

    return {
        "pipeline": pipeline,
        "tokenizer": tokenizer,
        "clf": clf,
    }


def run_attack(model, tokenizer, texts, labels, attack_name, seed, examples_out=None, max_examples=50):
    from textattack import Attacker, AttackArgs
    from textattack.attack_recipes import DeepWordBugGao2018
    from textattack.attack_results import SuccessfulAttackResult, FailedAttackResult, SkippedAttackResult
    from textattack.datasets import Dataset
    from textattack.models.wrappers import ModelWrapper

    class TfidfSklearnWrapper(ModelWrapper):
        def __init__(self, clf, tfidf):
            self.clf = clf
            self.tfidf = tfidf
            self.model = clf

        def __call__(self, text_input_list, batch_size=None):
            X = self.tfidf.transform(text_input_list)
            return self.clf.predict_proba(X)

    model_wrapper = TfidfSklearnWrapper(model, tokenizer)
    if attack_name == "deepwordbug":
        attack = DeepWordBugGao2018.build(model_wrapper)
    else:
        raise ValueError(f"Unsupported attack: {attack_name}")

    dataset = Dataset(list(zip(texts, labels)))
    attack_args = AttackArgs(
        num_examples=len(texts),
        random_seed=seed,
        shuffle=False,
        disable_stdout=True,
    )
    attacker = Attacker(attack, dataset, attack_args)

    perturbed_texts = []
    statuses = []
    success_count = 0
    example_rows = []

    for i, result in enumerate(attacker.attack_dataset()):
        orig_text = result.original_result.attacked_text.text
        pert_text = orig_text
        status = "unknown"

        if isinstance(result, SuccessfulAttackResult):
            pert_text = result.perturbed_result.attacked_text.text
            status = "successful"
            success_count += 1
        elif isinstance(result, FailedAttackResult):
            status = "failed"
        elif isinstance(result, SkippedAttackResult):
            status = "skipped"
        else:
            status = result.__class__.__name__.replace("AttackResult", "").lower()

        perturbed_texts.append(pert_text)
        statuses.append(status)

        if examples_out and len(example_rows) < max_examples:
            example_rows.append({
                "i": i,
                "label": int(labels[i]),
                "status": status,
                "original": orig_text,
                "perturbed": pert_text,
            })

    if examples_out:
        with open(examples_out, "w") as f:
            for row in example_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    success_rate = success_count / max(1, len(texts))
    return perturbed_texts, statuses, success_rate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sms_uci", choices=["sms_uci", "spamassassin"])
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--model-glob", default=None, help="Optional glob override for model selection")
    ap.add_argument("--n-samples", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--attack", default="deepwordbug", choices=["deepwordbug"])
    ap.add_argument("--out", default="results/textattack.csv")
    ap.add_argument("--examples-out", default="results/textattack_examples.jsonl")
    ap.add_argument("--max-examples", type=int, default=50)
    args = ap.parse_args()

    if args.data_dir:
        data_dir = args.data_dir
    else:
        data_dir = "dataset/spamassassin/processed" if args.dataset == "spamassassin" else "dataset/processed"

    df = load_split(data_dir)
    sample_df = stratified_sample(df, args.n_samples, args.seed)

    texts = sample_df["text"].tolist()
    labels = sample_df["label"].astype(int).tolist()

    if args.model_glob:
        model_paths = sorted(Path(".").glob(args.model_glob))
    else:
        if args.dataset == "sms_uci":
            model_paths = [Path("models/tfidf_word_lr.joblib")]
            if Path("models/tfidf_llm_lr.joblib").exists():
                model_paths.append(Path("models/tfidf_llm_lr.joblib"))
        else:
            model_paths = [Path("models/spamassassin_tfidf_word_lr.joblib")]

    model_paths = [p for p in model_paths if p.exists()]
    if not model_paths:
        raise SystemExit("No eligible models found. Train baselines or pass --model-glob.")

    os.makedirs(Path(args.out).parent, exist_ok=True)
    rows = []

    for model_path in model_paths:
        bundle = build_model_bundle(model_path)
        if bundle is None:
            continue
        model = bundle["pipeline"]
        tokenizer = bundle["tokenizer"]
        clf = bundle["clf"]

        # clean predictions
        clean_preds = predict_labels(model, texts)
        clean_f1 = f1_score(labels, clean_preds)

        # attack
        examples_out = args.examples_out
        if examples_out:
            stem = Path(examples_out).stem
            examples_out = str(Path(examples_out).with_name(f"{stem}_{model_path.stem}.jsonl"))

        pert_texts, statuses, success_rate = run_attack(
            clf,
            tokenizer,
            texts,
            labels,
            attack_name=args.attack,
            seed=args.seed,
            examples_out=examples_out,
            max_examples=args.max_examples,
        )

        adv_preds = predict_labels(model, pert_texts)
        adv_f1 = f1_score(labels, adv_preds)

        rows.append({
            "dataset": args.dataset,
            "model": model_path.name,
            "attack": args.attack,
            "n_samples": len(texts),
            "success_rate": round(success_rate, 4),
            "f1_clean": round(clean_f1, 4),
            "f1_attacked": round(adv_f1, 4),
            "delta_f1": round(adv_f1 - clean_f1, 4),
            "notes": "textattack",
        })

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "model",
                "attack",
                "n_samples",
                "success_rate",
                "f1_clean",
                "f1_attacked",
                "delta_f1",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
