#!/usr/bin/env python
"""
Evasion-Aware Training (EAT) Pipeline.

This script trains models on augmented (AttackMix) data and evaluates them
on both clean and adversarial test sets, producing comparison tables.

Usage:
    python -m src.train_eat --data-dir data/sms_spam/dedup/processed --prefix sms_dedup
    python -m src.train_eat --data-dir data/spamassassin/dedup/processed --prefix spamassassin_dedup
"""
import argparse
import csv
import os
from pathlib import Path

import pandas as pd
from joblib import dump, load
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .robustness.perturb import obfuscate, prompt_injection, simple_paraphrase_like


def load_splits(data_dir, use_augmix=False):
    """Load train/val/test splits. Optionally use augmented train set."""
    train_file = "train_augmix.csv" if use_augmix else "train.csv"
    tr = pd.read_csv(Path(data_dir) / train_file)
    va = pd.read_csv(Path(data_dir) / "val.csv")
    te = pd.read_csv(Path(data_dir) / "test.csv")
    return tr, va, te


def eval_metrics(y_true, y_pred):
    """Compute F1, precision, recall."""
    return {
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
    }


def apply_attack(texts, attack_name, seed=0):
    """Apply attack to a list of texts."""
    results = []
    for i, t in enumerate(texts):
        if attack_name == "obfuscate":
            results.append(obfuscate(t, seed=seed + i))
        elif attack_name == "prompt_injection":
            results.append(prompt_injection(t))
        elif attack_name == "paraphrase_like":
            results.append(simple_paraphrase_like(t))
        else:
            results.append(t)
    return results


def train_tfidf_word_lr(tr, prefix, suffix):
    """Train TF-IDF + Logistic Regression (word-level)."""
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
        ("clf", LogisticRegression(max_iter=2000)),
    ])
    pipe.fit(tr["text"], tr["label"])
    model_path = f"models/{prefix}_tfidf_word_lr{suffix}.joblib"
    dump(pipe, model_path)
    print(f"  Saved: {model_path}")
    return pipe


def train_tfidf_char_svm(tr, prefix, suffix):
    """Train TF-IDF + SVM (char-level)."""
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2)),
        ("clf", LinearSVC()),
    ])
    pipe.fit(tr["text"], tr["label"])
    model_path = f"models/{prefix}_tfidf_char_svm{suffix}.joblib"
    dump(pipe, model_path)
    print(f"  Saved: {model_path}")
    return pipe


def train_minilm_lr(tr, prefix, suffix):
    """Train MiniLM embeddings + Logistic Regression."""
    from sentence_transformers import SentenceTransformer

    embed_model = "sentence-transformers/all-MiniLM-L6-v2"
    st = SentenceTransformer(embed_model)
    X = st.encode(tr["text"].tolist(), batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X, tr["label"].values)
    model_path = f"models/{prefix}_minilm_lr{suffix}.joblib"
    dump({"embed_model": embed_model, "clf": clf, "encoder": st}, model_path)
    print(f"  Saved: {model_path}")
    return {"clf": clf, "encoder": st}


def evaluate_model(model, test_df, model_type, attack=None, seed=0):
    """Evaluate a model on test data, optionally with attack."""
    texts = test_df["text"].tolist()
    if attack:
        texts = apply_attack(texts, attack, seed=seed)

    if model_type == "minilm":
        X = model["encoder"].encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
        y_pred = model["clf"].predict(X)
    else:
        y_pred = model.predict(texts)

    return eval_metrics(test_df["label"].values, y_pred)


def main():
    ap = argparse.ArgumentParser(description="Train and evaluate EAT models")
    ap.add_argument("--data-dir", required=True, help="Path to processed dataset")
    ap.add_argument("--prefix", required=True, help="Model prefix (e.g., sms_dedup)")
    ap.add_argument("--skip-minilm", action="store_true", help="Skip MiniLM training (slow)")
    args = ap.parse_args()

    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    attacks = ["obfuscate", "prompt_injection", "paraphrase_like"]
    model_types = [
        ("tfidf_word_lr", train_tfidf_word_lr, "sklearn"),
        ("tfidf_char_svm", train_tfidf_char_svm, "sklearn"),
    ]
    if not args.skip_minilm:
        model_types.append(("minilm_lr", train_minilm_lr, "minilm"))

    results = []

    for training_mode, suffix in [("clean", ""), ("eat", "_eat")]:
        print(f"\n{'='*60}")
        print(f"Training Mode: {training_mode.upper()}")
        print(f"{'='*60}")

        use_augmix = (training_mode == "eat")
        tr, va, te = load_splits(args.data_dir, use_augmix=use_augmix)
        print(f"Train size: {len(tr)}, Val size: {len(va)}, Test size: {len(te)}")

        for model_name, train_fn, model_type in model_types:
            print(f"\n[{model_name}]")
            model = train_fn(tr, args.prefix, suffix)

            # Clean evaluation
            clean_metrics = evaluate_model(model, te, model_type)
            results.append({
                "model": f"{args.prefix}_{model_name}",
                "training": training_mode,
                "attack": "clean",
                "f1": clean_metrics["f1"],
                "precision": clean_metrics["precision"],
                "recall": clean_metrics["recall"],
            })
            print(f"  Clean F1: {clean_metrics['f1']:.4f}")

            # Robustness evaluation
            for attack in attacks:
                attack_metrics = evaluate_model(model, te, model_type, attack=attack)
                results.append({
                    "model": f"{args.prefix}_{model_name}",
                    "training": training_mode,
                    "attack": attack,
                    "f1": attack_metrics["f1"],
                    "precision": attack_metrics["precision"],
                    "recall": attack_metrics["recall"],
                })
                delta_f1 = attack_metrics["f1"] - clean_metrics["f1"]
                print(f"  {attack} F1: {attack_metrics['f1']:.4f} (Δ={delta_f1:+.4f})")

    # Save results
    results_df = pd.DataFrame(results)
    out_path = f"results/eat_results_{args.prefix}.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\n[OK] Saved results to {out_path}")

    # Generate comparison tables
    generate_comparison_tables(results_df, args.prefix)


def generate_comparison_tables(df, prefix):
    """Generate EAT gain tables."""
    # Pivot for easy comparison
    clean_df = df[df["training"] == "clean"].copy()
    eat_df = df[df["training"] == "eat"].copy()

    # Robustness gain table
    gain_rows = []
    for model in clean_df["model"].unique():
        for attack in df["attack"].unique():
            clean_row = clean_df[(clean_df["model"] == model) & (clean_df["attack"] == attack)]
            eat_row = eat_df[(eat_df["model"] == model) & (eat_df["attack"] == attack)]
            if len(clean_row) > 0 and len(eat_row) > 0:
                f1_clean = clean_row.iloc[0]["f1"]
                f1_eat = eat_row.iloc[0]["f1"]
                gain_rows.append({
                    "model": model,
                    "attack": attack,
                    "f1_clean_train": f1_clean,
                    "f1_eat_train": f1_eat,
                    "gain": f1_eat - f1_clean,
                })

    gain_df = pd.DataFrame(gain_rows)
    gain_path = f"results/eat_gain_{prefix}.csv"
    gain_df.to_csv(gain_path, index=False)
    print(f"[OK] Saved gain table to {gain_path}")

    # Trade-off table (clean performance vs robustness gain)
    tradeoff_rows = []
    for model in clean_df["model"].unique():
        # Clean F1 on clean test
        clean_clean = clean_df[(clean_df["model"] == model) & (clean_df["attack"] == "clean")]
        eat_clean = eat_df[(eat_df["model"] == model) & (eat_df["attack"] == "clean")]

        if len(clean_clean) > 0 and len(eat_clean) > 0:
            f1_clean_train = clean_clean.iloc[0]["f1"]
            f1_eat_train = eat_clean.iloc[0]["f1"]

            # Obfuscate robustness gain (main attack)
            obf_gain = gain_df[(gain_df["model"] == model) & (gain_df["attack"] == "obfuscate")]
            obf_robustness_gain = obf_gain["gain"].iloc[0] if len(obf_gain) > 0 else 0

            # Paraphrase_like robustness gain
            para_gain = gain_df[(gain_df["model"] == model) & (gain_df["attack"] == "paraphrase_like")]
            para_robustness_gain = para_gain["gain"].iloc[0] if len(para_gain) > 0 else 0

            tradeoff_rows.append({
                "model": model,
                "clean_f1_clean_train": f1_clean_train,
                "clean_f1_eat_train": f1_eat_train,
                "clean_diff": f1_eat_train - f1_clean_train,
                "obf_robustness_gain": obf_robustness_gain,
                "para_robustness_gain": para_robustness_gain,
            })

    tradeoff_df = pd.DataFrame(tradeoff_rows)
    tradeoff_path = f"results/eat_tradeoff_{prefix}.csv"
    tradeoff_df.to_csv(tradeoff_path, index=False)
    print(f"[OK] Saved trade-off table to {tradeoff_path}")


if __name__ == "__main__":
    main()
