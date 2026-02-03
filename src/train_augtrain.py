import argparse
import os
import random
from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .robustness.perturb import obfuscate, simple_paraphrase_like
from .robustness.defense import normalize_text


def load_split(data_dir):
    tr = pd.read_csv(Path(data_dir) / "train.csv")
    va = pd.read_csv(Path(data_dir) / "val.csv")
    te = pd.read_csv(Path(data_dir) / "test.csv")
    return tr, va, te


def augment_spam(text: str, seed: int):
    rng = random.Random(seed)
    return [
        obfuscate(text, seed=seed),
        simple_paraphrase_like(text),
        normalize_text(text),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs("models", exist_ok=True)

    tr, va, te = load_split(args.data_dir)
    texts = tr["text"].astype(str).tolist()
    labels = tr["label"].astype(int).tolist()

    aug_texts = []
    aug_labels = []
    for i, (t, y) in enumerate(zip(texts, labels)):
        if y == 1:
            for a in augment_spam(t, seed=args.seed + i):
                aug_texts.append(a)
                aug_labels.append(1)

    train_texts = texts + aug_texts
    train_labels = labels + aug_labels

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
        ("clf", LogisticRegression(max_iter=2000)),
    ])
    pipe.fit(train_texts, train_labels)

    out_path = f"models/{args.prefix}_aug_tfidf_word_lr.joblib"
    dump(pipe, out_path)
    print("Saved", out_path)


if __name__ == "__main__":
    main()
