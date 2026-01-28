import argparse, os
import pandas as pd
from joblib import dump
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

def load_split():
    tr = pd.read_csv("dataset/processed/train.csv")
    va = pd.read_csv("dataset/processed/val.csv")
    te = pd.read_csv("dataset/processed/test.csv")
    return tr, va, te

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs("models", exist_ok=True)

    tr, va, te = load_split()

    models = {
        "tfidf_word_lr": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=2)),
            ("clf", LogisticRegression(max_iter=2000))
        ]),
        "tfidf_char_svm": Pipeline([
            ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(3,5), min_df=2)),
            ("clf", LinearSVC())
        ])
    }

    for name, pipe in models.items():
        pipe.fit(tr["text"], tr["label"])
        dump(pipe, f"models/{name}.joblib")
        print("Saved", name)

if __name__ == "__main__":
    main()
