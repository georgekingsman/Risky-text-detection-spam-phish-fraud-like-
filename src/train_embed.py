import argparse, os
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer

def load_split():
    tr = pd.read_csv("dataset/processed/train.csv")
    va = pd.read_csv("dataset/processed/val.csv")
    te = pd.read_csv("dataset/processed/test.csv")
    return tr, va, te

def encode(model, texts, batch_size=64):
    embs = model.encode(texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)
    return np.asarray(embs, dtype=np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--embed_model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = ap.parse_args()

    os.makedirs("models", exist_ok=True)
    tr, va, te = load_split()

    st = SentenceTransformer(args.embed_model)
    Xtr = encode(st, tr["text"].tolist())
    ytr = tr["label"].values

    clf = LogisticRegression(max_iter=2000)
    clf.fit(Xtr, ytr)

    dump({"embed_model": args.embed_model, "clf": clf}, "models/minilm_lr.joblib")
    print("Saved minilm_lr")

if __name__ == "__main__":
    main()
