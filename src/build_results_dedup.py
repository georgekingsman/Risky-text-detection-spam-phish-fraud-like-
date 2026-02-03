import csv
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]


def load_dataset(data_dir):
    df = pd.read_csv(Path(data_dir) / "test.csv")
    texts = df["text"].tolist()
    labels = [1 if x == 1 else 0 for x in df["label"].tolist()]
    return texts, labels


def eval_model(model_path, texts, labels):
    m = joblib.load(model_path)
    proba = None
    if isinstance(m, dict):
        if "tfidf" in m and "clf" in m:
            vec = m["tfidf"]
            clf = m["clf"]
            X = vec.transform(texts)
            preds = clf.predict(X)
            if hasattr(clf, "predict_proba"):
                proba = clf.predict_proba(X)
            elif hasattr(clf, "decision_function"):
                proba = clf.decision_function(X)
        elif "embed_model" in m and "clf" in m:
            emb_name = m["embed_model"]
            clf = m["clf"]
            st = SentenceTransformer(emb_name)
            X = st.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
            preds = clf.predict(X)
            if hasattr(clf, "predict_proba"):
                proba = clf.predict_proba(X)
            elif hasattr(clf, "decision_function"):
                proba = clf.decision_function(X)
        else:
            raise ValueError(f"Unsupported model dict format: {model_path}")
    else:
        preds = m.predict(texts)
        if hasattr(m, "predict_proba"):
            proba = m.predict_proba(texts)
        elif hasattr(m, "decision_function"):
            proba = m.decision_function(texts)

    if isinstance(preds[0], str):
        ypred = [1 if p.lower().startswith("spam") else 0 for p in preds]
    else:
        ypred = [int(p) for p in preds]

    f1 = f1_score(labels, ypred)
    prec = precision_score(labels, ypred, zero_division=0)
    rec = recall_score(labels, ypred, zero_division=0)
    roc = ""
    try:
        if proba is not None:
            arr = np.array(proba)
            if arr.ndim == 1:
                roc = roc_auc_score(labels, arr)
            else:
                roc = roc_auc_score(labels, arr[:, 1])
    except Exception:
        roc = ""

    return f1, prec, rec, roc


def coral_transform(Xs: np.ndarray, Xt: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    Xs = Xs - Xs.mean(axis=0, keepdims=True)
    Xt_mean = Xt.mean(axis=0, keepdims=True)
    Xt_c = Xt - Xt_mean

    Cs = np.cov(Xs, rowvar=False) + eps * np.eye(Xs.shape[1])
    Ct = np.cov(Xt_c, rowvar=False) + eps * np.eye(Xt.shape[1])

    es, Us = np.linalg.eigh(Cs)
    et, Ut = np.linalg.eigh(Ct)
    Cs_inv_sqrt = Us @ np.diag(1.0 / np.sqrt(es)) @ Us.T
    Ct_sqrt = Ut @ np.diag(np.sqrt(et)) @ Ut.T

    Xs_aligned = Xs @ Cs_inv_sqrt @ Ct_sqrt
    Xs_aligned = Xs_aligned + Xt_mean
    return Xs_aligned


def eval_coral(source_dir: Path, target_dir: Path, st: SentenceTransformer):
    src_train = pd.read_csv(source_dir / "train.csv")
    src_test = pd.read_csv(source_dir / "test.csv")
    tgt_train = pd.read_csv(target_dir / "train.csv")
    tgt_test = pd.read_csv(target_dir / "test.csv")

    Xs_train = st.encode(src_train["text"].astype(str).tolist(), batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    Xt_train = st.encode(tgt_train["text"].astype(str).tolist(), batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    Xt_test = st.encode(tgt_test["text"].astype(str).tolist(), batch_size=64, show_progress_bar=False, normalize_embeddings=True)

    y_train = src_train["label"].astype(int).to_numpy()
    y_test = tgt_test["label"].astype(int).to_numpy()

    Xs_aligned = coral_transform(np.asarray(Xs_train, dtype=np.float32), np.asarray(Xt_train, dtype=np.float32))
    clf = LogisticRegression(max_iter=2000)
    clf.fit(Xs_aligned, y_train)

    preds = clf.predict(np.asarray(Xt_test, dtype=np.float32))
    f1 = f1_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    roc = ""
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(np.asarray(Xt_test, dtype=np.float32))[:, 1]
        try:
            roc = roc_auc_score(y_test, proba)
        except Exception:
            roc = ""
    return f1, prec, rec, roc


def main():
    out_path = ROOT / "results" / "results_dedup.csv"
    rows = []

    sms_dir = ROOT / "dataset" / "dedup" / "processed"
    spam_dir = ROOT / "dataset" / "spamassassin" / "dedup" / "processed"

    sms_models = {
        "tfidf_word_lr": ROOT / "models" / "sms_dedup_tfidf_word_lr.joblib",
        "tfidf_char_svm": ROOT / "models" / "sms_dedup_tfidf_char_svm.joblib",
        "minilm_lr": ROOT / "models" / "sms_dedup_minilm_lr.joblib",
        "tfidf_word_lr_augtrain": ROOT / "models" / "sms_dedup_aug_tfidf_word_lr.joblib",
    }
    sms_texts, sms_labels = load_dataset(sms_dir)
    for name, path in sms_models.items():
        if path.exists():
            f1, prec, rec, roc = eval_model(path, sms_texts, sms_labels)
            rows.append({
                "dataset": "sms_uci_dedup",
                "split": "test",
                "model": name,
                "seed": "",
                "f1": f1,
                "precision": prec,
                "recall": rec,
                "roc_auc": roc,
                "notes": "train=sms_uci_dedup in-domain",
            })

    spam_models = {
        "tfidf_word_lr": ROOT / "models" / "spamassassin_dedup_tfidf_word_lr.joblib",
        "tfidf_char_svm": ROOT / "models" / "spamassassin_dedup_tfidf_char_svm.joblib",
        "minilm_lr": ROOT / "models" / "spamassassin_dedup_minilm_lr.joblib",
        "tfidf_word_lr_augtrain": ROOT / "models" / "spamassassin_dedup_aug_tfidf_word_lr.joblib",
    }
    spam_texts, spam_labels = load_dataset(spam_dir)
    for name, path in spam_models.items():
        if path.exists():
            f1, prec, rec, roc = eval_model(path, spam_texts, spam_labels)
            rows.append({
                "dataset": "spamassassin_dedup",
                "split": "test",
                "model": name,
                "seed": "",
                "f1": f1,
                "precision": prec,
                "recall": rec,
                "roc_auc": roc,
                "notes": "train=spamassassin_dedup in-domain",
            })

    for name, path in sms_models.items():
        if path.exists():
            f1, prec, rec, roc = eval_model(path, spam_texts, spam_labels)
            rows.append({
                "dataset": "spamassassin_dedup",
                "split": "test",
                "model": name,
                "seed": "",
                "f1": f1,
                "precision": prec,
                "recall": rec,
                "roc_auc": roc,
                "notes": "train=sms_uci_dedup cross-domain",
            })

    for name, path in spam_models.items():
        if path.exists():
            f1, prec, rec, roc = eval_model(path, sms_texts, sms_labels)
            rows.append({
                "dataset": "sms_uci_dedup",
                "split": "test",
                "model": name,
                "seed": "",
                "f1": f1,
                "precision": prec,
                "recall": rec,
                "roc_auc": roc,
                "notes": "train=spamassassin_dedup cross-domain",
            })

    # CORAL MiniLM domain alignment (in-domain + cross-domain)
    embed_name = "sentence-transformers/all-MiniLM-L6-v2"
    st = SentenceTransformer(embed_name)

    # SMS -> SMS (in-domain, coral with same domain)
    f1, prec, rec, roc = eval_coral(sms_dir, sms_dir, st)
    rows.append({
        "dataset": "sms_uci_dedup",
        "split": "test",
        "model": "minilm_lr_coral",
        "seed": "",
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "roc_auc": roc,
        "notes": "train=sms_uci_dedup in-domain coral",
    })

    # SpamAssassin -> SpamAssassin (in-domain)
    f1, prec, rec, roc = eval_coral(spam_dir, spam_dir, st)
    rows.append({
        "dataset": "spamassassin_dedup",
        "split": "test",
        "model": "minilm_lr_coral",
        "seed": "",
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "roc_auc": roc,
        "notes": "train=spamassassin_dedup in-domain coral",
    })

    # SMS -> SpamAssassin (cross-domain)
    f1, prec, rec, roc = eval_coral(sms_dir, spam_dir, st)
    rows.append({
        "dataset": "spamassassin_dedup",
        "split": "test",
        "model": "minilm_lr_coral",
        "seed": "",
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "roc_auc": roc,
        "notes": "train=sms_uci_dedup cross-domain coral",
    })

    # SpamAssassin -> SMS (cross-domain)
    f1, prec, rec, roc = eval_coral(spam_dir, sms_dir, st)
    rows.append({
        "dataset": "sms_uci_dedup",
        "split": "test",
        "model": "minilm_lr_coral",
        "seed": "",
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "roc_auc": roc,
        "notes": "train=spamassassin_dedup cross-domain coral",
    })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "split", "model", "seed", "f1", "precision", "recall", "roc_auc", "notes"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print("Wrote", out_path)


if __name__ == "__main__":
    main()
