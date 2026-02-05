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

    # Dataset directories - now supporting 3 domains
    sms_dir = ROOT / "dataset" / "dedup" / "processed"
    spam_dir = ROOT / "dataset" / "spamassassin" / "dedup" / "processed"
    telegram_dir = ROOT / "dataset" / "telegram_spam_ham" / "dedup" / "processed"

    # Define all datasets with their models
    datasets_config = {
        "sms_uci_dedup": {
            "dir": sms_dir,
            "prefix": "sms_dedup",
        },
        "spamassassin_dedup": {
            "dir": spam_dir,
            "prefix": "spamassassin_dedup",
        },
        "telegram_dedup": {
            "dir": telegram_dir,
            "prefix": "telegram_dedup",
        },
    }

    model_types = ["tfidf_word_lr", "tfidf_char_svm", "minilm_lr", "tfidf_word_lr_augtrain"]

    # Build model paths for each dataset
    def get_models(prefix):
        return {
            "tfidf_word_lr": ROOT / "models" / f"{prefix}_tfidf_word_lr.joblib",
            "tfidf_char_svm": ROOT / "models" / f"{prefix}_tfidf_char_svm.joblib",
            "minilm_lr": ROOT / "models" / f"{prefix}_minilm_lr.joblib",
            "tfidf_word_lr_augtrain": ROOT / "models" / f"{prefix}_aug_tfidf_word_lr.joblib",
        }

    sms_models = get_models("sms_dedup")
    spam_models = get_models("spamassassin_dedup")
    telegram_models = get_models("telegram_dedup")

    # Load test sets for available datasets
    datasets_available = {}
    for ds_name, ds_config in datasets_config.items():
        test_file = ds_config["dir"] / "test.csv"
        if test_file.exists():
            datasets_available[ds_name] = {
                "dir": ds_config["dir"],
                "prefix": ds_config["prefix"],
                "texts": None,
                "labels": None,
                "models": get_models(ds_config["prefix"]),
            }
            texts, labels = load_dataset(ds_config["dir"])
            datasets_available[ds_name]["texts"] = texts
            datasets_available[ds_name]["labels"] = labels
        else:
            print(f"[WARN] Dataset {ds_name} not found at {test_file}")

    # Evaluate all dataset combinations (in-domain + cross-domain)
    all_model_prefixes = list(set(ds["prefix"] for ds in datasets_available.values()))
    
    for train_ds_name, train_ds in datasets_available.items():
        train_models = train_ds["models"]
        train_prefix = train_ds["prefix"]
        
        for test_ds_name, test_ds in datasets_available.items():
            test_texts = test_ds["texts"]
            test_labels = test_ds["labels"]
            
            is_cross = train_ds_name != test_ds_name
            note_suffix = "cross-domain" if is_cross else "in-domain"
            
            for model_name, model_path in train_models.items():
                if model_path.exists():
                    try:
                        f1, prec, rec, roc = eval_model(model_path, test_texts, test_labels)
                        rows.append({
                            "dataset": test_ds_name,
                            "split": "test",
                            "model": model_name,
                            "seed": "",
                            "f1": f1,
                            "precision": prec,
                            "recall": rec,
                            "roc_auc": roc,
                            "notes": f"train={train_ds_name} {note_suffix}",
                        })
                    except Exception as e:
                        print(f"[ERROR] {model_path} on {test_ds_name}: {e}")

    # CORAL MiniLM domain alignment (in-domain + cross-domain)
    embed_name = "sentence-transformers/all-MiniLM-L6-v2"
    st = SentenceTransformer(embed_name)

    # Evaluate CORAL for all dataset combinations
    for source_name, source_ds in datasets_available.items():
        source_dir = source_ds["dir"]
        for target_name, target_ds in datasets_available.items():
            target_dir = target_ds["dir"]
            is_cross = source_name != target_name
            note_suffix = "cross-domain coral" if is_cross else "in-domain coral"
            
            try:
                f1, prec, rec, roc = eval_coral(source_dir, target_dir, st)
                rows.append({
                    "dataset": target_name,
                    "split": "test",
                    "model": "minilm_lr_coral",
                    "seed": "",
                    "f1": f1,
                    "precision": prec,
                    "recall": rec,
                    "roc_auc": roc,
                    "notes": f"train={source_name} {note_suffix}",
                })
            except Exception as e:
                print(f"[ERROR] CORAL {source_name}->{target_name}: {e}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "split", "model", "seed", "f1", "precision", "recall", "roc_auc", "notes"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print("Wrote", out_path)


if __name__ == "__main__":
    main()
