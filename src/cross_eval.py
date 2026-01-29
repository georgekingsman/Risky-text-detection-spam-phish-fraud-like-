import argparse
import joblib
import pandas as pd
import os
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sentence_transformers import SentenceTransformer


def load_test_csv(p):
    df = pd.read_csv(p)
    return df['text'].tolist(), [1 if x==1 else 0 for x in df['label'].tolist()]


def predict_with_model(mpath, texts):
    m = joblib.load(mpath)
    proba = None
    preds = None
    if isinstance(m, dict):
        if 'tfidf' in m and 'clf' in m:
            vec = m['tfidf']
            clf = m['clf']
            X = vec.transform(texts)
            preds = clf.predict(X)
            if hasattr(clf, 'predict_proba'):
                proba = clf.predict_proba(X)
            elif hasattr(clf, 'decision_function'):
                proba = clf.decision_function(X)
        elif 'embed_model' in m and 'clf' in m:
            emb = m['embed_model']
            clf = m['clf']
            st = SentenceTransformer(emb)
            X = st.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
            preds = clf.predict(X)
            if hasattr(clf, 'predict_proba'):
                proba = clf.predict_proba(X)
            elif hasattr(clf, 'decision_function'):
                proba = clf.decision_function(X)
    else:
        # sklearn pipeline or estimator
        preds = m.predict(texts)
        if hasattr(m, 'predict_proba'):
            proba = m.predict_proba(texts)
        elif hasattr(m, 'decision_function'):
            proba = m.decision_function(texts)

    # normalize preds to 0/1
    if isinstance(preds[0], str):
        ypred = [1 if p.lower().startswith('spam') else 0 for p in preds]
    else:
        ypred = [int(p) for p in preds]
    return ypred, proba


def compute_metrics(ytrue, ypred, proba):
    f1 = f1_score(ytrue, ypred)
    prec = precision_score(ytrue, ypred, zero_division=0)
    rec = recall_score(ytrue, ypred, zero_division=0)
    roc = ''
    try:
        if proba is not None:
            import numpy as np
            pa = np.array(proba)
            if pa.ndim == 1:
                roc = roc_auc_score(ytrue, pa)
            else:
                roc = roc_auc_score(ytrue, pa[:,1])
    except Exception:
        roc = ''
    return f1, prec, rec, roc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, help='Path to model joblib')
    ap.add_argument('--test-csv', required=True, help='Path to test csv')
    ap.add_argument('--label', default=None, help='Label for results prefix')
    args = ap.parse_args()

    texts, ytrue = load_test_csv(args.test_csv)
    ypred, proba = predict_with_model(args.model, texts)
    f1, prec, rec, roc = compute_metrics(ytrue, ypred, proba)
    print('Eval on', args.test_csv, '-> F1', f1)

    # append to results
    outp = 'results/results.csv'
    import csv
    with open(outp, 'a') as f:
        w = csv.writer(f)
        name = args.label or os.path.basename(args.model)
        w.writerow([name, f1, prec, rec, roc])

if __name__ == '__main__':
    main()
