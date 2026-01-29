import os
import joblib
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sentence_transformers import SentenceTransformer

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'processed')
OUT_CSV = os.path.join(os.path.dirname(__file__), '..', 'results', 'results_clean.csv')
OUT_SUM = os.path.join(os.path.dirname(__file__), '..', 'results', 'metrics_summary.md')

def load_split(split):
    p = os.path.join(DATA_DIR, f"{split}.csv")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    # expect columns 'text' and 'label' (label: 1 spam, 0 ham)
    return df


def score_model_on_split(model_obj, texts, labels):
    # labels: 0/1
    ytrue = labels
    # model_obj can be several types:
    # - sklearn Pipeline / estimator with .predict
    # - dict with keys {'tfidf','clf'} where tfidf is vectorizer
    # - dict with keys {'embed_model','clf'} where embed_model is a model name
    preds = None
    proba_vals = None
    try:
        if isinstance(model_obj, dict):
            if 'tfidf' in model_obj and 'clf' in model_obj:
                vec = model_obj['tfidf']
                clf = model_obj['clf']
                Xvec = vec.transform(texts)
                preds = clf.predict(Xvec)
                if hasattr(clf, 'predict_proba'):
                    proba_vals = clf.predict_proba(Xvec)
                elif hasattr(clf, 'decision_function'):
                    proba_vals = clf.decision_function(Xvec)
            elif 'embed_model' in model_obj and 'clf' in model_obj:
                emb_name = model_obj['embed_model']
                clf = model_obj['clf']
                st = SentenceTransformer(emb_name)
                Xemb = st.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
                try:
                    preds = clf.predict(Xemb)
                except Exception:
                    preds = [clf.predict([x])[0] for x in Xemb]
                if hasattr(clf, 'predict_proba'):
                    proba_vals = clf.predict_proba(Xemb)
                elif hasattr(clf, 'decision_function'):
                    proba_vals = clf.decision_function(Xemb)
            else:
                raise ValueError('Unrecognized model dict format')
        else:
            # assume estimator or pipeline
            preds = model_obj.predict(texts)
            if hasattr(model_obj, 'predict_proba'):
                proba_vals = model_obj.predict_proba(texts)
            elif hasattr(model_obj, 'decision_function'):
                proba_vals = model_obj.decision_function(texts)
    except Exception:
        # fallback: elementwise predict
        try:
            preds = [model_obj.predict([t])[0] for t in texts]
        except Exception:
            raise
    # map textual labels to 0/1 if necessary
    # if preds are strings like 'spam'/'ham'
    if isinstance(preds[0], str):
        ypred = [1 if p.lower().startswith('spam') else 0 for p in preds]
    else:
        ypred = [int(p) for p in preds]

    f1 = f1_score(ytrue, ypred)
    prec = precision_score(ytrue, ypred, zero_division=0)
    rec = recall_score(ytrue, ypred, zero_division=0)

    # try ROC AUC via available proba_vals
    roc = ''
    try:
        if proba_vals is not None:
            if isinstance(proba_vals, list) or (hasattr(proba_vals, 'ndim') and proba_vals.ndim == 1):
                # decision_function single-dim
                roc = roc_auc_score(ytrue, proba_vals)
            else:
                # probas NxC
                p1 = proba_vals[:, 1]
                roc = roc_auc_score(ytrue, p1)
    except Exception:
        roc = ''

    return f1, prec, rec, roc


def main():
    # discover models
    model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.joblib') or f.endswith('.pkl')]
    rows = []
    splits = ['train', 'val', 'test']
    for mf in model_files:
        mp = os.path.join(MODEL_DIR, mf)
        try:
            m = joblib.load(mp)
        except Exception as e:
            print('Failed loading', mf, 'skipping', e)
            continue
        for split in splits:
            df = load_split(split)
            if df is None:
                continue
            texts = df['text'].tolist()
            labels = [1 if x == 1 else 0 for x in df['label'].tolist()]
            try:
                f1, prec, rec, roc = score_model_on_split(m, texts, labels)
            except Exception as e:
                print('Scoring failed for', mf, split, e)
                continue
            rows.append({
                'dataset': 'sms_uci',
                'split': split,
                'model': mf,
                'seed': '',
                'f1': float(f1),
                'precision': float(prec),
                'recall': float(rec),
                'roc_auc': float(roc) if roc != '' else '',
                'notes': ''
            })
    # also include any existing manual results not in models folder
    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_CSV, index=False)

    # write a small summary md
    best = out_df.loc[out_df.groupby('split')['f1'].idxmax()]
    with open(OUT_SUM, 'w') as f:
        f.write('# Metrics summary\n\n')
        for _, r in best.iterrows():
            f.write(f"- Split `{r['split']}` best model: {r['model']} -> F1={r['f1']:.4f}, ROC_AUC={r['roc_auc']}\n")
    print('Wrote', OUT_CSV, 'and', OUT_SUM)

if __name__ == '__main__':
    main()
