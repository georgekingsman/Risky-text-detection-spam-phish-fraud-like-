import os
import argparse
import glob
import csv
import random
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sentence_transformers import SentenceTransformer

from .perturb import obfuscate, prompt_injection, simple_paraphrase_like

PERTS = {
    "obfuscate": obfuscate,
    "paraphrase_like": simple_paraphrase_like,
    "prompt_injection": prompt_injection,
}

def apply_perturbation(func, texts, seed=0):
    out = []
    for i, t in enumerate(texts):
        if func is obfuscate:
            out.append(func(t, seed=seed + i))
        else:
            out.append(func(t))
    return out

def predict_with_model(m, texts, embed_cache=None):
    # m can be a sklearn pipeline or a dict (embed model)
    if isinstance(m, dict) and 'clf' in m:
        model_name = m.get('embed_model')
        if embed_cache is None or embed_cache.get('name') != model_name:
            embed_cache = {'name': model_name, 'model': SentenceTransformer(model_name)}
        X = embed_cache['model'].encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
        X = np.asarray(X, dtype=np.float32)
        preds = m['clf'].predict(X)
        return preds, embed_cache
    else:
        preds = m.predict(texts)
        return preds, embed_cache

def eval_model_on_texts(m, texts, labels, embed_cache=None):
    preds, embed_cache = predict_with_model(m, texts, embed_cache=embed_cache)
    f1 = f1_score(labels, preds)
    prec = precision_score(labels, preds)
    rec = recall_score(labels, preds)
    return dict(f1=f1, precision=prec, recall=rec), embed_cache

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', default='results/robustness.csv')
    args = ap.parse_args()

    os.makedirs('results', exist_ok=True)

    te = pd.read_csv('dataset/processed/test.csv')
    texts = te['text'].tolist()
    labels = te['label'].values

    rows = []
    embed_cache = None

    for p in sorted(glob.glob('models/*.joblib')):
        name = os.path.basename(p)
        m = joblib.load(p)
        # compute clean
        clean_res, embed_cache = eval_model_on_texts(m, texts, labels, embed_cache=embed_cache)
        base_f1 = clean_res['f1']
        rows.append({
            'model': name,
            'perturbation': 'clean',
            'f1': base_f1,
            'precision': clean_res['precision'],
            'recall': clean_res['recall'],
            'delta_f1': 0.0,
        })

        for pert_name, func in PERTS.items():
            pert_texts = apply_perturbation(func, texts, seed=args.seed)
            pert_res, embed_cache = eval_model_on_texts(m, pert_texts, labels, embed_cache=embed_cache)
            delta = pert_res['f1'] - base_f1
            rows.append({
                'model': name,
                'perturbation': pert_name,
                'f1': pert_res['f1'],
                'precision': pert_res['precision'],
                'recall': pert_res['recall'],
                'delta_f1': delta,
            })

    # write CSV
    keys = ['model', 'perturbation', 'f1', 'precision', 'recall', 'delta_f1']
    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print('Wrote', args.out)

if __name__ == '__main__':
    main()
