"""Final robust runner: import heavy libs inside suppressed stdout/stderr block
to avoid interleaved logging during model loads affecting output file.
"""
import os
import argparse
import glob
import csv
from contextlib import redirect_stdout, redirect_stderr

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', default='results/robustness.csv')
    args = ap.parse_args()

    os.makedirs('results', exist_ok=True)

    # suppress all stdout/stderr during heavy imports/loads
    with open(os.devnull, 'w') as devnull, redirect_stdout(devnull), redirect_stderr(devnull):
        import joblib
        import numpy as np
        import pandas as pd
        from sklearn.metrics import f1_score, precision_score, recall_score
        from sentence_transformers import SentenceTransformer

        te = pd.read_csv('dataset/processed/test.csv')
        texts = te['text'].tolist()
        labels = te['label'].values

        rows = []
        embed_cache = None

        for p in sorted(glob.glob('models/*.joblib')):
            name = os.path.basename(p)
            m = joblib.load(p)
            # eval clean
            if isinstance(m, dict) and 'clf' in m:
                model_name = m.get('embed_model')
                embed_cache = {'name': model_name, 'model': SentenceTransformer(model_name)}
                X = embed_cache['model'].encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
                X = np.asarray(X, dtype=np.float32)
                preds = m['clf'].predict(X)
            else:
                preds = m.predict(texts)
            from sklearn.metrics import f1_score, precision_score, recall_score
            f1 = f1_score(labels, preds)
            prec = precision_score(labels, preds)
            rec = recall_score(labels, preds)
            rows.append({'model': name, 'perturbation': 'clean', 'f1': f1, 'precision': prec, 'recall': rec, 'delta_f1': 0.0})

            # perturbs
            for pert_name, func in PERTS.items():
                if isinstance(m, dict) and 'clf' in m:
                    # use embed_cache
                    Xp = embed_cache['model'].encode(apply_perturbation(func, texts, seed=args.seed), batch_size=64, show_progress_bar=False, normalize_embeddings=True)
                    Xp = np.asarray(Xp, dtype=np.float32)
                    ppreds = m['clf'].predict(Xp)
                else:
                    ppreds = m.predict(apply_perturbation(func, texts, seed=args.seed))
                pf1 = f1_score(labels, ppreds)
                pprec = precision_score(labels, ppreds)
                prec_recall = recall_score(labels, ppreds)
                rows.append({'model': name, 'perturbation': pert_name, 'f1': pf1, 'precision': pprec, 'recall': prec_recall, 'delta_f1': pf1 - f1})

    # write CSV (outside suppressed block)
    keys = ['model', 'perturbation', 'f1', 'precision', 'recall', 'delta_f1']
    tmp = args.out + '.tmp'
    with open(tmp, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    os.replace(tmp, args.out)
    print('Wrote', args.out)

if __name__ == '__main__':
    main()
