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
    ap.add_argument('--dataset', default='sms_uci', help='Dataset name (sms_uci or spamassassin)')
    ap.add_argument('--data-dir', default=None, help='Override processed dataset directory')
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

        if args.data_dir:
            data_dir = args.data_dir
        else:
            if args.dataset == 'spamassassin':
                data_dir = 'dataset/spamassassin/processed'
            else:
                data_dir = 'dataset/processed'

        te = pd.read_csv(os.path.join(data_dir, 'test.csv'))
        texts = te['text'].tolist()
        labels = te['label'].values

        rows = []
        for p in sorted(glob.glob('models/*.joblib')):
            name = os.path.basename(p)
            m = joblib.load(p)
            # eval clean
            if isinstance(m, dict) and 'tfidf' in m and 'clf' in m:
                vec = m['tfidf']
                X = vec.transform(texts)
                preds = m['clf'].predict(X)
            elif isinstance(m, dict) and 'embed_model' in m and 'clf' in m:
                model_name = m.get('embed_model')
                embed_model = SentenceTransformer(model_name)
                X = embed_model.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
                X = np.asarray(X, dtype=np.float32)
                preds = m['clf'].predict(X)
            else:
                preds = m.predict(texts)
            from sklearn.metrics import f1_score, precision_score, recall_score
            f1 = f1_score(labels, preds)
            prec = precision_score(labels, preds)
            rec = recall_score(labels, preds)
            f1_clean = f1
            rows.append({
                'attack': 'clean',
                'dataset': args.dataset,
                'model': name,
                'f1_clean': f1_clean,
                'f1_attacked': f1_clean,
                'delta_f1': 0.0
            })

            # perturbs
            for pert_name, func in PERTS.items():
                if isinstance(m, dict) and 'tfidf' in m and 'clf' in m:
                    Xp = m['tfidf'].transform(apply_perturbation(func, texts, seed=args.seed))
                    ppreds = m['clf'].predict(Xp)
                elif isinstance(m, dict) and 'embed_model' in m and 'clf' in m:
                    embed_model = SentenceTransformer(m.get('embed_model'))
                    Xp = embed_model.encode(apply_perturbation(func, texts, seed=args.seed), batch_size=64, show_progress_bar=False, normalize_embeddings=True)
                    Xp = np.asarray(Xp, dtype=np.float32)
                    ppreds = m['clf'].predict(Xp)
                else:
                    ppreds = m.predict(apply_perturbation(func, texts, seed=args.seed))
                pf1 = f1_score(labels, ppreds)
                rows.append({
                    'attack': pert_name,
                    'dataset': args.dataset,
                    'model': name,
                    'f1_clean': f1_clean,
                    'f1_attacked': pf1,
                    'delta_f1': pf1 - f1_clean
                })

    # write CSV (outside suppressed block)
    keys = ['attack', 'dataset', 'model', 'f1_clean', 'f1_attacked', 'delta_f1']
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
