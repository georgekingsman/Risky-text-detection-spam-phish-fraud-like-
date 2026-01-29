"""Final robust runner: import heavy libs inside suppressed stdout/stderr block
to avoid interleaved logging during model loads affecting output file.
"""
import os
import argparse
import glob
import csv
from contextlib import redirect_stdout, redirect_stderr

from .perturb import obfuscate, prompt_injection, simple_paraphrase_like
from .defense import normalize_text

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


def apply_defense(texts, defense: str | None):
    if defense == 'normalize':
        return [normalize_text(t) for t in texts]
    return texts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--dataset', default='sms_uci', help='Dataset name (sms_uci or spamassassin)')
    ap.add_argument('--data-dir', default=None, help='Override processed dataset directory')
    ap.add_argument('--defense', default='none', choices=['none', 'normalize'], help='Optional defense to apply')
    ap.add_argument('--include-baseline', action='store_true', help='Include baseline (no defense) rows alongside defended rows')
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
                clean_texts = apply_defense(texts, args.defense if args.defense != 'none' else None)
                X = vec.transform(clean_texts)
                preds = m['clf'].predict(X)
            elif isinstance(m, dict) and 'embed_model' in m and 'clf' in m:
                model_name = m.get('embed_model')
                embed_model = SentenceTransformer(model_name)
                clean_texts = apply_defense(texts, args.defense if args.defense != 'none' else None)
                X = embed_model.encode(clean_texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
                X = np.asarray(X, dtype=np.float32)
                preds = m['clf'].predict(X)
            else:
                clean_texts = apply_defense(texts, args.defense if args.defense != 'none' else None)
                preds = m.predict(clean_texts)
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
                pert_texts = apply_perturbation(func, texts, seed=args.seed)
                pert_texts = apply_defense(pert_texts, args.defense if args.defense != 'none' else None)

                if isinstance(m, dict) and 'tfidf' in m and 'clf' in m:
                    Xp = m['tfidf'].transform(pert_texts)
                    ppreds = m['clf'].predict(Xp)
                elif isinstance(m, dict) and 'embed_model' in m and 'clf' in m:
                    embed_model = SentenceTransformer(m.get('embed_model'))
                    Xp = embed_model.encode(pert_texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
                    Xp = np.asarray(Xp, dtype=np.float32)
                    ppreds = m['clf'].predict(Xp)
                else:
                    ppreds = m.predict(pert_texts)
                pf1 = f1_score(labels, ppreds)
                attack_name = pert_name
                if args.defense != 'none':
                    attack_name = f"{pert_name}+{args.defense}"
                rows.append({
                    'attack': attack_name,
                    'dataset': args.dataset,
                    'model': name,
                    'f1_clean': f1_clean,
                    'f1_attacked': pf1,
                    'delta_f1': pf1 - f1_clean
                })

            # optional baseline rows
            if args.include_baseline and args.defense != 'none':
                base_args = argparse.Namespace(**vars(args))
                base_args.defense = 'none'
                # insert baseline clean and perturbed rows by reusing recursive call
                # Note: we approximate baseline by recomputing with defense=none
                base_texts = texts
                if isinstance(m, dict) and 'tfidf' in m and 'clf' in m:
                    Xb = m['tfidf'].transform(base_texts)
                    bpreds = m['clf'].predict(Xb)
                elif isinstance(m, dict) and 'embed_model' in m and 'clf' in m:
                    embed_model = SentenceTransformer(m.get('embed_model'))
                    Xb = embed_model.encode(base_texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
                    Xb = np.asarray(Xb, dtype=np.float32)
                    bpreds = m['clf'].predict(Xb)
                else:
                    bpreds = m.predict(base_texts)
                bf1 = f1_score(labels, bpreds)
                rows.append({
                    'attack': 'clean',
                    'dataset': args.dataset,
                    'model': name,
                    'f1_clean': bf1,
                    'f1_attacked': bf1,
                    'delta_f1': 0.0
                })
                for pert_name, func in PERTS.items():
                    bpert = apply_perturbation(func, base_texts, seed=args.seed)
                    if isinstance(m, dict) and 'tfidf' in m and 'clf' in m:
                        Xbp = m['tfidf'].transform(bpert)
                        bpreds_p = m['clf'].predict(Xbp)
                    elif isinstance(m, dict) and 'embed_model' in m and 'clf' in m:
                        embed_model = SentenceTransformer(m.get('embed_model'))
                        Xbp = embed_model.encode(bpert, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
                        Xbp = np.asarray(Xbp, dtype=np.float32)
                        bpreds_p = m['clf'].predict(Xbp)
                    else:
                        bpreds_p = m.predict(bpert)
                    bpf1 = f1_score(labels, bpreds_p)
                    rows.append({
                        'attack': pert_name,
                        'dataset': args.dataset,
                        'model': name,
                        'f1_clean': bf1,
                        'f1_attacked': bpf1,
                        'delta_f1': bpf1 - bf1
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
