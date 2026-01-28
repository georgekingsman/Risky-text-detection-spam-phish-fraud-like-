import argparse
import os
import json
from pathlib import Path
import random

import pandas as pd
import numpy as np
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score

from src import train_llm


def load_split():
    tr = pd.read_csv("dataset/processed/train.csv")
    va = pd.read_csv("dataset/processed/val.csv")
    te = pd.read_csv("dataset/processed/test.csv")
    return tr, va, te


def cached_reasons(texts, split_name, provider, model, sample_limit=None):
    os.makedirs('models', exist_ok=True)
    cache_path = Path(f'models/llm_reasons_{split_name}.json')
    if cache_path.exists():
        with open(cache_path, 'r') as f:
            data = json.load(f)
    else:
        data = {}

    to_query = []
    for i, t in enumerate(texts):
        if str(i) not in data:
            to_query.append((i, t))
    if sample_limit is not None and len(to_query) > sample_limit:
        to_query = to_query[:sample_limit]

    if to_query:
        idxs, batch_texts = zip(*to_query)
        reasons = train_llm.llm_extract_reasons(list(batch_texts), provider=provider, model=model, limit=len(batch_texts))
        for ii, r in zip(idxs, reasons):
            data[str(ii)] = r
        with open(cache_path, 'w') as f:
            json.dump(data, f)

    out = [data.get(str(i), '') for i in range(len(texts))]
    return out


def augment_texts(texts, reasons):
    out = []
    for t, r in zip(texts, reasons):
        if r:
            out.append(t + " \nREASONS: " + r)
        else:
            out.append(t)
    return out


def train_and_eval(Xtr, ytr, Xte, yte, model_name="tfidf_llm_lr"):
    pipe = TfidfVectorizer(ngram_range=(1,2), min_df=2)
    Xtr_vec = pipe.fit_transform(Xtr)
    Xte_vec = pipe.transform(Xte)
    clf = LogisticRegression(max_iter=2000)
    clf.fit(Xtr_vec, ytr)
    yp = clf.predict(Xte_vec)
    f1 = f1_score(yte, yp)
    prec = precision_score(yte, yp)
    rec = recall_score(yte, yp)
    dump({'tfidf': pipe, 'clf': clf}, f'models/{model_name}.joblib')
    return f1, prec, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--provider', default=None)
    ap.add_argument('--model', default='distilgpt2')
    ap.add_argument('--sample', type=int, default=200, help='How many examples per split to query LLM for (None = all)')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    tr, va, te = load_split()

    tr_reasons = cached_reasons(tr['text'].tolist(), 'train', args.provider, args.model, sample_limit=args.sample)
    va_reasons = cached_reasons(va['text'].tolist(), 'val', args.provider, args.model, sample_limit=args.sample)
    te_reasons = cached_reasons(te['text'].tolist(), 'test', args.provider, args.model, sample_limit=args.sample)

    tr_aug = augment_texts(tr['text'].tolist(), tr_reasons)
    va_aug = augment_texts(va['text'].tolist(), va_reasons)
    te_aug = augment_texts(te['text'].tolist(), te_reasons)

    ytr = tr['label'].values
    yva = va['label'].values
    yte = te['label'].values

    os.makedirs('results', exist_ok=True)

    f1_va, p_va, r_va = train_and_eval(tr_aug, ytr, va_aug, yva, model_name='tfidf_llm_lr')
    f1_te, p_te, r_te = train_and_eval(tr_aug, ytr, te_aug, yte, model_name='tfidf_llm_lr')

    import csv
    res_path = Path('results/results.csv')
    if not res_path.exists():
        with open(res_path, 'w') as f:
            f.write('model,f1,precision,recall,roc_auc\n')

    with open(res_path, 'a') as f:
        w = csv.writer(f)
        w.writerow(['tfidf_llm_lr_val', f1_va, p_va, r_va, ''])
        w.writerow(['tfidf_llm_lr_test', f1_te, p_te, r_te, ''])

    print('LLM-as-feature done. Val F1', f1_va, 'Test F1', f1_te)


if __name__ == '__main__':
    main()
