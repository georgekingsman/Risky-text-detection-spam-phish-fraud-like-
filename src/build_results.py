import os
import csv
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]


def load_dataset(data_dir):
    df = pd.read_csv(Path(data_dir) / 'test.csv')
    texts = df['text'].tolist()
    labels = [1 if x == 1 else 0 for x in df['label'].tolist()]
    return texts, labels


def eval_model(model_path, texts, labels):
    m = joblib.load(model_path)
    proba = None
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
            emb_name = m['embed_model']
            clf = m['clf']
            st = SentenceTransformer(emb_name)
            X = st.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
            preds = clf.predict(X)
            if hasattr(clf, 'predict_proba'):
                proba = clf.predict_proba(X)
            elif hasattr(clf, 'decision_function'):
                proba = clf.decision_function(X)
        else:
            raise ValueError(f'Unsupported model dict format: {model_path}')
    else:
        preds = m.predict(texts)
        if hasattr(m, 'predict_proba'):
            proba = m.predict_proba(texts)
        elif hasattr(m, 'decision_function'):
            proba = m.decision_function(texts)

    # normalize preds
    if isinstance(preds[0], str):
        ypred = [1 if p.lower().startswith('spam') else 0 for p in preds]
    else:
        ypred = [int(p) for p in preds]

    f1 = f1_score(labels, ypred)
    prec = precision_score(labels, ypred, zero_division=0)
    rec = recall_score(labels, ypred, zero_division=0)
    roc = ''
    try:
        if proba is not None:
            arr = np.array(proba)
            if arr.ndim == 1:
                roc = roc_auc_score(labels, arr)
            else:
                roc = roc_auc_score(labels, arr[:, 1])
    except Exception:
        roc = ''

    return f1, prec, rec, roc


def load_llm_jsonl(path):
    if not Path(path).exists():
        return None
    ytrue = []
    ypred = []
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            ytrue.append(int(obj['label']))
            ypred.append(int(obj['pred_label_int']))
    f1 = f1_score(ytrue, ypred)
    prec = precision_score(ytrue, ypred, zero_division=0)
    rec = recall_score(ytrue, ypred, zero_division=0)
    roc = roc_auc_score(ytrue, ypred) if len(set(ytrue)) > 1 else ''
    return f1, prec, rec, roc


def main():
    out_path = ROOT / 'results' / 'results.csv'
    rows = []

    # datasets
    sms_dir = ROOT / 'dataset' / 'processed'
    spam_dir = ROOT / 'dataset' / 'spamassassin' / 'processed'

    # in-domain SMS models
    sms_models = {
        'tfidf_word_lr': ROOT / 'models' / 'tfidf_word_lr.joblib',
        'tfidf_char_svm': ROOT / 'models' / 'tfidf_char_svm.joblib',
        'minilm_lr': ROOT / 'models' / 'minilm_lr.joblib',
        'tfidf_llm_lr': ROOT / 'models' / 'tfidf_llm_lr.joblib',
    }
    sms_texts, sms_labels = load_dataset(sms_dir)
    for name, path in sms_models.items():
        if path.exists():
            f1, prec, rec, roc = eval_model(path, sms_texts, sms_labels)
            rows.append({'dataset':'sms_uci','split':'test','model':name,'seed':'','f1':f1,'precision':prec,'recall':rec,'roc_auc':roc,'notes':'train=sms_uci in-domain'})

    # in-domain SpamAssassin models
    spam_models = {
        'tfidf_word_lr': ROOT / 'models' / 'spamassassin_tfidf_word_lr.joblib',
        'tfidf_char_svm': ROOT / 'models' / 'spamassassin_tfidf_char_svm.joblib',
        'minilm_lr': ROOT / 'models' / 'spamassassin_minilm_lr.joblib',
    }
    spam_texts, spam_labels = load_dataset(spam_dir)
    for name, path in spam_models.items():
        if path.exists():
            f1, prec, rec, roc = eval_model(path, spam_texts, spam_labels)
            rows.append({'dataset':'spamassassin','split':'test','model':name,'seed':'','f1':f1,'precision':prec,'recall':rec,'roc_auc':roc,'notes':'train=spamassassin in-domain'})

    # cross-domain: SMS models -> SpamAssassin
    for name, path in sms_models.items():
        if path.exists():
            f1, prec, rec, roc = eval_model(path, spam_texts, spam_labels)
            rows.append({'dataset':'spamassassin','split':'test','model':name,'seed':'','f1':f1,'precision':prec,'recall':rec,'roc_auc':roc,'notes':'train=sms_uci cross-domain'})

    # cross-domain: SpamAssassin models -> SMS
    for name, path in spam_models.items():
        if path.exists():
            f1, prec, rec, roc = eval_model(path, sms_texts, sms_labels)
            rows.append({'dataset':'sms_uci','split':'test','model':name,'seed':'','f1':f1,'precision':prec,'recall':rec,'roc_auc':roc,'notes':'train=spamassassin cross-domain'})

    # LLM zero-shot metrics if available
    sms_llm = ROOT / 'results' / 'llm_predictions_sms_test.jsonl'
    spam_llm = ROOT / 'results' / 'llm_predictions_spamassassin_test.jsonl'
    if sms_llm.exists():
        f1, prec, rec, roc = load_llm_jsonl(sms_llm)
        rows.append({'dataset':'sms_uci','split':'test','model':'llm_zeroshot','seed':'','f1':f1,'precision':prec,'recall':rec,'roc_auc':roc,'notes':'jsonl'})
    if spam_llm.exists():
        f1, prec, rec, roc = load_llm_jsonl(spam_llm)
        rows.append({'dataset':'spamassassin','split':'test','model':'llm_zeroshot','seed':'','f1':f1,'precision':prec,'recall':rec,'roc_auc':roc,'notes':'jsonl'})

    # write
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['dataset','split','model','seed','f1','precision','recall','roc_auc','notes'])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print('Wrote', out_path)


if __name__ == '__main__':
    main()
