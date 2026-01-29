import os
import argparse
import pandas as pd
from joblib import dump
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, precision_score, recall_score


def load_split(data_dir):
    tr = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    va = pd.read_csv(os.path.join(data_dir, 'val.csv'))
    te = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    return tr, va, te


def train_and_eval_word_lr(tr, va, te, out_prefix):
    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1,2), min_df=2)),
        ('clf', LogisticRegression(max_iter=2000))
    ])
    pipe.fit(tr['text'], tr['label'])
    dump(pipe, f'models/{out_prefix}_tfidf_word_lr.joblib')

    def eval_split(df):
        yp = pipe.predict(df['text'])
        return f1_score(df['label'], yp), precision_score(df['label'], yp), recall_score(df['label'], yp)
    return eval_split(tr), eval_split(va), eval_split(te)


def train_and_eval_char_svm(tr, va, te, out_prefix):
    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(analyzer='char', ngram_range=(3,5), min_df=2)),
        ('clf', LinearSVC())
    ])
    pipe.fit(tr['text'], tr['label'])
    dump(pipe, f'models/{out_prefix}_tfidf_char_svm.joblib')

    def eval_split(df):
        yp = pipe.predict(df['text'])
        return f1_score(df['label'], yp), precision_score(df['label'], yp), recall_score(df['label'], yp)
    return eval_split(tr), eval_split(va), eval_split(te)


def train_and_eval_minilm(tr, va, te, out_prefix, embed_model='sentence-transformers/all-MiniLM-L6-v2'):
    from sentence_transformers import SentenceTransformer
    import numpy as np
    st = SentenceTransformer(embed_model)
    Xtr = st.encode(tr['text'].tolist(), batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    Xva = st.encode(va['text'].tolist(), batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    Xte = st.encode(te['text'].tolist(), batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    clf = LogisticRegression(max_iter=2000)
    clf.fit(Xtr, tr['label'].values)
    dump({'embed_model': embed_model, 'clf': clf}, f'models/{out_prefix}_minilm_lr.joblib')

    def eval_arr(X, y):
        yp = clf.predict(X)
        from sklearn.metrics import f1_score, precision_score, recall_score
        return f1_score(y, yp), precision_score(y, yp), recall_score(y, yp)
    return eval_arr(Xtr, tr['label'].values), eval_arr(Xva, va['label'].values), eval_arr(Xte, te['label'].values)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', required=True, help='Path to processed dataset (train.csv, val.csv, test.csv)')
    ap.add_argument('--prefix', required=True, help='Prefix for model filenames (e.g., sms_uci, spamassassin)')
    args = ap.parse_args()

    os.makedirs('models', exist_ok=True)

    tr, va, te = load_split(args.data_dir)

    w_tr, w_va, w_te = train_and_eval_word_lr(tr, va, te, args.prefix)
    c_tr, c_va, c_te = train_and_eval_char_svm(tr, va, te, args.prefix)
    e_tr, e_va, e_te = train_and_eval_minilm(tr, va, te, args.prefix)

    # write results
    import csv
    outp = 'results/results.csv'
    if not os.path.exists(outp):
        with open(outp, 'w') as f:
            f.write('model,f1,precision,recall,roc_auc\n')
    with open(outp, 'a') as f:
        w = csv.writer(f)
        # in-domain test
        w.writerow([f'{args.prefix}_tfidf_word_lr_test', w_te[0], w_te[1], w_te[2], ''])
        w.writerow([f'{args.prefix}_tfidf_char_svm_test', c_te[0], c_te[1], c_te[2], ''])
        w.writerow([f'{args.prefix}_minilm_lr_test', e_te[0], e_te[1], e_te[2], ''])

    print('Done training on', args.data_dir)

if __name__ == '__main__':
    main()
