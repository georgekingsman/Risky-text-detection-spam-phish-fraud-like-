import os
import requests
import tarfile
import tempfile
import shutil
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

URL_HAM = 'http://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham.tar.bz2'
URL_SPAM = 'http://spamassassin.apache.org/old/publiccorpus/20030228_spam.tar.bz2'

OUT_DIR = Path(__file__).resolve().parents[1] / 'dataset' / 'spamassassin'
RAW_DIR = OUT_DIR / 'raw'
PROC_DIR = OUT_DIR / 'processed'


def download(url, dest):
    print('Downloading', url)
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    with open(dest, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)


def extract_tarball(path, dest):
    print('Extracting', path)
    with tarfile.open(path, 'r:*') as tar:
        tar.extractall(dest)


def collect_texts_from_dir(d, label):
    rows = []
    for root, _, files in os.walk(d):
        for fn in files:
            p = os.path.join(root, fn)
            try:
                with open(p, 'rb') as f:
                    raw = f.read()
                txt = raw.decode('utf-8', errors='ignore')
                # Heuristic: strip headers if present
                # Find first blank line
                parts = txt.split('\n\n', 1)
                body = parts[1] if len(parts) > 1 else parts[0]
                body = body.strip()
                if body:
                    rows.append({'text': body, 'label': label})
            except Exception:
                continue
    return rows


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        ham_tar = os.path.join(td, 'easy_ham.tar.bz2')
        spam_tar = os.path.join(td, 'spam.tar.bz2')
        download(URL_HAM, ham_tar)
        download(URL_SPAM, spam_tar)
        extract_tarball(ham_tar, td)
        extract_tarball(spam_tar, td)
        # Collect
        ham_rows = collect_texts_from_dir(td, 0)
        spam_rows = collect_texts_from_dir(td, 1)

    all_rows = ham_rows + spam_rows
    df = pd.DataFrame(all_rows)
    print('Collected', len(df), 'examples (ham:', len(ham_rows), 'spam:', len(spam_rows), ')')

    # Shuffle and stratified split 80/10/10
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    train, temp = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
    val, test = train_test_split(temp, test_size=0.5, stratify=temp['label'], random_state=42)

    train.to_csv(PROC_DIR / 'train.csv', index=False)
    val.to_csv(PROC_DIR / 'val.csv', index=False)
    test.to_csv(PROC_DIR / 'test.csv', index=False)

    print('Wrote processed files to', PROC_DIR)

if __name__ == '__main__':
    main()
