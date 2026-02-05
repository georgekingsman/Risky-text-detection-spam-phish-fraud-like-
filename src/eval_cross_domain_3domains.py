#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate Cross-Domain Generalization Across 3 Domains.

Trains on one domain, tests on all three, produces 9-cell matrix.
Output: results/cross_domain_3domain.csv
"""
import argparse
import warnings
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
from sentence_transformers import SentenceTransformer
import joblib

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]

DOMAINS = {
    "SMS": ROOT / "dataset/dedup/processed",
    "SpamAssassin": ROOT / "dataset/spamassassin/dedup/processed",
    "Telegram": ROOT / "dataset/telegram_spam_ham/dedup/processed",
}


def load_data(domain_path: Path) -> tuple:
    """Load train/val/test data for a domain."""
    train = pd.read_csv(domain_path / "train.csv")
    val = pd.read_csv(domain_path / "val.csv")
    test = pd.read_csv(domain_path / "test.csv")
    return train, val, test


def train_tfidf_lr(X_train, y_train, word_level=True):
    """Train TF-IDF + Logistic Regression."""
    if word_level:
        vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    else:
        vec = TfidfVectorizer(max_features=5000, ngram_range=(3, 5), analyzer='char')
    
    X_vec = vec.fit_transform(X_train)
    if word_level:
        clf = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    else:
        clf = LinearSVC(max_iter=2000, random_state=42, class_weight='balanced')
    clf.fit(X_vec, y_train)
    return vec, clf


def train_minilm(X_train, y_train):
    """Train MiniLM + Logistic Regression."""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(X_train.tolist(), show_progress_bar=False)
    clf = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    clf.fit(embeddings, y_train)
    return model, clf


def evaluate(vec, clf, X_test, y_test, is_minilm=False):
    """Evaluate and return F1 score."""
    if is_minilm:
        X_vec = vec.encode(X_test.tolist(), show_progress_bar=False)
    else:
        X_vec = vec.transform(X_test)
    y_pred = clf.predict(X_vec)
    return f1_score(y_test, y_pred, average='macro')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-minilm", action="store_true", help="Skip MiniLM to save time")
    args = parser.parse_args()

    results = []
    
    for train_domain, train_path in DOMAINS.items():
        print(f"\n{'='*60}")
        print(f"Training on: {train_domain}")
        print('='*60)
        
        if not train_path.exists():
            print(f"⚠️  {train_path} not found, skipping")
            continue
        
        train_df, val_df, test_df = load_data(train_path)
        
        # Combine train+val for training
        train_combined = pd.concat([train_df, val_df])
        X_train = train_combined["text"]
        y_train = train_combined["label"]
        
        # Train models on this domain
        print("Training TF-IDF Word LR...")
        vec_word, clf_word = train_tfidf_lr(X_train, y_train, word_level=True)
        
        print("Training TF-IDF Char SVM...")
        vec_char, clf_char = train_tfidf_lr(X_train, y_train, word_level=False)
        
        if not args.skip_minilm:
            print("Training MiniLM+LR...")
            model_mini, clf_mini = train_minilm(X_train, y_train)
        
        # Test on all domains
        for test_domain, test_path in DOMAINS.items():
            if not test_path.exists():
                continue
            
            _, _, test_df = load_data(test_path)
            X_test = test_df["text"]
            y_test = test_df["label"]
            
            f1_word = evaluate(vec_word, clf_word, X_test, y_test, is_minilm=False)
            f1_char = evaluate(vec_char, clf_char, X_test, y_test, is_minilm=False)
            
            if args.skip_minilm:
                f1_mini = None
            else:
                f1_mini = evaluate(model_mini, clf_mini, X_test, y_test, is_minilm=True)
            
            print(f"  {train_domain} → {test_domain}: Word={f1_word:.3f}, Char={f1_char:.3f}", end="")
            if f1_mini:
                print(f", MiniLM={f1_mini:.3f}")
            else:
                print()
            
            results.append({
                "Train": train_domain,
                "Test": test_domain,
                "TF-IDF Word LR": round(f1_word, 4),
                "TF-IDF Char SVM": round(f1_char, 4),
                "MiniLM+LR": round(f1_mini, 4) if f1_mini else None,
            })
    
    # Save results
    df = pd.DataFrame(results)
    out_path = ROOT / "results/cross_domain_3domain.csv"
    df.to_csv(out_path, index=False)
    print(f"\n✅ Saved to {out_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
