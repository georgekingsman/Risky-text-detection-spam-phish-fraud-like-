#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate JSD Domain Shift Across 3 Domains.
Output: results/domain_shift_js_3domains.csv
"""
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

ROOT = Path(__file__).resolve().parents[1]

DOMAINS = {
    "sms": ROOT / "dataset/dedup/processed",
    "spamassassin": ROOT / "dataset/spamassassin/dedup/processed",
    "telegram": ROOT / "dataset/telegram_spam_ham/dedup/processed",
}


def get_char_ngrams(texts, n_min=3, n_max=5):
    """Extract character n-grams from texts."""
    counter = Counter()
    for text in texts:
        text = str(text).lower()
        for n in range(n_min, n_max + 1):
            for i in range(len(text) - n + 1):
                counter[text[i:i+n]] += 1
    return counter


def compute_jsd(dist_a: dict, dist_b: dict) -> float:
    """Compute Jensen-Shannon Divergence between two distributions."""
    all_keys = set(dist_a.keys()) | set(dist_b.keys())
    
    vec_a = np.array([dist_a.get(k, 0) for k in all_keys], dtype=float)
    vec_b = np.array([dist_b.get(k, 0) for k in all_keys], dtype=float)
    
    # Normalize to probability distributions
    vec_a = vec_a / vec_a.sum() if vec_a.sum() > 0 else vec_a
    vec_b = vec_b / vec_b.sum() if vec_b.sum() > 0 else vec_b
    
    return jensenshannon(vec_a, vec_b)


def main():
    results = []
    
    # Load texts for each domain
    domain_texts = {}
    for name, path in DOMAINS.items():
        if path.exists():
            train = pd.read_csv(path / "train.csv")
            domain_texts[name] = train["text"].tolist()
            print(f"Loaded {len(domain_texts[name])} texts from {name}")
    
    # Compute pairwise JSD
    for (name_a, texts_a), (name_b, texts_b) in combinations(domain_texts.items(), 2):
        print(f"Computing JSD: {name_a} ↔ {name_b}")
        
        dist_a = get_char_ngrams(texts_a)
        dist_b = get_char_ngrams(texts_b)
        
        jsd = compute_jsd(dist_a, dist_b)
        
        results.append({
            "domain_a": name_a,
            "domain_b": name_b,
            "jsd_char_3to5": round(jsd, 4),
        })
        print(f"  JSD = {jsd:.4f}")
    
    df = pd.DataFrame(results)
    out_path = ROOT / "results/domain_shift_js_3domains.csv"
    df.to_csv(out_path, index=False)
    print(f"\n✅ Saved to {out_path}")
    print(df)


if __name__ == "__main__":
    main()
