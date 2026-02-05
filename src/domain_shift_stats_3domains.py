#!/usr/bin/env python
"""
Domain shift statistics for 3 domains: SMS, SpamAssassin, Telegram.

Computes pairwise JSD (Jensen-Shannon Divergence) and feature statistics
across all three domains to quantify domain shift.

Usage:
    python -m src.domain_shift_stats_3domains
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer

ROOT = Path(__file__).resolve().parents[1]

URL_RE = re.compile(r"(https?://|www\.)\S+")
EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
PHONE_RE = re.compile(r"\b(\+?\d[\d\-\s]{6,}\d)\b")


def featurize(text: str) -> dict:
    """Extract surface-level features from text."""
    t = "" if text is None else str(text)
    n = max(len(t), 1)
    tokens = re.findall(r"[A-Za-z0-9]+", t)
    ntok = max(len(tokens), 1)

    digits = sum(ch.isdigit() for ch in t)
    uppers = sum(ch.isupper() for ch in t)
    punct = sum((not ch.isalnum()) and (not ch.isspace()) for ch in t)

    return {
        "len_chars": len(t),
        "len_tokens": len(tokens),
        "digit_ratio": digits / n,
        "upper_ratio": uppers / n,
        "punct_ratio": punct / n,
        "url_cnt": len(URL_RE.findall(t)),
        "email_cnt": len(EMAIL_RE.findall(t)),
        "phone_cnt": len(PHONE_RE.findall(t)),
        "avg_tok_len": float(np.mean([len(x) for x in tokens])) if tokens else 0.0,
        "digit_per_tok": digits / ntok,
    }


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """Compute Jensen-Shannon divergence between two distributions."""
    p = p.astype(np.float64)
    q = q.astype(np.float64)
    p = p / (p.sum() + eps)
    q = q / (q.sum() + eps)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log((p + eps) / (m + eps)))
    kl_qm = np.sum(q * np.log((q + eps) / (m + eps)))
    return float(0.5 * (kl_pm + kl_qm))


def hashed_char_ngram_dist(texts: list[str], n_features: int = 2**18) -> np.ndarray:
    """Get character n-gram distribution using hashing vectorizer."""
    hv = HashingVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        n_features=n_features,
        alternate_sign=False,
        norm=None,
    )
    X = hv.transform(texts)
    v = np.asarray(X.sum(axis=0)).ravel()
    return v


def summarize(df: pd.DataFrame, domain_name: str, text_col: str) -> dict:
    """Compute summary statistics for a domain."""
    feats = df[text_col].map(featurize).apply(pd.Series)
    out = {"domain": domain_name, "n": len(df)}
    for c in feats.columns:
        out[f"{c}_mean"] = float(feats[c].mean())
        out[f"{c}_median"] = float(feats[c].median())
    return out


def main():
    ap = argparse.ArgumentParser(description="Compute domain shift stats for 3 domains")
    ap.add_argument("--text-col", default="text", help="Text column name")
    ap.add_argument("--out-stats", default="results/domain_shift_stats_3domains.csv",
                    help="Output CSV for feature statistics")
    ap.add_argument("--out-js", default="results/domain_shift_js_3domains.csv",
                    help="Output CSV for JSD matrix")
    args = ap.parse_args()

    # Define dataset paths (dedup versions)
    datasets = {
        "sms": ROOT / "dataset" / "dedup" / "processed" / "train.csv",
        "spamassassin": ROOT / "dataset" / "spamassassin" / "dedup" / "processed" / "train.csv",
        "telegram": ROOT / "dataset" / "telegram_spam_ham" / "dedup" / "processed" / "train.csv",
    }

    # Load available datasets
    data = {}
    for name, path in datasets.items():
        if path.exists():
            df = pd.read_csv(path)
            if args.text_col in df.columns:
                data[name] = df
                print(f"[INFO] Loaded {name}: {len(df)} samples")
            else:
                print(f"[WARN] {name}: missing column '{args.text_col}'")
        else:
            print(f"[WARN] {name} not found at {path}")

    if len(data) < 2:
        print("[ERROR] Need at least 2 datasets for comparison")
        return

    # Compute feature statistics for each domain
    stats_rows = []
    for name, df in data.items():
        row = summarize(df, name, args.text_col)
        stats_rows.append(row)
    
    stats_df = pd.DataFrame(stats_rows)
    Path(args.out_stats).parent.mkdir(parents=True, exist_ok=True)
    stats_df.to_csv(args.out_stats, index=False)
    print(f"[OK] Wrote {args.out_stats}")

    # Compute pairwise JSD
    ngram_vecs = {}
    for name, df in data.items():
        texts = df[args.text_col].astype(str).tolist()
        ngram_vecs[name] = hashed_char_ngram_dist(texts)
        print(f"[INFO] Computed n-gram distribution for {name}")

    js_rows = []
    domain_names = list(data.keys())
    for i, name_a in enumerate(domain_names):
        for name_b in domain_names[i+1:]:
            jsd = js_divergence(ngram_vecs[name_a], ngram_vecs[name_b])
            js_rows.append({
                "domain_a": name_a,
                "domain_b": name_b,
                "jsd_char_3to5": jsd,
            })
            print(f"  JSD({name_a}, {name_b}) = {jsd:.6f}")

    js_df = pd.DataFrame(js_rows)
    js_df.to_csv(args.out_js, index=False)
    print(f"[OK] Wrote {args.out_js}")

    # Print summary
    print("\n" + "="*60)
    print("Domain Shift Summary (JSD, lower=more similar)")
    print("="*60)
    for _, row in js_df.iterrows():
        print(f"  {row['domain_a']} <-> {row['domain_b']}: {row['jsd_char_3to5']:.4f}")


if __name__ == "__main__":
    main()
