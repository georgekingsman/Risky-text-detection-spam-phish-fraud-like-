import argparse
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer

URL_RE = re.compile(r"(https?://|www\.)\S+")
EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
PHONE_RE = re.compile(r"\b(\+?\d[\d\-\s]{6,}\d)\b")


def featurize(text: str) -> dict:
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
    p = p.astype(np.float64)
    q = q.astype(np.float64)
    p = p / (p.sum() + eps)
    q = q / (q.sum() + eps)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log((p + eps) / (m + eps)))
    kl_qm = np.sum(q * np.log((q + eps) / (m + eps)))
    return float(0.5 * (kl_pm + kl_qm))


def hashed_char_ngram_dist(texts: list[str], n_features: int = 2**18) -> np.ndarray:
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
    feats = df[text_col].map(featurize).apply(pd.Series)
    out = {"domain": domain_name, "n": len(df)}
    for c in feats.columns:
        out[f"{c}_mean"] = float(feats[c].mean())
        out[f"{c}_median"] = float(feats[c].median())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="CSV for domain A")
    ap.add_argument("--b", required=True, help="CSV for domain B")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--name-a", default="sms")
    ap.add_argument("--name-b", default="email")
    ap.add_argument("--out", required=True, help="Output CSV")
    ap.add_argument("--out-js", default=None, help="Output JS divergence CSV")
    args = ap.parse_args()

    da = pd.read_csv(args.a)
    db = pd.read_csv(args.b)

    assert args.text_col in da.columns, f"Missing {args.text_col} in {args.a}"
    assert args.text_col in db.columns, f"Missing {args.text_col} in {args.b}"

    row_a = summarize(da, args.name_a, args.text_col)
    row_b = summarize(db, args.name_b, args.text_col)
    pd.DataFrame([row_a, row_b]).to_csv(args.out, index=False)

    va = hashed_char_ngram_dist(da[args.text_col].astype(str).tolist())
    vb = hashed_char_ngram_dist(db[args.text_col].astype(str).tolist())
    jsd = js_divergence(va, vb)

    if args.out_js:
        pd.DataFrame([
            {"domain_a": args.name_a, "domain_b": args.name_b, "jsd_char_3to5": jsd}
        ]).to_csv(args.out_js, index=False)

    print("[OK] wrote:", args.out)
    if args.out_js:
        print("[OK] wrote:", args.out_js)
    print("[INFO] jsd_char_3to5:", jsd)


if __name__ == "__main__":
    main()
