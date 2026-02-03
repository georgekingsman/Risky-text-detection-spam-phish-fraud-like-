import argparse
import hashlib
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def canonicalize(
    s: str,
    *,
    lowercase: bool = True,
    unicode_nfkc: bool = True,
    collapse_ws: bool = True,
    strip_punct: bool = False,
    normalize_digits: bool = False,
) -> str:
    if s is None:
        return ""
    if unicode_nfkc:
        s = unicodedata.normalize("NFKC", s)
    if lowercase:
        s = s.lower()
    if normalize_digits:
        s = re.sub(r"\d+", "0", s)
    if strip_punct:
        s = re.sub(r"[^a-z0-9\s]+", " ", s)
    if collapse_ws:
        s = re.sub(r"\s+", " ", s).strip()
    return s


def stable_hash64(token: str) -> int:
    h = hashlib.md5(token.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "little", signed=False)


def tokenize_for_simhash(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", s)


def shingles(tokens: list[str], k: int = 3) -> list[str]:
    if len(tokens) <= k:
        return [" ".join(tokens)] if tokens else [""]
    return [" ".join(tokens[i : i + k]) for i in range(len(tokens) - k + 1)]


def simhash64(text: str, shingle_k: int = 3) -> int:
    toks = tokenize_for_simhash(text)
    feats = shingles(toks, k=shingle_k)
    v = np.zeros(64, dtype=np.int32)
    for f in feats:
        h = stable_hash64(f)
        for i in range(64):
            v[i] += 1 if ((h >> i) & 1) else -1
    out = 0
    for i in range(64):
        if v[i] >= 0:
            out |= (1 << i)
    return out


def hamming64(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def exact_dedup(keys: list[str]) -> tuple[np.ndarray, int]:
    seen = {}
    keep = np.ones(len(keys), dtype=bool)
    removed = 0
    for i, k in enumerate(keys):
        if k in seen:
            keep[i] = False
            removed += 1
        else:
            seen[k] = i
    return keep, removed


def near_dedup_simhash(
    canon_texts: list[str],
    keep_mask: np.ndarray,
    *,
    h_thresh: int = 3,
    band_bits: int = 16,
    shingle_k: int = 3,
) -> tuple[np.ndarray, int]:
    assert 64 % band_bits == 0
    n_bands = 64 // band_bits
    mask = (1 << band_bits) - 1

    buckets = [defaultdict(list) for _ in range(n_bands)]
    sigs = {}
    removed = 0
    keep2 = keep_mask.copy()

    for i, (t, keep) in enumerate(zip(canon_texts, keep_mask)):
        if not keep:
            continue
        sig = simhash64(t, shingle_k=shingle_k)
        sigs[i] = sig

        cand = set()
        for b in range(n_bands):
            key = (sig >> (b * band_bits)) & mask
            cand.update(buckets[b][key])

        is_dup = False
        for j in cand:
            if hamming64(sig, sigs[j]) <= h_thresh:
                is_dup = True
                break

        if is_dup:
            keep2[i] = False
            removed += 1
            continue

        for b in range(n_bands):
            key = (sig >> (b * band_bits)) & mask
            buckets[b][key].append(i)

    return keep2, removed


def stratified_split(labels: np.ndarray, seed: int, ratios=(0.8, 0.1, 0.1)) -> np.ndarray:
    assert abs(sum(ratios) - 1.0) < 1e-9
    n = len(labels)
    idx = np.arange(n)

    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=(1 - ratios[0]), random_state=seed)
    train_idx, rest_idx = next(sss1.split(idx, labels))

    rest_labels = labels[rest_idx]
    val_size = ratios[1] / (ratios[1] + ratios[2])

    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=(1 - val_size), random_state=seed)
    val_rel, test_rel = next(sss2.split(np.arange(len(rest_idx)), rest_labels))

    val_idx = rest_idx[val_rel]
    test_idx = rest_idx[test_rel]

    split = np.array(["train"] * n, dtype=object)
    split[val_idx] = "val"
    split[test_idx] = "test"
    return split


def cross_split_overlap(canon_text: np.ndarray, split: np.ndarray) -> dict:
    out = {}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        sa = set(canon_text[split == a])
        sb = set(canon_text[split == b])
        inter = sa.intersection(sb)
        out[f"overlap_{a}_{b}"] = len(inter)
    return out


def read_splits(data_dir: Path) -> pd.DataFrame:
    frames = []
    for split in ["train", "val", "test"]:
        p = data_dir / f"{split}.csv"
        df = pd.read_csv(p)
        df["_orig_split"] = split
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="Directory containing train/val/test CSVs")
    ap.add_argument("--out-dir", required=True, help="Output directory for dedup splits")
    ap.add_argument("--report", required=True, help="Output report CSV")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--strip-punct", action="store_true")
    ap.add_argument("--normalize-digits", action="store_true")
    ap.add_argument("--near", action="store_true", help="Enable near-duplicate filtering (SimHash)")
    ap.add_argument("--h-thresh", type=int, default=3)
    ap.add_argument("--band-bits", type=int, default=16)
    ap.add_argument("--shingle-k", type=int, default=3)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = read_splits(data_dir)
    assert args.text_col in df.columns, f"Missing text col: {args.text_col}"
    assert args.label_col in df.columns, f"Missing label col: {args.label_col}"

    canon = df[args.text_col].astype(str).map(
        lambda x: canonicalize(
            x,
            strip_punct=args.strip_punct,
            normalize_digits=args.normalize_digits,
        )
    )
    df["canon_text"] = canon

    n_in = len(df)

    orig_overlap = cross_split_overlap(df["canon_text"].to_numpy(), df["_orig_split"].to_numpy())

    keep1, n_exact_removed = exact_dedup(df["canon_text"].tolist())
    keep2 = keep1
    n_near_removed = 0
    if args.near:
        keep2, n_near_removed = near_dedup_simhash(
            df["canon_text"].tolist(),
            keep1,
            h_thresh=args.h_thresh,
            band_bits=args.band_bits,
            shingle_k=args.shingle_k,
        )

    df2 = df[keep2].copy().reset_index(drop=True)
    n_out = len(df2)

    labels = df2[args.label_col].to_numpy()
    split = stratified_split(labels, seed=args.seed)
    df2["split"] = split

    dedup_overlap = cross_split_overlap(df2["canon_text"].to_numpy(), df2["split"].to_numpy())

    for split_name in ["train", "val", "test"]:
        df2[df2["split"] == split_name].drop(columns=["canon_text"], errors="ignore").to_csv(
            out_dir / f"{split_name}.csv", index=False
        )

    report = {
        "data_dir": str(data_dir),
        "out_dir": str(out_dir),
        "n_in": n_in,
        "n_exact_removed": n_exact_removed,
        "n_near_removed": n_near_removed,
        "n_out": n_out,
        "seed": args.seed,
        "strip_punct": int(args.strip_punct),
        "normalize_digits": int(args.normalize_digits),
        "near_enabled": int(args.near),
        "h_thresh": args.h_thresh,
        "band_bits": args.band_bits,
        "shingle_k": args.shingle_k,
        **{f"orig_{k}": v for k, v in orig_overlap.items()},
        **{f"dedup_{k}": v for k, v in dedup_overlap.items()},
    }

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([report]).to_csv(args.report, index=False)

    print("[OK] wrote:", out_dir)
    print("[OK] report:", args.report)


if __name__ == "__main__":
    main()
