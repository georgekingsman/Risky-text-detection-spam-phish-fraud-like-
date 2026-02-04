#!/usr/bin/env python
"""
Evasion-Aware Training (EAT) / AttackMix: Build augmented training set.

This script generates augmented training data by applying threat-model-consistent
perturbations to spam samples. The output can be used to train models that are
more robust against evasion attacks.

Usage:
    python -m src.augtrain_build \
        --in_csv data/sms_spam/dedup/processed/train.csv \
        --out_csv data/sms_spam/dedup/processed/train_augmix.csv \
        --seed 0 --aug_prob_spam 0.7 --n_aug 1 \
        --mix "obfuscate:0.7,prompt_injection:0.3"
"""
import argparse
import random
from pathlib import Path

import pandas as pd

from .robustness.perturb import obfuscate, prompt_injection, simple_paraphrase_like


def parse_mix(s: str):
    """Parse mix string like 'obfuscate:0.6,prompt_injection:0.2,paraphrase_like:0.2'"""
    items = []
    for seg in s.split(","):
        name, w = seg.split(":")
        items.append((name.strip(), float(w)))
    total = sum(w for _, w in items)
    items = [(n, w / total) for n, w in items]
    return items


def pick_from_mix(mix, rng):
    """Randomly pick an attack type from the mix based on weights."""
    r = rng.random()
    acc = 0.0
    for name, w in mix:
        acc += w
        if r <= acc:
            return name
    return mix[-1][0]


def apply_attack(name: str, text: str, seed: int) -> str:
    """Apply a specific attack to the text."""
    if name == "obfuscate":
        return obfuscate(text, seed=seed)
    if name == "prompt_injection":
        return prompt_injection(text)
    if name in ("paraphrase_like", "simple_paraphrase_like"):
        return simple_paraphrase_like(text)
    return text


def main():
    ap = argparse.ArgumentParser(description="Build augmented training set for EAT")
    ap.add_argument("--in_csv", required=True, help="Input training CSV")
    ap.add_argument("--out_csv", required=True, help="Output augmented CSV")
    ap.add_argument("--seed", type=int, default=0, help="Random seed")
    ap.add_argument("--text_col", default=None, help="Text column name (auto-detect if None)")
    ap.add_argument("--label_col", default="label", help="Label column name")
    ap.add_argument("--spam_label", default="1", help="Spam label value (str or int)")
    ap.add_argument("--aug_prob_spam", type=float, default=0.7,
                    help="Probability of augmenting each spam sample")
    ap.add_argument("--n_aug", type=int, default=1,
                    help="Number of augmented copies per spam sample")
    ap.add_argument("--mix", default="obfuscate:0.7,prompt_injection:0.3",
                    help="Attack mix weights, e.g. 'obfuscate:0.7,prompt_injection:0.3'")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    df = pd.read_csv(args.in_csv)

    # Auto-detect text column
    if args.text_col is None:
        for cand in ["text", "message", "content", "body"]:
            if cand in df.columns:
                args.text_col = cand
                break
    if args.text_col is None:
        raise ValueError(f"Cannot infer text column. Columns: {list(df.columns)}")

    mix = parse_mix(args.mix)
    print(f"[INFO] Using mix: {mix}")
    print(f"[INFO] Text column: {args.text_col}, Label column: {args.label_col}")

    # Identify spam samples
    spam_mask = df[args.label_col].astype(str) == str(args.spam_label)
    spam_df = df[spam_mask]
    print(f"[INFO] Total samples: {len(df)}, Spam samples: {len(spam_df)}")

    out_rows = []
    aug_counts = {name: 0 for name, _ in mix}
    aug_counts["clean"] = 0

    # Keep all original rows
    for _, row in df.iterrows():
        row_dict = row.to_dict()
        row_dict["aug_type"] = "clean"
        out_rows.append(row_dict)
        aug_counts["clean"] += 1

    # Augment only spam samples
    for idx, (_, row) in enumerate(spam_df.iterrows()):
        if rng.random() > args.aug_prob_spam:
            continue
        for k in range(args.n_aug):
            attack_name = pick_from_mix(mix, rng)
            new_row = row.to_dict()
            original_text = str(row[args.text_col])
            new_row[args.text_col] = apply_attack(
                attack_name, original_text, seed=args.seed + idx * 1000 + k
            )
            new_row["aug_type"] = attack_name
            out_rows.append(new_row)
            aug_counts[attack_name] += 1

    out_df = pd.DataFrame(out_rows)

    # Ensure output directory exists
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)

    print(f"\n[OK] Wrote {len(out_df)} rows to {args.out_csv}")
    print(f"     Original: {len(df)}, Added: {len(out_df) - len(df)}")
    print(f"     Augmentation breakdown:")
    for name, cnt in sorted(aug_counts.items()):
        print(f"       {name}: {cnt}")


if __name__ == "__main__":
    main()
