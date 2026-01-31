import argparse
import csv
from pathlib import Path

import pandas as pd

from .textattack_baseline import run_textattack


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sms_uci", choices=["sms_uci", "spamassassin"])
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--n-samples", type=int, default=200)
    ap.add_argument("--seeds", default="0,1,2", help="Comma-separated seeds")
    ap.add_argument("--attack", default="deepwordbug")
    ap.add_argument("--out-dir", default="results/textattack_seeds")
    ap.add_argument("--examples", action="store_true", help="Save per-seed examples")
    args = ap.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for seed in seeds:
        out_csv = out_dir / f"textattack_{args.dataset}_seed{seed}.csv"
        examples_out = None
        if args.examples:
            examples_out = out_dir / f"textattack_{args.dataset}_seed{seed}_examples.jsonl"
        rows = run_textattack(
            dataset=args.dataset,
            data_dir=args.data_dir,
            n_samples=args.n_samples,
            seed=seed,
            attack=args.attack,
            out=str(out_csv),
            examples_out=str(examples_out) if examples_out else None,
        )
        for r in rows:
            r["seed"] = seed
        all_rows.extend(rows)

    if not all_rows:
        raise SystemExit("No rows produced. Ensure models exist.")

    agg_rows = []
    df = pd.DataFrame(all_rows)
    for (dataset, model, attack), sub in df.groupby(["dataset", "model", "attack"]):
        agg_rows.append({
            "dataset": dataset,
            "model": model,
            "attack": attack,
            "n_samples": int(sub["n_samples"].iloc[0]),
            "n_seeds": len(sub),
            "success_rate_mean": sub["success_rate"].mean(),
            "success_rate_std": sub["success_rate"].std(ddof=0),
            "f1_clean_mean": sub["f1_clean"].mean(),
            "f1_clean_std": sub["f1_clean"].std(ddof=0),
            "f1_attacked_mean": sub["f1_attacked"].mean(),
            "f1_attacked_std": sub["f1_attacked"].std(ddof=0),
            "delta_f1_mean": sub["delta_f1"].mean(),
            "delta_f1_std": sub["delta_f1"].std(ddof=0),
            "notes": "textattack_seeds",
        })

    out_agg = out_dir / f"textattack_{args.dataset}_agg.csv"
    with out_agg.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "model",
                "attack",
                "n_samples",
                "n_seeds",
                "success_rate_mean",
                "success_rate_std",
                "f1_clean_mean",
                "f1_clean_std",
                "f1_attacked_mean",
                "f1_attacked_std",
                "delta_f1_mean",
                "delta_f1_std",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerows(agg_rows)

    print(f"Wrote {out_agg}")


if __name__ == "__main__":
    main()
