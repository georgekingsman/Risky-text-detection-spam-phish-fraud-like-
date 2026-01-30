import argparse
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in-dir', default='results/robustness_seeds')
    ap.add_argument('--out', default='results/robustness_agg.csv')
    args = ap.parse_args()

    in_dir = ROOT / args.in_dir
    files = list(in_dir.glob('*.csv'))
    if not files:
        raise SystemExit('No seed robustness files found')

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df['seed_file'] = f.name
        dfs.append(df)

    all_df = pd.concat(dfs, ignore_index=True)
    group_cols = ['attack', 'dataset', 'model', 'defense']

    agg = all_df.groupby(group_cols).agg(
        f1_clean_mean=('f1_clean', 'mean'),
        f1_clean_std=('f1_clean', 'std'),
        f1_attacked_mean=('f1_attacked', 'mean'),
        f1_attacked_std=('f1_attacked', 'std'),
        delta_f1_mean=('delta_f1', 'mean'),
        delta_f1_std=('delta_f1', 'std'),
    ).reset_index()

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(out_path, index=False)
    print('Wrote', out_path)


if __name__ == '__main__':
    main()
