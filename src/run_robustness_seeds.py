import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(seed: int, dataset: str, out_path: Path):
    cmd = [
        'python3', '-m', 'src.robustness.run_robust_final',
        '--seed', str(seed),
        '--dataset', dataset,
        '--defense', 'normalize',
        '--include-baseline',
        '--out', str(out_path)
    ]
    if dataset == 'spamassassin':
        cmd.extend(['--data-dir', 'dataset/spamassassin/processed'])
    subprocess.check_call(cmd, cwd=ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', nargs='+', type=int, default=[0,1,2,3,4])
    args = ap.parse_args()

    out_dir = ROOT / 'results' / 'robustness_seeds'
    out_dir.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        sms_out = out_dir / f'robustness_sms_seed{seed}.csv'
        spam_out = out_dir / f'robustness_spamassassin_seed{seed}.csv'
        run(seed, 'sms_uci', sms_out)
        run(seed, 'spamassassin', spam_out)


if __name__ == '__main__':
    main()
