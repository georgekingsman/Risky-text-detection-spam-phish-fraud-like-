import argparse, os, random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

def load_sms_uci(raw_path: str) -> pd.DataFrame:
    df = pd.read_csv(raw_path, sep="\t", header=None, names=["label", "text"], quoting=3, engine='python')
    df["label"] = df["label"].map({"ham": 0, "spam": 1}).astype(int)
    df["text"] = df["text"].astype(str)
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sms", choices=["sms"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--raw_path", default="dataset/raw/SMSSpamCollection")
    ap.add_argument("--out_dir", default="dataset/processed")
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    df = load_sms_uci(args.raw_path)
    train, tmp = train_test_split(df, test_size=0.2, random_state=args.seed, stratify=df["label"])
    val, test  = train_test_split(tmp, test_size=0.5, random_state=args.seed, stratify=tmp["label"])

    train.to_csv(f"{args.out_dir}/train.csv", index=False)
    val.to_csv(f"{args.out_dir}/val.csv", index=False)
    test.to_csv(f"{args.out_dir}/test.csv", index=False)
    print("Saved:", len(train), len(val), len(test))

if __name__ == "__main__":
    main()
