#!/usr/bin/env python
"""
Cross-domain evaluation for EAT models.

Tests models trained on one dataset on another dataset to evaluate
generalization under domain shift.
"""
import argparse
import os
from pathlib import Path

import pandas as pd
from joblib import load
from sklearn.metrics import f1_score, precision_score, recall_score

from .robustness.perturb import obfuscate, prompt_injection, simple_paraphrase_like


def load_test_set(data_dir):
    """Load test set from a dataset."""
    return pd.read_csv(Path(data_dir) / "test.csv")


def eval_metrics(y_true, y_pred):
    """Compute metrics."""
    return {
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }


def apply_attack(texts, attack_name, seed=0):
    """Apply attack to texts."""
    results = []
    for i, t in enumerate(texts):
        if attack_name == "obfuscate":
            results.append(obfuscate(t, seed=seed + i))
        elif attack_name == "prompt_injection":
            results.append(prompt_injection(t))
        elif attack_name == "paraphrase_like":
            results.append(simple_paraphrase_like(t))
        else:
            results.append(t)
    return results


def evaluate_sklearn_model(model_path, test_df, attack=None, seed=0):
    """Evaluate a sklearn model."""
    model = load(model_path)
    texts = test_df["text"].tolist()
    if attack:
        texts = apply_attack(texts, attack, seed=seed)
    y_pred = model.predict(texts)
    return eval_metrics(test_df["label"].values, y_pred)


def evaluate_minilm_model(model_path, test_df, attack=None, seed=0):
    """Evaluate a MiniLM model."""
    from sentence_transformers import SentenceTransformer
    
    data = load(model_path)
    clf = data["clf"]
    embed_model = data.get("embed_model", "sentence-transformers/all-MiniLM-L6-v2")
    
    if "encoder" in data:
        st = data["encoder"]
    else:
        st = SentenceTransformer(embed_model)
    
    texts = test_df["text"].tolist()
    if attack:
        texts = apply_attack(texts, attack, seed=seed)
    
    X = st.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    y_pred = clf.predict(X)
    return eval_metrics(test_df["label"].values, y_pred)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-minilm", action="store_true")
    ap.add_argument("--full-threat-model", action="store_true", 
                    help="Run all attacks (obfuscate, paraphrase_like, prompt_injection)")
    args = ap.parse_args()

    os.makedirs("results", exist_ok=True)

    # Dataset paths - now supporting 3 domains
    datasets = {
        "sms": "dataset/dedup/processed",
        "spamassassin": "dataset/spamassassin/dedup/processed",
        "telegram": "dataset/telegram_spam_ham/dedup/processed",
    }
    
    # Filter to only available datasets
    available_datasets = {}
    for name, path in datasets.items():
        if Path(path).exists() and (Path(path) / "test.csv").exists():
            available_datasets[name] = path
        else:
            print(f"[INFO] Dataset {name} not found at {path}, skipping")
    datasets = available_datasets

    # Model configurations
    model_configs = [
        ("tfidf_word_lr", evaluate_sklearn_model),
        ("tfidf_char_svm", evaluate_sklearn_model),
    ]
    if not args.skip_minilm:
        model_configs.append(("minilm_lr", evaluate_minilm_model))

    training_modes = ["", "_eat"]  # Clean and EAT
    
    # Full threat model coverage
    if args.full_threat_model:
        attacks = ["clean", "obfuscate", "paraphrase_like", "prompt_injection"]
    else:
        attacks = ["clean", "obfuscate"]

    results = []

    for train_ds_name, train_ds_path in datasets.items():
        for test_ds_name, test_ds_path in datasets.items():
            test_df = load_test_set(test_ds_path)
            
            for model_type, eval_fn in model_configs:
                for suffix in training_modes:
                    model_prefix = f"{train_ds_name}_dedup"
                    model_path = f"models/{model_prefix}_{model_type}{suffix}.joblib"
                    
                    if not Path(model_path).exists():
                        print(f"[SKIP] {model_path} not found")
                        continue
                    
                    training_mode = "eat" if suffix == "_eat" else "clean"
                    
                    for attack in attacks:
                        atk = attack if attack != "clean" else None
                        try:
                            metrics = eval_fn(model_path, test_df, attack=atk)
                        except Exception as e:
                            print(f"[ERROR] {model_path} on {test_ds_name}: {e}")
                            continue
                        
                        results.append({
                            "train_dataset": train_ds_name,
                            "test_dataset": test_ds_name,
                            "model_type": model_type,
                            "training_mode": training_mode,
                            "attack": attack,
                            "f1": metrics["f1"],
                            "precision": metrics["precision"],
                            "recall": metrics["recall"],
                        })
                        
                        is_cross = "→" if train_ds_name != test_ds_name else "="
                        print(f"{train_ds_name}{is_cross}{test_ds_name} | "
                              f"{model_type}{suffix} | {attack}: F1={metrics['f1']:.4f}")

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv("results/eat_cross_domain.csv", index=False)
    print(f"\n[OK] Saved results/eat_cross_domain.csv")

    # Generate comparison table
    generate_cross_domain_summary(results_df)


def generate_cross_domain_summary(df):
    """Generate cross-domain comparison summary."""
    # Focus on cross-domain scenarios
    cross = df[df["train_dataset"] != df["test_dataset"]]
    
    if cross.empty:
        return
    
    # Compare clean vs EAT training
    summary_rows = []
    for (train_ds, test_ds, model_type, attack), group in cross.groupby(
        ["train_dataset", "test_dataset", "model_type", "attack"]
    ):
        clean_rows = group[group["training_mode"] == "clean"]
        eat_rows = group[group["training_mode"] == "eat"]
        
        if len(clean_rows) > 0 and len(eat_rows) > 0:
            f1_clean = clean_rows.iloc[0]["f1"]
            f1_eat = eat_rows.iloc[0]["f1"]
            summary_rows.append({
                "scenario": f"{train_ds}→{test_ds}",
                "model": model_type,
                "attack": attack,
                "f1_clean_train": f1_clean,
                "f1_eat_train": f1_eat,
                "gain": f1_eat - f1_clean,
            })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv("results/eat_cross_domain_gain.csv", index=False)
    print("[OK] Saved results/eat_cross_domain_gain.csv")
    
    # Print summary
    print("\n" + "="*60)
    print("Cross-Domain EAT Gain Summary")
    print("="*60)
    for _, row in summary_df.iterrows():
        gain_str = f"+{row['gain']:.2%}" if row['gain'] > 0 else f"{row['gain']:.2%}"
        print(f"{row['scenario']} | {row['model']} | {row['attack']}: {gain_str}")


if __name__ == "__main__":
    main()
