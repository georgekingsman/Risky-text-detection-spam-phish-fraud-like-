#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EAT Ablation Study: Analyze contribution of different components.

Ablations:
1. Attack mix ratios (obfuscate only, prompt_inj only, mixed)
2. Augmentation ratio (10%, 30%, 50%, 70%)
3. Train-only vs Train+Inference defense
4. Per-attack type effectiveness

Output: results/eat_ablation.csv, results/eat_ablation_summary.csv
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]

# Import attack functions from perturb module
from src.robustness.perturb import obfuscate as obfuscate_text, prompt_injection


DATASETS = {
    "sms": ROOT / "dataset/dedup/processed",
    "telegram": ROOT / "dataset/telegram_spam_ham/dedup/processed",
}

ATTACK_MIXES = {
    "obfuscate_only": {"obfuscate": 1.0, "prompt_injection": 0.0},
    "prompt_inj_only": {"obfuscate": 0.0, "prompt_injection": 1.0},
    "balanced": {"obfuscate": 0.5, "prompt_injection": 0.5},
    "weighted_obf": {"obfuscate": 0.7, "prompt_injection": 0.3},  # Default
    "weighted_pi": {"obfuscate": 0.3, "prompt_injection": 0.7},
}

AUG_RATIOS = [0.1, 0.3, 0.5, 0.7]


def load_data(dataset_path: Path):
    """Load train/val/test splits."""
    train = pd.read_csv(dataset_path / "train.csv")
    val = pd.read_csv(dataset_path / "val.csv")
    test = pd.read_csv(dataset_path / "test.csv")
    return train, val, test


def augment_data(texts: list, labels: list, attack_mix: dict, aug_ratio: float) -> tuple:
    """Augment training data with adversarial examples."""
    n_aug = int(len(texts) * aug_ratio)
    
    aug_texts = []
    aug_labels = []
    
    # Only augment spam samples
    spam_idx = [i for i, l in enumerate(labels) if l == 1]
    
    for _ in range(n_aug):
        idx = np.random.choice(spam_idx)
        text = texts[idx]
        label = labels[idx]
        
        # Choose attack based on mix
        r = np.random.random()
        if r < attack_mix.get("obfuscate", 0):
            aug_text = obfuscate_text(text)
        else:
            aug_text = prompt_injection(text)
        
        aug_texts.append(aug_text)
        aug_labels.append(label)
    
    return texts + aug_texts, labels + aug_labels


def train_eat_model(X_train: list, y_train: list, attack_mix: dict, aug_ratio: float,
                    model_type: str = "tfidf_char_svm"):
    """Train EAT model with specified configuration."""
    
    # Augment data
    X_aug, y_aug = augment_data(X_train, y_train, attack_mix, aug_ratio)
    
    # Vectorize
    if model_type == "tfidf_char_svm":
        vec = TfidfVectorizer(max_features=5000, ngram_range=(3, 5), analyzer='char')
        clf = LinearSVC(max_iter=2000, random_state=42, class_weight='balanced')
    else:  # tfidf_word_lr
        vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        clf = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    
    X_vec = vec.fit_transform(X_aug)
    clf.fit(X_vec, y_aug)
    
    return vec, clf


def evaluate_model(vec, clf, X_test: list, y_test: list, attacks: list) -> dict:
    """Evaluate model on clean and attacked data."""
    results = {}
    
    for attack in attacks:
        if attack == "clean":
            X_attacked = X_test
        elif attack == "obfuscate":
            X_attacked = [obfuscate_text(t) for t in X_test]
        elif attack == "prompt_injection":
            X_attacked = [prompt_injection(t) for t in X_test]
        else:
            continue
        
        X_vec = vec.transform(X_attacked)
        y_pred = clf.predict(X_vec)
        results[attack] = f1_score(y_test, y_pred, average='macro')
    
    return results


def run_ablation_attack_mix(dataset_name: str, dataset_path: Path) -> list:
    """Ablation 1: Different attack mixes."""
    print(f"\n  Ablation: Attack Mix Ratios")
    
    train_df, val_df, test_df = load_data(dataset_path)
    X_train = (pd.concat([train_df, val_df])["text"]).tolist()
    y_train = (pd.concat([train_df, val_df])["label"]).tolist()
    X_test = test_df["text"].tolist()
    y_test = test_df["label"].tolist()
    
    results = []
    attacks = ["clean", "obfuscate", "prompt_injection"]
    
    for mix_name, mix_config in ATTACK_MIXES.items():
        print(f"    Mix: {mix_name}")
        
        vec, clf = train_eat_model(X_train, y_train, mix_config, aug_ratio=0.3)
        scores = evaluate_model(vec, clf, X_test, y_test, attacks)
        
        for attack, f1 in scores.items():
            results.append({
                "dataset": dataset_name,
                "ablation": "attack_mix",
                "config": mix_name,
                "attack": attack,
                "f1": round(f1, 4),
            })
    
    return results


def run_ablation_aug_ratio(dataset_name: str, dataset_path: Path) -> list:
    """Ablation 2: Different augmentation ratios."""
    print(f"\n  Ablation: Augmentation Ratio")
    
    train_df, val_df, test_df = load_data(dataset_path)
    X_train = (pd.concat([train_df, val_df])["text"]).tolist()
    y_train = (pd.concat([train_df, val_df])["label"]).tolist()
    X_test = test_df["text"].tolist()
    y_test = test_df["label"].tolist()
    
    results = []
    attacks = ["clean", "obfuscate", "prompt_injection"]
    default_mix = {"obfuscate": 0.7, "prompt_injection": 0.3}
    
    # Also include no augmentation (baseline)
    ratios = [0.0] + AUG_RATIOS
    
    for ratio in ratios:
        print(f"    Ratio: {ratio}")
        
        if ratio == 0.0:
            # Baseline without EAT
            vec = TfidfVectorizer(max_features=5000, ngram_range=(3, 5), analyzer='char')
            clf = LinearSVC(max_iter=2000, random_state=42, class_weight='balanced')
            X_vec = vec.fit_transform(X_train)
            clf.fit(X_vec, y_train)
        else:
            vec, clf = train_eat_model(X_train, y_train, default_mix, aug_ratio=ratio)
        
        scores = evaluate_model(vec, clf, X_test, y_test, attacks)
        
        for attack, f1 in scores.items():
            results.append({
                "dataset": dataset_name,
                "ablation": "aug_ratio",
                "config": f"{int(ratio*100)}%",
                "attack": attack,
                "f1": round(f1, 4),
            })
    
    return results


def run_ablation_train_vs_inference(dataset_name: str, dataset_path: Path) -> list:
    """Ablation 3: Train-time defense vs Inference-time defense."""
    print(f"\n  Ablation: Train vs Inference Defense")
    
    train_df, val_df, test_df = load_data(dataset_path)
    X_train = (pd.concat([train_df, val_df])["text"]).tolist()
    y_train = (pd.concat([train_df, val_df])["label"]).tolist()
    X_test = test_df["text"].tolist()
    y_test = test_df["label"].tolist()
    
    results = []
    attacks = ["clean", "obfuscate", "prompt_injection"]
    default_mix = {"obfuscate": 0.7, "prompt_injection": 0.3}
    
    configs = {
        "baseline": {"train_aug": False, "infer_preprocess": False},
        "train_only": {"train_aug": True, "infer_preprocess": False},
        "infer_only": {"train_aug": False, "infer_preprocess": True},
        "train+infer": {"train_aug": True, "infer_preprocess": True},
    }
    
    for config_name, config in configs.items():
        print(f"    Config: {config_name}")
        
        if config["train_aug"]:
            vec, clf = train_eat_model(X_train, y_train, default_mix, aug_ratio=0.3)
        else:
            vec = TfidfVectorizer(max_features=5000, ngram_range=(3, 5), analyzer='char')
            clf = LinearSVC(max_iter=2000, random_state=42, class_weight='balanced')
            X_vec = vec.fit_transform(X_train)
            clf.fit(X_vec, y_train)
        
        for attack in attacks:
            if attack == "clean":
                X_attacked = X_test
            elif attack == "obfuscate":
                X_attacked = [obfuscate_text(t) for t in X_test]
            elif attack == "prompt_injection":
                X_attacked = [prompt_injection(t) for t in X_test]
            
            # Inference-time preprocessing (simple normalization)
            if config["infer_preprocess"]:
                X_attacked = [normalize_text(t) for t in X_attacked]
            
            X_vec = vec.transform(X_attacked)
            y_pred = clf.predict(X_vec)
            f1 = f1_score(y_test, y_pred, average='macro')
            
            results.append({
                "dataset": dataset_name,
                "ablation": "train_vs_infer",
                "config": config_name,
                "attack": attack,
                "f1": round(f1, 4),
            })
    
    return results


def normalize_text(text: str) -> str:
    """Simple text normalization for inference-time defense."""
    import re
    
    # Remove zero-width characters
    text = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', text)
    
    # Normalize common substitutions
    subs = {
        '0': 'o', '1': 'i', '3': 'e', '4': 'a', '5': 's', 
        '7': 't', '@': 'a', '$': 's', '!': 'i',
    }
    for old, new in subs.items():
        text = text.replace(old, new)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["sms", "telegram"],
                        help="Datasets to evaluate")
    parser.add_argument("--ablations", nargs="+", 
                        default=["attack_mix", "aug_ratio", "train_vs_infer"],
                        help="Ablations to run")
    args = parser.parse_args()
    
    np.random.seed(42)
    
    all_results = []
    
    for dataset in args.datasets:
        if dataset not in DATASETS:
            print(f"⚠️  Dataset {dataset} not found, skipping")
            continue
        
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset}")
        print('='*60)
        
        dataset_path = DATASETS[dataset]
        
        if "attack_mix" in args.ablations:
            results = run_ablation_attack_mix(dataset, dataset_path)
            all_results.extend(results)
        
        if "aug_ratio" in args.ablations:
            results = run_ablation_aug_ratio(dataset, dataset_path)
            all_results.extend(results)
        
        if "train_vs_infer" in args.ablations:
            results = run_ablation_train_vs_inference(dataset, dataset_path)
            all_results.extend(results)
    
    # Save results
    df = pd.DataFrame(all_results)
    output_path = ROOT / "results/eat_ablation.csv"
    df.to_csv(output_path, index=False)
    print(f"\n✅ Results saved to {output_path}")
    
    # Generate summary pivot tables
    print("\n" + "="*60)
    print("📊 EAT Ablation Summary")
    print("="*60)
    
    for ablation in df["ablation"].unique():
        print(f"\n### {ablation.upper()}")
        subset = df[df["ablation"] == ablation]
        pivot = subset.pivot_table(
            index=["dataset", "config"],
            columns="attack",
            values="f1"
        )
        print(pivot.round(4))
    
    # Save summary
    summary_path = ROOT / "results/eat_ablation_summary.md"
    with open(summary_path, 'w') as f:
        f.write("# EAT Ablation Study Results\n\n")
        
        for ablation in df["ablation"].unique():
            f.write(f"## {ablation.replace('_', ' ').title()}\n\n")
            subset = df[df["ablation"] == ablation]
            pivot = subset.pivot_table(
                index=["dataset", "config"],
                columns="attack",
                values="f1"
            )
            f.write(pivot.round(4).to_markdown() + "\n\n")
    
    print(f"\n✅ Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
