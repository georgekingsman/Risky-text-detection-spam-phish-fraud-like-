#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Seed Evaluation for Neural Baselines.

Runs DistilBERT, MiniLM+LR, and EAT variants across multiple random seeds,
reports mean±std, and performs statistical significance testing (paired t-test).

Output: results/multiseed_results.csv, results/multiseed_significance.csv
"""
import argparse
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]

DATASETS = {
    "sms": ROOT / "dataset/dedup/processed",
    "spamassassin": ROOT / "dataset/spamassassin/dedup/processed",
    "telegram": ROOT / "dataset/telegram_spam_ham/dedup/processed",
}

SEEDS = [0, 1, 2]


def load_data(dataset_path: Path):
    """Load train/val/test splits."""
    train = pd.read_csv(dataset_path / "train.csv")
    val = pd.read_csv(dataset_path / "val.csv")
    test = pd.read_csv(dataset_path / "test.csv")
    return train, val, test


def train_minilm(X_train, y_train, seed: int):
    """Train MiniLM + Logistic Regression with specific seed."""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(X_train.tolist(), show_progress_bar=False)
    clf = LogisticRegression(max_iter=1000, random_state=seed, class_weight='balanced')
    clf.fit(embeddings, y_train)
    return model, clf


def evaluate_minilm(model, clf, X_test, y_test):
    """Evaluate MiniLM model."""
    embeddings = model.encode(X_test.tolist(), show_progress_bar=False)
    y_pred = clf.predict(embeddings)
    return f1_score(y_test, y_pred, average='macro')


def train_distilbert(dataset_name: str, seed: int, output_dir: Path):
    """Train DistilBERT using HuggingFace Trainer."""
    import torch
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        TrainingArguments, Trainer, set_seed
    )
    from datasets import Dataset
    
    set_seed(seed)
    
    dataset_path = DATASETS[dataset_name]
    train_df, val_df, test_df = load_data(dataset_path)
    
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2
    )
    
    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)
    
    train_dataset = Dataset.from_pandas(train_df[["text", "label"]])
    val_dataset = Dataset.from_pandas(val_df[["text", "label"]])
    test_dataset = Dataset.from_pandas(test_df[["text", "label"]])
    
    train_dataset = train_dataset.map(tokenize, batched=True)
    val_dataset = val_dataset.map(tokenize, batched=True)
    test_dataset = test_dataset.map(tokenize, batched=True)
    
    model_output = output_dir / f"distilbert_{dataset_name}_seed{seed}"
    
    training_args = TrainingArguments(
        output_dir=str(model_output),
        num_train_epochs=3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        warmup_steps=100,
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        seed=seed,
        report_to="none",
    )
    
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {"f1": f1_score(labels, predictions, average='macro')}
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()
    
    # Evaluate on test set
    test_results = trainer.evaluate(test_dataset)
    return test_results["eval_f1"]


def run_multiseed_minilm(dataset_name: str, seeds: List[int]) -> Dict:
    """Run MiniLM+LR across multiple seeds."""
    results = {"model": "MiniLM+LR", "dataset": dataset_name, "seeds": {}}
    
    dataset_path = DATASETS[dataset_name]
    train_df, val_df, test_df = load_data(dataset_path)
    
    X_train = pd.concat([train_df, val_df])["text"]
    y_train = pd.concat([train_df, val_df])["label"]
    X_test = test_df["text"]
    y_test = test_df["label"]
    
    f1_scores = []
    for seed in seeds:
        print(f"  MiniLM+LR seed={seed}...")
        model, clf = train_minilm(X_train, y_train, seed)
        f1 = evaluate_minilm(model, clf, X_test, y_test)
        results["seeds"][seed] = f1
        f1_scores.append(f1)
        print(f"    F1 = {f1:.4f}")
    
    results["mean"] = np.mean(f1_scores)
    results["std"] = np.std(f1_scores)
    return results


def run_multiseed_distilbert(dataset_name: str, seeds: List[int], output_dir: Path) -> Dict:
    """Run DistilBERT across multiple seeds."""
    results = {"model": "DistilBERT", "dataset": dataset_name, "seeds": {}}
    
    f1_scores = []
    for seed in seeds:
        print(f"  DistilBERT seed={seed}...")
        f1 = train_distilbert(dataset_name, seed, output_dir)
        results["seeds"][seed] = f1
        f1_scores.append(f1)
        print(f"    F1 = {f1:.4f}")
    
    results["mean"] = np.mean(f1_scores)
    results["std"] = np.std(f1_scores)
    return results


def paired_ttest(scores_a: List[float], scores_b: List[float]) -> Dict:
    """Perform paired t-test between two sets of scores."""
    t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
    return {
        "t_statistic": t_stat,
        "p_value": p_value,
        "significant_005": p_value < 0.05,
        "significant_001": p_value < 0.01,
    }


def bootstrap_ci(scores: List[float], n_bootstrap: int = 1000, ci: float = 0.95) -> tuple:
    """Compute bootstrap confidence interval."""
    np.random.seed(42)
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(scores, size=len(scores), replace=True)
        bootstrap_means.append(np.mean(sample))
    
    alpha = (1 - ci) / 2
    lower = np.percentile(bootstrap_means, alpha * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha) * 100)
    return lower, upper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["sms", "telegram"],
                        help="Datasets to evaluate")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2],
                        help="Random seeds to use")
    parser.add_argument("--skip-distilbert", action="store_true",
                        help="Skip DistilBERT (slow)")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models/multiseed",
                        help="Output directory for models")
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    for dataset in args.datasets:
        if dataset not in DATASETS:
            print(f"⚠️  Dataset {dataset} not found, skipping")
            continue
        
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset}")
        print('='*60)
        
        # MiniLM+LR
        minilm_results = run_multiseed_minilm(dataset, args.seeds)
        all_results.append({
            "dataset": dataset,
            "model": "MiniLM+LR",
            "seed_0": minilm_results["seeds"].get(0),
            "seed_1": minilm_results["seeds"].get(1),
            "seed_2": minilm_results["seeds"].get(2),
            "mean": minilm_results["mean"],
            "std": minilm_results["std"],
        })
        
        # DistilBERT
        if not args.skip_distilbert:
            distilbert_results = run_multiseed_distilbert(dataset, args.seeds, args.output_dir)
            all_results.append({
                "dataset": dataset,
                "model": "DistilBERT",
                "seed_0": distilbert_results["seeds"].get(0),
                "seed_1": distilbert_results["seeds"].get(1),
                "seed_2": distilbert_results["seeds"].get(2),
                "mean": distilbert_results["mean"],
                "std": distilbert_results["std"],
            })
    
    # Save results
    df = pd.DataFrame(all_results)
    output_path = ROOT / "results/multiseed_results.csv"
    df.to_csv(output_path, index=False)
    print(f"\n✅ Results saved to {output_path}")
    
    # Format as mean±std
    print("\n📊 Multi-Seed Results (mean±std):")
    for _, row in df.iterrows():
        print(f"  {row['dataset']} | {row['model']}: {row['mean']:.4f} ± {row['std']:.4f}")
    
    # Statistical significance (if both models available for a dataset)
    significance_results = []
    for dataset in df["dataset"].unique():
        subset = df[df["dataset"] == dataset]
        if len(subset) >= 2:
            minilm = subset[subset["model"] == "MiniLM+LR"].iloc[0]
            distilbert_row = subset[subset["model"] == "DistilBERT"]
            
            if not distilbert_row.empty:
                distilbert = distilbert_row.iloc[0]
                scores_minilm = [minilm["seed_0"], minilm["seed_1"], minilm["seed_2"]]
                scores_distil = [distilbert["seed_0"], distilbert["seed_1"], distilbert["seed_2"]]
                
                ttest = paired_ttest(scores_minilm, scores_distil)
                significance_results.append({
                    "dataset": dataset,
                    "comparison": "MiniLM+LR vs DistilBERT",
                    "t_statistic": ttest["t_statistic"],
                    "p_value": ttest["p_value"],
                    "significant_005": ttest["significant_005"],
                })
    
    if significance_results:
        sig_df = pd.DataFrame(significance_results)
        sig_path = ROOT / "results/multiseed_significance.csv"
        sig_df.to_csv(sig_path, index=False)
        print(f"\n✅ Significance tests saved to {sig_path}")
        print(sig_df.to_string(index=False))


if __name__ == "__main__":
    main()
