#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strong Baseline Evaluation: RoBERTa-base and DeBERTa-v3-base.

Evaluates transformer-based strong baselines for:
1. In-domain performance
2. Cross-domain transfer (1 pair)
3. Adversarial robustness (attack suite)

Output: results/strong_baselines.csv, results/strong_baselines_robustness.csv
"""
import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, set_seed
)
from datasets import Dataset

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]

DATASETS = {
    "sms": ROOT / "dataset/dedup/processed",
    "telegram": ROOT / "dataset/telegram_spam_ham/dedup/processed",
}

MODELS = {
    "roberta-base": "roberta-base",
    "deberta-v3-base": "microsoft/deberta-v3-base",
}

# Import attack functions
from src.robustness.perturb import obfuscate as obfuscate_text, simple_paraphrase_like as paraphrase_like, prompt_injection


def load_data(dataset_path: Path):
    """Load train/val/test splits."""
    train = pd.read_csv(dataset_path / "train.csv")
    val = pd.read_csv(dataset_path / "val.csv")
    test = pd.read_csv(dataset_path / "test.csv")
    return train, val, test


def train_transformer(model_name: str, model_id: str, dataset_path: Path, 
                      output_dir: Path, seed: int = 42) -> tuple:
    """Train a transformer model and return the trainer + tokenizer."""
    set_seed(seed)
    
    train_df, val_df, test_df = load_data(dataset_path)
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=2
    )
    
    def tokenize(examples):
        return tokenizer(
            examples["text"], 
            truncation=True, 
            padding="max_length", 
            max_length=128
        )
    
    train_dataset = Dataset.from_pandas(train_df[["text", "label"]])
    val_dataset = Dataset.from_pandas(val_df[["text", "label"]])
    
    train_dataset = train_dataset.map(tokenize, batched=True)
    val_dataset = val_dataset.map(tokenize, batched=True)
    
    model_output = output_dir / f"{model_name}_{dataset_path.parent.name}"
    
    training_args = TrainingArguments(
        output_dir=str(model_output),
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        warmup_steps=100,
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        seed=seed,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )
    
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {
            "f1": f1_score(labels, predictions, average='macro'),
            "precision": precision_score(labels, predictions, average='macro'),
            "recall": recall_score(labels, predictions, average='macro'),
        }
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    print(f"  Training {model_name}...")
    trainer.train()
    
    return trainer, tokenizer


def evaluate_on_test(trainer, tokenizer, test_df: pd.DataFrame) -> dict:
    """Evaluate trained model on test set."""
    def tokenize(examples):
        return tokenizer(
            examples["text"], 
            truncation=True, 
            padding="max_length", 
            max_length=128
        )
    
    test_dataset = Dataset.from_pandas(test_df[["text", "label"]])
    test_dataset = test_dataset.map(tokenize, batched=True)
    
    results = trainer.evaluate(test_dataset)
    return {
        "f1": results["eval_f1"],
        "precision": results["eval_precision"],
        "recall": results["eval_recall"],
    }


def evaluate_robustness(trainer, tokenizer, test_df: pd.DataFrame, attacks: list) -> list:
    """Evaluate model robustness under different attacks."""
    results = []
    
    for attack_name in attacks:
        print(f"    Attack: {attack_name}")
        
        if attack_name == "clean":
            attacked_texts = test_df["text"].tolist()
        elif attack_name == "obfuscate":
            attacked_texts = [obfuscate_text(t) for t in test_df["text"]]
        elif attack_name == "paraphrase_like":
            attacked_texts = [paraphrase_like(t) for t in test_df["text"]]
        elif attack_name == "prompt_injection":
            attacked_texts = [prompt_injection(t) for t in test_df["text"]]
        else:
            continue
        
        attacked_df = pd.DataFrame({"text": attacked_texts, "label": test_df["label"].values})
        
        def tokenize(examples):
            return tokenizer(
                examples["text"], 
                truncation=True, 
                padding="max_length", 
                max_length=128
            )
        
        attacked_dataset = Dataset.from_pandas(attacked_df)
        attacked_dataset = attacked_dataset.map(tokenize, batched=True)
        
        eval_results = trainer.evaluate(attacked_dataset)
        results.append({
            "attack": attack_name,
            "f1": eval_results["eval_f1"],
        })
    
    return results


def measure_latency(trainer, tokenizer, test_texts: list, n_samples: int = 100) -> float:
    """Measure average inference latency."""
    import torch
    
    model = trainer.model
    device = next(model.parameters()).device
    
    # Warm up
    sample_texts = test_texts[:10]
    inputs = tokenizer(sample_texts, return_tensors="pt", truncation=True, 
                       padding=True, max_length=128).to(device)
    with torch.no_grad():
        _ = model(**inputs)
    
    # Measure
    sample_texts = test_texts[:n_samples]
    start = time.time()
    for text in sample_texts:
        inputs = tokenizer([text], return_tensors="pt", truncation=True, 
                          padding=True, max_length=128).to(device)
        with torch.no_grad():
            _ = model(**inputs)
    end = time.time()
    
    return (end - start) / n_samples * 1000  # ms per message


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["roberta-base"],
                        choices=list(MODELS.keys()),
                        help="Models to evaluate")
    parser.add_argument("--train-dataset", default="sms",
                        help="Dataset to train on")
    parser.add_argument("--cross-domain", default="telegram",
                        help="Dataset for cross-domain evaluation")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models/strong_baselines",
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    robustness_results = []
    
    attacks = ["clean", "obfuscate", "paraphrase_like", "prompt_injection"]
    
    for model_name in args.models:
        model_id = MODELS[model_name]
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print('='*60)
        
        # Train on primary dataset
        train_path = DATASETS[args.train_dataset]
        trainer, tokenizer = train_transformer(
            model_name, model_id, train_path, args.output_dir, args.seed
        )
        
        # Count parameters
        n_params = sum(p.numel() for p in trainer.model.parameters())
        
        # In-domain evaluation
        print(f"\n  In-domain evaluation ({args.train_dataset})...")
        _, _, test_df = load_data(train_path)
        in_domain = evaluate_on_test(trainer, tokenizer, test_df)
        
        # Measure latency
        latency = measure_latency(trainer, tokenizer, test_df["text"].tolist())
        
        all_results.append({
            "model": model_name,
            "train_dataset": args.train_dataset,
            "test_dataset": args.train_dataset,
            "type": "in-domain",
            "f1": round(in_domain["f1"], 4),
            "precision": round(in_domain["precision"], 4),
            "recall": round(in_domain["recall"], 4),
            "latency_ms": round(latency, 2),
            "params_M": round(n_params / 1e6, 1),
        })
        
        print(f"    F1: {in_domain['f1']:.4f}, Latency: {latency:.2f}ms, Params: {n_params/1e6:.1f}M")
        
        # Cross-domain evaluation
        if args.cross_domain and args.cross_domain in DATASETS:
            print(f"\n  Cross-domain evaluation ({args.train_dataset}→{args.cross_domain})...")
            cross_path = DATASETS[args.cross_domain]
            _, _, cross_test_df = load_data(cross_path)
            cross_domain = evaluate_on_test(trainer, tokenizer, cross_test_df)
            
            all_results.append({
                "model": model_name,
                "train_dataset": args.train_dataset,
                "test_dataset": args.cross_domain,
                "type": "cross-domain",
                "f1": round(cross_domain["f1"], 4),
                "precision": round(cross_domain["precision"], 4),
                "recall": round(cross_domain["recall"], 4),
                "latency_ms": round(latency, 2),
                "params_M": round(n_params / 1e6, 1),
            })
            
            print(f"    F1: {cross_domain['f1']:.4f}")
        
        # Robustness evaluation (in-domain)
        print(f"\n  Robustness evaluation...")
        robust = evaluate_robustness(trainer, tokenizer, test_df, attacks)
        for r in robust:
            robustness_results.append({
                "model": model_name,
                "dataset": args.train_dataset,
                "attack": r["attack"],
                "f1": round(r["f1"], 4),
            })
    
    # Save results
    df = pd.DataFrame(all_results)
    output_path = ROOT / "results/strong_baselines.csv"
    df.to_csv(output_path, index=False)
    print(f"\n✅ Results saved to {output_path}")
    print(df.to_string(index=False))
    
    robust_df = pd.DataFrame(robustness_results)
    robust_path = ROOT / "results/strong_baselines_robustness.csv"
    robust_df.to_csv(robust_path, index=False)
    print(f"\n✅ Robustness results saved to {robust_path}")
    print(robust_df.to_string(index=False))
    
    # Summary comparison with TF-IDF baselines
    print("\n" + "="*60)
    print("📊 Comparison Summary")
    print("="*60)
    print(f"{'Model':<20} {'In-Domain F1':<15} {'Latency (ms)':<15} {'Params (M)':<12}")
    print("-"*60)
    print(f"{'TF-IDF Char SVM':<20} {'0.986':<15} {'0.04':<15} {'<0.1':<12}")
    print(f"{'MiniLM+LR':<20} {'0.923':<15} {'24.55':<15} {'22.7':<12}")
    for _, row in df[df["type"] == "in-domain"].iterrows():
        print(f"{row['model']:<20} {row['f1']:<15.4f} {row['latency_ms']:<15.2f} {row['params_M']:<12.1f}")


if __name__ == "__main__":
    main()
