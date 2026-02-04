#!/usr/bin/env python
"""
Threshold calibration for EAT models on cross-domain evaluation.

This script calibrates classification thresholds on validation set
to address the common cross-domain issue of high recall but low precision.
Uses Youden's J statistic (sensitivity + specificity - 1) for optimal threshold.
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import f1_score, precision_score, recall_score, roc_curve

from .robustness.perturb import obfuscate, prompt_injection, simple_paraphrase_like


def load_split(data_dir, split="val"):
    """Load a specific split."""
    return pd.read_csv(Path(data_dir) / f"{split}.csv")


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


def get_proba_sklearn(model, texts):
    """Get probability scores from sklearn model."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(texts)[:, 1]
    elif hasattr(model, "decision_function"):
        # For SVM, use decision function and sigmoid
        scores = model.decision_function(texts)
        return 1 / (1 + np.exp(-scores))
    else:
        # Fallback to binary predictions
        return model.predict(texts).astype(float)


def get_proba_minilm(model_data, texts):
    """Get probability scores from MiniLM model."""
    from sentence_transformers import SentenceTransformer
    
    clf = model_data["clf"]
    embed_model = model_data.get("embed_model", "sentence-transformers/all-MiniLM-L6-v2")
    
    if "encoder" in model_data:
        st = model_data["encoder"]
    else:
        st = SentenceTransformer(embed_model)
    
    X = st.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)[:, 1]
    else:
        return clf.predict(X).astype(float)


def find_optimal_threshold(y_true, y_proba, method="youden"):
    """Find optimal threshold using specified method."""
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    
    if method == "youden":
        # Youden's J = sensitivity + specificity - 1 = TPR - FPR
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
    elif method == "f1":
        # Find threshold that maximizes F1
        f1_scores = []
        for thresh in thresholds:
            y_pred = (y_proba >= thresh).astype(int)
            f1_scores.append(f1_score(y_true, y_pred, zero_division=0))
        optimal_idx = np.argmax(f1_scores)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return thresholds[optimal_idx]


def eval_with_threshold(y_true, y_proba, threshold=0.5):
    """Evaluate with a specific threshold."""
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "threshold": threshold,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-minilm", action="store_true")
    ap.add_argument("--method", default="youden", choices=["youden", "f1"],
                    help="Threshold optimization method")
    args = ap.parse_args()

    os.makedirs("results", exist_ok=True)

    datasets = {
        "sms": "data/sms_spam/dedup/processed",
        "spamassassin": "data/spamassassin/dedup/processed",
    }

    model_configs = [
        ("tfidf_word_lr", get_proba_sklearn, load),
        ("tfidf_char_svm", get_proba_sklearn, load),
    ]
    if not args.skip_minilm:
        model_configs.append(("minilm_lr", get_proba_minilm, load))

    training_modes = ["", "_eat"]
    attacks = ["clean", "obfuscate"]

    results = []
    calibration_info = []

    for train_ds_name, train_ds_path in datasets.items():
        for test_ds_name, test_ds_path in datasets.items():
            if train_ds_name == test_ds_name:
                continue  # Only cross-domain
            
            val_df = load_split(test_ds_path, "val")
            test_df = load_split(test_ds_path, "test")
            
            for model_type, get_proba_fn, load_fn in model_configs:
                for suffix in training_modes:
                    model_prefix = f"{train_ds_name}_dedup"
                    model_path = f"models/{model_prefix}_{model_type}{suffix}.joblib"
                    
                    if not Path(model_path).exists():
                        print(f"[SKIP] {model_path} not found")
                        continue
                    
                    model = load_fn(model_path)
                    training_mode = "eat" if suffix == "_eat" else "clean"
                    
                    for attack in attacks:
                        atk = attack if attack != "clean" else None
                        
                        # Get val probabilities for calibration
                        val_texts = val_df["text"].tolist()
                        if atk:
                            val_texts = apply_attack(val_texts, atk)
                        
                        try:
                            if model_type == "minilm_lr":
                                val_proba = get_proba_fn(model, val_texts)
                            else:
                                val_proba = get_proba_fn(model, val_texts)
                        except Exception as e:
                            print(f"[ERROR] {model_path}: {e}")
                            continue
                        
                        # Find optimal threshold on val
                        optimal_thresh = find_optimal_threshold(
                            val_df["label"].values, val_proba, method=args.method
                        )
                        
                        # Evaluate on test with default (0.5) and calibrated threshold
                        test_texts = test_df["text"].tolist()
                        if atk:
                            test_texts = apply_attack(test_texts, atk)
                        
                        if model_type == "minilm_lr":
                            test_proba = get_proba_fn(model, test_texts)
                        else:
                            test_proba = get_proba_fn(model, test_texts)
                        
                        # Default threshold (0.5)
                        default_metrics = eval_with_threshold(
                            test_df["label"].values, test_proba, 0.5
                        )
                        
                        # Calibrated threshold
                        calibrated_metrics = eval_with_threshold(
                            test_df["label"].values, test_proba, optimal_thresh
                        )
                        
                        scenario = f"{train_ds_name}→{test_ds_name}"
                        
                        results.append({
                            "scenario": scenario,
                            "model": model_type,
                            "training": training_mode,
                            "attack": attack,
                            "calibration": "default",
                            "threshold": 0.5,
                            "f1": default_metrics["f1"],
                            "precision": default_metrics["precision"],
                            "recall": default_metrics["recall"],
                        })
                        
                        results.append({
                            "scenario": scenario,
                            "model": model_type,
                            "training": training_mode,
                            "attack": attack,
                            "calibration": "calibrated",
                            "threshold": optimal_thresh,
                            "f1": calibrated_metrics["f1"],
                            "precision": calibrated_metrics["precision"],
                            "recall": calibrated_metrics["recall"],
                        })
                        
                        # Log calibration improvement
                        f1_gain = calibrated_metrics["f1"] - default_metrics["f1"]
                        print(f"{scenario} | {model_type}{suffix} | {attack}: "
                              f"thresh {optimal_thresh:.3f}, F1 gain: {f1_gain:+.4f}")
                        
                        calibration_info.append({
                            "scenario": scenario,
                            "model": model_type,
                            "training": training_mode,
                            "attack": attack,
                            "optimal_threshold": optimal_thresh,
                            "f1_default": default_metrics["f1"],
                            "f1_calibrated": calibrated_metrics["f1"],
                            "f1_gain": f1_gain,
                            "precision_default": default_metrics["precision"],
                            "precision_calibrated": calibrated_metrics["precision"],
                            "recall_default": default_metrics["recall"],
                            "recall_calibrated": calibrated_metrics["recall"],
                        })

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv("results/eat_cross_domain_calibrated.csv", index=False)
    print(f"\n[OK] Saved results/eat_cross_domain_calibrated.csv")

    calibration_df = pd.DataFrame(calibration_info)
    calibration_df.to_csv("results/eat_calibration_summary.csv", index=False)
    print("[OK] Saved results/eat_calibration_summary.csv")

    # Print summary
    print("\n" + "="*60)
    print("Calibration Summary (EAT models only)")
    print("="*60)
    eat_cal = calibration_df[calibration_df["training"] == "eat"]
    if not eat_cal.empty:
        avg_f1_gain = eat_cal["f1_gain"].mean()
        print(f"Average F1 gain from calibration: {avg_f1_gain:+.4f}")
        print("\nPer-scenario gains:")
        for _, row in eat_cal.iterrows():
            print(f"  {row['scenario']} | {row['model']} | {row['attack']}: "
                  f"F1 {row['f1_default']:.3f} → {row['f1_calibrated']:.3f} "
                  f"({row['f1_gain']:+.3f})")


if __name__ == "__main__":
    main()
