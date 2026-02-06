#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Error Taxonomy Analysis: Categorize FP/FN by error type.

Samples 30-50 FP/FN per domain, categorizes by:
- URL obfuscation
- Character substitution (l33t speak)
- Semantic rewriting
- Short text (<20 chars)
- Template marketing
- Code/technical content
- Ambiguous context

Output: results/error_taxonomy.csv, results/error_examples.md
"""
import argparse
import re
import warnings
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]

DATASETS = {
    "sms": ROOT / "dataset/dedup/processed",
    "spamassassin": ROOT / "dataset/spamassassin/dedup/processed",
    "telegram": ROOT / "dataset/telegram_spam_ham/dedup/processed",
}

MODELS = {
    "sms": ROOT / "models/sms_dedup_tfidf_char_svm.joblib",
    "spamassassin": ROOT / "models/spamassassin_dedup_tfidf_char_svm.joblib",
    "telegram": ROOT / "models/telegram_dedup_tfidf_char_svm.joblib",
}


# Error category patterns
ERROR_PATTERNS = {
    "url_obfuscation": [
        r'https?://[^\s]+',
        r'www\.[^\s]+',
        r'bit\.ly|tinyurl|goo\.gl|t\.co',
        r'\[url\]|\[link\]',
    ],
    "char_substitution": [
        r'[0-9]+[a-zA-Z]+[0-9]+',  # Mixed alphanumeric like "fr33" or "w1n"
        r'[@$!][a-zA-Z]',  # Symbol substitution like "@mazing"
        r'(\w)\1{2,}',  # Repeated chars like "freeee"
    ],
    "short_text": [],  # Handled by length check
    "template_marketing": [
        r'(?i)(limited time|act now|click here|free|winner|congratulations)',
        r'(?i)(unsubscribe|opt-out|reply stop)',
        r'(?i)(discount|offer|deal|save \d+%)',
    ],
    "technical_content": [
        r'<[^>]+>',  # HTML tags
        r'\{[^}]+\}',  # JSON/code blocks
        r'import |def |class |function',
        r'error|exception|traceback',
    ],
    "phone_numbers": [
        r'\+?\d{10,}',
        r'\d{3}[-.\s]?\d{3}[-.\s]?\d{4}',
    ],
    "currency_symbols": [
        r'[$£€¥]\s*\d+',
        r'\d+\s*[$£€¥]',
        r'(?i)(bitcoin|btc|eth|usd)',
    ],
}


def classify_error(text: str, label: int, pred: int) -> str:
    """Classify the error type for a misclassified sample."""
    text_lower = str(text).lower()
    
    # Short text
    if len(text) < 20:
        return "short_text"
    
    # Check each pattern category
    for category, patterns in ERROR_PATTERNS.items():
        if category == "short_text":
            continue
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return category
    
    # If FP (predicted spam but actually ham)
    if label == 0 and pred == 1:
        # Check for ambiguous promotional content
        if any(word in text_lower for word in ['meeting', 'schedule', 'reminder', 'update']):
            return "ambiguous_business"
        return "false_positive_other"
    
    # If FN (predicted ham but actually spam)
    if label == 1 and pred == 0:
        # Check for subtle spam
        if len(text) > 200:
            return "long_subtle_spam"
        return "false_negative_other"
    
    return "unknown"


def analyze_errors(dataset_name: str, model_path: Path, dataset_path: Path, 
                   n_samples: int = 50) -> tuple:
    """Analyze errors for a dataset and return categorized results."""
    
    # Load data
    test_df = pd.read_csv(dataset_path / "test.csv")
    train_df = pd.read_csv(dataset_path / "train.csv")
    val_df = pd.read_csv(dataset_path / "val.csv")
    
    # Combine train+val
    train_combined = pd.concat([train_df, val_df])
    
    # Always train fresh to avoid serialization issues
    vec = TfidfVectorizer(max_features=5000, ngram_range=(3, 5), analyzer='char')
    X_train = vec.fit_transform(train_combined["text"])
    
    from sklearn.svm import LinearSVC
    clf = LinearSVC(max_iter=2000, random_state=42, class_weight='balanced')
    clf.fit(X_train, train_combined["label"])
    
    # Predict
    X_test = vec.transform(test_df["text"])
    y_pred = clf.predict(X_test)
    y_true = test_df["label"].values
    
    # Find errors
    errors_idx = (y_pred != y_true)
    error_df = test_df[errors_idx].copy()
    error_df["pred"] = y_pred[errors_idx]
    
    # Sample if too many
    if len(error_df) > n_samples:
        error_df = error_df.sample(n=n_samples, random_state=42)
    
    # Classify errors
    error_categories = []
    examples = []
    
    for idx, row in error_df.iterrows():
        category = classify_error(row["text"], row["label"], row["pred"])
        error_categories.append({
            "dataset": dataset_name,
            "category": category,
            "error_type": "FP" if row["label"] == 0 else "FN",
            "text_length": len(str(row["text"])),
        })
        
        # Store example
        examples.append({
            "dataset": dataset_name,
            "category": category,
            "error_type": "FP" if row["label"] == 0 else "FN",
            "text": str(row["text"])[:200] + "..." if len(str(row["text"])) > 200 else str(row["text"]),
            "true_label": "spam" if row["label"] == 1 else "ham",
            "pred_label": "spam" if row["pred"] == 1 else "ham",
        })
    
    return error_categories, examples


def anonymize_text(text: str) -> str:
    """Anonymize PII in text for paper examples."""
    # Replace phone numbers
    text = re.sub(r'\+?\d{10,}', '[PHONE]', text)
    text = re.sub(r'\d{3}[-.\s]?\d{3}[-.\s]?\d{4}', '[PHONE]', text)
    
    # Replace emails
    text = re.sub(r'[\w.-]+@[\w.-]+\.\w+', '[EMAIL]', text)
    
    # Replace URLs
    text = re.sub(r'https?://[^\s]+', '[URL]', text)
    text = re.sub(r'www\.[^\s]+', '[URL]', text)
    
    # Replace currency amounts
    text = re.sub(r'[$£€¥]\s*\d+(?:,\d{3})*(?:\.\d{2})?', '[AMOUNT]', text)
    
    # Replace names (simple heuristic)
    text = re.sub(r'\b[A-Z][a-z]+\s[A-Z][a-z]+\b', '[NAME]', text)
    
    return text


def generate_markdown_report(examples: list, taxonomy: pd.DataFrame) -> str:
    """Generate markdown report with error taxonomy and examples."""
    
    report = """# Error Taxonomy Analysis

## Summary Statistics

"""
    
    # Summary table
    summary = taxonomy.groupby(["dataset", "category"]).size().unstack(fill_value=0)
    report += summary.to_markdown() + "\n\n"
    
    # Error type distribution
    report += "## Error Type Distribution\n\n"
    
    for dataset in taxonomy["dataset"].unique():
        report += f"### {dataset.upper()}\n\n"
        
        subset = taxonomy[taxonomy["dataset"] == dataset]
        dist = subset.groupby(["error_type", "category"]).size().unstack(fill_value=0)
        report += dist.to_markdown() + "\n\n"
    
    # Representative examples
    report += "## Representative Error Examples\n\n"
    report += "*Note: All examples have been anonymized to remove PII.*\n\n"
    
    # Select diverse examples
    example_df = pd.DataFrame(examples)
    
    for dataset in example_df["dataset"].unique():
        report += f"### {dataset.upper()}\n\n"
        
        subset = example_df[example_df["dataset"] == dataset]
        
        # Get 2-3 examples per category
        for category in subset["category"].unique():
            cat_examples = subset[subset["category"] == category].head(2)
            
            for _, ex in cat_examples.iterrows():
                anon_text = anonymize_text(ex["text"])
                report += f"**{category}** ({ex['error_type']}: predicted {ex['pred_label']}, actual {ex['true_label']})\n"
                report += f"> {anon_text}\n\n"
    
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["sms", "telegram"],
                        help="Datasets to analyze")
    parser.add_argument("--n-samples", type=int, default=50,
                        help="Max samples per dataset")
    args = parser.parse_args()
    
    all_categories = []
    all_examples = []
    
    for dataset in args.datasets:
        if dataset not in DATASETS:
            print(f"⚠️  Dataset {dataset} not found, skipping")
            continue
        
        print(f"\n{'='*60}")
        print(f"Analyzing errors for: {dataset}")
        print('='*60)
        
        model_path = MODELS.get(dataset, ROOT / f"models/{dataset}_dedup_tfidf_char_svm.joblib")
        dataset_path = DATASETS[dataset]
        
        categories, examples = analyze_errors(dataset, model_path, dataset_path, args.n_samples)
        all_categories.extend(categories)
        all_examples.extend(examples)
        
        # Print summary
        cat_counts = Counter([c["category"] for c in categories])
        print(f"  Total errors sampled: {len(categories)}")
        print(f"  Category distribution:")
        for cat, count in cat_counts.most_common():
            print(f"    {cat}: {count}")
    
    # Save taxonomy CSV
    taxonomy_df = pd.DataFrame(all_categories)
    taxonomy_path = ROOT / "results/error_taxonomy.csv"
    taxonomy_df.to_csv(taxonomy_path, index=False)
    print(f"\n✅ Taxonomy saved to {taxonomy_path}")
    
    # Generate markdown report
    report = generate_markdown_report(all_examples, taxonomy_df)
    report_path = ROOT / "results/error_examples.md"
    report_path.write_text(report)
    print(f"✅ Examples saved to {report_path}")
    
    # Print final summary
    print("\n" + "="*60)
    print("📊 Error Taxonomy Summary")
    print("="*60)
    
    pivot = taxonomy_df.pivot_table(
        index="category", 
        columns="dataset", 
        aggfunc="size", 
        fill_value=0
    )
    print(pivot)
    
    # Key insights
    print("\n🔍 Key Insights:")
    top_errors = taxonomy_df["category"].value_counts().head(3)
    for cat, count in top_errors.items():
        pct = count / len(taxonomy_df) * 100
        print(f"  - {cat}: {count} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
