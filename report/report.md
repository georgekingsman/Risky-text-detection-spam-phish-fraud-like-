# Risky Text Detection Benchmark Report

## 1. Motivation and scope
This project builds a reproducible benchmark for risky text detection (spam/phish/fraud-like) with an explicit focus on **cross-domain generalization** and **robustness under perturbations**. We combine strong classical baselines, embedding baselines, and LLM zero-shot baselines, and provide consistent evaluation artifacts across datasets.

## 2. Datasets and splits
We evaluate on two public datasets with fixed 80/10/10 stratified splits and consistent schema (`text,label,split`).

- SMS (UCI SMSSpamCollection): short, informal mobile messages. See [data/data_card_sms.md](data/data_card_sms.md).
- SpamAssassin public corpus: email-like messages with longer context. See [data/data_card_spamassassin.md](data/data_card_spamassassin.md).

Processed data lives under:
- [data/sms_spam/processed/train.csv](data/sms_spam/processed/train.csv)
- [data/sms_spam/processed/val.csv](data/sms_spam/processed/val.csv)
- [data/sms_spam/processed/test.csv](data/sms_spam/processed/test.csv)
- [data/spamassassin/processed/train.csv](data/spamassassin/processed/train.csv)
- [data/spamassassin/processed/val.csv](data/spamassassin/processed/val.csv)
- [data/spamassassin/processed/test.csv](data/spamassassin/processed/test.csv)

## 3. Models and baselines
We train three main baselines:
1) TF-IDF (word n-grams) + Logistic Regression
2) TF-IDF (character n-grams) + Linear SVM
3) Sentence-Transformer embeddings (MiniLM) + Logistic Regression

We also include:
- LLM zero-shot classification with rationales (local `distilgpt2`)
- LLM-as-feature: LLM rationales appended to text before TF-IDF+LR

## 4. Evaluation protocol and schema
All results are recorded in a standardized schema:
`dataset, split, model, seed, f1, precision, recall, roc_auc, notes`

The consolidated table is stored at [results/results.csv](results/results.csv). This file contains in-domain results, cross-domain results, and LLM baselines with consistent field names.

## 5. In-domain results (SMS)
On SMS test split, classical baselines are strong:
- Character TF-IDF + SVM reaches F1 ≈ 0.973.
- Word TF-IDF + LR reaches F1 ≈ 0.921.
- MiniLM + LR reaches F1 ≈ 0.952.

These values are visible in [results/results.csv](results/results.csv).

## 6. Cross-domain generalization
We evaluate cross-domain transfer in both directions:
- Train on SMS, test on SpamAssassin
- Train on SpamAssassin, test on SMS

The cross-domain summary is documented in [results/robustness_cross_domain.md](results/robustness_cross_domain.md). Key observation: **F1 drops substantially under domain shift**, with character n-grams retaining comparatively better performance than word TF-IDF in SMS → SpamAssassin transfer.

## 7. Robustness under perturbations
We generate three perturbation families: obfuscation, paraphrase-like, and prompt-injection style perturbations. The full robustness table is in [results/robustness.csv](results/robustness.csv).

Findings:
- Character n-grams are most resilient to obfuscation (smallest ΔF1).
- Word TF-IDF suffers larger drops under obfuscation and paraphrase-like perturbations.
- MiniLM shows moderate robustness, especially on paraphrase-like changes.

## 8. LLM zero-shot baseline
We run local zero-shot LLM classification with rationales:
- SMS test (limit=200) and SpamAssassin test (limit=100)
- Outputs are stored as JSONL in [results/llm_predictions_sms_test.jsonl](results/llm_predictions_sms_test.jsonl) and [results/llm_predictions_spamassassin_test.jsonl](results/llm_predictions_spamassassin_test.jsonl)

Results are appended to [results/results.csv](results/results.csv). As expected for a small local model, F1 is low but provides useful interpretability signals for failure analysis.

## 9. Failure mode analysis
We curated 10 error cases from LLM zero-shot outputs in [results/cases.md](results/cases.md). Common failure patterns include:
- Over-triggering on informal or playful language
- Poor handling of long, structured email-style messages
- Prompt-echo artifacts in rationale outputs

These failure modes motivate future work on domain-adaptive prompting or larger instruction-tuned models.

## 10. Reproducibility
Key scripts:
- Dataset preparation: [src/data_prep.py](src/data_prep.py), [src/fetch_prepare_spamassassin.py](src/fetch_prepare_spamassassin.py)
- Baselines: [src/train_ml.py](src/train_ml.py), [src/train_embed.py](src/train_embed.py), [src/train_baselines_on_dataset.py](src/train_baselines_on_dataset.py)
- Evaluation and standardization: [src/eval_all.py](src/eval_all.py), [src/standardize_results.py](src/standardize_results.py)
- LLM zero-shot: [src/llm_zero_shot_rationale.py](src/llm_zero_shot_rationale.py)

A unified results table is maintained in [results/results.csv](results/results.csv).

## 11. Limitations and next steps
- LLM baselines are limited by local model capacity; larger instruction-tuned models should improve accuracy and rationale quality.
- Cross-domain performance remains low, suggesting opportunities for domain adaptation or contrastive pretraining.
- Robustness suite could be expanded to include adversarial character substitutions and multilingual obfuscation.

---

**Artifacts:**
- Primary results table: [results/results.csv](results/results.csv)
- Robustness table: [results/robustness.csv](results/robustness.csv)
- Cross-domain summary: [results/robustness_cross_domain.md](results/robustness_cross_domain.md)
- LLM cases: [results/cases.md](results/cases.md)
