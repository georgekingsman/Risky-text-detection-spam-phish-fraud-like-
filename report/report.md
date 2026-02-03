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

### 2.1 DedupShift protocol (deduplicated splits + leakage control)
We introduce a **deduplicated split protocol** that removes exact and near-duplicates (SimHash) before re-splitting into 80/10/10 stratified splits. This reduces cross-split leakage in template-heavy corpora and yields more conservative estimates.

Artifacts:
- Dedup reports: [results/dedup_report_sms.csv](results/dedup_report_sms.csv), [results/dedup_report_spamassassin.csv](results/dedup_report_spamassassin.csv)
- Dedup splits (SMS): [dataset/dedup/processed/train.csv](dataset/dedup/processed/train.csv), [dataset/dedup/processed/val.csv](dataset/dedup/processed/val.csv), [dataset/dedup/processed/test.csv](dataset/dedup/processed/test.csv)
- Dedup splits (SpamAssassin): [dataset/spamassassin/dedup/processed/train.csv](dataset/spamassassin/dedup/processed/train.csv), [dataset/spamassassin/dedup/processed/val.csv](dataset/spamassassin/dedup/processed/val.csv), [dataset/spamassassin/dedup/processed/test.csv](dataset/spamassassin/dedup/processed/test.csv)

## 3. Models and baselines
We train three main baselines:
1) TF-IDF (word n-grams) + Logistic Regression
2) TF-IDF (character n-grams) + Linear SVM
3) Sentence-Transformer embeddings (MiniLM) + Logistic Regression

We also include a modern neural baseline:
- DistilBERT fine-tuning on deduplicated splits (in-domain + cross-domain).

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

We also provide a compact cross-domain table for paper-ready reporting in [results/cross_domain_table.csv](results/cross_domain_table.csv).

Dedup cross-domain table (protocol-controlled): [results/cross_domain_table_dedup.csv](results/cross_domain_table_dedup.csv).

Neural baseline outputs (DistilBERT, dedup):
- [results/nn_distilbert_sms_train.csv](results/nn_distilbert_sms_train.csv)
- [results/nn_distilbert_spam_train.csv](results/nn_distilbert_spam_train.csv)

## 7. Robustness under perturbations
We generate three perturbation families: obfuscation, paraphrase-like, and prompt-injection style perturbations. The full robustness table (SMS + SpamAssassin) is in [results/robustness.csv](results/robustness.csv).

Findings:

Note: `delta_f1` is computed as $(\text{F1}_{attacked} - \text{F1}_{clean})$. When interpreting drops, use $\text{F1}_{drop} = -\text{delta_f1}$.

### 7.1 Normalization defense (CPU-only)
We add a lightweight normalization defense (Unicode normalization, lowercasing, common obfuscation replacements, punctuation/space cleanup) and re-run robustness on both datasets. The defense reduces the drop for obfuscation on word-based models, indicating that simple preprocessing can mitigate common evasion patterns. Results are included in [results/robustness.csv](results/robustness.csv) with `defense=normalize` and a paired `defense=none` baseline.

![Robustness delta](report/fig_robustness_delta.png)

### 7.2 Multi-seed stability
We run robustness under 5 seeds (0–4) and aggregate mean±std in [results/robustness_agg.csv](results/robustness_agg.csv). Key conclusions (most robust model, defense impact, and worst attack type) remain stable across seeds.

We also visualize mean±std across seeds for in-domain models (defense=none):

![Robustness delta (agg)](report/fig_robustness_delta_agg.png)

### 7.4 DedupShift robustness impact
We re-run robustness under the deduplicated splits and compare against the original protocol:
- Dedup robustness table: [results/robustness_dedup.csv](results/robustness_dedup.csv)
- Dedup robustness plot: [report/fig_robustness_delta_dedup.png](report/fig_robustness_delta_dedup.png)
- Dedup effect (ΔF1 change): [results/dedup_effect.csv](results/dedup_effect.csv)

The dedup robustness table also includes DistilBERT fine-tuning (train-domain specific) under the same attack/defense suite for a neural anchor comparison.

### 7.3 Adversarial baseline (TextAttack)
We run a CPU-only TextAttack baseline using DeepWordBug on 200 sampled test messages per dataset. Results are saved in:
- [results/textattack_sms.csv](results/textattack_sms.csv)
- [results/textattack_spamassassin.csv](results/textattack_spamassassin.csv)

Highlights:
- SMS (TF-IDF word LR): F1 drops from 0.9615 to 0.0000 on the attacked sample; success rate 0.155.
- SpamAssassin (TF-IDF word LR): F1 drops from 0.1508 to 0.0101; success rate 0.140.

These results are sample-based and should be interpreted as a lightweight adversarial sanity check rather than a full-scale attack evaluation.

We further repeat the 200-sample evaluation across multiple seeds (0,1,2) and aggregate mean±std in:
- [results/textattack_seeds/textattack_sms_agg.csv](results/textattack_seeds/textattack_sms_agg.csv)
- [results/textattack_seeds/textattack_spamassassin_agg.csv](results/textattack_seeds/textattack_spamassassin_agg.csv)

## 8. LLM zero-shot baseline
We run local zero-shot LLM classification with rationales:
- SMS test (limit=200) and SpamAssassin test (limit=100)
- Outputs are stored as JSONL in [results/llm_predictions_sms_test.jsonl](results/llm_predictions_sms_test.jsonl) and [results/llm_predictions_spamassassin_test.jsonl](results/llm_predictions_spamassassin_test.jsonl)

Results are appended to [results/results.csv](results/results.csv). As expected for a small local model, F1 is low but provides useful interpretability signals for failure analysis.

## 9. Failure mode analysis
We curated a taxonomy of 12 cases that compare model predictions before and after perturbations in [results/cases.md](results/cases.md). Each case includes original/perturbed text, predictions for word-TFIDF and char-TFIDF, and a failure tag (obfuscation, paraphrase shift, or prompt injection).

Key patterns:
- Word TF-IDF is more sensitive to spacing/symbol obfuscation.
- Character n-grams are comparatively stable under obfuscation but can be affected by paraphrase substitutions.
- Prompt-injection style prefixes can flip decisions for one model while leaving another stable.

## 10. Reproducibility
Key scripts:
- Dataset preparation: [src/data_prep.py](src/data_prep.py), [src/fetch_prepare_spamassassin.py](src/fetch_prepare_spamassassin.py)
- Baselines: [src/train_ml.py](src/train_ml.py), [src/train_embed.py](src/train_embed.py), [src/train_baselines_on_dataset.py](src/train_baselines_on_dataset.py)
- Evaluation and standardization: [src/eval_all.py](src/eval_all.py), [src/standardize_results.py](src/standardize_results.py)
- LLM zero-shot: [src/llm_zero_shot_rationale.py](src/llm_zero_shot_rationale.py)

A unified results table is maintained in [results/results.csv](results/results.csv).

Data QA checks (exact-duplicate rates within splits and across splits) are recorded in [results/duplicate_check.csv](results/duplicate_check.csv).

Clean performance vs. normalization defense trade-off is summarized in [results/defense_tradeoff.csv](results/defense_tradeoff.csv).

Domain shift diagnostics (length, symbol density, URL/email/phone frequency) are summarized in [results/domain_shift_stats.csv](results/domain_shift_stats.csv) with distributional divergence in [results/domain_shift_js.csv](results/domain_shift_js.csv).

## 11. Limitations and next steps
- LLM baselines are limited by local model capacity; larger instruction-tuned models should improve accuracy and rationale quality.
- Cross-domain performance remains low, suggesting opportunities for domain adaptation or contrastive pretraining.
- Robustness suite could be expanded to include adversarial character substitutions and multilingual obfuscation.

---

**Artifacts:**
- Primary results table: [results/results.csv](results/results.csv)
- Dedup results table: [results/results_dedup.csv](results/results_dedup.csv)
- Robustness table: [results/robustness.csv](results/robustness.csv)
- Dedup robustness table: [results/robustness_dedup.csv](results/robustness_dedup.csv)
- Cross-domain summary: [results/robustness_cross_domain.md](results/robustness_cross_domain.md)
- Dedup cross-domain table: [results/cross_domain_table_dedup.csv](results/cross_domain_table_dedup.csv)
- LLM cases: [results/cases.md](results/cases.md)
