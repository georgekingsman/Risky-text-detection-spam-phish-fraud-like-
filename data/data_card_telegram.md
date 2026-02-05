# Telegram Spam or Ham Data Card

## Overview
- **Source**: Kaggle - Telegram Spam or Ham Dataset
- **URL**: https://www.kaggle.com/datasets/mexwell/telegram-spam-or-ham
- **Domain**: Telegram chat/messaging platform
- **Task**: Binary spam classification (ham=0, spam=1)

## Dataset Characteristics
- **Format**: CSV with text messages and spam/ham labels
- **Schema**: `text,label` where `label` ∈ {0: ham, 1: spam}
- **Language**: Primarily English with some multilingual content
- **Split**: 80% train, 10% val, 10% test (stratified, seed=0)

## Why This Dataset?
1. **Modern Chat Domain**: Telegram represents contemporary messaging patterns (2020s), complementing older SMS (2000s) and email (2000s-2010s) corpora.
2. **Domain Shift**: Chat-style communication has distinct characteristics from SMS and email, enabling cross-domain generalization studies.
3. **Real-world Relevance**: Telegram is widely used and increasingly targeted by spammers/scammers.

## Processing Pipeline
1. **Download**: `make telegram_download` (requires Kaggle API)
2. **Standardize**: `make telegram_prepare` → unified `text,label` format
3. **Dedup Split**: `make telegram_dedup` → DedupShift protocol applied

## Expected Domain Characteristics
- Longer messages compared to SMS
- More URLs and links (crypto scams, phishing)
- Modern slang and emoji usage
- Group chat dynamics (broadcast spam patterns)

## Ethical Considerations
- Dataset is publicly available on Kaggle under specified license
- Contains anonymized/synthetic user communications
- Used for research purposes only

## Citation
```
@misc{telegram-spam-ham-kaggle,
  title = {Telegram Spam or Ham Dataset},
  author = {Mexwell},
  year = {2023},
  howpublished = {Kaggle},
  url = {https://www.kaggle.com/datasets/mexwell/telegram-spam-or-ham}
}
```

## Integration Notes
- Dedup report: `results/dedup_report_telegram.csv`
- Domain shift stats: `results/domain_shift_stats_3domains.csv`
- JSD comparison: `results/domain_shift_js_3domains.csv`
