# EAT (Evasion-Aware Training) Results Summary
## Key Findings
- **Obfuscate attack robustness gain**: 15.23% (avg)
- **Clean performance change**: +3.34% (avg)

## Detailed Results by Model
| Model | Attack | F1 (Clean Train) | F1 (EAT Train) | Gain |
|-------|--------|-----------------|----------------|------|
| sms_dedup_tfidf_word_lr | clean | 0.8929 | 0.9402 | +0.0473 |
| sms_dedup_tfidf_word_lr | obfuscate | 0.7647 | 0.8750 | +0.1103 |
| sms_dedup_tfidf_word_lr | paraphrase_like | 0.8624 | 0.9027 | +0.0403 |
| sms_dedup_tfidf_char_svm | clean | 0.9836 | 0.9836 | +0.0000 |
| sms_dedup_tfidf_char_svm | obfuscate | 0.9402 | 0.9752 | +0.0350 |
| sms_dedup_tfidf_char_svm | paraphrase_like | 0.9836 | 0.9836 | +0.0000 |
| spamassassin_dedup_tfidf_word_lr | clean | 0.4723 | 0.5519 | +0.0796 |
| spamassassin_dedup_tfidf_word_lr | obfuscate | 0.3719 | 0.6620 | +0.2901 |
| spamassassin_dedup_tfidf_word_lr | paraphrase_like | 0.4723 | 0.5476 | +0.0753 |
| spamassassin_dedup_tfidf_char_svm | clean | 0.4549 | 0.4615 | +0.0067 |
| spamassassin_dedup_tfidf_char_svm | obfuscate | 0.4689 | 0.6427 | +0.1738 |
| spamassassin_dedup_tfidf_char_svm | paraphrase_like | 0.4549 | 0.4561 | +0.0013 |
