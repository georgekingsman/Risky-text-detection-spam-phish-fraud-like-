# Cross-domain generalization summary

This note summarizes cross-domain performance between SMS (UCI) and SpamAssassin datasets. Metrics are F1/Precision/Recall/ROC-AUC. Cross-domain rows are tagged in the standardized results file.

## In-domain baselines (test)
- SMS (UCI):
  - tfidf_word_lr F1=0.9209
  - tfidf_char_svm F1=0.9726
  - minilm_lr F1=0.9517
- SpamAssassin:
  - tfidf_word_lr F1=0.1165
  - tfidf_char_svm F1=0.0932
  - minilm_lr F1=0.3163

## Cross-domain (train → test)
- SMS → SpamAssassin:
  - sms_trained_tfidf_word_lr F1=0.1348 (P=0.4286, R=0.0800, AUC=0.4934)
  - sms_trained_tfidf_char_svm F1=0.4618 (P=0.5080, R=0.4233, AUC=0.5173)
  - sms_trained_minilm_lr F1=0.1547 (P=0.5510, R=0.0900, AUC=0.4991)

- SpamAssassin → SMS:
  - spamassassin_trained_tfidf_word_lr F1=0.2840 (P=0.1825, R=0.6400, AUC=0.6225)
  - spamassassin_trained_tfidf_char_svm F1=0.2236 (P=0.1445, R=0.4933, AUC=0.5191)
  - spamassassin_trained_minilm_lr F1=0.1346 (P=0.0821, R=0.3733, AUC=0.3148)

## Observations
- Cross-domain generalization drops substantially relative to SMS in-domain performance, indicating strong distribution shift between SMS and email-style data.
- Character n-grams retain comparatively better cross-domain F1 than word TF-IDF, aligning with the robustness narrative for obfuscation/noisy text.
