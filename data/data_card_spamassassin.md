# SpamAssassin Public Corpus Data Card

- Source: Apache SpamAssassin public corpus (old/publiccorpus)
- URL: http://spamassassin.apache.org/old/publiccorpus/
- Instances: Combined easy_ham + spam tarballs, processed into train/val/test (stratified)
- Schema: `text,label,split` where `label` ∈ {0: ham, 1: spam}
- Split: 80% train, 10% val, 10% test; `seed=42`
- Notes: Longer email-style messages with headers and quoted text; suitable for cross-domain evaluation vs SMS.
