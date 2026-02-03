This folder contains processed datasets used in the benchmark.

Structure:
- data/sms_spam/processed/{train.csv,val.csv,test.csv}
- data/sms_spam/dedup/processed/{train.csv,val.csv,test.csv}
- data/spamassassin/processed/{train.csv,val.csv,test.csv}
- data/spamassassin/dedup/processed/{train.csv,val.csv,test.csv}

CSV schema: `text,label,split` where `label` is 1 for spam, 0 for ham and `split` is one of train/val/test.

To regenerate datasets, run the scripts under `src/`:
- `src/data_prep.py` for UCI SMS dataset
- `src/fetch_prepare_spamassassin.py` for SpamAssassin public corpus

All splits use fixed `seed=42` and stratified 80/10/10 splits.

Deduplicated splits are generated with SimHash-based near-duplicate filtering (see `src/dedup_split.py`) and then re-split with `seed=0`.
