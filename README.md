# Risk Text Robust Benchmark

Reproducible benchmarks for risky-text detection (spam/phish/fraud-like) with cross-domain evaluation and robustness suites.

## Quickstart (CPU)
```bash
make data
make train
make eval
```

Expected outputs:
- Results table: [results/results.csv](results/results.csv)
- Robustness table: [results/robustness.csv](results/robustness.csv)
- Cross-domain summary: [results/robustness_cross_domain.md](results/robustness_cross_domain.md)
- Report: [report/report.md](report/report.md)

## One-command run
```bash
make all
```

## Optional LLM baselines (local, offline)
```bash
make llm
```

## Optional adversarial baseline (TextAttack, CPU)
```bash
make textattack
```

## Paper reproduction (longer, CPU)
This regenerates DedupShift splits, classical baselines, DistilBERT anchor, robustness tables, and domain shift diagnostics.
```bash
make paper_repro
```

## Third Domain: Telegram (Optional)

Add the **Telegram Spam or Ham** dataset from Kaggle as a third modern chat domain:

```bash
# 1. Download from Kaggle (requires kaggle API configured)
make telegram_download

# 2. Prepare and run full pipeline
make telegram_full
```

Or run step-by-step:
```bash
make telegram_prepare     # Standardize raw CSV
make telegram_dedup       # DedupShift split
make telegram_train       # Train baselines
make telegram_sync        # Sync to data/ directory
make telegram_eat_augment # EAT augmentation
make telegram_eat_train   # Train EAT models
make telegram_robust      # Robustness evaluation
```

After integration, `make paper_repro` will automatically include Telegram if data is present.

Includes optional P2 experimental completeness features:
- **DedupShift sensitivity analysis** (h_thresh 2/3/4): `make sensitivity_dedup`
- **DistilBERT multi-seed training** (seeds 0/1/2 with mean±std): `make distilbert_multiseed`
- **Generate sensitivity tables/figures**: `make generate_sensitivity_tables`

## Paper Submission Ready

See [P3_PACKAGING_CHECKLIST.md](P3_PACKAGING_CHECKLIST.md) for pre-submission polish:
- ✅ Figure captions with quantitative conclusions
- ✅ Abstract with key numbers (F1 ranges, dedup rate, etc.)
- ✅ CITATION.cff for GitHub-based citing
- ✅ Release v1.0.0 tag
- ✅ Data compliance documentation

## Documentation & Guides

- [P2_SENSITIVITY_ANALYSIS.md](P2_SENSITIVITY_ANALYSIS.md) - Hyperparameter sensitivity and multi-seed details
- [P2_QUICKSTART.md](P2_QUICKSTART.md) - Quick guide for P2 features
- [P2_IMPLEMENTATION_SUMMARY.md](P2_IMPLEMENTATION_SUMMARY.md) - Technical implementation details
- [P3_PACKAGING_CHECKLIST.md](P3_PACKAGING_CHECKLIST.md) - Pre-submission checklist
- [DATA_COMPLIANCE.md](DATA_COMPLIANCE.md) - Data licensing and PII compliance
- [CITATION.cff](CITATION.cff) - Citation metadata (auto-used by GitHub)
- [LICENSE](LICENSE) - MIT license for code

See [Makefile](Makefile) for all targets.
