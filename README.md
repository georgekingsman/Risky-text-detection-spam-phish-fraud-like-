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

Includes optional P2 experimental completeness features:
- **DedupShift sensitivity analysis** (h_thresh 2/3/4): `make sensitivity_dedup`
- **DistilBERT multi-seed training** (seeds 0/1/2 with mean±std): `make distilbert_multiseed`
- **Generate sensitivity tables/figures**: `make generate_sensitivity_tables`

See [P2_SENSITIVITY_ANALYSIS.md](P2_SENSITIVITY_ANALYSIS.md) for details on optional experimental completeness.

See [Makefile](Makefile) for all targets.
