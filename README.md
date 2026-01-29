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

See [Makefile](Makefile) for all targets.
