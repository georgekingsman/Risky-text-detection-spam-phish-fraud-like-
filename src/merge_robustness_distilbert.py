import csv
from pathlib import Path

root = Path(__file__).resolve().parents[1]
base = root / "results" / "robustness_dedup.csv"
sm = root / "results" / "robustness_distilbert_sms_dedup.csv"
sp = root / "results" / "robustness_distilbert_spam_dedup.csv"
out = root / "results" / "robustness_dedup.csv"

rows = []
header = None
for path in [base, sm, sp]:
    if not path.exists():
        continue
    with path.open() as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        for row in reader:
            rows.append(row)

if not header:
    raise SystemExit("No robustness files found to merge")

with out.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=header)
    writer.writeheader()
    for r in rows:
        writer.writerow(r)

print("Wrote", out)
