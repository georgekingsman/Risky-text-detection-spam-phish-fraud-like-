import csv
from pathlib import Path

root = Path(__file__).resolve().parents[1]

# Support 3 domains: SMS, SpamAssassin, and Telegram
robustness_files = [
    root / "results" / "robustness_dedup_sms.csv",
    root / "results" / "robustness_dedup_spamassassin.csv",
    root / "results" / "robustness_dedup_telegram.csv",
]
out = root / "results" / "robustness_dedup.csv"

rows = []
header = None
for path in robustness_files:
    if not path.exists():
        print(f"[WARN] {path} not found, skipping")
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
