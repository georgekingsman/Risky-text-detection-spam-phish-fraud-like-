import json
from pathlib import Path
import random

root = Path(__file__).resolve().parents[1]
files = [
    ('sms_uci', root / 'results' / 'llm_predictions_sms_test.jsonl'),
    ('spamassassin', root / 'results' / 'llm_predictions_spamassassin_test.jsonl'),
]

def clean_rationale(text: str) -> str:
    if not text:
        return ''
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    # drop prompt echoes
    lines = [ln for ln in lines if not ln.lower().startswith('list up to 5') and not ln.lower().startswith('respond with') and not ln.lower().startswith('message:')]
    cleaned = ' '.join(lines)
    if not cleaned:
        cleaned = text.strip()
    return cleaned[:200]


cases = []
for dataset, path in files:
    if not path.exists():
        continue
    with path.open() as f:
        for line in f:
            obj = json.loads(line)
            if obj.get('label') != obj.get('pred_label_int'):
                cases.append({
                    'dataset': dataset,
                    'text': obj.get('text', ''),
                    'label': obj.get('label'),
                    'pred': obj.get('pred_label'),
                    'rationale': clean_rationale(obj.get('rationale', ''))
                })

random.seed(42)
random.shuffle(cases)
selected = cases[:10]

out_path = root / 'results' / 'cases.md'
with out_path.open('w') as f:
    f.write('# LLM zero-shot error cases\n\n')
    f.write('Examples of LLM zero-shot misclassifications with model rationales.\n\n')
    for i, c in enumerate(selected, 1):
        txt = c['text'].replace('\n', ' ')
        if len(txt) > 400:
            txt = txt[:400] + '...'
        f.write(f"## Case {i} ({c['dataset']})\n")
        f.write(f"- True label: {c['label']}\n")
        f.write(f"- Pred label: {c['pred']}\n")
        f.write(f"- Rationale: {c['rationale'].strip()}\n")
        f.write(f"- Text: {txt}\n\n")

print('Wrote', out_path)
