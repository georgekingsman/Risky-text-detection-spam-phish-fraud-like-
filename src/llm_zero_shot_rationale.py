import argparse
import json
import os
import sys
from pathlib import Path
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

# allow running as a script
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src import train_llm


def load_split(csv_path):
    df = pd.read_csv(csv_path)
    texts = df['text'].tolist()
    labels = df['label'].tolist()
    return texts, labels


def label_to_int(label):
    if isinstance(label, str):
        return 1 if label.lower().startswith('spam') else 0
    return int(label)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='Path to test CSV')
    ap.add_argument('--dataset', required=True, help='Dataset name (e.g., sms_uci, spamassassin)')
    ap.add_argument('--out', required=True, help='Output JSONL path')
    ap.add_argument('--provider', default='local')
    ap.add_argument('--model', default='distilgpt2')
    ap.add_argument('--limit', type=int, default=200)
    args = ap.parse_args()

    texts, labels = load_split(args.data)
    limit = min(args.limit, len(texts)) if args.limit else len(texts)

    # LLM zero-shot labels
    preds = train_llm.llm_zero_shot(texts[:limit], labels=['spam','ham'], provider=args.provider, model=args.model, limit=limit)
    # LLM rationales
    rationales = train_llm.llm_extract_reasons(texts[:limit], provider=args.provider, model=args.model, limit=limit)

    # write JSONL
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open('w') as f:
        for t, y, p, r in zip(texts[:limit], labels[:limit], preds, rationales):
            rec = {
                'text': t,
                'label': int(y),
                'pred_label': p,
                'pred_label_int': label_to_int(p),
                'rationale': r
            }
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    # metrics
    ytrue = [label_to_int(y) for y in labels[:limit]]
    ypred = [label_to_int(p) for p in preds]
    f1 = f1_score(ytrue, ypred)
    prec = precision_score(ytrue, ypred, zero_division=0)
    rec = recall_score(ytrue, ypred, zero_division=0)
    roc = ''
    try:
        roc = roc_auc_score(ytrue, ypred)
    except Exception:
        roc = ''

    # append results
    results_path = Path('results/results.csv')
    if not results_path.exists():
        results_path.write_text('dataset,split,model,seed,f1,precision,recall,roc_auc,notes\n')
    with results_path.open('a') as f:
        f.write(f"{args.dataset},test,llm_zeroshot,,{f1},{prec},{rec},{roc},limit={limit}\n")

    print('Wrote', outp, 'F1', f1)


if __name__ == '__main__':
    main()
