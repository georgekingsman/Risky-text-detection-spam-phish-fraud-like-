"""Provider-agnostic LLM baseline interfaces.

Supports three provider styles (chosen automatically or via `--provider`):
- OpenAI (OPENAI_API_KEY)
- Hugging Face Inference API (HF_API_KEY + model)
- Local transformer model (LLM_LOCAL_MODEL)

This module exposes `llm_zero_shot` and `llm_extract_reasons` and a small CLI to
run zero-shot/evidence-extraction on the test split. The user must provide API keys
via env vars to call remote providers.
"""

import os
import time
import argparse
from typing import List, Optional

import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

try:
    import requests
except Exception:
    requests = None


def _choose_provider(explicit: Optional[str] = None) -> str:
    if explicit:
        return explicit
    if os.getenv('OPENAI_API_KEY'):
        return 'openai'
    if os.getenv('HF_API_KEY'):
        return 'hf'
    if os.getenv('LLM_LOCAL_MODEL'):
        return 'local'
    return 'none'


def _call_openai_chat(prompt: str, model: str = 'gpt-3.5-turbo') -> str:
    key = os.getenv('OPENAI_API_KEY')
    if not key:
        raise RuntimeError('OPENAI_API_KEY not set')
    if requests is None:
        raise RuntimeError('requests package required for OpenAI calls')
    url = 'https://api.openai.com/v1/chat/completions'
    headers = {'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'}
    payload = {
        'model': model,
        'messages': [{'role': 'user', 'content': prompt}],
        'temperature': 0.0,
        'max_tokens': 256,
    }
    r = requests.post(url, headers=headers, json=payload, timeout=30)
    r.raise_for_status()
    j = r.json()
    return j['choices'][0]['message']['content'].strip()


def _call_hf_inference(prompt: str, model: str) -> str:
    key = os.getenv('HF_API_KEY')
    if not key:
        raise RuntimeError('HF_API_KEY not set')
    if requests is None:
        raise RuntimeError('requests package required for HF inference calls')
    url = f'https://api-inference.huggingface.co/models/{model}'
    headers = {'Authorization': f'Bearer {key}'}
    r = requests.post(url, headers=headers, json={'inputs': prompt}, timeout=30)
    r.raise_for_status()
    return r.text


def llm_zero_shot(texts: List[str], labels: List[str] = ['spam', 'ham'], few_shot: Optional[List[dict]] = None,
                  provider: Optional[str] = None, model: Optional[str] = None, limit: Optional[int] = None) -> List[str]:
    """Classify each text as one of `labels` using an LLM provider.

    Returns: list of predicted labels (strings).
    """
    prov = _choose_provider(provider)
    out = []
    limit = limit or len(texts)

    base_prompt = (
        "Classify the following message as 'spam' or 'ham' (legitimate).\n"
        "Return only the label: spam or ham.\n\nMessage:\n"
    )

    pipe = None
    if prov == 'local' and model:
        from transformers import pipeline
        pipe = pipeline('text-generation', model=model)

    for i, t in enumerate(texts[:limit]):
        prompt = base_prompt + t
        if few_shot:
            ex = '\n\n'.join([f"Example: {e['text']} -> {e['label']}" for e in few_shot])
            prompt = ex + '\n\n' + prompt

        if prov == 'openai':
            candidate = _call_openai_chat(prompt, model=(model or 'gpt-3.5-turbo'))
        elif prov == 'hf' and model:
            candidate = _call_hf_inference(prompt, model=model)
        elif prov == 'local' and model:
            # use text-generation locally so prompts can instruct the model
            outp = pipe(prompt, max_new_tokens=16)
            # transformers returns list of dicts for text2text
            if isinstance(outp, list):
                candidate = outp[0].get('generated_text', '') or outp[0].get('text', '')
            else:
                candidate = str(outp)
            if candidate.startswith(prompt):
                candidate = candidate[len(prompt):]
        else:
            raise RuntimeError('No LLM provider configured (set OPENAI_API_KEY, HF_API_KEY, or LLM_LOCAL_MODEL)')

        c = candidate.lower()
        chosen = None
        for lab in labels:
            if lab in c:
                chosen = lab
                break
        if chosen is None:
            chosen = c.split()[0] if c else labels[-1]
        out.append(chosen)
        time.sleep(0.1)
    return out


def llm_extract_reasons(texts: List[str], provider: Optional[str] = None, model: Optional[str] = None, limit: Optional[int] = None) -> List[str]:
    """Ask the LLM to return a short comma-separated list of risky keywords/reasons.

    Returns list of strings (one per input).
    """
    prov = _choose_provider(provider)
    out = []
    limit = limit or len(texts)
    prompt_template = (
        "List up to 5 short keywords or brief reasons why the following message may be risky (spam/phish/fraud)."
        " Respond with a short comma-separated list.\n\nMessage:\n"
    )
    pipe = None
    if prov == 'local' and model:
        from transformers import pipeline
        # use text-generation for decoder-only models (distilgpt2, GPT-2, etc.)
        pipe = pipeline('text-generation', model=model)

    for t in texts[:limit]:
        prompt = prompt_template + t
        if prov == 'openai':
            candidate = _call_openai_chat(prompt, model=(model or 'gpt-3.5-turbo'))
        elif prov == 'hf' and model:
            candidate = _call_hf_inference(prompt, model=model)
        elif prov == 'local' and model:
            res = pipe(prompt, max_new_tokens=32)
            if isinstance(res, list) and res:
                candidate = res[0].get('generated_text', '') or res[0].get('text', '')
            else:
                candidate = str(res)
            if candidate.startswith(prompt):
                candidate = candidate[len(prompt):]
        else:
            raise RuntimeError('No LLM provider configured (set OPENAI_API_KEY, HF_API_KEY, or LLM_LOCAL_MODEL)')
        out.append(candidate.strip())
        time.sleep(0.1)
    return out


def _eval_zero_shot_on_test(provider: Optional[str], model: Optional[str], limit: Optional[int], outpath: str):
    te = pd.read_csv('dataset/processed/test.csv')
    texts = te['text'].tolist()
    labels = ['spam' if x == 1 else 'ham' for x in te['label'].tolist()]
    preds = llm_zero_shot(texts, labels=['spam', 'ham'], provider=provider, model=model, limit=limit)
    preds = preds + ['ham'] * (len(labels) - len(preds))
    ytrue = [1 if l == 'spam' else 0 for l in labels[:len(preds)]]
    ypred = [1 if p == 'spam' else 0 for p in preds]
    f1 = f1_score(ytrue, ypred)
    prec = precision_score(ytrue, ypred)
    rec = recall_score(ytrue, ypred)
    pd.DataFrame({'pred': preds[:len(ypred)], 'label': labels[:len(ypred)]}).to_csv(outpath, index=False)
    print('LLM zero-shot results ->', outpath)
    print('F1', f1, 'prec', prec, 'rec', rec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--provider', default=None, choices=['openai', 'hf', 'local', 'none'])
    ap.add_argument('--model', default=None, help='Model name for HF/local or OpenAI model id')
    ap.add_argument('--task', choices=['zero_shot', 'extract_reasons'], default='zero_shot')
    ap.add_argument('--limit', type=int, default=50)
    ap.add_argument('--out', default='results/llm_results.csv')
    args = ap.parse_args()

    if args.task == 'zero_shot':
        _eval_zero_shot_on_test(args.provider, args.model, args.limit, args.out)
    else:
        te = pd.read_csv('dataset/processed/test.csv')
        texts = te['text'].tolist()[: args.limit]
        reasons = llm_extract_reasons(texts, provider=args.provider, model=args.model, limit=args.limit)
        pd.DataFrame({'text': texts, 'reasons': reasons}).to_csv(args.out, index=False)
        print('Wrote', args.out)


if __name__ == '__main__':
    main()

