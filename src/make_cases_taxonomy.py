import json
from pathlib import Path
import random
import pandas as pd
import joblib
from src.robustness.perturb import obfuscate, simple_paraphrase_like, prompt_injection

ROOT = Path(__file__).resolve().parents[1]

MODELS = {
    'tfidf_word_lr': ROOT / 'models' / 'tfidf_word_lr.joblib',
    'tfidf_char_svm': ROOT / 'models' / 'tfidf_char_svm.joblib',
}

ATTACKS = {
    'obfuscate': obfuscate,
    'paraphrase_like': simple_paraphrase_like,
    'prompt_injection': prompt_injection,
}

TAGS = {
    'obfuscate': 'obfuscation/spacing/symbols',
    'paraphrase_like': 'paraphrase/synonym shift',
    'prompt_injection': 'instruction/prompt injection',
}


def predict(model, texts):
    preds = model.predict(texts)
    if isinstance(preds[0], str):
        return [1 if p.lower().startswith('spam') else 0 for p in preds]
    return [int(p) for p in preds]


def main():
    df = pd.read_csv(ROOT / 'dataset' / 'processed' / 'test.csv')
    texts = df['text'].tolist()
    labels = df['label'].tolist()

    loaded = {k: joblib.load(v) for k, v in MODELS.items() if v.exists()}

    cases = []
    for attack_name, attack_fn in ATTACKS.items():
        pert_texts = [attack_fn(t) if attack_fn is not obfuscate else attack_fn(t, seed=i) for i, t in enumerate(texts)]
        for idx, (orig, pert, ytrue) in enumerate(zip(texts, pert_texts, labels)):
            row = {
                'attack': attack_name,
                'tag': TAGS[attack_name],
                'orig': orig,
                'pert': pert,
                'label': int(ytrue)
            }
            changed = False
            for model_name, model in loaded.items():
                p_clean = predict(model, [orig])[0]
                p_pert = predict(model, [pert])[0]
                row[f'{model_name}_clean'] = p_clean
                row[f'{model_name}_pert'] = p_pert
                if p_clean != p_pert:
                    changed = True
            if changed:
                cases.append(row)

    random.seed(42)
    random.shuffle(cases)
    selected = cases[:12]

    out_path = ROOT / 'results' / 'cases.md'
    with out_path.open('w') as f:
        f.write('# Case taxonomy (model sensitivity under perturbations)\n\n')
        f.write('Each case shows original vs perturbed text and predictions from two models.\n\n')
        for i, c in enumerate(selected, 1):
            f.write(f"## Case {i} ({c['attack']})\n")
            f.write(f"- Failure tag: {c['tag']}\n")
            f.write(f"- True label: {c['label']}\n")
            f.write(f"- tfidf_word_lr clean→pert: {c['tfidf_word_lr_clean']} → {c['tfidf_word_lr_pert']}\n")
            f.write(f"- tfidf_char_svm clean→pert: {c['tfidf_char_svm_clean']} → {c['tfidf_char_svm_pert']}\n")
            f.write(f"- Original: {c['orig'].replace('\n',' ')}\n")
            f.write(f"- Perturbed: {c['pert'].replace('\n',' ')}\n\n")

    print('Wrote', out_path)


if __name__ == '__main__':
    main()
