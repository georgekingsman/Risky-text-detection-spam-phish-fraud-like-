import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import joblib

def report_model(name, model, df):
    ypred = model.predict(df['text'])
    ytrue = df['label'].values
    f1 = f1_score(ytrue, ypred)
    prec = precision_score(ytrue, ypred)
    rec = recall_score(ytrue, ypred)
    try:
        proba = model.predict_proba(df['text'])[:,1]
        auc = roc_auc_score(ytrue, proba)
    except Exception:
        auc = float('nan')
    return dict(name=name, f1=f1, precision=prec, recall=rec, roc_auc=auc)

def main():
    tr = pd.read_csv('dataset/processed/train.csv')
    va = pd.read_csv('dataset/processed/val.csv')
    te = pd.read_csv('dataset/processed/test.csv')

    results = []
    import glob
    for p in glob.glob('models/*.joblib'):
        name = p.split('/')[-1]
        m = joblib.load(p)
        # support dict for embed model
        if isinstance(m, dict) and 'clf' in m:
            clf = m['clf']
            # need to load embedder to encode; skip A for now
            print('Skipping embed model in eval_all:', name)
            continue
        else:
            res = report_model(name, m, te)
            results.append(res)

    df = pd.DataFrame(results)
    df.to_csv('results/results.csv', index=False)
    print('Wrote results/results.csv')

if __name__ == '__main__':
    main()
