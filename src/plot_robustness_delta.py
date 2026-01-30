import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

root = Path(__file__).resolve().parents[1]
robust_path = root / 'results' / 'robustness.csv'
fig_path = root / 'report' / 'fig_robustness_delta.png'

_df = pd.read_csv(robust_path)
# exclude clean
_df = _df[_df['attack'] != 'clean']

# make grouped bar chart by dataset/model/attack with defense hue
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
for ax, dataset in zip(axes, sorted(_df['dataset'].unique())):
    sub = _df[_df['dataset'] == dataset]
    sub['label'] = sub['model'] + ' / ' + sub['attack']
    labels = sub['label'].unique().tolist()
    x = range(len(labels))

    # defense groups
    for i, defense in enumerate(['none', 'normalize']):
        dsub = sub[sub['defense'] == defense]
        dmap = dict(zip(dsub['label'], dsub['delta_f1']))
        vals = [dmap.get(l, 0) for l in labels]
        ax.bar([p + (i-0.5)*0.35 for p in x], vals, width=0.35, label=defense)

    ax.set_title(dataset)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=70, ha='right')
    ax.set_ylabel('ΔF1')
    ax.grid(axis='y', linestyle='--', alpha=0.3)

axes[0].legend(title='defense', loc='best')
fig.tight_layout()
fig_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(fig_path, dpi=200)
print('Wrote', fig_path)
