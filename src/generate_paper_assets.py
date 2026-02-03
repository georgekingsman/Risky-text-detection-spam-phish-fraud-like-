from pathlib import Path
import shutil

root = Path(__file__).resolve().parents[1]
fig_src = root / "report"
fig_dst = root / "paper" / "figs"
fig_dst.mkdir(parents=True, exist_ok=True)

files = [
    "Pipeline.png",
    "fig_robustness_delta.png",
    "fig_robustness_delta_agg.png",
    "fig_robustness_delta_dedup.png",
]

for name in files:
    src = fig_src / name
    if not src.exists():
        continue
    shutil.copy2(src, fig_dst / name)
    print(f"Copied {src} -> {fig_dst / name}")
