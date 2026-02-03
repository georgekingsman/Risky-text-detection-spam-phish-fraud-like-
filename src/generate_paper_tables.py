import math
from pathlib import Path

import pandas as pd

root = Path(__file__).resolve().parents[1]


def fmt(x, nd=3):
    if x == "" or x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    return f"{float(x):.{nd}f}"


def write_tex(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    print(f"Wrote {path}")


def table_cross_domain():
    path = root / "results" / "cross_domain_table_dedup.csv"
    df = pd.read_csv(path)
    model_map = {
        "tfidf_word_lr": "TF-IDF word LR",
        "tfidf_char_svm": "TF-IDF char SVM",
        "minilm_lr": "MiniLM + LR",
        "distilbert_ft": "DistilBERT-FT",
    }
    df["model"] = df["model"].map(lambda x: model_map.get(x, x))

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Cross-domain F1 on deduplicated splits.}",
        "\\label{tab:cross_domain_dedup}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Model & SMS in-domain & SpamAssassin in-domain & SMS$\\to$SpamAssassin & SpamAssassin$\\to$SMS \\",
        "\\midrule",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"{r['model']} & {fmt(r['sms_in_domain_f1'])} & {fmt(r['spam_in_domain_f1'])} & {fmt(r['sms_to_spam_f1'])} & {fmt(r['spam_to_sms_f1'])} \\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write_tex(root / "paper" / "tables" / "cross_domain_table_dedup.tex", "\n".join(lines))


def table_dedup_effect():
    path = root / "results" / "dedup_effect.csv"
    df = pd.read_csv(path)
    if df.empty:
        return

    df["f1_drop_change"] = -df["delta_f1_change"]
    agg = (
        df.groupby(["dataset", "attack", "defense"], as_index=False)["f1_drop_change"]
        .mean()
        .sort_values(["dataset", "attack", "defense"])
    )

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Average change in F1 drop after DedupShift (positive means larger drop).}",
        "\\label{tab:dedup_effect}",
        "\\begin{tabular}{lccr}",
        "\\toprule",
        "Dataset & Attack & Defense & $\\Delta$F1$_{drop}$ \\",
        "\\midrule",
    ]
    for _, r in agg.iterrows():
        lines.append(
            f"{r['dataset']} & {r['attack']} & {r['defense']} & {fmt(r['f1_drop_change'])} \\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write_tex(root / "paper" / "tables" / "dedup_effect.tex", "\n".join(lines))


def table_domain_shift():
    stats_path = root / "results" / "domain_shift_stats.csv"
    js_path = root / "results" / "domain_shift_js.csv"
    df = pd.read_csv(stats_path)
    js = pd.read_csv(js_path)

    cols = [
        "len_chars_mean",
        "len_tokens_mean",
        "digit_ratio_mean",
        "punct_ratio_mean",
        "url_cnt_mean",
        "email_cnt_mean",
        "phone_cnt_mean",
    ]

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Domain shift diagnostics on deduplicated train splits.}",
        "\\label{tab:domain_shift}",
        "\\begin{tabular}{lrrrrrrr}",
        "\\toprule",
        "Domain & Len & Tokens & Digit & Punct & URL & Email & Phone \\",
        "\\midrule",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"{r['domain']} & {fmt(r['len_chars_mean'])} & {fmt(r['len_tokens_mean'])} & {fmt(r['digit_ratio_mean'])} & {fmt(r['punct_ratio_mean'])} & {fmt(r['url_cnt_mean'])} & {fmt(r['email_cnt_mean'])} & {fmt(r['phone_cnt_mean'])} \\")
    lines += ["\\bottomrule", "\\end{tabular}"]

    jsd = js.iloc[0]["jsd_char_3to5"] if not js.empty else ""
    lines += ["\\vspace{2pt}", f"\\footnotesize{{Char n-gram JS divergence: {fmt(jsd, nd=3)}}}", "\\end{table}"]
    write_tex(root / "paper" / "tables" / "domain_shift_stats.tex", "\n".join(lines))


def table_textattack():
    sms = root / "results" / "textattack_seeds" / "textattack_sms_uci_agg.csv"
    spam = root / "results" / "textattack_seeds" / "textattack_spamassassin_agg.csv"
    df = pd.concat([pd.read_csv(sms), pd.read_csv(spam)], ignore_index=True)

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        r"\\caption{TextAttack (DeepWordBug) on 200-sample subsets, 3 seeds (mean$\\pm$std).}",
        "\\label{tab:textattack}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Dataset & Model & Success rate & F1 clean & F1 attacked \\",
        "\\midrule",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"{r['dataset']} & {r['model']} & {fmt(r['success_rate_mean'])}$\\pm${fmt(r['success_rate_std'])} & {fmt(r['f1_clean_mean'])}$\\pm${fmt(r['f1_clean_std'])} & {fmt(r['f1_attacked_mean'])}$\\pm${fmt(r['f1_attacked_std'])} \\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write_tex(root / "paper" / "tables" / "textattack_table.tex", "\n".join(lines))


def main():
    table_cross_domain()
    table_dedup_effect()
    table_domain_shift()
    table_textattack()


if __name__ == "__main__":
    main()
