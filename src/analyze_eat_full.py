#!/usr/bin/env python3
"""Generate EAT full threat model analysis summary."""
import pandas as pd

# Load cross-domain gain data (full threat model)
gain = pd.read_csv("results/eat_cross_domain_gain.csv")

print("=" * 70)
print("EAT Full Threat Model Cross-Domain Analysis")
print("=" * 70)

# Cross-attack robustness
print("\n1. Cross-Attack Robustness")
print("-" * 70)

# SMS -> SpamAssassin direction (where EAT helps)
sms_spam = gain[gain["scenario"] == "sms→spamassassin"]
print("\nSMS→SpamAssassin (EAT effective):")
for _, row in sms_spam.iterrows():
    print(f"  {row['model']:15} | {row['attack']:18} | gain: {row['gain']:+.2%}")

# SpamAssassin -> SMS direction
spam_sms = gain[gain["scenario"] == "spamassassin→sms"]
print("\nSpamAssassin→SMS (EAT limited):")
for _, row in spam_sms.iterrows():
    print(f"  {row['model']:15} | {row['attack']:18} | gain: {row['gain']:+.2%}")

# Calibration analysis
print("\n" + "=" * 70)
print("2. Threshold Calibration Analysis")
print("-" * 70)

cal = pd.read_csv("results/eat_calibration_summary.csv")
eat_cal = cal[cal["training"] == "eat"]

# Cases with high recall (overdetection)
high_recall = eat_cal[eat_cal["recall_default"] > 0.9]
print("\nOverdetection cases (Recall > 0.9):")
for _, row in high_recall.iterrows():
    print(f"  {row['scenario']} | {row['model']}:")
    print(f"    Before: P={row['precision_default']:.3f}, R={row['recall_default']:.3f}")
    print(f"    After:  P={row['precision_calibrated']:.3f}, R={row['recall_calibrated']:.3f}")
    print(f"    Threshold: 0.5 -> {row['optimal_threshold']:.3f}")

# Summary statistics
print("\n" + "=" * 70)
print("3. Key Paper Data Points")
print("-" * 70)

# Best cross-domain gain
best = sms_spam.loc[sms_spam["gain"].idxmax()]
print(f"\nBest cross-domain gain: {best['scenario']} | {best['model']} | {best['attack']}: {best['gain']:+.2%}")

# Cross-attack transfer for tfidf_word_lr
word_lr = sms_spam[sms_spam["model"] == "tfidf_word_lr"]
if not word_lr.empty:
    obf = word_lr[word_lr["attack"] == "obfuscate"]["gain"].values
    para = word_lr[word_lr["attack"] == "paraphrase_like"]["gain"].values
    pi = word_lr[word_lr["attack"] == "prompt_injection"]["gain"].values
    
    print(f"\nCross-attack transfer (SMS->SpamAssassin, tfidf_word_lr):")
    if len(obf) > 0:
        print(f"  obfuscate (trained): {obf[0]:+.2%}")
    if len(para) > 0:
        print(f"  paraphrase_like (untrained): {para[0]:+.2%}")
    if len(pi) > 0:
        print(f"  prompt_injection (untrained): {pi[0]:+.2%}")

print("\n" + "=" * 70)
print("Paper claim:")
print("  'Training-time augmentation on obfuscation yields cross-attack")
print("   robustness gains on unseen attack types (paraphrase, prompt_injection)'")
print("=" * 70)
