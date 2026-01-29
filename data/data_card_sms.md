# SMS (UCI SMSSpamCollection) Data Card

- Source: UCI Machine Learning Repository - SMS Spam Collection Dataset
- URL: https://archive.ics.uci.edu/ml/datasets/sms+spam+collection
- Instances: train/val/test split created from original dataset (stratified)
- Schema: `text,label,split` where `label` ∈ {0: ham, 1: spam}
- Split: 80% train, 10% val, 10% test; `seed=42`
- Notes: Short informal messages, often with abbreviations and phone-numeric tokens. Good for channel-specific spam detection.
