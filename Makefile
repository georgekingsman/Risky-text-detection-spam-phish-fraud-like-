PY=python3

data: data_sms data_spamassassin sync_data

data_sms:
	$(PY) -m src.data_prep --dataset sms --seed 42

data_spamassassin:
	$(PY) -m src.fetch_prepare_spamassassin

sync_data:
	mkdir -p data/sms_spam/processed data/spamassassin/processed
	cp dataset/processed/*.csv data/sms_spam/processed/
	cp dataset/spamassassin/processed/*.csv data/spamassassin/processed/

train: train_sms train_spamassassin

train_sms:
	$(PY) -m src.train_ml --seed 42
	$(PY) -m src.train_embed --seed 42

train_spamassassin:
	$(PY) -m src.train_baselines_on_dataset --data-dir dataset/spamassassin/processed --prefix spamassassin

eval:
	$(PY) -m src.build_results

robust:
	$(PY) -m src.robustness.run_robust_final --seed 42 --dataset sms_uci --out results/robustness.csv

robust_spamassassin:
	$(PY) -m src.robustness.run_robust_final --seed 42 --dataset spamassassin --out results/robustness_spamassassin.csv

llm:
	TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 $(PY) -m src.llm_zero_shot_rationale --data data/sms_spam/processed/test.csv --dataset sms_uci --out results/llm_predictions_sms_test.jsonl --provider local --model distilgpt2 --limit 200
	TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 $(PY) -m src.llm_zero_shot_rationale --data data/spamassassin/processed/test.csv --dataset spamassassin --out results/llm_predictions_spamassassin_test.jsonl --provider local --model distilgpt2 --limit 100
	$(PY) -m src.build_results

report:
	@echo "Report located at report/report.md"

all: data train eval robust report
