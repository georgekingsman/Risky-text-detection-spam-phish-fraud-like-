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
	$(PY) -m src.robustness.run_robust_final --seed 42 --dataset sms_uci --defense normalize --include-baseline --out results/robustness_sms.csv
	$(PY) -m src.robustness.run_robust_final --seed 42 --dataset spamassassin --data-dir dataset/spamassassin/processed --defense normalize --include-baseline --out results/robustness_spamassassin.csv
	$(PY) -m src.merge_robustness
	$(PY) -m src.plot_robustness_delta

robust_seeds:
	$(PY) -m src.run_robustness_seeds
	$(PY) -m src.aggregate_robustness
	$(PY) -m src.plot_robustness_delta_agg

robust_spamassassin:
	$(PY) -m src.robustness.run_robust_final --seed 42 --dataset spamassassin --out results/robustness_spamassassin.csv

llm:
	TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 $(PY) -m src.llm_zero_shot_rationale --data data/sms_spam/processed/test.csv --dataset sms_uci --out results/llm_predictions_sms_test.jsonl --provider local --model distilgpt2 --limit 200
	TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 $(PY) -m src.llm_zero_shot_rationale --data data/spamassassin/processed/test.csv --dataset spamassassin --out results/llm_predictions_spamassassin_test.jsonl --provider local --model distilgpt2 --limit 100
	$(PY) -m src.build_results

textattack:
	$(PY) -m src.textattack_baseline --dataset sms_uci --n-samples 200 --out results/textattack_sms.csv --examples-out results/textattack_sms_examples.jsonl
	$(PY) -m src.textattack_baseline --dataset spamassassin --data-dir dataset/spamassassin/processed --n-samples 200 --out results/textattack_spamassassin.csv --examples-out results/textattack_spamassassin_examples.jsonl

textattack_seeds:
	$(PY) -m src.textattack_seeds --dataset sms_uci --n-samples 200 --seeds 0,1,2 --out-dir results/textattack_seeds
	$(PY) -m src.textattack_seeds --dataset spamassassin --data-dir dataset/spamassassin/processed --n-samples 200 --seeds 0,1,2 --out-dir results/textattack_seeds

defense_tradeoff:
	$(PY) -m src.build_defense_tradeoff

dup_check:
	$(PY) -m src.check_duplicates

cross_domain_table:
	$(PY) -m src.build_cross_domain_table

dedup:
	$(PY) -m src.dedup_split --data-dir dataset/processed --out-dir dataset/dedup/processed --report results/dedup_report_sms.csv --text-col text --label-col label --seed 0 --near --h-thresh 3
	$(PY) -m src.dedup_split --data-dir dataset/spamassassin/processed --out-dir dataset/spamassassin/dedup/processed --report results/dedup_report_spamassassin.csv --text-col text --label-col label --seed 0 --near --h-thresh 3

dedup_train:
	$(PY) -m src.train_baselines_on_dataset --data-dir dataset/dedup/processed --prefix sms_dedup
	$(PY) -m src.train_baselines_on_dataset --data-dir dataset/spamassassin/dedup/processed --prefix spamassassin_dedup

dedup_eval:
	$(PY) -m src.build_results_dedup
	$(PY) -m src.build_cross_domain_table --results results/results_dedup.csv --out results/cross_domain_table_dedup.csv

dedup_robust:
	$(PY) -m src.robustness.run_robust_final --seed 42 --dataset sms_uci_dedup --data-dir dataset/dedup/processed --defense normalize --include-baseline --model-glob "models/*dedup*.joblib" --out results/robustness_dedup_sms.csv
	$(PY) -m src.robustness.run_robust_final --seed 42 --dataset spamassassin_dedup --data-dir dataset/spamassassin/dedup/processed --defense normalize --include-baseline --model-glob "models/*dedup*.joblib" --out results/robustness_dedup_spamassassin.csv
	$(PY) -m src.merge_robustness_dedup
	$(PY) -m src.plot_robustness_delta_dedup
	$(PY) -m src.compare_robustness_dedup

shift_stats:
	$(PY) -m src.domain_shift_stats --a dataset/dedup/processed/train.csv --b dataset/spamassassin/dedup/processed/train.csv --text-col text --name-a sms --name-b spamassassin --out results/domain_shift_stats.csv --out-js results/domain_shift_js.csv

report:
	@echo "Report located at report/report.md"

all: data train eval robust report
