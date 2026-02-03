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
	$(PY) -m src.build_dedup_all --data-dir dataset/dedup/processed --out dataset/dedup/processed/all.csv
	$(PY) -m src.build_dedup_all --data-dir dataset/spamassassin/dedup/processed --out dataset/spamassassin/dedup/processed/all.csv

dedup_train:
	$(PY) -m src.train_baselines_on_dataset --data-dir dataset/dedup/processed --prefix sms_dedup
	$(PY) -m src.train_baselines_on_dataset --data-dir dataset/spamassassin/dedup/processed --prefix spamassassin_dedup
	$(PY) -m src.train_augtrain --data-dir dataset/dedup/processed --prefix sms_dedup --seed 0
	$(PY) -m src.train_augtrain --data-dir dataset/spamassassin/dedup/processed --prefix spamassassin_dedup --seed 0

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

nn_distilbert_dedup:
	$(PY) -m src.nn_distilbert_ft --train_csv dataset/dedup/processed/all.csv --train_domain sms \
	  --eval_csvs dataset/dedup/processed/all.csv dataset/spamassassin/dedup/processed/all.csv \
	  --eval_domains sms spamassassin --out_dir models/distilbert_sms_dedup \
	  --results_csv results/nn_distilbert_sms_train.csv --seed 0 --epochs 2 --batch 8 --grad_accum 2 --max_len 128
	$(PY) -m src.nn_distilbert_ft --train_csv dataset/spamassassin/dedup/processed/all.csv --train_domain spamassassin \
	  --eval_csvs dataset/spamassassin/dedup/processed/all.csv dataset/dedup/processed/all.csv \
	  --eval_domains spamassassin sms --out_dir models/distilbert_spam_dedup \
	  --results_csv results/nn_distilbert_spam_train.csv --seed 0 --epochs 2 --batch 8 --grad_accum 2 --max_len 128

sensitivity_dedup:
	$(PY) -m src.sensitivity_analysis_dedup

distilbert_multiseed:
	$(PY) src/train_distilbert_multiseed.py \
	  --train_csv dataset/dedup/processed/all.csv \
	  --train_domain sms \
	  --eval_csvs dataset/dedup/processed/all.csv dataset/spamassassin/dedup/processed/all.csv \
	  --eval_domains sms spamassassin \
	  --out_dir models/distilbert_sms_dedup_multiseed \
	  --results_csv results/distilbert_multiseed.csv \
	  --seeds 0 1 2 --epochs 2 --batch 8 --max_len 128

generate_sensitivity_tables:
	$(PY) -m src.generate_sensitivity_tables

generate_leakage_table:
	$(PY) src/generate_leakage_table.py

generate_dedup_robustness_summary:
	$(PY) src/generate_dedup_robustness_summary.py

paper_repro:
	$(PY) -m src.dedup_split --data-dir dataset/processed --out-dir dataset/dedup/processed --report results/dedup_report_sms.csv --text-col text --label-col label --seed 0 --near --h-thresh 3
	$(PY) -m src.dedup_split --data-dir dataset/spamassassin/processed --out-dir dataset/spamassassin/dedup/processed --report results/dedup_report_spamassassin.csv --text-col text --label-col label --seed 0 --near --h-thresh 3
	$(PY) -m src.build_dedup_all --data-dir dataset/dedup/processed --out dataset/dedup/processed/all.csv
	$(PY) -m src.build_dedup_all --data-dir dataset/spamassassin/dedup/processed --out dataset/spamassassin/dedup/processed/all.csv
	$(PY) -m src.train_baselines_on_dataset --data-dir dataset/dedup/processed --prefix sms_dedup
	$(PY) -m src.train_baselines_on_dataset --data-dir dataset/spamassassin/dedup/processed --prefix spamassassin_dedup
	$(PY) -m src.train_augtrain --data-dir dataset/dedup/processed --prefix sms_dedup --seed 0
	$(PY) -m src.train_augtrain --data-dir dataset/spamassassin/dedup/processed --prefix spamassassin_dedup --seed 0
	$(PY) -m src.build_results_dedup
	$(PY) -m src.build_cross_domain_table --results results/results_dedup.csv --out results/cross_domain_table_dedup.csv
	$(PY) -m src.robustness.run_robust_final --seed 42 --dataset sms_uci_dedup --data-dir dataset/dedup/processed --defense normalize --include-baseline --model-glob "models/*dedup*.joblib" --out results/robustness_dedup_sms.csv
	$(PY) -m src.robustness.run_robust_final --seed 42 --dataset spamassassin_dedup --data-dir dataset/spamassassin/dedup/processed --defense normalize --include-baseline --model-glob "models/*dedup*.joblib" --out results/robustness_dedup_spamassassin.csv
	$(PY) -m src.merge_robustness_dedup
	$(PY) -m src.nn_distilbert_ft --train_csv dataset/dedup/processed/all.csv --train_domain sms --eval_csvs dataset/dedup/processed/all.csv dataset/spamassassin/dedup/processed/all.csv --eval_domains sms spamassassin --out_dir models/distilbert_sms_dedup --results_csv results/nn_distilbert_sms_train.csv --seed 0 --epochs 2 --batch 8 --grad_accum 2 --max_len 128
	$(PY) -m src.nn_distilbert_ft --train_csv dataset/spamassassin/dedup/processed/all.csv --train_domain spamassassin --eval_csvs dataset/spamassassin/dedup/processed/all.csv dataset/dedup/processed/all.csv --eval_domains spamassassin sms --out_dir models/distilbert_spam_dedup --results_csv results/nn_distilbert_spam_train.csv --seed 0 --epochs 2 --batch 8 --grad_accum 2 --max_len 128
	$(PY) -m src.merge_distilbert_results
	$(PY) -m src.nn_distilbert_ft --train_csv dataset/dedup/processed/all.csv --train_domain sms --eval_csvs dataset/dedup/processed/all.csv dataset/spamassassin/dedup/processed/all.csv --eval_domains sms spamassassin --out_dir models/distilbert_sms_dedup --results_csv results/nn_distilbert_sms_train.csv --seed 0 --epochs 2 --batch 8 --grad_accum 2 --max_len 128 --robust --robust_out results/robustness_distilbert_sms_dedup.csv
	$(PY) -m src.nn_distilbert_ft --train_csv dataset/spamassassin/dedup/processed/all.csv --train_domain spamassassin --eval_csvs dataset/spamassassin/dedup/processed/all.csv dataset/dedup/processed/all.csv --eval_domains spamassassin sms --out_dir models/distilbert_spam_dedup --results_csv results/nn_distilbert_spam_train.csv --seed 0 --epochs 2 --batch 8 --grad_accum 2 --max_len 128 --robust --robust_out results/robustness_distilbert_spam_dedup.csv
	$(PY) -m src.merge_robustness_distilbert
	$(PY) -m src.plot_robustness_delta_dedup
	$(PY) -m src.compare_robustness_dedup
	$(PY) -m src.domain_shift_stats --a dataset/dedup/processed/train.csv --b dataset/spamassassin/dedup/processed/train.csv --text-col text --name-a sms --name-b spamassassin --out results/domain_shift_stats.csv --out-js results/domain_shift_js.csv
	$(PY) -m src.sensitivity_analysis_dedup
	$(PY) src/train_distilbert_multiseed.py --train_csv dataset/dedup/processed/all.csv --train_domain sms --eval_csvs dataset/dedup/processed/all.csv dataset/spamassassin/dedup/processed/all.csv --eval_domains sms spamassassin --out_dir models/distilbert_sms_dedup_multiseed --results_csv results/distilbert_multiseed.csv --seeds 0 1 2 --epochs 2 --batch 8 --max_len 128
	$(PY) -m src.generate_sensitivity_tables
	$(PY) src/generate_leakage_table.py
	$(PY) src/generate_dedup_robustness_summary.py
	$(PY) -m src.generate_paper_tables
	$(PY) -m src.generate_paper_assets

report:
	@echo "Report located at report/report.md"

all: data train eval robust report
