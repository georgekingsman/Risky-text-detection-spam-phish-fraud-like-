PY=python3

data: data_sms data_spamassassin sync_data

data_sms:
	$(PY) -m src.data_prep --dataset sms --seed 42

data_spamassassin:
	$(PY) -m src.fetch_prepare_spamassassin

sync_data:
	mkdir -p data/sms_spam/processed data/spamassassin/processed data/telegram_spam_ham/processed
	cp dataset/processed/*.csv data/sms_spam/processed/
	cp dataset/spamassassin/processed/*.csv data/spamassassin/processed/
	-cp dataset/telegram_spam_ham/processed/*.csv data/telegram_spam_ham/processed/ 2>/dev/null || true

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
	$(PY) -m src.dedup_split --data-dir dataset/telegram_spam_ham/processed --out-dir dataset/telegram_spam_ham/dedup/processed --report results/dedup_report_telegram.csv --text-col text --label-col label --seed 0 --near --h-thresh 3
	$(PY) -m src.build_dedup_all --data-dir dataset/dedup/processed --out dataset/dedup/processed/all.csv
	$(PY) -m src.build_dedup_all --data-dir dataset/spamassassin/dedup/processed --out dataset/spamassassin/dedup/processed/all.csv
	$(PY) -m src.build_dedup_all --data-dir dataset/telegram_spam_ham/dedup/processed --out dataset/telegram_spam_ham/dedup/processed/all.csv

dedup_train:
	$(PY) -m src.train_baselines_on_dataset --data-dir dataset/dedup/processed --prefix sms_dedup
	$(PY) -m src.train_baselines_on_dataset --data-dir dataset/spamassassin/dedup/processed --prefix spamassassin_dedup
	$(PY) -m src.train_baselines_on_dataset --data-dir dataset/telegram_spam_ham/dedup/processed --prefix telegram_dedup
	$(PY) -m src.train_augtrain --data-dir dataset/dedup/processed --prefix sms_dedup --seed 0
	$(PY) -m src.train_augtrain --data-dir dataset/spamassassin/dedup/processed --prefix spamassassin_dedup --seed 0
	$(PY) -m src.train_augtrain --data-dir dataset/telegram_spam_ham/dedup/processed --prefix telegram_dedup --seed 0

dedup_eval:
	$(PY) -m src.build_results_dedup
	$(PY) -m src.build_cross_domain_table --results results/results_dedup.csv --out results/cross_domain_table_dedup.csv

dedup_robust:
	$(PY) -m src.robustness.run_robust_final --seed 42 --dataset sms_uci_dedup --data-dir dataset/dedup/processed --defense normalize --include-baseline --model-glob "models/sms_dedup*.joblib" --out results/robustness_dedup_sms.csv
	$(PY) -m src.robustness.run_robust_final --seed 42 --dataset spamassassin_dedup --data-dir dataset/spamassassin/dedup/processed --defense normalize --include-baseline --model-glob "models/spamassassin_dedup*.joblib" --out results/robustness_dedup_spamassassin.csv
	$(PY) -m src.robustness.run_robust_final --seed 42 --dataset telegram_dedup --data-dir dataset/telegram_spam_ham/dedup/processed --defense normalize --include-baseline --model-glob "models/telegram_dedup*.joblib" --out results/robustness_dedup_telegram.csv
	$(PY) -m src.merge_robustness_dedup
	$(PY) -m src.plot_robustness_delta_dedup
	$(PY) -m src.compare_robustness_dedup

shift_stats:
	$(PY) -m src.domain_shift_stats --a dataset/dedup/processed/train.csv --b dataset/spamassassin/dedup/processed/train.csv --text-col text --name-a sms --name-b spamassassin --out results/domain_shift_stats.csv --out-js results/domain_shift_js.csv

shift_stats_3domains:
	$(PY) -m src.domain_shift_stats_3domains

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

# ============================================================
# Telegram Dataset Integration (Third Domain)
# ============================================================

# Step 1: Download Telegram dataset from Kaggle
# Requires: pip install kaggle && configure ~/.kaggle/kaggle.json
telegram_download:
	@echo "[INFO] Downloading Telegram Spam or Ham dataset from Kaggle..."
	@mkdir -p dataset/telegram_spam_ham/raw
	kaggle datasets download -d mexwell/telegram-spam-or-ham -p dataset/telegram_spam_ham/raw --unzip
	@echo "[OK] Downloaded to dataset/telegram_spam_ham/raw/"

# Step 2: Prepare/standardize the raw CSV
telegram_prepare:
	$(PY) -m src.prepare_telegram \
	  --in_csv dataset/telegram_spam_ham/raw/telegram_spam_or_ham.csv \
	  --out_csv dataset/telegram_spam_ham/processed/data.csv

# Step 3: Dedup split for Telegram
telegram_dedup:
	$(PY) -m src.dedup_split \
	  --data-dir dataset/telegram_spam_ham/processed \
	  --out-dir dataset/telegram_spam_ham/dedup/processed \
	  --report results/dedup_report_telegram.csv \
	  --text-col text --label-col label --seed 0 --near --h-thresh 3
	$(PY) -m src.build_dedup_all \
	  --data-dir dataset/telegram_spam_ham/dedup/processed \
	  --out dataset/telegram_spam_ham/dedup/processed/all.csv

# Step 4: Train baselines on Telegram
telegram_train:
	$(PY) -m src.train_baselines_on_dataset \
	  --data-dir dataset/telegram_spam_ham/dedup/processed \
	  --prefix telegram_dedup
	$(PY) -m src.train_augtrain \
	  --data-dir dataset/telegram_spam_ham/dedup/processed \
	  --prefix telegram_dedup --seed 0

# Step 5: Robustness evaluation on Telegram
telegram_robust:
	$(PY) -m src.robustness.run_robust_final \
	  --seed 42 \
	  --dataset telegram_dedup \
	  --data-dir dataset/telegram_spam_ham/dedup/processed \
	  --defense normalize \
	  --include-baseline \
	  --model-glob "models/*telegram_dedup*.joblib" \
	  --out results/robustness_dedup_telegram.csv

# Step 6: Sync Telegram dedup data to data/ directory
telegram_sync:
	mkdir -p data/telegram_spam_ham/dedup/processed
	cp dataset/telegram_spam_ham/dedup/processed/*.csv data/telegram_spam_ham/dedup/processed/

# Step 7: EAT augmentation for Telegram
telegram_eat_augment:
	$(PY) -m src.augtrain_build \
	  --in_csv data/telegram_spam_ham/dedup/processed/train.csv \
	  --out_csv data/telegram_spam_ham/dedup/processed/train_augmix.csv \
	  --seed 0 --aug_prob_spam 0.7 --n_aug 1 \
	  --mix "obfuscate:0.7,prompt_injection:0.3"

# Step 8: Train EAT models for Telegram
telegram_eat_train:
	$(PY) -m src.train_eat \
	  --data-dir data/telegram_spam_ham/dedup/processed \
	  --prefix telegram_dedup

# Full Telegram pipeline (after manual download or telegram_download)
telegram_full: telegram_prepare telegram_dedup telegram_train telegram_sync telegram_eat_augment telegram_eat_train telegram_robust
	@echo "[OK] Telegram dataset fully integrated"

# ============================================================
# EAT (Evasion-Aware Training) - Attack-Aware Augmentation
# ============================================================

eat_augment:
	$(PY) -m src.augtrain_build \
	  --in_csv data/sms_spam/dedup/processed/train.csv \
	  --out_csv data/sms_spam/dedup/processed/train_augmix.csv \
	  --seed 0 --aug_prob_spam 0.7 --n_aug 1 \
	  --mix "obfuscate:0.7,prompt_injection:0.3"
	$(PY) -m src.augtrain_build \
	  --in_csv data/spamassassin/dedup/processed/train.csv \
	  --out_csv data/spamassassin/dedup/processed/train_augmix.csv \
	  --seed 0 --aug_prob_spam 0.7 --n_aug 1 \
	  --mix "obfuscate:0.7,prompt_injection:0.3"
	-$(PY) -m src.augtrain_build \
	  --in_csv data/telegram_spam_ham/dedup/processed/train.csv \
	  --out_csv data/telegram_spam_ham/dedup/processed/train_augmix.csv \
	  --seed 0 --aug_prob_spam 0.7 --n_aug 1 \
	  --mix "obfuscate:0.7,prompt_injection:0.3" 2>/dev/null || true

eat_train:
	$(PY) -m src.train_eat --data-dir data/sms_spam/dedup/processed --prefix sms_dedup
	$(PY) -m src.train_eat --data-dir data/spamassassin/dedup/processed --prefix spamassassin_dedup
	-$(PY) -m src.train_eat --data-dir data/telegram_spam_ham/dedup/processed --prefix telegram_dedup 2>/dev/null || true

eat_cross_domain:
	$(PY) -m src.eval_eat_cross_domain --full-threat-model

eat_calibration:
	$(PY) -m src.eval_eat_calibration

eat_summary:
	$(PY) -m src.generate_eat_summary

eat: eat_augment eat_train eat_cross_domain eat_calibration eat_summary
	@echo "EAT pipeline complete. Results in results/eat_*.csv"

# ============================================================

paper_repro:
	# ========== SMS + SpamAssassin (original 2-domain) ==========
	$(PY) -m src.dedup_split --data-dir dataset/processed --out-dir dataset/dedup/processed --report results/dedup_report_sms.csv --text-col text --label-col label --seed 0 --near --h-thresh 3
	$(PY) -m src.dedup_split --data-dir dataset/spamassassin/processed --out-dir dataset/spamassassin/dedup/processed --report results/dedup_report_spamassassin.csv --text-col text --label-col label --seed 0 --near --h-thresh 3
	$(PY) -m src.build_dedup_all --data-dir dataset/dedup/processed --out dataset/dedup/processed/all.csv
	$(PY) -m src.build_dedup_all --data-dir dataset/spamassassin/dedup/processed --out dataset/spamassassin/dedup/processed/all.csv
	# ========== Telegram (3rd domain - optional, requires data download) ==========
	@if [ -f dataset/telegram_spam_ham/processed/data.csv ]; then \
	  echo "[INFO] Telegram data found, including in pipeline..."; \
	  $(PY) -m src.dedup_split --data-dir dataset/telegram_spam_ham/processed --out-dir dataset/telegram_spam_ham/dedup/processed --report results/dedup_report_telegram.csv --text-col text --label-col label --seed 0 --near --h-thresh 3; \
	  $(PY) -m src.build_dedup_all --data-dir dataset/telegram_spam_ham/dedup/processed --out dataset/telegram_spam_ham/dedup/processed/all.csv; \
	else \
	  echo "[WARN] Telegram data not found. Run 'make telegram_download telegram_prepare' first to include 3rd domain."; \
	fi
	# ========== Train baselines ==========
	$(PY) -m src.train_baselines_on_dataset --data-dir dataset/dedup/processed --prefix sms_dedup
	$(PY) -m src.train_baselines_on_dataset --data-dir dataset/spamassassin/dedup/processed --prefix spamassassin_dedup
	@if [ -f dataset/telegram_spam_ham/dedup/processed/train.csv ]; then \
	  $(PY) -m src.train_baselines_on_dataset --data-dir dataset/telegram_spam_ham/dedup/processed --prefix telegram_dedup; \
	fi
	$(PY) -m src.train_augtrain --data-dir dataset/dedup/processed --prefix sms_dedup --seed 0
	$(PY) -m src.train_augtrain --data-dir dataset/spamassassin/dedup/processed --prefix spamassassin_dedup --seed 0
	@if [ -f dataset/telegram_spam_ham/dedup/processed/train.csv ]; then \
	  $(PY) -m src.train_augtrain --data-dir dataset/telegram_spam_ham/dedup/processed --prefix telegram_dedup --seed 0; \
	fi
	# ========== Evaluation ==========
	$(PY) -m src.build_results_dedup
	$(PY) -m src.build_cross_domain_table --results results/results_dedup.csv --out results/cross_domain_table_dedup.csv
	# ========== Robustness ==========
	$(PY) -m src.robustness.run_robust_final --seed 42 --dataset sms_uci_dedup --data-dir dataset/dedup/processed --defense normalize --include-baseline --model-glob "models/*dedup*.joblib" --out results/robustness_dedup_sms.csv
	$(PY) -m src.robustness.run_robust_final --seed 42 --dataset spamassassin_dedup --data-dir dataset/spamassassin/dedup/processed --defense normalize --include-baseline --model-glob "models/*dedup*.joblib" --out results/robustness_dedup_spamassassin.csv
	@if [ -f dataset/telegram_spam_ham/dedup/processed/test.csv ]; then \
	  $(PY) -m src.robustness.run_robust_final --seed 42 --dataset telegram_dedup --data-dir dataset/telegram_spam_ham/dedup/processed --defense normalize --include-baseline --model-glob "models/*telegram_dedup*.joblib" --out results/robustness_dedup_telegram.csv; \
	fi
	$(PY) -m src.merge_robustness_dedup
	# ========== DistilBERT ==========
	$(PY) -m src.nn_distilbert_ft --train_csv dataset/dedup/processed/all.csv --train_domain sms --eval_csvs dataset/dedup/processed/all.csv dataset/spamassassin/dedup/processed/all.csv --eval_domains sms spamassassin --out_dir models/distilbert_sms_dedup --results_csv results/nn_distilbert_sms_train.csv --seed 0 --epochs 2 --batch 8 --grad_accum 2 --max_len 128
	$(PY) -m src.nn_distilbert_ft --train_csv dataset/spamassassin/dedup/processed/all.csv --train_domain spamassassin --eval_csvs dataset/spamassassin/dedup/processed/all.csv dataset/dedup/processed/all.csv --eval_domains spamassassin sms --out_dir models/distilbert_spam_dedup --results_csv results/nn_distilbert_spam_train.csv --seed 0 --epochs 2 --batch 8 --grad_accum 2 --max_len 128
	@if [ -f dataset/telegram_spam_ham/dedup/processed/all.csv ]; then \
	  $(PY) -m src.nn_distilbert_ft --train_csv dataset/telegram_spam_ham/dedup/processed/all.csv --train_domain telegram --eval_csvs dataset/telegram_spam_ham/dedup/processed/all.csv dataset/dedup/processed/all.csv dataset/spamassassin/dedup/processed/all.csv --eval_domains telegram sms spamassassin --out_dir models/distilbert_telegram_dedup --results_csv results/nn_distilbert_telegram_train.csv --seed 0 --epochs 2 --batch 8 --grad_accum 2 --max_len 128; \
	fi
	$(PY) -m src.merge_distilbert_results
	$(PY) -m src.nn_distilbert_ft --train_csv dataset/dedup/processed/all.csv --train_domain sms --eval_csvs dataset/dedup/processed/all.csv dataset/spamassassin/dedup/processed/all.csv --eval_domains sms spamassassin --out_dir models/distilbert_sms_dedup --results_csv results/nn_distilbert_sms_train.csv --seed 0 --epochs 2 --batch 8 --grad_accum 2 --max_len 128 --robust --robust_out results/robustness_distilbert_sms_dedup.csv
	$(PY) -m src.nn_distilbert_ft --train_csv dataset/spamassassin/dedup/processed/all.csv --train_domain spamassassin --eval_csvs dataset/spamassassin/dedup/processed/all.csv dataset/dedup/processed/all.csv --eval_domains spamassassin sms --out_dir models/distilbert_spam_dedup --results_csv results/nn_distilbert_spam_train.csv --seed 0 --epochs 2 --batch 8 --grad_accum 2 --max_len 128 --robust --robust_out results/robustness_distilbert_spam_dedup.csv
	@if [ -f dataset/telegram_spam_ham/dedup/processed/all.csv ]; then \
	  $(PY) -m src.nn_distilbert_ft --train_csv dataset/telegram_spam_ham/dedup/processed/all.csv --train_domain telegram --eval_csvs dataset/telegram_spam_ham/dedup/processed/all.csv dataset/dedup/processed/all.csv dataset/spamassassin/dedup/processed/all.csv --eval_domains telegram sms spamassassin --out_dir models/distilbert_telegram_dedup --results_csv results/nn_distilbert_telegram_train.csv --seed 0 --epochs 2 --batch 8 --grad_accum 2 --max_len 128 --robust --robust_out results/robustness_distilbert_telegram_dedup.csv; \
	fi
	$(PY) -m src.merge_robustness_distilbert
	$(PY) -m src.plot_robustness_delta_dedup
	$(PY) -m src.compare_robustness_dedup
	# ========== Domain Shift Stats ==========
	$(PY) -m src.domain_shift_stats --a dataset/dedup/processed/train.csv --b dataset/spamassassin/dedup/processed/train.csv --text-col text --name-a sms --name-b spamassassin --out results/domain_shift_stats.csv --out-js results/domain_shift_js.csv
	@if [ -f dataset/telegram_spam_ham/dedup/processed/train.csv ]; then \
	  $(PY) -m src.domain_shift_stats_3domains; \
	fi
	$(PY) -m src.sensitivity_analysis_dedup
	$(PY) src/train_distilbert_multiseed.py --train_csv dataset/dedup/processed/all.csv --train_domain sms --eval_csvs dataset/dedup/processed/all.csv dataset/spamassassin/dedup/processed/all.csv --eval_domains sms spamassassin --out_dir models/distilbert_sms_dedup_multiseed --results_csv results/distilbert_multiseed.csv --seeds 0 1 2 --epochs 2 --batch 8 --max_len 128
	$(PY) -m src.generate_sensitivity_tables
	$(PY) src/generate_leakage_table.py
	$(PY) src/generate_dedup_robustness_summary.py
	# ========== New Paper Tables (B-level) ==========
	$(PY) src/generate_dataset_stats_table.py
	$(PY) src/generate_cross_domain_3domain.py
	$(PY) src/generate_jsd_table.py
	$(PY) src/generate_robustness_summary_table.py
	$(PY) src/cost_throughput_analysis.py --dataset sms
	@if [ -f dataset/telegram_spam_ham/dedup/processed/test.csv ]; then \
	  $(PY) src/cost_throughput_analysis.py --dataset telegram; \
	fi
	$(PY) -m src.generate_paper_tables
	$(PY) -m src.generate_paper_assets

report:
	@echo "Report located at report/report.md"

all: data train eval robust report
