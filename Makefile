PY=python3

data:
	$(PY) -m src.data_prep --dataset sms --seed 42

train_ml:
	$(PY) -m src.train_ml --seed 42

train_embed:
	$(PY) -m src.train_embed --seed 42

eval:
	$(PY) -m src.eval_all

robust:
	$(PY) -m src.robustness.run_robust --seed 42

all: data train_ml train_embed eval robust
