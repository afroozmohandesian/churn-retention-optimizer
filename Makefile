install:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check src tests scripts
	ruff format --check src tests scripts

analyze:
	python -m odin analyze --out outputs > outputs/summary.txt && cat outputs/summary.txt

synthetic:
	python -m odin generate --n 1000 --seed 2024 --out data/synthetic/odin_customers_1000.csv
	mkdir -p outputs/synthetic_1000
	python -m odin analyze --data data/synthetic/odin_customers_1000.csv --budget 5000 --budget-scenarios 2500,5000,10000 --out outputs/synthetic_1000 > outputs/synthetic_1000/summary.txt

benchmark:
	python scripts/benchmark.py

notebook:
	cd notebooks && jupyter nbconvert --to notebook --execute --inplace Project_Odin_Professional.ipynb

all: lint test analyze synthetic benchmark notebook

.PHONY: install test lint analyze synthetic benchmark notebook all
