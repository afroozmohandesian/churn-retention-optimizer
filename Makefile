install:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check src tests scripts app
	ruff format --check src tests scripts app

analyze:
	python -m odin analyze --out outputs > outputs/summary.txt && cat outputs/summary.txt

synthetic:
	python -m odin generate --n 5000 --seed 7 --out data/synthetic/customer_base_5000.csv
	mkdir -p outputs/customer_base_5000
	python -m odin analyze --data data/synthetic/customer_base_5000.csv --budget 25000 --budget-scenarios 10000,25000,40000 --out outputs/customer_base_5000 > outputs/customer_base_5000/summary.txt

app:
	streamlit run app/streamlit_app.py

benchmark:
	python scripts/benchmark.py

notebook:
	python scripts/build_report_notebook.py
	cd notebooks && jupyter nbconvert --to notebook --execute --inplace Project_Odin_Report.ipynb

all: lint test analyze synthetic benchmark notebook

.PHONY: install test lint analyze synthetic app benchmark notebook all
