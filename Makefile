.PHONY: install download-data prepare-data build-index test lint typecheck run-pipeline run-baselines run-ablations run-all verify-env clean

install:
	pip install -r requirements.txt
	python -m spacy download en_core_web_sm

download-data:
	python scripts/download_data.py

prepare-data:
	python scripts/prepare_data.py

build-index:
	python scripts/build_index.py

test:
	pytest tests/

lint:
	ruff check .

typecheck:
	mypy src/

run-pipeline:
	python scripts/run_pipeline.py

run-baselines:
	python scripts/run_baselines.py

run-ablations:
	python scripts/run_ablations.py

run-all: download-data prepare-data build-index run-pipeline run-baselines run-ablations

verify-env:
	python scripts/verify_env.py

clean:
	rm -rf data/cache/*
	rm -rf outputs/*
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
