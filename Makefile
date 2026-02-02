.PHONY: setup test test-unit test-integration ingest graph features train all clean

PYTHON := python
UV := uv

setup:
	$(UV) venv
	$(UV) pip install -e ".[dev,graph]"

test:
	$(PYTHON) -m pytest tests/ -v

test-unit:
	$(PYTHON) -m pytest tests/ -v -m "not integration and not slow"

test-integration:
	$(PYTHON) -m pytest tests/ -v -m "integration"

ingest:
	$(PYTHON) -m src.ingestion

graph:
	$(PYTHON) -m src.graph_construction

features:
	$(PYTHON) -m src.feature_extraction

train:
	$(PYTHON) -m src.prediction

all: ingest graph features train

clean:
	rm -rf .venv __pycache__ .pytest_cache .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
