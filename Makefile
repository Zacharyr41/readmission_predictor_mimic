.PHONY: setup test test-unit test-integration ingest graph analyze features train pipeline all clean

UV := uv
PYTHON := $(UV) run python

setup:
	$(UV) venv
	$(UV) pip install -e ".[dev,graph]"

# Run unit tests only (excludes integration tests)
test:
	$(PYTHON) -m pytest tests/ -v --ignore=tests/test_integration

# Run fast unit tests (excludes slow and integration tests)
test-unit:
	$(PYTHON) -m pytest tests/ -v -m "not integration and not slow"

# Run integration tests only
test-integration:
	$(PYTHON) -m pytest tests/test_integration/ -v -m "integration"

# Stage 1: Ingest MIMIC-IV CSVs to DuckDB
ingest:
	$(PYTHON) -m src.ingestion

# Stage 2: Build RDF knowledge graph
graph:
	$(PYTHON) -m src.graph_construction

# Stage 3: Generate graph analysis report
analyze:
	$(PYTHON) -m src.graph_analysis

# Stage 4: Extract features from RDF graph
features:
	$(PYTHON) -m src.feature_extraction

# Stage 5: Train and evaluate models
train:
	$(PYTHON) -m src.prediction

# Run full pipeline via main orchestrator
pipeline:
	$(PYTHON) -m src.main

# Run all stages sequentially
all: ingest graph analyze features train

clean:
	rm -rf .venv __pycache__ .pytest_cache .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
