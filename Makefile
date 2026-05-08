.PHONY: setup test test-unit test-integration test-tier3 test-tier2 test-tier1 test-dashboard clean-reports ingest graph analyze features train pipeline all clean export-graph train-gnn run-experiment

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

# ---- Dashboard 3-tier smoke suite -------------------------------------------
# tests/dashboard/ replaces the manual phase-h-smoke-test.md walkthroughs.
# Each test writes a paste-able markdown report at
# tests/dashboard/reports/<test_name>.md.
#
# These targets call .venv/bin/python directly (bypassing uv) because
# `uv run` re-resolves torch-scatter on every invocation, which is slow
# and unrelated to dashboard testing.

DASHBOARD_PY := .venv/bin/python -m pytest

# Tier 3 (~5s, $0): SQL emission regression — runs every commit.
test-tier3:
	$(DASHBOARD_PY) tests/dashboard/test_tier3_sql_emission.py -v

# Tier 2 (~30s, ~$0.30/run): real Anthropic critique() smoke — daily / PR.
# Sources .env so ANTHROPIC_API_KEY + OMOPHUB_API_KEY are loaded.
test-tier2:
	bash -c 'set -a && source .env && set +a && $(DASHBOARD_PY) tests/dashboard/test_tier2_critic_verdicts.py -v'

# Tier 1 (~3-5min, ~$0.50/run): full AppTest end-to-end against live
# pipeline — pre-release / on-demand. Requires DuckDB present.
test-tier1:
	bash -c 'set -a && source .env && set +a && DATA_SOURCE=local RUN_LIVE_DASHBOARD=1 $(DASHBOARD_PY) tests/dashboard/test_tier1_dashboard_e2e.py -v'

# All cheap dashboard tests (Tier 2 + Tier 3). Fast feedback for the hot
# paths the team cares about. Tier 1 is opt-in via test-tier1.
test-dashboard: test-tier3 test-tier2

clean-reports:
	rm -f tests/dashboard/reports/*.md

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

# Stage 5b: Export RDF graph to PyG format
export-graph:
	$(PYTHON) -m src.gnn.graph_export

# Stage 5b: Train GNN model
train-gnn:
	$(PYTHON) -m src.gnn.train

# Stage 5b: Run GNN experiment
run-experiment:
	$(PYTHON) -m src.gnn.experiments --run $(EXP)

clean:
	rm -rf .venv __pycache__ .pytest_cache .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
