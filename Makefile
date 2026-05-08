.PHONY: help setup test test-unit test-integration test-tier3 test-tier2 test-tier1 test-dashboard clean-reports ingest graph analyze features train pipeline all clean export-graph train-gnn run-experiment

# Default target — `make` with no args prints the help summary.
.DEFAULT_GOAL := help

UV := uv
PYTHON := $(UV) run python

# `make` (or `make help`) prints a per-target summary so a teammate can
# orient without grepping the file. Each target's docstring lives in the
# trailing-comment column ("## …"); the awk below extracts and aligns
# them. New targets should keep the format.
help:
	@printf "\nUsage: make <target>\n\n"
	@printf "Setup\n"
	@grep -E '^(setup|clean):.*?##.*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "} {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@printf "\nFast tests\n"
	@grep -E '^(test|test-unit|test-integration|test-tier3):.*?##.*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "} {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@printf "\nDashboard smoke (replaces docs/phase-h-smoke-test.md)\n"
	@grep -E '^(test-dashboard|test-tier2|test-tier1|clean-reports):.*?##.*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "} {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@printf "\nPipeline stages\n"
	@grep -E '^(ingest|graph|analyze|features|train|pipeline|all|export-graph|train-gnn|run-experiment):.*?##.*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "} {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@printf "\nReports written by Tier 1/2/3: tests/dashboard/reports/<test>.md\n"
	@printf "Each is a paste-able markdown file with Q/A/SQL/verdict/assertions.\n\n"

setup: ## Install editable package + dev/graph extras into a fresh .venv via uv
	$(UV) venv
	$(UV) pip install -e ".[dev,graph]"

test: ## Run unit tests (excludes tests/test_integration). General-purpose.
	$(PYTHON) -m pytest tests/ -v --ignore=tests/test_integration

test-unit: ## Fast unit tests only (-m "not integration and not slow")
	$(PYTHON) -m pytest tests/ -v -m "not integration and not slow"

test-integration: ## Integration tests only (-m integration)
	$(PYTHON) -m pytest tests/test_integration/ -v -m "integration"

# ---- Dashboard 3-tier smoke suite -------------------------------------------
# tests/dashboard/ replaces the manual phase-h-smoke-test.md walkthroughs.
# Each test writes a paste-able markdown report at
# tests/dashboard/reports/<test_name>.md so a teammate or reviewing agent
# can qualitatively assess what the dashboard actually did.
#
# These targets call .venv/bin/python directly (bypassing uv) because
# `uv run` re-resolves torch-scatter on every invocation, which is slow
# and unrelated to dashboard testing.

DASHBOARD_PY := .venv/bin/python -m pytest

test-tier3: ## Tier 3 (~5s, $0). No LLM. SQL-shape regression for Inc 4/7/9 wiring.
	$(DASHBOARD_PY) tests/dashboard/test_tier3_sql_emission.py -v

test-tier2: ## Tier 2 (~30s, ~$0.30). Real Anthropic critique() — §4a/4b/4c. Sources .env.
	bash -c 'set -a && source .env && set +a && $(DASHBOARD_PY) tests/dashboard/test_tier2_critic_verdicts.py -v'

test-tier1: ## Tier 1 (~3-5min, ~$0.50). Full AppTest E2E. Sources .env, sets RUN_LIVE_DASHBOARD=1.
	bash -c 'set -a && source .env && set +a && DATA_SOURCE=local RUN_LIVE_DASHBOARD=1 $(DASHBOARD_PY) tests/dashboard/test_tier1_dashboard_e2e.py -v'

test-dashboard: test-tier3 test-tier2 ## Cheap dashboard tiers (3 + 2). Default pre-PR check.

clean-reports: ## Wipe all generated tests/dashboard/reports/*.md
	rm -f tests/dashboard/reports/*.md

# ---- Pipeline stages --------------------------------------------------------

ingest: ## Stage 1: ingest MIMIC-IV CSVs into DuckDB
	$(PYTHON) -m src.ingestion

graph: ## Stage 2: build the RDF knowledge graph
	$(PYTHON) -m src.graph_construction

analyze: ## Stage 3: generate the graph-analysis report
	$(PYTHON) -m src.graph_analysis

features: ## Stage 4: extract features from the RDF graph
	$(PYTHON) -m src.feature_extraction

train: ## Stage 5: train + evaluate readmission models
	$(PYTHON) -m src.prediction

pipeline: ## Run the full pipeline via src.main
	$(PYTHON) -m src.main

all: ingest graph analyze features train ## Run every pipeline stage sequentially

export-graph: ## Stage 5b: export RDF graph to PyG format
	$(PYTHON) -m src.gnn.graph_export

train-gnn: ## Stage 5b: train the GNN model
	$(PYTHON) -m src.gnn.train

run-experiment: ## Stage 5b: run a GNN experiment (pass via EXP=name)
	$(PYTHON) -m src.gnn.experiments --run $(EXP)

clean: ## Remove .venv, caches, and all *.pyc / __pycache__ artifacts
	rm -rf .venv __pycache__ .pytest_cache .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
