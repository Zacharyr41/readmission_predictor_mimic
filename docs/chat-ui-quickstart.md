# NeuroGraph Chat UI — Quickstart

> **See also:**
> - [`e2e-bigquery-testing.md`](./e2e-bigquery-testing.md) — detailed
>   step-by-step guide for testing the full pipeline against MIMIC-IV on
>   BigQuery, with sample questions covering the single-CQ, clarifying,
>   and multi-CQ "big question" paths.
> - [`decomposer-refactor-changelog.md`](./decomposer-refactor-changelog.md)
>   — what changed across Phases 0–6 and why.

## Prerequisites

```bash
# Install conversational extras (streamlit, plotly, anthropic)
uv pip install --python .venv/bin/python -e ".[conversational]"
```

## Option A: Local DuckDB

```bash
# 1. Make sure your .env has the API key set
#    ANTHROPIC_API_KEY=sk-ant-...

# 2. Launch
bash scripts/run_chat.sh
```

In the sidebar:
- Data source: **Local DuckDB**
- DuckDB path: `data/processed/mimiciv.duckdb` (default)
- API key auto-fills from .env
- Click **Connect**

## Option B: BigQuery

```bash
# 1. Authenticate with Google Cloud
gcloud auth login
gcloud auth application-default login

# 2. Set your project (should match BIGQUERY_PROJECT in .env)
gcloud config set project <your-gcp-project-with-mimic-iv-access>

# 3. Verify access to MIMIC-IV dataset (if it hangs, press Enter a few times —
#    bq may be waiting for you to accept a prompt that isn't visible)
bq ls physionet-data:mimiciv_3_1_hosp

# 4. Launch
bash scripts/run_chat.sh
```

In the sidebar:
- Data source: **BigQuery**
- GCP project ID: `<your-gcp-project-with-mimic-iv-access>` (auto-fills from .env)
- API key auto-fills from .env
- Click **Connect**

## Pipeline Configuration

The pipeline now returns every admission matching your filters — there is no artificial cohort cap. Admissions are fetched in batches to keep per-query payloads reasonable. Settings are configurable in `src/conversational/orchestrator.py`:

| Setting | Default | Description |
|---|---|---|
| `batch_size` | 2000 | Admissions per `IN (…)` batch sent to the database. Lower if you hit BigQuery parameter limits; raise for fewer round-trips. |
| `cohort_strategy` | `"recent"` | Ordering only: `"recent"` sorts by admittime DESC, `"random"` shuffles. Does not change the final result set. |
| `max_workers` | 1 | Parallel graph build workers. Set to 4+ for large cohorts. |

Allen temporal relations are automatically skipped for non-temporal queries, which speeds up most lookups significantly.

### Supported patient filters

The decomposer recognizes these filter fields — use them in your questions for targeted cohorts:

- **age** — e.g. "patients over 65"
- **gender** — e.g. "female patients"
- **diagnosis** — e.g. "patients with sepsis", "ICD code I63"
- **admission_type** — e.g. "emergency admissions"
- **subject_id** — e.g. "patient 12345"
- **readmitted_30d** — e.g. "patients readmitted within 30 days"
- **readmitted_60d** — e.g. "patients readmitted within 60 days"

## Demo Questions

### Readmission analysis
```
What is the average creatinine for patients readmitted within 30 days?
Compare albumin levels between patients readmitted within 30 days and those who were not
```

### Cohort filtering
```
What are the lactate values for patients over 65 with sepsis?
Show hemoglobin trends for emergency admission patients
```

### Temporal reasoning
```
Which antibiotics were prescribed during the first 48 hours of ICU admission?
What were the creatinine values before intubation?
```

### Visualization
```
Plot creatinine trends over the ICU stay for patients with acute kidney injury
Visualize lactate levels over time for patients readmitted within 30 days
```

### Multi-turn conversation
```
What is the average creatinine for patients readmitted within 30 days?
→ Now compare that to patients who were not readmitted
→ Show me the trend over time instead
→ What about lactate for the same population?
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: plotly` | `uv pip install --python .venv/bin/python -e ".[conversational]"` |
| `Database not found` | Check the DuckDB path exists: `ls data/processed/mimiciv.duckdb` |
| BigQuery `403 Access Denied` | Run `gcloud auth application-default login` and verify project access |
| `ANTHROPIC_API_KEY must be set` | Add it to `.env` or paste it in the sidebar |
| Query is slow (>2 min) | Narrow your filters, reduce `batch_size` for memory pressure, or increase `max_workers` |
