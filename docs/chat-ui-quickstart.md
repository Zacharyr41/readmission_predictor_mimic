# NeuroGraph Chat UI — Quickstart

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

## Example Queries

### Biomarkers
```
What are the creatinine values?
What is the average albumin level?
What is the median lactate?
How many hemoglobin measurements are there?
```

### Vitals
```
Show heart rate trends over time
What was the maximum blood pressure recorded?
```

### Medications
```
What medications include heparin?
Show all atorvastatin prescriptions
```

### Diagnoses
```
Which patients have ICD code I63?
Show patients diagnosed with sepsis
```

### Visualizations
```
Plot creatinine over time
Show a chart of hemoglobin trends
```

### Comparisons
```
Compare creatinine levels between readmitted and non-readmitted patients
```

### Follow-up questions (conversation memory)
```
What is the creatinine?
→ Now show the trend over time
→ What about sodium instead?
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: plotly` | `uv pip install --python .venv/bin/python -e ".[conversational]"` |
| `Database not found` | Check the DuckDB path exists: `ls data/processed/mimiciv.duckdb` |
| BigQuery `403 Access Denied` | Run `gcloud auth application-default login` and verify project access |
| `ANTHROPIC_API_KEY must be set` | Add it to `.env` or paste it in the sidebar |
