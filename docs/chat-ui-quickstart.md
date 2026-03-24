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

### Biomarkers & Vitals
```
What are the creatinine values?
What is the average albumin level?
What is the median lactate?
How many hemoglobin measurements are there?
Show heart rate trends over time
What was the maximum blood pressure recorded?
```

### Medications
```
What medications include heparin?
Show all atorvastatin prescriptions
What vasopressors were administered?
```

### Diagnoses & Patient Populations
```
Which patients have ICD code I63?
Show patients diagnosed with sepsis
Which patients have a traumatic brain injury diagnosis (S06)?
Show all patients with acute kidney injury
```

### Visualizations
```
Plot creatinine over time
Show a chart of hemoglobin trends
Visualize lactate levels over the ICU stay
Plot a box plot of length of stay by admission type
```

### Research-Style Questions
```
What is the average creatinine for patients readmitted within 30 days vs those who were not?
Compare albumin levels between readmitted and non-readmitted patients
How does length of stay differ across ICU admission types?
What are the most common organisms found in microbiology cultures?
Show the distribution of ICU length of stay
What microbiology results include staph?
```

### Multi-Turn Conversations
```
What is the creatinine?
→ Now show the trend over time
→ What about sodium instead?
→ Plot that as a line chart
```

```
Which patients were diagnosed with cerebral infarction (I63)?
→ What medications were they on?
→ Compare their albumin levels to non-readmitted patients
```

```
Show all microbiology results
→ Which ones involved blood cultures?
→ What organisms were found?
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: plotly` | `uv pip install --python .venv/bin/python -e ".[conversational]"` |
| `Database not found` | Check the DuckDB path exists: `ls data/processed/mimiciv.duckdb` |
| BigQuery `403 Access Denied` | Run `gcloud auth application-default login` and verify project access |
| `ANTHROPIC_API_KEY must be set` | Add it to `.env` or paste it in the sidebar |
