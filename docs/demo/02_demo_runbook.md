# Demo Runbook (for the developer)

> A linear, do-this-then-that guide to **rehearse the demo by hand** and catch
> any issue before Dr. Mansour is in the room. Companion to the script in
> `01_demo_outline.md` and the architecture in `00_chatbot_architecture.md`.
>
> Rule of thumb: **green at every step below ⇒ you're demo-ready.** If a step is
> red, the "If it's wrong" column tells you what to do.

---

## 0. What you need

- The local **DuckDB** copy of MIMIC-IV (`data/processed/mimiciv.duckdb`, ~16 GB)
  **or** BigQuery access (`gcloud` auth + a project with the MIMIC-IV dataset).
- An **`ANTHROPIC_API_KEY`** in `.env`.
- The `conversational` extras installed in `.venv`.

> **Recommendation:** run the demo against **Local DuckDB**. It removes BigQuery
> latency and cost from the live demo and makes graph turns more predictable.

---

## 1. One-time setup (≈5 min)

```bash
# From the repo root.
python -m venv .venv                     # if you don't already have one
.venv/bin/python -m pip install -e ".[dev,conversational]"

# .env must contain at least:
#   ANTHROPIC_API_KEY=sk-ant-...
#   DATA_SOURCE=local              # for the recommended local run
```

Confirm the database and mappings are present:

```bash
ls -lh data/processed/mimiciv.duckdb                       # ~16 GB
ls data/mappings/ | grep -E '(drug|icd10cm|labitem|loinc)_to_snomed.json'
```

> If you'll demo on **BigQuery** instead: `gcloud auth application-default login`,
> set `BIGQUERY_PROJECT` in `.env`, and verify with
> `bq ls physionet-data:mimiciv_3_1_hosp`. See `docs/chat-ui-quickstart.md`.

---

## 2. Pre-flight: the programmatic check (≈10 s, do this first, every time)

This is the hermetic harness that verifies **every scripted demo question still
routes the way the script claims**, and that **every fast-path question's SQL
compiles and executes** against a MIMIC-shaped schema. No API key, no database,
no network — so it can never fail for environment reasons, only for real ones.

```bash
make demo-check
# (equivalently: .venv/bin/python -m pytest tests/test_demo/ -q)
```

**Expect:** `36 passed`.

| If it's wrong | What it means / do |
|---|---|
| a `test_demo_routing` case fails | a scripted question no longer routes as the outline claims (a planner/fixture change). Re-read the failure: fix the question wording in `tests/test_demo/demo_questions.py` and `01_demo_outline.md`, or investigate the planner change. |
| a `test_demo_sql_fastpath` case fails | a fast-path question's SQL no longer compiles/executes — a real backend bug. Do **not** demo that question until fixed. |
| `test_dashboard_welcome` fails | the welcome panel / example prompts regressed. |

> Then optionally run the broader regression once:
> `.venv/bin/python -m pytest tests/test_conversational -q` (the conversational
> engine the demo rides on). One pre-existing failure in
> `test_health_evidence/test_tools.py` only appears when the local DuckDB is
> **absent** — ignore it if you have the DuckDB in place.

---

## 3. Launch the dashboard (≈30 s)

```bash
bash scripts/run_chat.sh
# (equivalently: .venv/bin/streamlit run src/conversational/app.py)
```

A browser opens at `http://localhost:8501`.

1. In the sidebar, choose **Local DuckDB** (or **BigQuery**).
2. The **API key** auto-fills from `.env`; the DuckDB path defaults correctly.
3. Click **Connect** → the sidebar should show a green **Connected**.
4. The main pane shows the **welcome panel** with example questions. Good — that
   means `app.py` loaded cleanly.

**Smoke test before anything else** — type:

> *What is the average creatinine for patients over 65?*

You should get a single mean value within a few seconds, and the live status
should pass through "Interpreting your question…" → "Running the query…" →
"Reviewing the result…" → green **Analysis complete**. If this errors, fix it
before going further (almost always the API key or the DuckDB path).

---

## 4. Hand-run the five phases

Type each prompt and confirm the **Expect** column. Watch the **live status
label** — it tells you which engine is running.

### Phase 1 — Basic SQL fast-path

| Type | Expect | If it's wrong |
|---|---|---|
| *What is the average creatinine for patients over 65?* | one number, fast (<5s) | check API key / DuckDB path |
| *How many patients had a positive blood culture?* | one count | open Query Details — confirm it hit `microbiologyevents` |
| *What was the highest heart rate recorded?* | one max value | — |

✅ **Good signs:** instant; **Query Details** shows clean SQL; labs grounded to
`itemid IN (...)`.

### Phase 2 — Complex SQL fast-path

| Type | Expect | If it's wrong |
|---|---|---|
| *Compare creatinine between male and female patients.* | a 2-row grouped table | confirm status never said "Building the knowledge graph" (should stay fast-path) |
| *Compare mean lactate between readmitted and non-readmitted patients.* | a table keyed on the readmission flag | — |
| *Average creatinine for female sepsis patients over 50 readmitted within 30 days.* | one number over the 4-filter cohort | if 0/empty, the cohort is just sparse — say so; the SQL is still correct |

✅ **Good signs:** still instant (no graph build); the interpretation block echoes
all the filters you asked for.

### Phase 3 — Visualization

| Type | Expect | If it's wrong |
|---|---|---|
| *Show the distribution of creatinine for sepsis patients as a histogram.* | a **histogram** renders below the text | if slow, narrow the cohort (add a more specific diagnosis) and re-ask |
| *Plot heart rate over the first 7 days of the ICU stay.* | a **line/scatter** trend | status will say "Building the knowledge graph…" — expected for trends |

✅ **Good signs:** an interactive Plotly chart appears. ⚠️ **Watch-out:** charts
over a broad cohort can be slow — keep cohorts narrow.

### Phase 4 — Knowledge-graph (temporal) path

| Type | Expect | If it's wrong |
|---|---|---|
| *Which antibiotics were prescribed during the first 48 hours of ICU admission?* | a table of drugs; status shows graph build | if very slow, narrow (add "for stroke patients") |
| *What are the creatinine levels during the ICU stay for sepsis patients?* | timestamped values; Query Details shows graph stats | — |
| *Creatinine values within the first 24 hours of ICU admission.* | a bounded table | — |

✅ **Good signs:** live status shows **"Building the knowledge graph…"** then
**"Reasoning over the data…"**; Query Details shows graph stats (triples/nodes).
⚠️ **Watch-out:** this is the slowest path — *always* narrow the cohort.

### Phase 5 — Cohort / patient similarity

| Type | Expect | If it's wrong |
|---|---|---|
| *Find admissions similar to a 68-year-old woman with atrial fibrillation and chronic kidney disease.* | a ranked table + **⬇ Download cohort (CSV)** button | status shows "Building the cohort definition…" → "Scoring the candidate cohort…" |
| *(follow-up)* *What was the average creatinine in this cohort?* | a metric interpreted against that profile | if it drifts, fall back to the CSV as the source of truth |
| *Find patients similar to a 75-year-old man admitted for septic shock with acute kidney injury.* | a second, different cohort + CSV | — |

✅ **Good signs:** the table is ranked by distance (nearest first); the CSV
downloads; **no internal IDs** appear in the chat prose. ⚠️ **Watch-out:** cohort
scoring over the full pool can take a while — the live status makes it clear it's
working, not hung.

---

## 5. Troubleshooting (fast lookups)

| Symptom | Likely cause | Fix |
|---|---|---|
| `ANTHROPIC_API_KEY must be set` | key missing | add to `.env` or paste in the sidebar |
| `Database not found` | wrong DuckDB path | confirm `ls data/processed/mimiciv.duckdb` |
| BigQuery `403 Access Denied` | auth/project | `gcloud auth application-default login`; check `BIGQUERY_PROJECT` |
| `ModuleNotFoundError: plotly` (or streamlit) | extras not installed | `.venv/bin/python -m pip install -e ".[conversational]"` |
| a turn spins for minutes | broad cohort on the graph/cohort path | narrow the cohort (a diagnosis, a window); status shows the live stage |
| chart doesn't appear | the answer wasn't a visualization shape | rephrase to explicitly ask to "plot" / "show a histogram of" |
| red "Analysis failed" | a pipeline exception | the error message hints the cause; re-ask after rephrasing; check the terminal log |
| answer has a yellow ⚠️ critic note | the critic flagged plausibility | this is a *feature* — read it aloud; it shows the system self-checks |

---

## 6. Day-before checklist

Run this top to bottom the day before. Every line should be ✅.

1. `make demo-check` → **36 passed**.
2. `.venv/bin/python -m pytest tests/test_conversational -q` → green (ignore the
   one `test_health_evidence/test_tools.py` case only if your DuckDB is absent).
3. `ls -lh data/processed/mimiciv.duckdb` → present (or BigQuery auth verified).
4. `bash scripts/run_chat.sh` → app loads, welcome panel shows, **Connect** → green.
5. Smoke prompt (*average creatinine for patients over 65*) → returns a value.
6. Run **one** prompt from each phase end-to-end:
   - P1: *How many patients had a positive blood culture?*
   - P2: *Compare mean lactate between readmitted and non-readmitted patients.*
   - P3: *Show the distribution of creatinine for sepsis patients as a histogram.*
   - P4: *Which antibiotics were prescribed during the first 48 hours of ICU admission?*
   - P5: *Find admissions similar to a 68-year-old woman with atrial fibrillation and chronic kidney disease.* → confirm the **CSV** downloads.
7. Pick your **narrow cohorts** for P3/P4/P5 and note them — broad cohorts are the
   only real live-demo risk.

If all seven pass, you're ready. Keep `01_demo_outline.md` open on a second screen
as your script, and the one-screen cheat sheet at its bottom as your fallback.
