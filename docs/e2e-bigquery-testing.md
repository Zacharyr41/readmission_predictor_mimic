# End-to-End Testing Against MIMIC-IV on BigQuery

This guide walks through running the conversational pipeline end-to-end
against MIMIC-IV **without downloading any data locally**. Queries execute
against the public `physionet-data.mimiciv_*` BigQuery datasets.

Target audience: clinicians / data scientists who want to validate the
pipeline against real MIMIC data before wiring up a local DuckDB mirror.

## What "end-to-end" means here

One `ask()` call in the conversational pipeline runs all 5 stages:

1. **Decompose** — LLM turns natural language into one or more
   `CompetencyQuestion`s.
2. **Extract** — SQL against MIMIC-IV: patients + admissions + ICU stays +
   events matching the CQs. Rows come back in batches.
3. **Graph build** — RDF knowledge graph of the extracted data (one graph
   per turn, even when the decomposer produced multiple CQs).
4. **Reason** — SPARQL against the graph.
5. **Answer** — Claude writes a 1–3 sentence summary + optional table + chart.

The E2E test below exercises all 5 stages against BigQuery.

## Prerequisites

1. **Anthropic API key** for Claude. Export it or paste it into the
   Streamlit sidebar later:
   ```bash
   export ANTHROPIC_API_KEY="sk-ant-..."
   ```

2. **Google Cloud access to the PhysioNet MIMIC-IV dataset.** You need:
   - A GCP project (billing enabled — BigQuery charges per byte scanned).
   - Access to `physionet-data` — follow the
     [PhysioNet credentialed access instructions](https://physionet.org/content/mimiciv/)
     and link your Google account on your profile page.
   - Application Default Credentials set up locally:
     ```bash
     gcloud auth application-default login
     ```

3. **Python environment** — from the repo root:
   ```bash
   uv venv --python 3.11
   uv pip install --python .venv/bin/python -e ".[conversational,dev]"
   ```

4. **Optional: SNOMED hierarchy cache.** If you have
   `data/ontology_cache/snomed_hierarchy.json` generated (see
   `scripts/build_snomed_hierarchy.py`), the resolver will use it as a
   fallback for concepts not in `category_to_snomed.json`. Absent = the
   resolver just uses the curated category map. No setup needed to get
   started.

## Launch the Streamlit UI

```bash
streamlit run src/conversational/app.py
```

In the sidebar:

1. **Data source** → select **"BigQuery"**.
2. **GCP project ID** — type your project. The sidebar pre-fills from the
   `BIGQUERY_PROJECT` env var if set.
3. **Anthropic API key** — paste it if not exported.
4. **Advanced Settings** (optional):
   - **Batch size** (default 2000): controls how many `hadm_id`s go into
     each `IN (…)` clause. Lower if you hit BigQuery parameter limits;
     higher for fewer round-trips.
   - **Cohort strategy**: `recent` sorts by admittime desc; `random`
     shuffles. Ordering only — the cohort cap is gone, all matching
     admissions are returned.
   - **Parallel workers**: graph-build concurrency.
5. Click **Connect**.

## Smoke test — three questions

Run these three questions in order. Each exercises a different code path.

### 1. Simple single-CQ question

Type:

> **What is the average creatinine for patients over 65?**

**Expect:**
- Sidebar status: "Connected" (green).
- Above the answer, a blue info box: **“Interpreting as:** Mean serum
  creatinine in admissions with patient anchor_age > 65.”
- A 1–3 sentence answer citing the mean value.
- A "Query Details" expander at the bottom with the SPARQL + graph stats.

**If this fails:** look at the Streamlit terminal for a traceback. The
most common issues are listed in the troubleshooting section below.

### 2. Ambiguous question → clarifying response

Type:

> **Show me the labs.**

**Expect:**
- Blue info box: **“Interpreting as:** Unclear: no specific lab, cohort,
  or time window provided.”
- Bold text in the body: **"Which lab(s) would you like to see, and for
  which patients or time window?"**
- A caption: *"Reply with more detail and I'll re-run the analysis."*
- **No Query Details expander** — the pipeline short-circuited and never
  executed any SQL or SPARQL.

This proves the Phase 4 clarifying-question short-circuit works against
BigQuery: no wasted query cost on ambiguous input.

### 3. Big question → multi-CQ decomposition with one shared graph

Type:

> **Why do our sepsis patients keep getting readmitted within 30 days?**

**Expect:**
- Blue info box with a one-sentence narrative explaining the breakdown.
- Top-level text summarising the multi-part answer.
- A **"Breakdown:"** divider, followed by 2–4 expanders labelled "Part 1",
  "Part 2", … Each expander contains:
  - Its own *Sub-question interpretation* block.
  - A per-sub-CQ text summary + optional table.
- A single **Query Details** expander at the bottom aggregating graph
  stats + every SPARQL query executed across sub-CQs.

**Correctness guarantees:**
- Exactly **one** knowledge graph is constructed for this turn — not one
  per sub-CQ. Confirm this by checking `graph_stats` in the Query
  Details expander: it's a single aggregate dict, not a list.
- All Allen temporal relations are computed once across the union of
  events from every sub-CQ.
- Cross-CQ relationships hold: the same patient's readmission flag and
  the same patient's labs live in the same graph.

## Cost expectations

BigQuery charges per byte scanned. Rough estimates for the three smoke
tests above:

| Question | Data scanned | Cost (us-central1 $6.25/TB) |
|---|---|---|
| 1. Creatinine over 65 | ~1–3 GB | ~$0.02 |
| 2. "Show me the labs" (clarify) | 0 bytes (short-circuit) | **$0.00** |
| 3. Sepsis readmission (big Q, 3 sub-CQs) | ~5–10 GB | ~$0.05 |

Phase 2 removed the 500-row cohort cap — queries now return every
matching admission. For broad filters ("all patients", "anyone over 18")
scanning costs can climb. Always narrow with filters (diagnosis, age,
readmission status) before running expensive questions.

## Observability in the UI

Every answer carries metadata you can inspect:

- **Interpretation block** (always present) — verify the pipeline
  understood your question before reading the result.
- **Query Details expander**:
  - `graph_stats`: triples written per class (Patient, Admission, ICU,
    BioMarkerEvent, …). For a multi-CQ turn, these are the merged counts.
  - Each SPARQL query emitted (one block per template + temporal overlay).
  - For multi-CQ turns, queries from every sub-CQ are aggregated here.

If an answer looks wrong, check the interpretation block first. If the
pipeline misunderstood, restate the question more concretely. If the
interpretation is right but the answer is wrong, inspect the SPARQL and
graph stats.

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `google.api_core.exceptions.PermissionDenied: 403` | GCP account not linked to PhysioNet, or ADC credentials missing. Run `gcloud auth application-default login` and confirm your PhysioNet profile shows the Google account. |
| `google.api_core.exceptions.BadRequest 400 ... ORDER BY` | Should not occur after Phase 2 (the ORDER BY is on a column visible after DISTINCT). If it does, file a bug — include the full SQL from the exception message. |
| Streamlit hangs on "Analyzing..." for >3 min | BigQuery query is scanning too much. Narrow your filters, reduce `batch_size` for memory pressure, or cancel. Check the BigQuery console UI for the running job. |
| Answer says "An error occurred while processing your question" | Generic fallback. Check the Streamlit terminal for the traceback. |
| `ModuleNotFoundError: plotly` | Install conversational extras: `uv pip install --python .venv/bin/python -e ".[conversational]"` |
| Blue interpretation block says "Unclear" for a concrete question | The LLM's decomposer couldn't map your concept to a supported concept_type. Rephrase using terms from the "Concept Types" list in the system prompt (see `tests/test_conversational/fixtures/prompts_snapshot.txt`). |
| Multi-CQ response always collapses to one sub-CQ | The LLM is conservative on decomposition; only clearly multi-part questions trigger Shape B. Try broader phrasings ("why do our … ?", "how do … differ?"). |

## Programmatic E2E (no UI)

If you want to script the pipeline instead of using the UI:

```python
from pathlib import Path
import os
from src.conversational.orchestrator import ConversationalPipeline
from src.conversational.models import ExtractionConfig

pipeline = ConversationalPipeline(
    db_path=Path("/unused/for/bigquery"),     # placeholder, not read
    ontology_dir=Path("ontology/definition"),
    api_key=os.environ["ANTHROPIC_API_KEY"],
    data_source="bigquery",
    bigquery_project="your-gcp-project",
    extraction_config=ExtractionConfig(batch_size=2000),
    max_workers=1,
)

answer = pipeline.ask("What is the average creatinine for patients over 65?")
print(answer.interpretation_summary)
print(answer.text_summary)
if answer.sub_answers:
    for i, sub in enumerate(answer.sub_answers, 1):
        print(f"\n--- Part {i} ---")
        print(sub.text_summary)
```

For multi-turn conversations the pipeline maintains its own
`conversation_history`; call `pipeline.reset()` to clear.

## Regression tests (no BigQuery required)

The full regression suite runs offline against a synthetic DuckDB fixture
and mock LLM responses — no GCP credentials needed:

```bash
pytest tests/test_conversational/ -q
# expect: 400+ passed
```

The suite covers:
- Every supported filter field (parity against the pre-refactor SQL).
- Every aggregate keyword (routed to the right SPARQL template).
- Every comparison axis (clause parity with the old `_COMPARISON_FIELD_MAP`).
- Shape A and Shape B decomposition parsing.
- Multi-CQ orchestration (ONE graph per turn, booby-trapped mocks verify
  no extra graph builds).
- Clarify short-circuit (every downstream stage booby-trapped to assert
  it didn't run).
- Interpretation-summary synthesis across 40+ behavioural fixtures.
- hypothesis-driven fuzz over `_validate_return_type`,
  `_synthesise_interpretation`, `_extract_json`, and JSON round-trips.

Adding a new behavioural scenario is usually a one-file drop under
`tests/test_conversational/fixtures/decomposer_cases/` — see the plan's
"Tests exercise behaviour, not mock shape" principle.

## Live-LLM manual tests

A small set of cases are marked `@pytest.mark.live_llm` — disabled by
default. To hand-check real Claude responses against the current prompt:

```bash
pytest tests/test_conversational/ -m live_llm
```

These are sanity checks, not gate-kept CI tests. They require
`ANTHROPIC_API_KEY` and cost a few cents per run.
