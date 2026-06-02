# Outlier Screening — Manual Test Guide

> Hands-on guide for verifying the **pre-aggregation biological-impossibility
> screening** added to the chat SQL fast-path. Three independent checks, fastest
> first: the automated suite, a no-dependencies headless demo, and the live
> Streamlit chat UI.

## What you're testing

A clinician asked for the **average lactate among patients aged ≥ 65** and got a
mean > 8 mmol/L. The cause was a single `labevents` row whose lactate `valuenum`
had been entered as **1,000,000** (a data-entry error). Because the SQL fast-path
aggregates *in the database* (`AVG(valuenum)`), that one poison value polluted
the result and the per-row values never reached Python where anything could
catch them.

The fix screens each measurement against an **absolute biological-possibility
envelope** for the analyte (lactate: `[0, 40] mmol/L`) *before* aggregation, in
SQL, via a portable `CASE WHEN valuenum BETWEEN lo AND hi` predicate:

1. **Identify** — resolve the `[low, high]` envelope for the analyte (LOINC-keyed
   cache, `data/ontology_cache/biological_limits.json`).
2. **Log** — fetch only the out-of-envelope rows, with context columns.
3. **Remove pre-aggregation** — outliers never enter `AVG/MAX/MIN/COUNT`.
4. **Aggregate** — the clean value flows downstream unchanged (`mean_value`).
5. **Present** — an expander lists the removed rows; a live toggle flips the
   answer to *include* them.

High-but-real values (e.g. sepsis lactate 12) are **kept** — only physiologically
impossible values are removed.

### Units guard

Bounds are unit-tagged. A row is screened **only** if its recorded unit is NULL
or matches the bound's unit (case/whitespace-insensitive). A row carrying a
*different* present unit (e.g. a `mg/dL` value where the bound is `mmol/L`) is
**never silently dropped** — mismatched-unit rows are kept, not falsely removed.

---

## 1. Automated test suite (fastest)

```bash
.venv/bin/python -m pytest tests/test_conversational/test_outlier_detection.py -v
```

**Expected:** `20 passed` in well under a second. Covers bug replication, the
removed-row log, the bound resolver, the emitted SQL (envelope + units guard),
GROUP-BY removal, the no-false-removal correctness guard, the grounded fallback,
graceful skip, COUNT screening, and the units guard.

Full suite (no regressions expected from this feature):

```bash
.venv/bin/python -m pytest tests/ -q
```

> Note: two `test_decomposer_fuzz::TestJsonRoundTrip` failures are **pre-existing**
> and unrelated to outlier screening (a triple-backtick fence breaks
> `extract_json`). Three neo4j tests skip. Everything else is green.

---

## 2. Headless demo (no API key, no Streamlit)

This proves the screen end-to-end against a throwaway DuckDB — it exercises the
real `compile_sql` fast-path, so it's the most direct way to *see* the numbers
move. Save the script below as `outlier_demo.py` at the repo root and run it:

```bash
.venv/bin/python outlier_demo.py
```

<details>
<summary><code>outlier_demo.py</code> (click to expand)</summary>

```python
"""Headless demo of pre-aggregation outlier screening (no API key needed)."""
import os
import tempfile

import duckdb

from src.conversational.extractor import _DuckDBBackend
from src.conversational.models import (
    ClinicalConcept,
    CompetencyQuestion,
    PatientFilter,
)
from src.conversational.operations import get_default_registry
from src.conversational.sql_fastpath import OutlierScreen, compile_sql


class ConnBackend(_DuckDBBackend):
    def __init__(self, conn):
        self._conn = conn

    def close(self):
        pass


def build_db():
    path = os.path.join(tempfile.mkdtemp(), "outlier_demo.duckdb")
    conn = duckdb.connect(path)
    conn.execute(
        "CREATE TABLE patients (subject_id INTEGER, gender VARCHAR, "
        "anchor_age INTEGER, anchor_year INTEGER, dod DATE)"
    )
    conn.execute(
        "INSERT INTO patients VALUES "
        "(1,'M',65,2150,NULL),(2,'F',72,2151,NULL),(5,'M',80,2151,NULL)"
    )
    conn.execute(
        "CREATE TABLE admissions (hadm_id INTEGER, subject_id INTEGER, "
        "admittime TIMESTAMP, dischtime TIMESTAMP, admission_type VARCHAR, "
        "discharge_location VARCHAR, hospital_expire_flag INTEGER)"
    )
    conn.execute(
        "INSERT INTO admissions VALUES "
        "(101,1,NULL,NULL,'EMERGENCY','HOME',0),"
        "(103,2,NULL,NULL,'ELECTIVE','SNF',0),"
        "(106,5,NULL,NULL,'EMERGENCY','HOSPICE',1)"
    )
    conn.execute(
        "CREATE TABLE d_labitems (itemid INTEGER, label VARCHAR, "
        "fluid VARCHAR, category VARCHAR)"
    )
    conn.execute("INSERT INTO d_labitems VALUES (50813,'Lactate','Blood','Chemistry')")
    conn.execute(
        "CREATE TABLE labevents (labevent_id INTEGER, subject_id INTEGER, "
        "hadm_id INTEGER, stay_id INTEGER, itemid INTEGER, charttime TIMESTAMP, "
        "valuenum DOUBLE, valueuom VARCHAR, ref_range_lower DOUBLE, "
        "ref_range_upper DOUBLE)"
    )
    conn.execute(
        "INSERT INTO labevents VALUES "
        "(50,1,101,1001,50813,'2150-01-16 06:00:00',1.2,'mmol/L',0.5,2.0),"
        "(51,2,103,1002,50813,'2151-03-03 08:00:00',2.5,'mmol/L',0.5,2.0),"
        "(52,5,106,1003,50813,'2151-04-12 10:00:00',12.0,'mmol/L',0.5,2.0),"
        "(53,1,101,1001,50813,'2150-01-16 07:00:00',1000000.0,'mmol/L',0.5,2.0),"
        "(54,2,103,1002,50813,'2151-03-03 09:00:00',90.0,'mg/dL',0.5,2.0)"
    )
    return ConnBackend(conn)


def run(label, screen):
    backend = build_db()
    cq = CompetencyQuestion(
        original_question="average lactate among patients aged >= 65",
        clinical_concepts=[
            ClinicalConcept(name="lactate", concept_type="biomarker", loinc_code="2524-7")
        ],
        patient_filters=[PatientFilter(field="age", operator=">=", value="65")],
        aggregation="mean",
        scope="cohort",
        comparison_field=None,
        return_type="text_and_table",
    )
    q = compile_sql(
        cq, backend, get_default_registry(),
        resolved_itemids=[50813], outlier_screen=screen,
    )
    cols = q.outlier_agg_columns or q.columns
    row = dict(zip(cols, backend.execute(q.sql, q.params)[0]))
    print(f"\n=== {label} ===")
    print(f"  clean mean        = {row['mean_value']:.4f}")
    print(f"  with-outliers mean= {row['mean_value_with_outliers']:.4f}")
    print(f"  n_outliers        = {row['n_outliers']}   n_total = {row['n_total']}")
    if q.outlier_rows_sql:
        removed = [
            dict(zip(q.outlier_rows_columns, r))
            for r in backend.execute(q.outlier_rows_sql, q.outlier_rows_params)
        ]
        print(f"  removed rows      = {[(r['valuenum'], r['valueuom']) for r in removed]}")


run("WITH units guard (units='mmol/L')", OutlierScreen(low=0.0, high=40.0, units="mmol/L"))
run("WITHOUT units guard (units=None)", OutlierScreen(low=0.0, high=40.0))
```

</details>

### Expected output

```text
=== WITH units guard (units='mmol/L') ===
  clean mean        = 26.4250
  with-outliers mean= 200021.1400
  n_outliers        = 1   n_total = 5
  removed rows      = [(1000000.0, 'mmol/L')]

=== WITHOUT units guard (units=None) ===
  clean mean        = 5.2333
  with-outliers mean= 200021.1400
  n_outliers        = 2   n_total = 5
  removed rows      = [(1000000.0, 'mmol/L'), (90.0, 'mg/dL')]
```

### How to read it

The cohort has five lactate rows: `1.2`, `2.5`, `12.0` (all `mmol/L`, all kept),
`1,000,000` (`mmol/L`, the poison row), and `90.0` (`mg/dL`, a wrong-unit row).

- **`with-outliers mean = 200021.14`** is the polluted answer the chatbot used to
  return — the `1e6` row dominates `AVG()`. This is the bug.
- **WITH the units guard**, only the `mmol/L` `1e6` row is impossible, so it's the
  sole removal. `clean mean = (1.2 + 2.5 + 12.0 + 90.0) / 4 = 26.425`. The `90.0`
  is kept because its unit (`mg/dL`) differs from the bound's (`mmol/L`) — the
  guard refuses to judge a value it can't compare.
- **WITHOUT the guard**, both `> 40` rows are removed regardless of unit, so the
  `90.0 mg/dL` row is **falsely** dropped too (`n_outliers = 2`). `clean mean =
  (1.2 + 2.5 + 12.0) / 3 = 5.233`. This is exactly the false-removal the units
  guard prevents.

> In production the units come from the resolved analyte
> (`biological_limits.json` → lactate → `mmol/L`), so the guard is **on** by
> default. The "without" run is shown only to demonstrate what the guard buys you.

When you're done, delete the throwaway script: `rm outlier_demo.py`.

---

## 3. Streamlit chat UI (the real thing)

This is the end-to-end path a clinician actually sees — the expander and the live
include/exclude toggle.

### Launch

```bash
bash scripts/run_chat.sh
```

In the sidebar: **Data source → Local DuckDB**, point at your DuckDB
(`data/processed/mimiciv.duckdb`), let the API key auto-fill from `.env`, and
click **Connect**. (See [`chat-ui-quickstart.md`](./chat-ui-quickstart.md) for the
full launch options, including BigQuery.)

> To reproduce the exact demo numbers you'd need a lactate row with
> `valuenum = 1e6` in your data. Against real MIMIC-IV the screen still fires on
> whatever impossible values exist; the *behavior* below is identical regardless
> of the specific poison value.

### Ask

> **What is the average lactate among patients aged 65 and older?**

### What you should see

1. **The answer + data table.** The text summary reports the **clean** mean, and
   the table shows a **Mean Value** column with that screened value — *not* the
   polluted one.
2. **An expander: `🔍 Removed 1 impossible outlier`.** Expand it to see:
   - a caption naming the envelope, e.g. *"Screened out values outside the
     biological-possibility envelope `[0, 40] mmol/L` for **lactate** (source:
     seed:literature). These are physiologically impossible / data-entry errors,
     not high-but-real values."*
   - a table of the removed row(s) with their context columns (subject, hadm,
     charttime, label, value, unit).
3. **A checkbox: `Include outliers in the result`.** Tick it and the answer
   **instantly** re-renders to the with-outliers value, with a blue note:
   *"Including 1 outlier, the value is **200021 mmol/L** (vs. the screened answer
   above)."* Untick it to return to the clean default.

The toggle is an instant re-render — both values are precomputed on the backend,
so there's **no second query and no second LLM call**.

### Pass criteria

- [ ] Default answer is the **clean** mean (the impossible value is excluded).
- [ ] The expander appears and lists the removed row(s) with bounds + source.
- [ ] Toggling **Include outliers** swaps to the polluted value and back, live.
- [ ] A high-but-real value (e.g. an actual sepsis lactate of 12) is **kept** —
      never appears in the removed-rows table.

---

## Where the pieces live

| Concern | Location |
| --- | --- |
| Bound envelopes (LOINC-keyed cache + seed) | `data/ontology_cache/biological_limits.json` |
| Resolver (`BiologicalLimitsResolver`) | `src/conversational/outliers.py` |
| SQL emission (dual aggregate + companion rows + units guard) | `src/conversational/sql_fastpath.py` (`_compile_event_aggregate`, `OutlierScreen`) |
| Orchestrator wiring (`_resolve_outlier_screen`, report build) | `src/conversational/orchestrator.py` |
| `OutlierReport` model | `src/conversational/models.py` |
| UI expander + toggle | `src/conversational/app.py` (`_render_outlier_panel`) |
| Config knobs (`outlier_*`) | `config/settings.py` |
| Tests | `tests/test_conversational/test_outlier_detection.py` |
