# Phase 7a Smoke Test — BigQuery E2E

Quick verification that the Phase 7a query planner + SQL fast-path cut
latency on real MIMIC-IV. Target: the question *"average creatinine for
patients over 65"* — the original pain point that took 10–30+ minutes
pre-7a — should now complete in **under 30 seconds**.

## Launch

```bash
set -a && source .env && set +a && .venv/bin/streamlit run src/conversational/app.py
```

Sidebar: **BigQuery** · GCP project auto-fills from `.env` · Anthropic API
key auto-fills · **Connect**.

## Smoke questions

Run in order. Each exercises a different code path. The hypothesis-style
questions (Q4–Q7) are the ones a clinical researcher actually asks — "is
signal X different between patients with/without property Y?" — so they're
the real test of whether the system is useful, not just fast.

| # | Question | Planner route | Target |
|---|---|---|---|
| 1 | **Show me the labs.** | clarify short-circuit | <5s, $0 |
| 2 | **How many sepsis admissions did we have?** | `SQL_FAST` (diagnosis count) | <15s |
| 3 | **What is the average creatinine for patients over 65?** | `SQL_FAST` (biomarker AVG) | **<30s** ← the original pain point |
| 4 | **Compare mean lactate between patients readmitted within 30 days and those not readmitted.** | `SQL_FAST` (GROUP BY readmitted_30d) | <30s |
| 5 | **Compare mean serum creatinine between emergency and elective admissions.** | `SQL_FAST` (GROUP BY admission_type) | <30s |
| 6 | **How many sepsis patients received vancomycin?** | `SQL_FAST` (drug count w/ diagnosis filter) | <30s |
| 7 | **What is the mean heart rate in admissions with sepsis?** | `SQL_FAST` (vital AVG w/ diagnosis filter) | <30s |
| 8 | **Why do our sepsis patients keep getting readmitted within 30 days?** | mixed big-Q (3 sub-CQs) | <1 min |

### Graph-complex questions (Phase 7b's real target)

These are where the graph is the computation — not just SQL in disguise.
They all take the **graph path** (temporal Allen relations, time-series
rendering, median with Python post-processor, or multi-event relational
reasoning). Phase 7b's parallel batched extraction is designed to make
these viable on BigQuery where they were previously painful.

| # | Question | Why it's graph-complex | Target |
|---|---|---|---|
| G1 | **Plot the creatinine trajectory over time for patients admitted with sepsis.** | Time-series viz (`value_with_timestamps`). Graph owns timestamp+interval semantics. Parallel batches matter a lot. | <3 min |
| G2 | **What was the median ICU length of stay for patients readmitted within 30 days?** | `aggregation=median` forces graph path (no portable SQL percentile). Python post-processor. | <2 min |
| G3 | **Which antibiotics were prescribed during the first 24 hours of ICU admission for sepsis patients?** | Temporal `within 24h` + `drug_lookup` — Allen relations on ICU intervals. | <3 min |
| G4 | **What were the creatinine levels before intubation for septic patients?** | Temporal `before` relation on a clinical event. **SQL genuinely can't answer this cleanly** — the killer graph use case. | <3 min |
| G5 | **Plot how heart rate changed over the first 48 hours of ICU admission for sepsis patients.** | Time-series viz + temporal `within 48h` + filter. Demonstrates multi-patient aggregate trajectory. | <3 min |
| G6 | **How does the lactate trajectory differ between patients who survived and those who died from sepsis?** | Big-Q that likely decomposes into survivor vs non-survivor cohorts, both with time-series. Exercises Phase 4.5 (ONE graph) + Phase 7b (parallel batches) together. | <5 min |
| G7 | **Which medications were given in the 12 hours before a positive blood culture in sepsis patients?** | Temporal `before` + drug concept + microbiology concept + diagnosis filter. Multi-concept, multi-temporal — exactly what the graph is for. | <5 min |
| G8 | **Compare the distribution of ICU length of stay between patients who received vasopressors and those who didn't.** | Drug exposure as a cohort definer (not a comparison axis — the LLM has to emit a filter) + LOS distribution. Stretches what the system can do. | <5 min |

### How to interpret graph-path latency

Graph-path time breaks down as: **cohort query** (≤10s) · **parallel
batched event fetches** (Phase 7b target — 30s-2min) · **graph
construction** (scales with events; 30s-5min) · **SPARQL reason** (seconds).

The remaining bottleneck post-7b is **graph construction** for very
wide cohorts. If G1/G3/G5 take 3-5 minutes, that's expected until a
future Phase 7c adds graph-builder parallelism / lazy event placement.
The fact they finish AT ALL on BigQuery in <5 min is the Phase 7b win —
pre-7b, the sequential batched extraction alone would be 10-20 min
before the graph build even started.

### Why these hypothesis-style questions matter

- **Q4** — classic research hook: lactate is a sepsis-severity marker.
  If readmitted patients had higher lactate on their index admission,
  that's a signal the severity biomarker might predict bounceback. This
  is the clinical-research shape the system is ultimately designed for.
- **Q5** — acuity effect on renal function. Emergency admissions should
  have higher creatinine (acute kidney injury is more common on
  unplanned admission). A null result would be suspicious.
- **Q6** — drug exposure count in a cohort. Answers "how common is X in
  condition Y." Fast-path even though it has a filter + drug concept.
- **Q7** — vital-sign aggregate with a diagnosis filter. Forces the
  fast-path to compose a diagnosis JOIN with a chartevents aggregate.

### Limitation worth knowing — "does drug X affect outcome in condition Y?"

The obvious next research question is: *"Does metformin improve sepsis
mortality?"* — compare outcomes in metformin-treated vs untreated
sepsis patients. **This doesn't route cleanly today** because the
system's comparison axes are gender, age, readmitted_30d/60d,
admission_type, and discharge_location — **"received drug X"** is not a
registered comparison axis.

Workaround: ask two questions and compare manually.

1. *"How many sepsis patients who received metformin were readmitted within 30 days?"*
2. *"How many sepsis patients who did NOT receive metformin were readmitted within 30 days?"*

Question 1 routes `SQL_FAST` (drug count + diagnosis + readmitted filter).
Question 2 depends on how the LLM expresses "not receiving" — may hit
the graph path or generate a clarifying question.

Adding a **drug-exposure comparison axis** is a natural follow-up; logged
as a candidate for Phase 8 if the clinical-research use case proves
hot.

### What to look for in the UI

- **Blue "Interpreting as:"** info block above every answer (Phase 4 UX).
- Question 1: **bold clarifying follow-up** + the "Reply with more detail"
  caption. **No** data table, **no** Query Details expander.
- Question 5: **"Breakdown:"** divider with 2–4 per-sub-CQ expanders, each
  with its own interpretation and summary. Query Details at the top level
  aggregates SQL + SPARQL across sub-CQs.

### What to look for in Query Details

- **SQL_FAST** route → expander shows a single SQL statement (e.g.
  `SELECT AVG(l.valuenum) AS mean_value FROM …`). **No SPARQL** present.
- **GRAPH** route → SPARQL queries plus `graph_stats` (triples per class).
- Multi-CQ mixed → mix of SQL + SPARQL, one per sub-CQ.

## Reporting

For each question report back:

1. **Wall-clock** (rough, e.g. "~20s").
2. **Blue interpretation block present?** y/n + the displayed text.
3. **Errors in the Streamlit terminal?** paste traceback.
4. **For #3** specifically: under 30s ⇒ Phase 7a works end-to-end. Over
   30s ⇒ something's still routing to graph or BigQuery itself is slow;
   share what the Query Details expander shows.
5. **For #5**: did the "Breakdown:" render with multiple expanders?

### Copy-paste template

```
Q1 ("Show me the labs."):
  wall-clock:
  interpretation block:
  render OK? (y/n):
  notes:

Q2 (sepsis count):
  wall-clock:
  interpretation:
  Query Details — SQL or SPARQL?:
  numeric answer (rough):
  notes:

Q3 (avg creatinine over 65):
  wall-clock:
  interpretation:
  Query Details — SQL or SPARQL?:
  numeric answer (rough mg/dL):
  notes:

Q4 (lactate by readmission_30d):
  wall-clock:
  interpretation:
  Query Details — SQL or SPARQL?:
  numeric answer — readmitted arm vs not:
  clinical plausibility (higher in readmitted? y/n):
  notes:

Q5 (creatinine by admission_type):
  wall-clock:
  interpretation:
  Query Details — SQL or SPARQL?:
  numeric answer — EMERGENCY vs ELECTIVE vs URGENT:
  clinical plausibility (higher in EMERGENCY? y/n):
  notes:

Q6 (sepsis pts with vancomycin):
  wall-clock:
  interpretation:
  Query Details — SQL or SPARQL?:
  numeric answer:
  notes:

Q7 (mean heart rate in sepsis):
  wall-clock:
  interpretation:
  Query Details — SQL or SPARQL?:
  numeric answer (bpm):
  clinical plausibility (tachycardic >90? y/n):
  notes:

Q8 (sepsis readmission big-Q):
  wall-clock:
  narrative shown:
  # of sub-answer expanders:
  Query Details — mix of SQL + SPARQL?:
  notes:

--- Graph-complex questions ---

G1 (creatinine trajectory, sepsis):
  wall-clock:
  Query Details — SPARQL present?:
  plot rendered? (y/n):
  notes:

G2 (median ICU LOS, readmitted):
  wall-clock:
  Query Details — SPARQL + median post-processor?:
  numeric answer (days):
  notes:

G3 (antibiotics first 24h ICU, sepsis):
  wall-clock:
  Query Details — SPARQL + temporal_during/within?:
  table rendered? (y/n):
  notes:

G4 (creatinine before intubation):
  wall-clock:
  Query Details — SPARQL + temporal_before?:
  intuitively sensible values? (y/n):
  notes:

G5 (heart rate trajectory first 48h):
  wall-clock:
  plot rendered? (y/n):
  notes:

G6 (lactate survivor vs non-survivor, big-Q):
  wall-clock (total):
  # of sub-CQs:
  graph built once? (check Query Details for single graph_stats):
  notes:

G7 (meds before positive blood culture):
  wall-clock:
  Query Details — SPARQL multi-event?:
  rows returned:
  notes:

G8 (LOS by vasopressor exposure):
  wall-clock:
  did the LLM clarify? (expected — "received X" isn't a comparison axis):
  notes:
```

## If something is slow or wrong

- **>30s on Q3 / Q4 / Q5 / Q6 / Q7** — planner probably routed to graph.
  Open Query Details: if SPARQL is present, the planner misrouted; if
  it's SQL and still slow, BigQuery itself is cold-starting or the
  project is constrained.
- **Misclassification on Q4 / Q5** — these rely on `readmitted_30d` and
  `admission_type` being registered comparison axes with `sql_group_by`.
  Both are seeded in `operations_comparison.py`. If routed to graph,
  check the LLM's CQ: it may have emitted an unregistered axis name
  (e.g. "readmission_status" instead of "readmitted_30d"), which the
  planner treats as graph-path.
- **Clinical implausibility on Q4 / Q5 / Q7** — if the numbers look way
  off (e.g. sepsis HR of 60 bpm, creatinine 10x expected), capture the
  SQL from Query Details and share it. The fast-path joins admissions
  + optional patients + filter-compiled WHERE; a join bug would show up
  here first.
- **Errors** — the Streamlit terminal has the full traceback. Paste it
  here; the most useful lines are the final exception + the frame in
  `src/conversational/*.py`.
- **No clarify on Q1** — the LLM produced a CQ instead of a clarifying
  question; check the blue interpretation block for what it "thought"
  the question was asking.

## Stop the server

```bash
lsof -ti :8501 | xargs kill -9
```
