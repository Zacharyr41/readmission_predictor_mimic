# Cohort-by-Similarity — Hand-Testing Guide

> Drive the **anchorless cohort** path end-to-end in the Streamlit chat UI:
> describe a set of patient characteristics in plain English and get back a
> *ranked cohort* of similar admissions, scored by **Gower's distance** to a
> synthesized reference profile. This guide gives exact questions to type, what
> the answer should look like, and how to audit and tune it.
>
> **See also:** [`chat-ui-quickstart.md`](./chat-ui-quickstart.md) (launching the
> app), [`gower_distance_design.md`](./gower_distance_design.md) (the distance
> math), [`phase9-patient-similarity.md`](./phase9-patient-similarity.md) (the
> anchor-on-a-patient special case this generalizes).

---

## 0. What you are testing

A free-text request like *"emergency patients like a 68-year-old whose creatinine
ran high"* is turned — by a single **Sonnet** call (the *definition builder*) —
into a strict `CohortDefinition`: a list of **prefilters** (a cheap Boolean gate
that narrows the candidate pool) and a list of **traits** (the Gower columns).
Each candidate admission is then scored by its Gower distance to the profile
those traits describe, and the **cohort is every candidate with
`distance ≤ distance_threshold`**, ranked nearest-first and capped at `top_k`.

Two kinds of trait exist, and the routing differs:

- **`source="sql"`** — pulled straight from the database (age, gender,
  `creatinine_max`, `icu_los_hours`, …). A pure-SQL cohort **skips the graph**.
- **`source="graph_temporal"`** — a *temporal* feature (a lab slope, a dose
  trajectory, ICU length-of-stay, a precedence count) computed from a
  per-question **RDF graph** built over the candidate pool. Any one such trait
  makes the run build the graph.

Each trait also carries a **kernel** chosen from the clause's *wording*:

| You wrote… | Kernel (`direction`) | Meaning |
|---|---|---|
| "68 years old", "sodium ~140" | `symmetric` | closeness to a point |
| "female", "emergency admission" | nominal match | equal or not |
| "creatinine ran **high**", "**worsening**", "**escalating**" | `higher_more_similar` | more-extreme = at-least-as-similar; **never** penalized for exceeding |
| "platelets **dropped**", "stay on the **shorter** side" | `lower_more_similar` | mirror of the above |

This is the load-bearing correctness behavior: a one-sided kernel must **not**
treat a patient who is *worse than* the reference as *less* similar.

---

## 1. Prerequisites & launch

```bash
# Conversational extras (streamlit, plotly, anthropic) + an API key in .env
uv pip install --python .venv/bin/python -e ".[conversational]"
#   ANTHROPIC_API_KEY=sk-ant-...   (auto-fills the sidebar)
```

The cohort code path is **identical** on either backend — only the data source
differs. Pick one:

### Option A — BigQuery (full MIMIC-IV; use this for real temporal cohorts)

BigQuery carries the **complete** lab/chart history, so lab-slope and
dose-trajectory traits actually populate (see the caveat below for why the
bundled DuckDB does not). This is the right backend for the §3 "worsening labs"
test.

```bash
# 1. Authenticate
gcloud auth login
gcloud auth application-default login

# 2. Point at your MIMIC-IV project (should match BIGQUERY_PROJECT in .env)
gcloud config set project <your-gcp-project-with-mimic-iv-access>

# 3. Verify dataset access (if it hangs, press Enter — bq may be waiting on a
#    prompt you can't see)
bq ls physionet-data:mimiciv_3_1_hosp

# 4. Launch
bash scripts/run_chat.sh
```

In the **sidebar**:

- **Data source:** `BigQuery`
- **GCP project ID:** `<your-gcp-project-with-mimic-iv-access>` (auto-fills from `.env`)
- Click **Connect** → you should see **Connected**.

> **Scale & cost (read before the temporal test).** BigQuery bills **per byte
> scanned**, and a cohort scores the *whole* matching pool — there is no row cap.
> A graph-temporal cohort additionally builds an RDF graph over that pool, so an
> unfiltered temporal request can scan a lot and run for minutes. Two habits keep
> it cheap and fast:
> - **Always lead with a crisp prefilter** ("**ICU** patients", "with **sepsis**",
>   "**emergency** admissions") so the pool is narrowed *before* scoring. Confirm
>   the prefilter landed in the JSONL log (§4).
> - For temporal cohorts, raise **Concurrent batches** (e.g. 8→16) and **Parallel
>   workers** (e.g. 1→4) in the sidebar to parallelize the graph build.
>
> If Streamlit hangs on "Analyzing…" for >3 min the query is scanning too much —
> tighten the filters or cancel and watch the job in the BigQuery console. See
> [`e2e-bigquery-testing.md`](./e2e-bigquery-testing.md) for cost expectations.

### Option B — Local DuckDB (fast, offline; contextual cohorts only)

```bash
bash scripts/run_chat.sh
```

In the **sidebar**: **Data source:** `Local DuckDB`; **DuckDB path:**
`data/processed/mimiciv.duckdb` (default); **Connect**.

> **Local-data caveat:** the bundled DuckDB has labs for only ~9.4k of 546k
> admissions, so lab-slope traits are sparse (most candidates → `NaN` → not in
> the cohort). Good enough for the contextual Test A and the **ICU-LOS** temporal
> variant (every ICU admission has a length-of-stay), but for a real
> *worsening-lactate* cohort use **BigQuery** (Option A).

### Sidebar controls that matter for cohorts (both backends)

- **Cohort strategy** (`recent` | `random`) — how the pool is *ordered* into
  batches. Affects which admissions are pulled when the pool exceeds one batch,
  **not** the final scoring. Leave on `recent`.
- **Concurrent batches** — parallel DB fetches; raise it for large BigQuery pools.
- **Parallel workers** — graph-build parallelism for `graph_temporal` cohorts.
  `1` for a first run; `4+` for large temporal cohorts.

---

## 2. Test A — a contextual-only cohort (no graph)

**Type this into the chat box:**

```
Find emergency patients similar to a 68-year-old whose creatinine ran high and
whose platelets dropped, with a relatively short hospital stay.
```

### What you should see

**1) A text summary** that opens with the cohort size and distance spread, e.g.:

```
Found 24 of 60 candidates within Gower distance 0.35 of the described profile.
Cohort distance — nearest 0.041, median 0.188, farthest 0.342.
```

**2) A per-trait contribution table** (this is the "per-trait contributions" to
look for — it is rendered as markdown *in the summary*, above the data table):

```
| trait          | weight | reference | cohort mean similarity |
| age            | 0.8    | 68        | 0.72 |
| gender         | 1.0    | F         | 0.55 |
| creatinine_max | 1.5    | null      | 0.66 |
| platelet_min   | 1.5    | null      | 0.61 |
| icu_los_hours  | 0.5    | null      | 0.70 |
```

Read this as: *each trait's average per-column similarity across the cohort*
(1.0 = identical to the profile, 0 = maximally far). A `reference` of `null` is
expected for directional traits — the comparison point comes from the **frozen
population stats**, not a number you typed.

**3) The ranked cohort table** (a Streamlit dataframe) with exactly these
columns — **nearest first**:

| rank | hadm_id | subject_id | distance |
|---|---|---|---|
| 1 | … | … | 0.041 |
| 2 | … | … | 0.067 |

- `distance` is monotonically non-decreasing down the table.
- Every `distance ≤ distance_threshold` (the number in the summary).

**4) A download button:** **⬇ Download cohort (CSV)**. Click it → `cohort.csv`
whose header is `rank,subject_id,hadm_id,distance`.

> This CSV is the **only** place the database keys (`hadm_id` / `subject_id`)
> appear — by design. The clinician describes patients in words and never has to
> read or type an internal ID in chat.

### Sanity checks

- Flipping the wording from "creatinine ran **high**" to "creatinine ran **low**"
  should change *which* admissions rank highest (the one-sided kernel flips
  direction) — the cohort is **not** symmetric around the reference.
- A patient whose creatinine is *even higher* than typical-high should **not**
  fall in rank for that trait. Spot-check by sorting the CSV.

---

## 3. Test B — a temporal cohort (builds the RDF graph)

A `graph_temporal` trait makes the run **build a per-question RDF graph** over
the candidate pool and score the temporal feature off it. Pick the variant that
matches your backend.

### B1 — worsening labs (BigQuery; the real motivating case)

This is the plan's headline example and the reason to test on BigQuery — it needs
the full multi-reading lab history that only BigQuery has.

**Type this into the chat box:**

```
Find ICU patients with sepsis like one whose lactate got progressively worse
over the first 48 hours and who needed escalating vasopressors.
```

What to confirm:

- **Prefilters in the log (§4):** `ICU` and `sepsis` become a Boolean gate that
  narrows the pool *before* scoring — this is what keeps the BigQuery scan cheap.
- **Two `graph_temporal` traits** in the log: `lactate_slope_48h`
  (`template="sim_series_by_admission"`, `concept="lactate"`,
  `graph_params={"agg":"slope","window_hours":48}`,
  `direction="higher_more_similar"`) and a vasopressor dose-trajectory trait
  (`template="sim_dose_series"`).
- A **non-empty** cohort (with full labs the slopes are defined), ranked nearest
  the described worsening profile.
- The one-sided kernel: an admission whose lactate worsened *more* steeply than
  the reference is **not** penalized for it.

> Expect this to take longer than a contextual cohort — it scans lab + chart +
> prescription history and builds a graph over the pool. If it drags, tighten the
> prefilter ("**only** ICU admissions with sepsis"), and raise **Concurrent
> batches** / **Parallel workers** (§1).

### B2 — ICU length-of-stay (works on DuckDB too)

A dense temporal trait that every ICU admission has, so it populates even on the
partial-labs DuckDB. Good for a quick smoke test of the graph path.

**Type this into the chat box:**

```
Find ICU patients similar to one with a long ICU stay.
```

This yields a single `graph_temporal` trait (`icu_los_hours` via the
`sim_icu_los` template).

### What you should see (either variant)

- Same shape as Test A: summary + per-trait table + ranked dataframe + CSV.
- The per-trait table lists the temporal trait; the logged criteria (§4) show
  `"source": "graph_temporal"`.
- Only admissions that **have** the temporal feature appear — those without it
  yield `NaN` and are correctly **excluded** from the cohort (for B2, only
  admissions with an ICU stay).
- A larger **Parallel workers** value should speed the build up (and must not
  change membership).

---

## 4. Audit the criteria (JSONL activity log)

Every cohort definition is logged — one line — so the selection is reproducible
from the log alone. Default path: **`logs/dashboard_queries.jsonl`** (override
with `export NEUROGRAPH_QUERY_LOG=/path/to/file.jsonl` before launch).

```bash
# The criteria record for the most recent cohort query:
grep '"kind": "cohort_definition"' logs/dashboard_queries.jsonl | tail -1 | \
  .venv/bin/python -m json.tool
```

You should see the prefilters and every trait's kernel/weight/reference plus the
threshold and cap:

```json
{
  "kind": "cohort_definition",
  "question": "Find emergency patients similar to a 68-year-old ...",
  "distance_threshold": 0.35,
  "top_k": 30,
  "prefilters": [
    {"field": "admission_type", "operator": "=", "value": "EMERGENCY"}
  ],
  "traits": [
    {"name": "age", "source": "sql", "kind": "quantitative",
     "direction": "symmetric", "weight": 0.8, "reference_value": 68},
    {"name": "creatinine_max", "source": "sql", "kind": "quantitative",
     "direction": "higher_more_similar", "weight": 1.5, "reference_value": null}
  ]
}
```

**What to verify in the log:**

- "ran high" / "worsening" / "escalating" → `direction: "higher_more_similar"`.
- "dropped" / "shorter" / "low" → `direction: "lower_more_similar"`.
- "68-year-old" → an `age` trait with `direction: "symmetric"`,
  `reference_value: 68`.
- A crisp phrase ("emergency", "ICU", "sepsis") shows up as a **prefilter**
  (and, if it is a presence/identity, *also* as a high-weight trait).
- A temporal phrase → a trait with `source: "graph_temporal"`.

---

## 5. Flipping `distance_threshold` and `top_k`

The threshold and cap are **LLM-proposed and user-overridable** (logged in §4).
There is no separate slider — you steer them with the **wording of the request**,
and the values land in the JSONL record so you can confirm what was used.

| To change | Phrase it like… | Effect |
|---|---|---|
| **Looser** cohort (higher threshold) | "*a broad / loose net of patients roughly like…*" | more candidates admitted; larger cohort |
| **Tighter** cohort (lower threshold) | "*only very close matches to…*", "*near-identical patients*" | fewer, closer members |
| **Cap the count** (`top_k`) | "*the 10 most similar patients*", "*top 15…*" | at most N rows even if more pass the threshold |

Workflow to see the effect:

1. Ask the loose version → note `distance_threshold` / `top_k` and the cohort
   size in the summary, and the values in the JSONL line.
2. Re-ask the tight version → the threshold in the log should drop and the
   cohort should shrink; the farthest `distance` in the table should fall at or
   under the new threshold.

> Defaults when you don't steer it: `distance_threshold` ≈ **0.30–0.40**,
> `top_k` = **30**. `top_k` is a *cap*, not a target — a tight, small cohort can
> return far fewer than `top_k` rows.

---

## 6. Troubleshooting

| Symptom | Likely cause | What to do |
|---|---|---|
| *"Could not assemble the cohort: …"* | a quantitative trait has no frozen range and no stated value | check the trait name in the log against the frozen-ranges set (`src/similarity/reference_ranges.py`); rephrase to a supported feature |
| Empty cohort, temporal trait (lab slope) | running on **DuckDB**, whose partial labs → slope `NaN` for most candidates | switch to **BigQuery** (Option A, §1) for B1; or use the **ICU-LOS** variant B2 (§3) on DuckDB |
| Streamlit hangs on "Analyzing…" >3 min (BigQuery) | the query is scanning too much of MIMIC-IV before the cohort narrows | tighten the prefilter ("**only** ICU admissions with sepsis"), lower `batch_size`, or cancel and watch the job in the BigQuery console |
| BigQuery temporal cohort is slow but works | graph build over a large pool is single-threaded | raise **Concurrent batches** and **Parallel workers** in the sidebar (§1) |
| No download button | the cohort came back empty (`download_csv` is only set for non-empty cohorts) | loosen the request (§5) or relax a prefilter |
| A generic error answer, no table | the turn hit the orchestrator's swallow-all guard | check `logs/dashboard_queries.jsonl` and the app console for the underlying exception |
| Members include admissions you expected a prefilter to exclude | the phrase wasn't read as crisp | confirm a `prefilters` entry exists in the JSONL; make the phrase explicit ("**only** ICU admissions") |

---

## 7. One-glance checklist

- [ ] Contextual question → summary with **size + distance spread**.
- [ ] **Per-trait** table shows the right `direction` per clause (high→
      `higher_more_similar`, dropped→`lower_more_similar`, age→`symmetric`).
- [ ] Ranked table columns are `rank, hadm_id, subject_id, distance`,
      **nearest-first**, all `≤ threshold`.
- [ ] **⬇ Download cohort (CSV)** → `cohort.csv`, header
      `rank,subject_id,hadm_id,distance`.
- [ ] Temporal question pulls in only admissions **with** the temporal feature;
      others (`NaN`) are excluded.
- [ ] (BigQuery) crisp phrases ("ICU", "sepsis") land as **prefilters** in the
      log so the pool narrows before scoring — and B1 returns a non-empty cohort.
- [ ] `logs/dashboard_queries.jsonl` has a `cohort_definition` line whose
      threshold/top_k/traits match what you asked.
- [ ] Tightening the wording **lowers** the logged threshold and **shrinks** the
      cohort.
