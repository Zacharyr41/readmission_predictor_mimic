# UChicago Demonstration — Detailed Outline

> **Audience:** Dr. Mansour, UChicago Medicine.
> **Goal:** a hiccup-free walk-through of the NeuroGraph chatbot against
> MIMIC-IV that motivates the IRB process for a UChicago-EHR version.
> **Arc:** start with instant, simple answers; build to comparisons; show
> visualizations; open the knowledge-graph (temporal) engine; finish with the
> headline capability — assembling a cohort of *similar* patients.

Each prompt below is tagged with an ID (e.g. **D1.1**) that matches the
programmatic harness in `tests/test_demo/demo_questions.py`, so every scripted
question here is one whose routing and (for fast-path) SQL is verified
automatically. Run `make demo-check` (see the runbook) before presenting.

**Reading the tags:** `→ SQL fast-path` = one instant SQL query;
`→ Graph path` = builds a temporal knowledge graph; `→ Similarity` = cohort
discovery. See `00_chatbot_architecture.md` §3 for the routing rule.

---

## Phase 0 — Framing (≈1 min, before typing anything)

On the empty chat screen, the **welcome panel** lists the capability groups and
example prompts. Use it to set expectations in one sentence:

> "I'll ask plain-English clinical questions against MIMIC-IV. Simple metrics
> come back instantly via SQL; anything with a *time* element builds a temporal
> knowledge graph behind the scenes; and at the end I'll ask it to find me a
> cohort of *similar* patients."

Confirm the sidebar shows **Connected** (green). Then begin.

---

## Phase 1 — Basic questions, SQL fast-path (≈3 min)

**Demonstrates:** instant answers across *different clinical domains*, each one a
single SQL aggregate. Point out the **Query Details** expander so Dr. Mansour
sees the actual SQL — transparency, not a black box.

| ID | Ask this | Domain | What appears |
|----|----------|--------|--------------|
| **D1.1** | *What is the average creatinine for patients over 65?* | Renal / chemistry | a single mean value |
| **D1.2** | *How many patients had a positive blood culture?* | Infectious disease | a single count |
| **D1.3** | *What was the highest heart rate recorded?* | Cardiology / vitals | a single max |

**Optional breadth (pick if time / interest):**

| ID | Ask this | Domain |
|----|----------|--------|
| **D1.4** | *How many patients died during their hospital stay?* | Outcomes / mortality |
| **D1.5** | *List the patients with cerebral infarction.* | Neurology (anchor cohort) |

**Talking points.**
- Each answer is one SQL query — *no* graph is built, which is why it's instant.
- These span renal, infectious disease, cardiology (and optionally mortality and
  neurology) — *breadth of clinical domains* from one interface.
- Open **Query Details** once to show the emitted SQL and that the lab was
  grounded to a specific MIMIC `itemid` (not a fuzzy text match).

---

## Phase 2 — More complex questions, still SQL fast-path (≈3 min)

**Demonstrates:** the fast-path handles real analytic shape — *stratified
comparisons* and *multi-criteria cohorts* — still in one query, still instant,
because there is **no temporal element**.

| ID | Ask this | What it shows |
|----|----------|---------------|
| **D2.1** | *Compare creatinine between male and female patients.* | a `GROUP BY` comparison (two rows: mean + n per group) |
| **D2.2** | *Compare mean lactate between readmitted and non-readmitted patients.* | grouping on a computed 30-day readmission flag — the product's pilot question |
| **D2.3** | *Average creatinine for female sepsis patients over 50 readmitted within 30 days.* | four stacked filters (sex + diagnosis + age + readmission) in one query |

**Optional:** **D2.4** *Compare systolic BP between emergency and elective
admissions.* (grouping on admission type).

**Talking points.**
- The system understood "readmitted within 30 days" as a structured cohort
  definition and computed the readmission label on the fly.
- D2.3 shows it stacking four independent criteria correctly — the kind of cohort
  a clinician actually wants — and still returning instantly.
- This is the boundary of the fast-path: still **no time element**, so no graph.

---

## Phase 3 — Visualization questions (≈3 min)

**Demonstrates:** the dashboard renders **charts**, not just tables. Visualization
is an *output type*; the most compelling charts (trends) are produced by the
graph path. **Keep the cohort narrow** so the chart returns quickly.

| ID | Ask this | Chart |
|----|----------|-------|
| **D3.2** | *Show the distribution of creatinine for sepsis patients as a histogram.* | a **histogram** of the cohort's values |
| **D3.1** | *Plot heart rate over the first 7 days of the ICU stay.* | a **line/scatter** trend over time |

**Talking points.**
- "Distribution / histogram" and "plot over time" both ask for a chart — the
  system picks an appropriate Plotly chart type automatically.
- The trend (D3.1) has a time axis, so under the hood this already uses the
  knowledge-graph engine — a natural segue into Phase 4.
- *(If a chart is slow, narrow further — e.g. add a specific diagnosis — and
  re-ask. See the runbook's "narrow the cohort" note.)*

---

## Phase 4 — Simple questions, knowledge-graph (temporal) path (≈4 min)

**Demonstrates:** the headline backend capability — when a question involves
**time**, the system builds an RDF **knowledge graph** with temporal
relationships, then queries it. These are deliberately *simple* questions on a
*narrow* cohort so the graph build is quick.

| ID | Ask this | Temporal element |
|----|----------|------------------|
| **D4.2** | *Which antibiotics were prescribed during the first 48 hours of ICU admission?* | "first 48 hours" window |
| **D4.1** | *What are the creatinine levels during the ICU stay for sepsis patients?* | "during the ICU stay" + narrow cohort |
| **D4.3** | *Creatinine values within the first 24 hours of ICU admission.* | "within 24 hours" window |

**Talking points.**
- Watch the live status: it shows **"Building the knowledge graph…"** — this is
  the graph being constructed for *this* question, with temporal (Allen)
  relationships embedded.
- The temporal phrase ("first 48 hours", "during the ICU stay") is exactly what
  forces this richer path — the SQL fast-path can't express interval logic.
- This is the engine that will let us ask genuinely temporal questions of the
  UChicago EHR (medication timing, lab trajectories, event ordering).
- Pre-filtering to a narrow cohort *before* graph construction is what keeps it
  fast — a key design choice for scaling to a full EHR.

---

## Phase 5 — Complex cohort creation / patient similarity (≈5 min)

**Demonstrates:** the capstone — describe a patient *in clinical terms* and the
system assembles a ranked cohort of **similar** admissions, with a downloadable
patient list. No database IDs are ever typed.

| ID | Ask this | What appears |
|----|----------|--------------|
| **D5.1** | *Find admissions similar to a 68-year-old woman with atrial fibrillation and chronic kidney disease.* | a ranked cohort table + **⬇ Download cohort (CSV)** |
| **D5.2** | *Find patients similar to a 75-year-old man admitted for septic shock with acute kidney injury.* | a second cohort with a different clinical profile |

**Optional:** **D5.3** *…and only show me ones with similarity above 0.7.*
(demonstrates a similarity threshold tightening the cohort).

**Talking points.**
- The clinical description was translated into a formal cohort definition
  (demographics + comorbidities + severity), then every admission was scored by
  similarity to that profile — *one-vs-many*, nearest first.
- The ranked list + **CSV** is the take-away: a reproducible cohort for
  downstream analysis. Database keys live only in the file, never in the chat.
- This is the differentiator for UChicago: "find me patients like this one" is
  the question that powers comparative-effectiveness and matched-cohort studies.

### Follow-up questions (per cohort)

The chat keeps conversational context, so you can interrogate the cohort you
just described. Ask these *immediately after* the matching cohort prompt:

- **After D5.1** (68-year-old woman, AF + CKD):
  - *What was the average creatinine in this cohort?*
  - *How many of them were readmitted within 30 days?*
- **After D5.2** (75-year-old man, septic shock + AKI):
  - *What was the in-hospital mortality for this cohort?*

> **Honest framing for these follow-ups.** The system re-interprets the
> follow-up against the prior cohort *description* (it carries the last several
> turns as context), not by silently re-using the exact returned ID list. So
> "this cohort" is understood as "patients matching that same profile." The
> downloaded **CSV** is the precise, frozen patient list for any analysis that
> needs the exact membership. Phrase the follow-ups as above and they read
> naturally; if one drifts, fall back to the CSV as the source of truth.

---

## Closing (≈1 min)

> "Everything you saw — instant metrics, stratified comparisons, charts, temporal
> reasoning over a knowledge graph, and similar-patient cohort discovery — runs
> through one chat interface, with the underlying SQL and graph queries fully
> inspectable. This is the prototype; the production version runs the same
> engine against UChicago Medicine's EHR. That's what the IRB approval unlocks."

---

## One-screen cheat sheet

| Phase | Lead question | Path |
|---|---|---|
| 1 Basic | *Average creatinine for patients over 65?* | SQL fast-path |
| 1 Basic | *How many patients had a positive blood culture?* | SQL fast-path |
| 1 Basic | *Highest heart rate recorded?* | SQL fast-path |
| 2 Complex | *Compare creatinine between male and female patients.* | SQL fast-path |
| 2 Complex | *Compare mean lactate between readmitted and non-readmitted patients.* | SQL fast-path |
| 2 Complex | *Avg creatinine for female sepsis patients over 50 readmitted within 30 days.* | SQL fast-path |
| 3 Viz | *Distribution of creatinine for sepsis patients as a histogram.* | Graph → histogram |
| 3 Viz | *Plot heart rate over the first 7 days of the ICU stay.* | Graph → line |
| 4 Graph | *Which antibiotics were prescribed during the first 48 hours of ICU admission?* | Graph (temporal) |
| 4 Graph | *Creatinine levels during the ICU stay for sepsis patients.* | Graph (temporal) |
| 5 Cohort | *Find admissions similar to a 68-yo woman with AF and CKD.* | Similarity → CSV |
| 5 Cohort | *Find patients similar to a 75-yo man with septic shock and AKI.* | Similarity → CSV |
