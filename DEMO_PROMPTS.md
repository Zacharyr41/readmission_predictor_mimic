# Clinical Conversational Demo — Verified Prompt Set

Every prompt below is backed by a **passing live BigQuery end-to-end test** in
`tests/test_conversational/test_bq_unseen_questions.py` (`-m bigquery`). Use these
with confidence. Prompts that the system can't yet answer **decline gracefully**
(the bot asks you to clarify or says it can't) rather than fabricating a number —
those are listed under *Guardrails*, and the missing capabilities under *Planned*.

> Backend: BigQuery (`mimic-485500`, `physionet-data.mimiciv_3_1_*`).
> Diagnosis grounding is now **ICD-anchored + consistent** (see the highlight below).

---

## Headline: consistent, correct diagnosis grounding

The demo bug — *"subarachnoid hemorrhage"* returning **4,033** as a count but a
**9,362**-patient cohort for mortality — is fixed. A diagnosis term now grounds to
the **same, clinically-correct ICD codes** everywhere (candidate disambiguation:
OMOPHub + cohort registry, codes-only, both ICD-9 and ICD-10).

- **"How many patients were diagnosed with a subarachnoid hemorrhage?"** → **1,576**
  (nontraumatic SAH, ICD-10 `I60` ∪ ICD-9 `430`) — *not* the 4,033 that swept in
  traumatic SAH.
- **"What's the in-hospital mortality for subarachnoid hemorrhage patients?"** →
  same **1,576** cohort (≈20.8% mortality). Count and mortality finally agree.

Ambiguous terms are **surfaced for you to pick** rather than silently guessed —
e.g. *"stroke"* → "did you mean ischemic (I63) or hemorrhagic (I60–I62)?"

---

## Cohort counts (diagnosis grounding)

- "How many patients were diagnosed with an **ischemic stroke**?"  (I63 / 433–434)
- "How many patients were diagnosed with a **hemorrhagic stroke**?"  (I60–I62 / 430–432)
- "How many patients were diagnosed with a **subarachnoid hemorrhage**?"  (**1,576**)
- "How many patients had an **acute MI**?" · "…**DKA**?" · "…**AKI**?" · "…**ARDS**?"
- "How many patients had **acute MI** as a **primary** diagnosis?"  (seq_num = 1)

## Mortality & outcomes

- "What's the **in-hospital mortality** for **subarachnoid hemorrhage** patients?"
- "What's the in-hospital **mortality rate** for **sepsis** patients?"
- "What's the in-hospital mortality for **intracerebral hemorrhage** patients?"
- "What's the in-hospital mortality for **cirrhosis** patients?"
- "What's the **overall 30-day readmission rate**?"

## Severity-defined cohort (#4)

- "What's the in-hospital mortality for patients with **severe traumatic brain injury
  and an admission GCS of 8 or below**?"  (**~748 cases, ~29% mortality** — stable.
  GCS now grounds to the derived-table TOTAL score (not the 3 components), and "severe
  TBI" grounds via the `traumatic_brain_injury` registry — both from fixed sources.
  The pre-fix ~9.7% was the *wrong* component-matched cohort.)

## Lab / biomarker aggregates in a cohort

- "What's the **median lactate** in **sepsis** patients?"  (outlier-screened)
- "What's the **mean hemoglobin** in **sepsis** patients?"
- "What's the **mean creatinine / mean bilirubin / mean INR** in **cirrhosis** patients?"
- "What's the **peak troponin** in **myocardial infarction** patients?"
- "What's the **average body temperature** in **pneumonia** patients?"
- "What's the **average length of stay** in **heart failure** patients?"

## Comparisons (built-in axes)

- "Compare in-hospital **mortality by gender**."
- "Compare **creatinine by gender**."  (keeps LOINC grounding)
- "Compare **creatinine by 30-day readmission** in heart-failure patients."

## Split a cohort by ANY condition (#2, #3)

Mortality (or any outcome) split by presence/absence of a sub-condition — a
procedure, a chronic-use diagnosis, a lab threshold — not just the fixed axes:

- **#2** "What's the in-hospital mortality for **subarachnoid hemorrhage** patients,
  **split by whether they required mechanical ventilation**?"  (~40% vent vs ~9%)
- **#3** "Across **intracerebral hemorrhage** admissions, compare in-hospital
  mortality between patients with **documented chronic use of anticoagulants or
  antiplatelets** and those without."  (~21% vs ~17%; grounds Z79.01/02 ∪ V58.61/63)

## Event ordering — temporal sequence (#7)

- **#7** "Across ICU patients with **intracerebral hemorrhage**, what's the most
  common **temporal order of intubation, first hyperosmolar therapy** (mannitol or
  hypertonic saline), **and the first GCS drop of 2+ points**?"  (returns the
  most-common sequence + which-came-first fractions + median inter-event gap)

## Drugs (brand → generic grounding)

- "How many patients were prescribed **warfarin** (in atrial fibrillation)?"
- "How many patients were prescribed **Coumadin**?"  (resolves to the generic cohort)
- "How many distinct admissions had a **Lasix** prescription?"

## Microbiology (specimen-aware)

- "How many patients had a **positive blood culture**?"  (counts growth only)
- "How many had an **E. coli blood culture**?"  (organism-grounded)
- "How many had a **positive sputum culture** in **pneumonia**?"  (specimen-matched)

## Timelines (single small cohorts)

- **#5 (⚠️ not for live)** "Among patients with **spontaneous (non-traumatic)
  intracerebral hemorrhage** who had an **elevated admission INR (>1.7)** and received
  a coagulation-reversal agent — 4-factor PCC, vitamin K, or FFP — map the timeline of
  INR correction, the reversal-agent administration, and any neurologic change."
  *(Grounding + extraction now correct and fast (~40s): cohort=150, reversal agents
  and GCS-total all extract. But the 150-patient COHORT-timeline answer assembly still
  returns empty — a graph-reasoning follow-up. **Use #6 for a reliable timeline
  walkthrough.**)*
- **#6 (patient walkthrough)** "Walk me through the entire ICU course of patient
  **18744840** as a timeline — GCS, coagulation labs, reversal agents,
  blood-pressure control, and any procedures."  *(slower: ~60–90 s; the patient is
  a severe-ICH + ICP-monitored case with dense data.)*

## Metadata

- **#1a** "How many distinct **ICU stays** are in the database, and what's the
  **median ICU length of stay**?"

---

## Guardrails — the bot declines gracefully (good to show!)

These return an honest *"I can't answer that as asked / which did you mean?"*
instead of a confident wrong number:

- "Which individual **attending physician** had the lowest sepsis mortality?"  (PHI/identifiability)
- "How many patients were admitted **in 2017**?"  (MIMIC dates are de-identified/shifted)
- Avoid: **"ischemic stroke vs hemorrhagic stroke"** as a single comparison (routes to a
  whole-DB graph build) — ask the two counts separately instead.

---

## Planned (not yet demo-ready — real features, scoped)

- **Count-of-A-vs-B routing** (#1b): "how many ischemic vs hemorrhagic stroke" should
  decompose to two SQL counts instead of a graph comparison (it currently routes to a
  whole-DB graph build).
- **EVD-specific split** (#2 primary): mechanical ventilation grounds (above), but the
  external ventricular drain procedure code isn't wired yet — use the ventilation split
  for the demo.
- **Cohort-timeline answer assembly** (#5): grounding + extraction are fixed (cohort
  grounds, all concepts extract, fast ~40s), but the graph-reasoning layer doesn't
  assemble a 150-patient cohort timeline into an answer (single-patient timelines, #6,
  do work). Needs a cohort-timeline answer shape.
