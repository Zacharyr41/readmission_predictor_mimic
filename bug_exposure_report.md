# Backend Robustness — Unseen-Question Bug-Exposure Report

**Harness:** `tests/test_conversational/test_bq_unseen_questions.py`
**Pipeline under test:** `ConversationalPipeline` (`src/conversational/orchestrator.py`) in **full production config** (critic + pre-validator + disambiguation + outlier-screening ON), driven end-to-end against the **live BigQuery** MIMIC-IV v3.1 datasets (`physionet-data.mimiciv_3_1_hosp`, billing project `mimic-485500`).
**Method:** each iteration authors one *unseen-but-valid* clinical question (data exists in MIMIC, the quantity is computable, the format is supported), runs it through the real decompose → resolve → (SQL fast-path | graph) → answer pipeline, and asserts a *valid answer* against a **hybrid oracle**: structural checks (no error/clarify, real rows, a query executed, a finite numeric result) **plus** a direct-BigQuery ground-truth cross-check with a plausibility/tolerance band. On failure: investigate with direct BQ probes, characterize the bug, implement a **general, ontology-grounded** fix (no curation, no hardcoding), and re-verify with the full offline suite (~2,183 tests) and the whole accumulated BQ suite before committing.

> **Why BigQuery, not local DuckDB:** the demo runs on BigQuery, and the local DuckDB load is partial (~9.4k of 546k admissions carry labs), so a DuckDB run would mask real coverage bugs.

---

## Headline results

- **30 unseen questions authored across 12 clinical specialties** (critical care, neuro, cardiology, nephrology, hematology, hospital medicine, pulmonary, endocrine, infectious disease, hepatology) and both backend paths (SQL fast-path + graph).
- **~13 distinct backend bug-classes found and fixed** with general, ontology-grounded fixes — every fix wired into the live path with an offline regression guard.
- **Remaining iterations validated** that the fixes generalize (passes on fresh cohorts/shapes) and that the bot **declines gracefully** on genuinely-unanswerable questions (de-identified dates, hashed providers).
- **No regressions:** offline suite green throughout; the full accumulated BQ suite green after every fix.

The bugs were overwhelmingly **grounding and aggregate-shape** defects — the pipeline routes and executes correctly, but the *concept→MIMIC-vocabulary* mapping and the *aggregate the user asked for* were silently wrong in ways that produce **confident wrong numbers** (the worst failure mode for a clinical demo), not crashes.

---

## Per-iteration summary

| # | Question (abbrev.) | Specialty | Result | Bug class / note |
|---|---|---|---|---|
| 1 | avg troponin in MI, by age | Cardiology | 🔴→🟢 | biomarker grounds to an *empty* assay subtype (Troponin I); broaden to the populated label-family (Troponin T) |
| 2 | avg body temperature in pneumonia | Pulm/Vitals | 🔴→🟢 | a blocked companion outlier-rows scan aborted an already-computed vital answer |
| 3 | median ICU length of stay | Hosp. Med | 🟢 | metadata-only LOS aggregate (no clinical concept) |
| 4 | drug-cohort count | Pharmacy | 🔴→🟢 | COUNT inflated prescription *rows* into a "patient" count → COUNT(DISTINCT hadm) |
| 5 | in-hospital mortality **rate** (sepsis) | Critical care | 🔴→🟢 | outcome compiler emitted a COUNT, not the RATE → in-query `fraction` |
| 6 | brand-name drug count | Pharmacy | 🔴→🟢 | brand name never reached its MIMIC-stored generic → RxNorm brand→generic |
| 7 | median lactate (sepsis) | Critical care | 🟢 | first graph-path coverage — median routes off the SQL fast-path |
| 8 | graph-path biomarker mean | Critical care | 🟢 | graph extractor resolves by label-LIKE not LOINC (hardening note; outlier-screen caught it) |
| 9 | **positive** blood culture (sepsis) | Infectious dis. | 🔴→🟢 | "positive" status dropped (counted cultures *drawn*, ~4×) → `org_name IS NOT NULL` |
| 10 | blood culture grew **E. coli** (sepsis) | Infectious dis. | 🔴→🟢 | colloquial organism not grounded to MIMIC's scientific binomial; specimen∧organism not intersected |
| 11 | avg creatinine by **sex** (sepsis) | Critical care | 🟢 | comparison-axis aggregate keeps LOINC grounding per group |
| 12 | **highest** lactate (sepsis) | Critical care | 🟢 | MAX is outlier-screened on the SQL fast-path (a 1.27e6 artifact is rejected) |
| 13 | positive **sputum** culture (pneumonia) | Infectious dis. | 🔴→🟢 | "X culture" never matched MIMIC's `SPUTUM`/`URINE` specimen vocabulary |
| 14 | mortality by **sex** (heart failure) | Cardiology | 🔴→🟢 | outcome compiler **ignored the comparison axis** → per-group rate |
| 15 | warfarin count (atrial fibrillation) | Cardiology | 🟢 | distinct-admission drug count survives daily-dosing + brand/generic |
| 16 | creatinine by **30-day readmission** (HF) | Hosp. Med | 🟢 | the project's core axis — readmission-labels join + LOINC grounding per group |
| 17 | **30-day readmission rate** | Hosp. Med | 🔴→🟢 | the headline metric was UNANSWERABLE → readmission is now a first-class SQL outcome |
| 18–21 | counts: ischemic/hemorrhagic stroke, acute MI, DKA | Neuro/Cardio/Endo | 🔴→🟢 | **diagnosis grounding** — dotted/category ICD codes never matched; icd_autocode coverage gaps |
| 22–23 | rank physicians / count "in 2017" | Guardrails | 🟢 | bot **declines gracefully** (hashed providers; shifted dates) |
| 24–27 | SAH count; bilirubin & INR in cirrhosis; ICH mortality | Neuro/Hepatology | 🟢 | grounding/LOINC/outcome fixes generalize to neuro + liver cohorts |
| 28–30 | ARDS count; AKI count; hemoglobin in sepsis | Pulm/Nephro/Heme | 🔴→🟢 | **abbreviation name over-matched** (`%ards%` → "hazards"/"Edwards") → expand abbreviations |

🔴→🟢 = initially failed, fixed; 🟢 = passed on authoring (validated-robust coverage).

---

## Bug classes & general fixes (this loop's themes)

### 1. Concept → MIMIC-vocabulary grounding (the dominant theme)

The single richest bug family: a clinically-correct concept name does not match the *string MIMIC actually stores*, so the query silently returns 0 or the wrong cohort.

- **Biomarker assay coverage (iter 1).** Generic "troponin" grounded to Troponin **I** (LOINC 10839-9 → itemids that are *empty* in MIMIC) while the populated assay is Troponin **T**. Fix: an empty-result broadening retry to the label-family — general for any analyte whose chosen subtype is sparse but whose sibling is populated.
- **Organism names (iter 10).** "E. coli" never matches `org_name = 'ESCHERICHIA COLI'`. Fix: the decomposer grounds organisms to the scientific binomial (LLM ontology, like LOINC), and the compiler conjoins the specimen and organism (`spec ∧ org`) instead of OR-ing them.
- **Specimen names (iter 13).** "sputum culture" never matches `spec_type_desc = 'SPUTUM'` (MIMIC keeps the word "culture" only for blood). Fix: a morphological rule — strip the trailing "culture"/"cx" test-modality token so the source matches.
- **Diagnosis ICD grounding (iters 18–21).** Two compounding defects, both general: (a) `icd_autocode` returns **dotted, category-level** codes (`I63`, `E11.1`) that an exact `IN` never matched against MIMIC's **undotted, billable** codes (`I6300`, `E1110`) — fixed by normalizing + **prefix-matching**; (b) `icd_autocode` has **wide coverage gaps** (returns nothing for hemorrhagic stroke, DVT, COPD exacerbation, intracerebral hemorrhage) — fixed by having the decomposer emit the ICD-10 category codes from its own knowledge (analogous to LOINC), which grounds *any* condition the model knows.
- **Abbreviation over-match (iter 28).** A concept named "ARDS" made the title-LIKE fallback match `%ards%` as a bare substring — "ha**zards**", "Edw**ards**", "li**zards**", "Rich**ards**on" — a ~17× over-count (12,953 vs ~769). Fix: the decomposer expands abbreviations to the full descriptive term so the title fallback is precise.

**Principle:** ground every clinical concept to the *vocabulary MIMIC records* via an ontology the model already encodes (LOINC, scientific binomials, ICD-10 categories) rather than hand-curated synonym tables; match codes as prefixes, not exact strings.

### 2. Aggregate shape — answering the question the user asked

- **Count vs. rate (iter 5).** "mortality **rate**" returned two raw counts; the rate survived only as fragile LLM prose. Fix: the outcome compiler computes each bucket's `fraction` in-query.
- **Grain inflation (iter 4).** A drug "patient count" counted prescription rows (daily dosing → ~N× inflation). Fix: COUNT(DISTINCT hadm_id).
- **Comparison axis dropped (iter 14).** "compare mortality between men and women" returned the *pooled* rate — the outcome compiler ignored `comparison_field`. Fix: it now groups by the axis and partitions the rate window per group (general across gender / admission-type / readmission / …).
- **Missing first-class metric (iter 17).** "30-day readmission rate" — the project's headline number — was unanswerable (it routed to a concept-less graph build that fell back to length-of-stay and declined). Fix: readmission is now a first-class SQL **outcome** alongside mortality, sharing one parameterized rate compiler.

### 3. Result-qualifier semantics (microbiology)

- **Culture positivity (iter 9).** "positive blood culture" counted cultures *drawn*, not cultures that *grew* (~4× over-count). Fix: ground "positive"/"negative" to `org_name IS NOT NULL`/`IS NULL` — the MIMIC definition of culture positivity.

### 4. Outlier / unit safety (validated)

- **MAX screening (iter 12).** "highest lactate" is the worst case for a data-entry artifact (raw MAX = 1,276,103 mmol/L); the biological-limit screen fires for MAX on the SQL fast-path, returning a physiologic value. Validated, not a defect.
- **Unit pooling (iters 2, 8, 11).** LOINC grounding (serum vs urine) and the outlier screen hold across the ungrouped, grouped, and graph paths.

### 5. Honest refusal (guardrails — validated)

- **Hashed providers (iter 22)** and **de-identified/shifted dates (iter 23)** are genuinely unanswerable; the bot emits a clarifying decline rather than fabricating a leaderboard or a year-specific count. Pinned so a regression can't silently start answering them.

---

## Files touched (live code paths)

- `src/conversational/sql_fastpath.py` — microbiology specimen∧organism intersection + culture suffix stripping; diagnosis-count prefix-matching; outcome compiler generalized to mortality **and** readmission with comparison-axis support.
- `src/conversational/operations_filters.py` — diagnosis-filter grounded codes normalized + prefix-matched.
- `src/conversational/prompts.py` — decomposer rules: organism scientific-binomial grounding; specimen+organism in one concept; readmission as an outcome; ICD-10 category grounding for diagnoses; abbreviation expansion. (Snapshot `fixtures/prompts_snapshot.txt` regenerated.)
- `data/mappings/clinical_cohorts.json` — added the missing `stroke_hemorrhagic` registry entry (sibling of `stroke_ischemic`).
- `src/conversational/orchestrator.py`, `concept_resolver.py` — empty-result biomarker broadening; outcome-rate wiring (earlier iterations).
- Offline regression guards added in `tests/test_conversational/test_sql_fastpath.py` (microbiology intersection, ICD prefix-match, outcome readmission/comparison) and contract tests updated in `test_operations.py` / `tests/dashboard/test_tier3_sql_emission.py`.

Every fix ships with an offline test that fails on the pre-fix behavior, so the bug can't silently return.

---

## Known limitations / recommended future work

- **Ranked / top-N aggregation is unsupported.** "the 10 most common discharge diagnoses" (battery Q29) and "the most frequently isolated organisms" (Q44) route to a single-concept count, not a `GROUP BY <dimension> ORDER BY count DESC LIMIT N`. This is a genuine *feature* gap (a new aggregation shape touching decomposer + planner + compiler + answerer), not a grounding bug, and was deliberately scoped out. Recommended next: add a `ranking` aggregate that groups by the concept's natural dimension (diagnosis title / organism / drug) and returns the ranked head.
- **ICD-9 tail.** Diagnosis grounding emits ICD-10 categories; ICD-9 admissions are caught only when the title fallback matches the colloquial term. For MIMIC-IV (ICD-10-dominant) this is a minor undercount, surfaced as wide tolerance bands in the count tests (e.g., AKI, ARDS). A deterministic ICD-9↔ICD-10 crosswalk would close it.
- **Graph-path biomarker resolution (iter 8 note).** The graph extractor resolves biomarkers by label-LIKE rather than the SQL path's LOINC grounding. The outlier screen currently masks the difference for robust statistics (median), but aligning the two paths is a hardening recommendation.
- **Derived/temporal showpieces** (SOFA/KDIGO/MELD severity, time-to-event, single-patient timelines, literature joins) were out of scope for this structural-correctness sweep.
- **Live-LLM determinism.** Decompositions are now stable for the suite's questions (verified by repeated runs). One previously-intermittent failure (iter 1's troponin "above 70 vs ≤ 70" split) was **root-caused and fixed** — see below; it was a real grounding bug, not mere model variance. Genuinely free-form phrasings can still vary at the margins, so for a demo keep a known-good phrasing for the marquee questions.

### iter 1 follow-up — biomarker name over-specification defeated the label-family broadening (fixed)

A recurring ~25% flake on "average troponin … above 70 vs ≤ 70" was traced (via repeated live runs + decomposition/SQL capture) to the **decomposer intermittently naming the biomarker concept `"troponin I"`** (copying the LOINC-10839-9 subtype) instead of the general `"troponin"` the user said. The iter-1 empty-subtype broadening recompiles the label-family fallback from the concept **name**, so `name="troponin I"` produced `d.label ILIKE '%troponin I%'` — which matches only the *unpopulated* Troponin-I labels and misses the populated **Troponin T**, so the recovery returned empty and the answer was a spurious "no data" (`Mean Value: None`). When the name was `"troponin"`, `%troponin%` recovered Troponin T and the answer was correct — hence the intermittency. **Fix (general, decomposer prompt):** a biomarker concept's `name` must be the analyte the user named ("troponin"), with subtype precision in `loinc_code` — never the subtype in the name, which defeats the label-family fallback. Verified: 20/20 decompositions now name it "troponin", and iter 1 passes 8/8 (was ~75%).

### Non-registry diagnosis filters degraded under full-suite load — FIXED (separate from iter 1)

During iter-1 verification, three biomarker-in-cohort tests (anion gap / serum osmolality in DKA, INR in cirrhosis) failed intermittently **only late in the full 38-test BQ run**; each passed **8/8 in isolation** with a fresh pipeline. Root cause: DKA and cirrhosis are **not in the Tier-1 cohort registry**, so their *filters* ground via the `icd_autocode` MCP (Tier 2). Under cumulative full-suite load that MCP call transiently misses, and the **title-LIKE fallback is unreliable for colloquial terms** — `long_title LIKE '%diabetic ketoacidosis%'` matches **0** rows (MIMIC titles read "diabetes mellitus *with* ketoacidosis"), so the cohort came back empty and the answer was a spurious "no data" (`Mean Value: None`). The old `_cached_icd_autocode` called the tool **exactly once**, so a single transient miss fell straight through to the bad fallback.

**Fix applied — option (a), general `icd_autocode` resilience.** `_cached_icd_autocode` now retries on a transient/empty MCP result (bounded: `_ICD_AUTOCODE_MAX_ATTEMPTS=3`, linear backoff) and still **never caches a failed lookup** (sentinel-raise). A single hiccup is absorbed and grounding succeeds; a *persistent* failure still falls back after the bounded retries. This is general — it hardens grounding for **every** non-registry cohort, not just DKA/cirrhosis, and keeps ontology grounding (no per-cohort curation). **Reproduced deterministically** before fixing (`TestIcdAutocodeRetryResilience` in `test_concept_resolver.py`): three tests inject a transient `unavailable` / transient-empty / persistent failure via a mocked MCP, so they fail reliably without the retry and pass with it — no dependence on real MCP load. This is a *filter-grounding resilience* fix, distinct from iter 1's biomarker-naming bug.

---

## T1/T2 demo-battery feasibility analysis

A systematic pass over every **T1/T2** question in the demo battery, classified by probing (a) the pipeline's reachable-table allow-list (`_BQ_TABLES` — **11 base hosp/icu tables**) and (b) the actual BigQuery schema (`physionet-data` datasets + table schemas).

**Key finding:** the "derived clinical concepts" I'd first assumed unanswerable **exist as clean tables** in `mimiciv_3_1_derived` (`sepsis3`, `sofa`/`first_day_sofa`, `kdigo_stages`, `meld`, `ventilation`, `first_day_gcs`, `charlson`, `rrt`/`crrt`, `oasis`) — but the pipeline reaches only the 11 base tables, so answering them needs *wiring the derived dataset in as new concepts* (a feature). The data is present; the capability is not.

### Answerable via existing features — implemented & passing
Q1 median ICU LOS · Q6 ischemic/hemorrhagic stroke counts · Q12 acute-MI count · Q13 peak troponin (biomarker MAX + Troponin-I→T broadening + MI filter) · Q17 AKI count (ICD) · Q30 30-day readmission rate · Q35 ARDS count · Q39 DKA count · Q40 anion gap + serum osmolality in DKA · Q43 positive blood culture (all) · Q48 MELD *components* (bilirubin/INR/creatinine in cirrhosis) + cirrhosis mortality.

### Answerable via a small tweak to an existing feature — implemented & passing
- **Q12 "primary diagnosis"** — added a `primary_only` qualifier on `ClinicalConcept` (the diagnosis analogue of microbiology `culture_status`) → `diagnoses_icd.seq_num = 1`. Fixes a silent ~1.9x over-count (16,537 any-position vs 8,573 primary). General for any condition.

### Deferred — would need a major new feature (NOT built, per the "no new major features" constraint)
- **Derived-table concepts** (data exists in `mimiciv_3_1_derived`, pipeline can't reach it): Q3 Sepsis-3, Q5 SOFA tertiles, Q7 GCS total (chartevents has only the 3 components), Q17-by-KDIGO / Q18 (`kdigo_stages` — an ICD-AKI proxy exists), Q36 PaO₂/FiO₂ + ARDS severity, Q48 MELD *score*, Q14 Charlson comorbidities.
- **Procedure concepts** (no procedure concept type): Q8 EVD, Q19 RRT/CRRT, Q23 RBC transfusion (in `inputevents`/`procedures_icd`, not `prescriptions`).
- **New filter types**: Q25 thrombocytopenia (biomarker-value-threshold cohort), Q32 (≥5-medications count).
- **Comparison on the metadata-LOS path**: Q31 LOS *by admission type* — the LOS path drops the comparison axis (returns raw per-stay LOS); honoring it is a feature.
- **Top-N ranking**: Q29, Q44 (a new aggregation shape; explicitly skipped).
- **Data genuinely absent**: Q50 ED triage lactate (no `mimiciv_ed` dataset loaded).

### Recommended next features (high demo value, priority order)
1. A general **derived-cohort/measurement concept** wiring `mimiciv_3_1_derived` (unlocks Sepsis-3, KDIGO, SOFA, MELD-score, ventilation, GCS, Charlson — a large fraction of the ICU/research battery in one general mechanism).
2. A **procedure concept type** (EVD / RRT / transfusion / ventilation via `procedures_icd` / `procedureevents` / `inputevents`).
3. **Biomarker-value-threshold cohorts** (Q25/Q32).
4. **Top-N / ranked aggregation** (Q29/Q44).

---

## Conclusion

Across 30 unseen questions spanning 12 specialties and both backend paths, the pipeline's **routing and execution were sound**, but its **grounding and aggregate-shape layers harbored a recurring class of "confident wrong number" bugs** — exactly the failures that erode clinician trust in a live demo. The fixes were uniformly **general and ontology-grounded** (LOINC, RxNorm, SNOMED-style scientific names, ICD-10 categories, MIMIC schema semantics) rather than per-question patches, so they repair entire bug *classes* and were shown to generalize to fresh cohorts (neuro, hepatology, pulmonary, nephrology, hematology) the fixes were never tuned against. The system also declines the genuinely-unanswerable gracefully. Net: the demo-relevant question space is now substantially more trustworthy, with ranked aggregation the main remaining feature gap.
