#!/usr/bin/env python
"""Seed the labeled routing corpus (§8 "D0" of ``querytriagesystem.md``).

The corpus is a set of real clinical questions tagged with the route they
*should* take (``desired_plan``), plus the route the planner produces *today*
(``current_plan``). It turns "some queries are misrouted" into a measurable rate
(``test_routing_corpus.py``) and a regression target for the later fixes
(Directions A/B/C).

This script is the corpus's **provenance**: each entry's ``cq`` is built here as a
real ``CompetencyQuestion`` (the same shapes the decomposer emits), serialized via
``model_dump(mode="json")``, and written to
``tests/test_conversational/fixtures/routing_corpus/<name>.json``. ``desired_plan``
is the human-assigned ground truth; ``current_plan`` is snapshotted from the live
planner at seed time.

    .venv/bin/python scripts/seed_routing_corpus.py

WARNING — re-running this **re-snapshots** ``current_plan`` from the current
planner. The flip-gate (``test_no_unexpected_route_flips``) relies on the
committed ``current_plan`` to catch *any* routing change, so only re-run when you
are *intentionally* re-baselining after a reviewed routing change — and review the
resulting fixture diff.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.conversational.models import (
    ClinicalConcept,
    CompetencyQuestion,
    InterventionSpec,
    PatientFilter,
    TemporalConstraint,
)
from src.conversational.planner import QueryPlanner

OUT_DIR = (
    Path(__file__).resolve().parents[1]
    / "tests"
    / "test_conversational"
    / "fixtures"
    / "routing_corpus"
)


def _cc(name: str, ctype: str) -> ClinicalConcept:
    return ClinicalConcept(name=name, concept_type=ctype)


def _pf(field: str, op: str, value: str) -> PatientFilter:
    return PatientFilter(field=field, operator=op, value=value)


def _tc(relation: str, ref: str, window: str | None = None) -> TemporalConstraint:
    return TemporalConstraint(relation=relation, reference_event=ref, time_window=window)


# Each entry: (name, question, CompetencyQuestion, desired_plan, tags, notes).
# current_plan is derived from the live planner below.
SEED: list[tuple[str, str, CompetencyQuestion, str, list[str], str]] = [
    # -- Fast-path, correct (desired == current == sql_fast) -----------------
    (
        "mean_creatinine_over_65",
        "What is the average creatinine for patients over 65?",
        CompetencyQuestion(
            original_question="What is the average creatinine for patients over 65?",
            clinical_concepts=[_cc("creatinine", "biomarker")],
            patient_filters=[_pf("age", ">", "65")],
            aggregation="mean", scope="cohort",
        ),
        "sql_fast", ["fast_path", "biomarker"], "Textbook single-concept aggregate.",
    ),
    (
        "max_heart_rate_cohort",
        "What is the maximum heart rate across admissions?",
        CompetencyQuestion(
            original_question="What is the maximum heart rate across admissions?",
            clinical_concepts=[_cc("heart rate", "vital")],
            aggregation="max", scope="cohort",
        ),
        "sql_fast", ["fast_path", "vital"], "MAX over a single vital.",
    ),
    (
        "min_lactate_cohort",
        "What is the lowest lactate recorded?",
        CompetencyQuestion(
            original_question="What is the lowest lactate recorded?",
            clinical_concepts=[_cc("lactate", "biomarker")],
            aggregation="min", scope="cohort",
        ),
        "sql_fast", ["fast_path", "biomarker"], "MIN over a single biomarker.",
    ),
    (
        "count_sepsis",
        "How many patients were diagnosed with sepsis?",
        CompetencyQuestion(
            original_question="How many patients were diagnosed with sepsis?",
            clinical_concepts=[_cc("sepsis", "diagnosis")],
            aggregation="count", scope="cohort",
        ),
        "sql_fast", ["fast_path", "diagnosis"], "Diagnosis COUNT.",
    ),
    (
        "count_ischemic_stroke",
        "How many patients had an ischemic stroke?",
        CompetencyQuestion(
            original_question="How many patients had an ischemic stroke?",
            clinical_concepts=[_cc("ischemic stroke", "diagnosis")],
            aggregation="count", scope="cohort",
        ),
        "sql_fast", ["fast_path", "diagnosis"], "Diversify cohorts beyond sepsis.",
    ),
    (
        "mean_creatinine_by_gender",
        "Compare average creatinine by gender.",
        CompetencyQuestion(
            original_question="Compare average creatinine by gender.",
            clinical_concepts=[_cc("creatinine", "biomarker")],
            aggregation="mean", scope="comparison", comparison_field="gender",
        ),
        "sql_fast", ["fast_path", "comparison"], "Registered axis with sql_group_by.",
    ),
    (
        "mean_lactate_by_readmitted_30d",
        "Compare average lactate between 30-day readmits and non-readmits.",
        CompetencyQuestion(
            original_question="Compare average lactate between 30-day readmits and non-readmits.",
            clinical_concepts=[_cc("lactate", "biomarker")],
            aggregation="mean", scope="comparison", comparison_field="readmitted_30d",
        ),
        "sql_fast", ["fast_path", "comparison"], "Readmission-label axis.",
    ),
    (
        "mean_creatinine_by_age",
        "Compare average creatinine across age groups.",
        CompetencyQuestion(
            original_question="Compare average creatinine across age groups.",
            clinical_concepts=[_cc("creatinine", "biomarker")],
            aggregation="mean", scope="comparison", comparison_field="age",
        ),
        "sql_fast", ["fast_path", "comparison"], "Age axis.",
    ),
    (
        "mortality_by_admission_type",
        "How does mortality differ by admission type?",
        CompetencyQuestion(
            original_question="How does mortality differ by admission type?",
            clinical_concepts=[_cc("in-hospital mortality", "outcome")],
            aggregation="count", scope="comparison", comparison_field="admission_type",
        ),
        "sql_fast", ["fast_path", "comparison", "outcome"], "Outcome by admission_type.",
    ),
    (
        "single_patient_mean_creatinine",
        "What is the average creatinine for this patient?",
        CompetencyQuestion(
            original_question="What is the average creatinine for this patient?",
            clinical_concepts=[_cc("creatinine", "biomarker")],
            patient_filters=[_pf("subject_id", "=", "12345")],
            aggregation="mean", scope="single_patient",
        ),
        "sql_fast", ["fast_path", "single_patient"], "Single-patient aggregate.",
    ),
    (
        "mortality_count_cohort",
        "How many patients died in hospital?",
        CompetencyQuestion(
            original_question="How many patients died in hospital?",
            clinical_concepts=[_cc("in-hospital mortality", "outcome")],
            aggregation="count", scope="cohort",
        ),
        "sql_fast", ["fast_path", "outcome"], "Outcome COUNT.",
    ),
    (
        "drug_count_warfarin",
        "How many patients received warfarin?",
        CompetencyQuestion(
            original_question="How many patients received warfarin?",
            clinical_concepts=[_cc("warfarin", "drug")],
            aggregation="count", scope="cohort",
        ),
        "sql_fast", ["fast_path", "drug"], "Drug COUNT.",
    ),
    (
        "microbiology_count_ecoli",
        "How many patients had an E. coli positive culture?",
        CompetencyQuestion(
            original_question="How many patients had an E. coli positive culture?",
            clinical_concepts=[_cc("Escherichia coli", "microbiology")],
            aggregation="count", scope="cohort",
        ),
        "sql_fast", ["fast_path", "microbiology"], "Microbiology COUNT.",
    ),
    (
        "diagnosis_list_subarachnoid_hemorrhage",
        "List patients diagnosed with subarachnoid hemorrhage.",
        CompetencyQuestion(
            original_question="List patients diagnosed with subarachnoid hemorrhage.",
            clinical_concepts=[_cc("subarachnoid hemorrhage", "diagnosis")],
            aggregation=None, scope="cohort",
        ),
        "sql_fast", ["fast_path", "diagnosis"], "Bare diagnosis list (rule 8).",
    ),
    (
        "peak_troponin_mi",
        "What is the peak troponin in myocardial infarction patients?",
        CompetencyQuestion(
            original_question="What is the peak troponin in myocardial infarction patients?",
            clinical_concepts=[_cc("troponin", "biomarker")],
            patient_filters=[_pf("diagnosis", "contains", "myocardial infarction")],
            aggregation="max", scope="cohort",
        ),
        "sql_fast", ["fast_path", "biomarker"], "Peak (MAX) with a diagnosis filter.",
    ),
    (
        "avg_hemoglobin_pneumonia",
        "What is the mean hemoglobin in pneumonia patients?",
        CompetencyQuestion(
            original_question="What is the mean hemoglobin in pneumonia patients?",
            clinical_concepts=[_cc("hemoglobin", "biomarker")],
            patient_filters=[_pf("diagnosis", "contains", "pneumonia")],
            aggregation="mean", scope="cohort",
        ),
        "sql_fast", ["fast_path", "biomarker"], "Diversify cohorts (pneumonia).",
    ),
    (
        "event_ordering_intubation_mannitol",
        "Which comes first, intubation or hyperosmolar therapy?",
        CompetencyQuestion(
            original_question="Which comes first, intubation or hyperosmolar therapy?",
            clinical_concepts=[_cc("intubation", "procedure"), _cc("mannitol", "drug")],
            aggregation="event_ordering", scope="cohort",
        ),
        "sql_fast", ["fast_path", "event_ordering"], "Dedicated rule 3 branch.",
    ),
    (
        "mortality_split_by_ventilation",
        "Compare mortality between mechanically ventilated and non-ventilated patients.",
        CompetencyQuestion(
            original_question="Compare mortality between mechanically ventilated and non-ventilated patients.",
            clinical_concepts=[_cc("in-hospital mortality", "outcome")],
            aggregation="count", scope="comparison", comparison_field="condition",
            split_condition=_pf("procedure", "contains", "mechanical ventilation"),
        ),
        "sql_fast", ["fast_path", "split_condition"], "Dynamic condition axis (rule 12b).",
    ),
    (
        "causal_two_vasopressors",
        "What is the causal effect of norepinephrine vs vasopressin on mortality?",
        CompetencyQuestion(
            original_question="What is the causal effect of norepinephrine vs vasopressin on mortality?",
            clinical_concepts=[_cc("in-hospital mortality", "outcome")],
            scope="causal_effect",
            intervention_set=[
                InterventionSpec(label="norepinephrine", kind="drug", rxnorm_ingredient="7512"),
                InterventionSpec(label="vasopressin", kind="drug", rxnorm_ingredient="11149"),
            ],
        ),
        "causal", ["causal"], "Well-formed causal CQ (|I|>=2, rule 2).",
    ),
    (
        "patient_similarity_query",
        "Find patients similar to a 68-year-old with acute kidney injury.",
        CompetencyQuestion(
            original_question="Find patients similar to a 68-year-old with acute kidney injury.",
            scope="patient_similarity",
        ),
        "similarity", ["similarity"], "Similarity scope short-circuits (rule 1).",
    ),
    # -- Graph-path, correct (desired == current == graph) -------------------
    (
        "median_lactate_sepsis",
        "What is the median lactate in sepsis patients?",
        CompetencyQuestion(
            original_question="What is the median lactate in sepsis patients?",
            clinical_concepts=[_cc("lactate", "biomarker")],
            patient_filters=[_pf("diagnosis", "contains", "sepsis")],
            aggregation="median", scope="cohort",
        ),
        "graph", ["graph", "median"], "median has no sql_fn (rule 10) — genuinely graph.",
    ),
    (
        "creatinine_before_intubation",
        "What was the creatinine before intubation?",
        CompetencyQuestion(
            original_question="What was the creatinine before intubation?",
            clinical_concepts=[_cc("creatinine", "biomarker")],
            temporal_constraints=[_tc("before", "intubation")],
            scope="cohort",
        ),
        "graph", ["graph", "temporal", "allen"],
        "Relational/Allen temporal (before an EVENT) — genuinely needs the graph.",
    ),
    (
        "lactate_after_dialysis",
        "What was the lactate after dialysis?",
        CompetencyQuestion(
            original_question="What was the lactate after dialysis?",
            clinical_concepts=[_cc("lactate", "biomarker")],
            temporal_constraints=[_tc("after", "dialysis")],
            scope="cohort",
        ),
        "graph", ["graph", "temporal", "allen"], "Event-relative ordering — graph.",
    ),
    (
        "multi_concept_creatinine_lactate",
        "Compare creatinine and lactate together.",
        CompetencyQuestion(
            original_question="Compare creatinine and lactate together.",
            clinical_concepts=[_cc("creatinine", "biomarker"), _cc("lactate", "biomarker")],
            aggregation="mean", scope="cohort",
        ),
        "graph", ["graph", "multi_concept"], "Two concepts (rule 6).",
    ),
    (
        "visualization_lactate_timeline",
        "Plot this patient's lactate over time.",
        CompetencyQuestion(
            original_question="Plot this patient's lactate over time.",
            clinical_concepts=[_cc("lactate", "biomarker")],
            patient_filters=[_pf("subject_id", "=", "123")],
            return_type="visualization", scope="single_patient",
        ),
        "graph", ["graph", "raw_value"], "Raw-value/timeseries lookup (rule 9).",
    ),
    (
        "comparison_unregistered_axis_ethnicity",
        "Compare average creatinine by ethnicity.",
        CompetencyQuestion(
            original_question="Compare average creatinine by ethnicity.",
            clinical_concepts=[_cc("creatinine", "biomarker")],
            aggregation="mean", scope="comparison", comparison_field="ethnicity",
        ),
        "graph", ["graph", "comparison"], "Axis not SQL-compilable yet (rule 13).",
    ),
    (
        "sum_urine_output",
        "What is the total urine output?",
        CompetencyQuestion(
            original_question="What is the total urine output?",
            clinical_concepts=[_cc("urine output", "vital")],
            aggregation="sum", scope="cohort",
        ),
        "graph", ["graph", "sum"], "sum has no sql_fn today (rule 10).",
    ),
    (
        "event_ordering_single_concept",
        "When did intubation happen?",
        CompetencyQuestion(
            original_question="When did intubation happen?",
            clinical_concepts=[_cc("intubation", "procedure")],
            aggregation="event_ordering", scope="cohort",
        ),
        "graph", ["graph", "event_ordering"], "event_ordering needs >=2 concepts (rule 3b).",
    ),
    (
        "condition_split_missing",
        "Compare mortality by condition.",
        CompetencyQuestion(
            original_question="Compare mortality by condition.",
            clinical_concepts=[_cc("in-hospital mortality", "outcome")],
            aggregation="count", scope="comparison", comparison_field="condition",
            split_condition=None,
        ),
        "graph", ["graph", "underspecified"], "condition axis with no split (rule 12a).",
    ),
    (
        "raw_value_creatinine_single",
        "Show this patient's creatinine values.",
        CompetencyQuestion(
            original_question="Show this patient's creatinine values.",
            clinical_concepts=[_cc("creatinine", "biomarker")],
            patient_filters=[_pf("subject_id", "=", "999")],
            aggregation=None, scope="single_patient",
        ),
        "graph", ["graph", "raw_value"], "Raw-value lookup, no aggregation (rule 9).",
    ),
    # -- Known misroutes (desired != current) --------------------------------
    (
        "avg_creatinine_during_icu_stay",
        "What's the average creatinine during the ICU stay for patients over 65?",
        CompetencyQuestion(
            original_question="What's the average creatinine during the ICU stay for patients over 65?",
            clinical_concepts=[_cc("creatinine", "biomarker")],
            patient_filters=[_pf("age", ">", "65")],
            temporal_constraints=[_tc("during", "ICU stay")],
            aggregation="mean", scope="cohort",
        ),
        "sql_fast", ["known_misroute", "temporal", "window"],
        "§6.1 canonical: 'during ICU stay' is a filter window, not an Allen relation. "
        "Temporal veto (rule 4) sends it to graph; desired is a charttime-bounded SQL aggregate.",
    ),
    (
        "lactate_first_24h_admission",
        "What is the mean lactate in the first 24 hours of admission?",
        CompetencyQuestion(
            original_question="What is the mean lactate in the first 24 hours of admission?",
            clinical_concepts=[_cc("lactate", "biomarker")],
            temporal_constraints=[_tc("within", "admission", "24 hours")],
            aggregation="mean", scope="cohort",
        ),
        "sql_fast", ["known_misroute", "temporal", "window"],
        "§6.1: 'first 24h' relative to admission is a time-bound filter, not Allen.",
    ),
    (
        "mean_glucose_during_admission",
        "What is the average glucose during the admission?",
        CompetencyQuestion(
            original_question="What is the average glucose during the admission?",
            clinical_concepts=[_cc("glucose", "biomarker")],
            temporal_constraints=[_tc("during", "admission")],
            aggregation="mean", scope="cohort",
        ),
        "sql_fast", ["known_misroute", "temporal", "window"],
        "§6.1: 'during the admission' window is expressible as a WHERE bound.",
    ),
    (
        "los_heart_failure_metadata",
        "What is the average length of stay for heart failure patients?",
        CompetencyQuestion(
            original_question="What is the average length of stay for heart failure patients?",
            clinical_concepts=[],
            patient_filters=[_pf("diagnosis", "contains", "heart failure")],
            aggregation="mean", scope="cohort",
        ),
        "sql_fast", ["known_misroute", "metadata"],
        "§6.3 / Direction C: metadata-only LOS has a fast-path compiler branch; "
        "rule 5 keeps it on graph 'for now'.",
    ),
    # -- Causal degenerate (diagnostic; desired == current) ------------------
    (
        "causal_degenerate_single_intervention",
        "What is the effect of norepinephrine on mortality?",
        CompetencyQuestion(
            original_question="What is the effect of norepinephrine on mortality?",
            clinical_concepts=[_cc("creatinine", "biomarker")],
            aggregation="mean", scope="causal_effect",
            intervention_set=[
                InterventionSpec(label="norepinephrine", kind="drug", rxnorm_ingredient="7512"),
            ],
        ),
        "sql_fast", ["causal_degenerate"],
        "|I| < 2: degenerate causal falls through to the legacy dispatch and answers "
        "something rather than erroring (had_causal_fallthrough).",
    ),
]


def main() -> None:
    planner = QueryPlanner()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Clear stale fixtures so a removed seed entry doesn't linger.
    for stale in OUT_DIR.glob("*.json"):
        stale.unlink()

    names: set[str] = set()
    written = 0
    for name, question, cq, desired, tags, notes in SEED:
        if name in names:
            raise SystemExit(f"duplicate corpus entry name: {name!r}")
        names.add(name)

        cq_dump = cq.model_dump(mode="json")
        # Provenance guard: the dump must round-trip and re-route identically,
        # else the committed fixture would not faithfully replay this CQ.
        replayed = CompetencyQuestion.model_validate(cq_dump)
        current = planner.classify(replayed).value
        if current != planner.classify(cq).value:
            raise SystemExit(f"{name}: round-trip changed the route")

        entry = {
            "name": name,
            "question": question,
            "cq": cq_dump,
            "desired_plan": desired,
            "current_plan": current,
            "tags": tags,
            "notes": notes,
        }
        (OUT_DIR / f"{name}.json").write_text(
            json.dumps(entry, indent=2, sort_keys=False) + "\n"
        )
        written += 1

    misrouted = sum(
        1 for n, _q, cq, desired, *_ in SEED
        if planner.classify(cq).value != desired
    )
    print(f"wrote {written} corpus entries to {OUT_DIR}")
    print(f"known misroutes (current != desired): {misrouted}")
    print(f"misrouting rate: {misrouted / written:.3f}")


if __name__ == "__main__":
    main()
