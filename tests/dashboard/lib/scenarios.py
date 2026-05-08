"""Reusable scenario builders for the dashboard test suite.

Mirrors the §4a / §4b / §4c examples in ``docs/phase-h-smoke-test.md`` so
each Tier 2 / Tier 3 test can construct its scenario in one line, and so
the test source documents what the scenario is even when read in
isolation.

Helpers return ``(CompetencyQuestion, AnswerResult)`` pairs ready to feed
into ``critique(client, cq, answer)`` (Tier 2) or to ``compile_sql(...)``
(Tier 3) without further setup. For Tier 1 dashboard tests we expose the
plain natural-language questions plus the deterministic SQL-shape
predicates the e2e tests assert against.
"""

from __future__ import annotations

from src.conversational.models import (
    AnswerResult,
    ClinicalConcept,
    CompetencyQuestion,
    PatientFilter,
)


# ---------------------------------------------------------------------------
# Tier 2 scenarios — for direct critique() smoke
# ---------------------------------------------------------------------------


def build_dexmedetomidine_overdose_cq() -> tuple[CompetencyQuestion, AnswerResult]:
    """§4a — implausible drug dose. ~6× FDA-labeled max maintenance
    infusion (0.2-0.7 mcg/kg/hr). Critic should fire ``severity=warn``,
    ``plausible=False``, citing the FDA label and/or rxnorm_lookup."""
    cq = CompetencyQuestion(
        original_question="Mean dexmedetomidine dose in our septic shock cohort",
        clinical_concepts=[
            ClinicalConcept(name="dexmedetomidine", concept_type="drug"),
        ],
        return_type="text_and_table",
        scope="cohort",
        interpretation_summary=(
            "Cohort: ICU septic shock. Drug: dexmedetomidine. "
            "Aggregate: mean infusion dose."
        ),
    )
    answer = AnswerResult(
        text_summary="Mean dexmedetomidine dose 4.5 mcg/kg/hr (n=124) in septic shock cohort.",
        data_table=[{"mean_dose_mcg_kg_hr": 4.5, "n": 124}],
    )
    return cq, answer


def build_cardiogenic_shock_cq() -> tuple[CompetencyQuestion, AnswerResult]:
    """§4b — diagnosis-code mismatch. Interpretation says R57.0 (correct
    ICD-10 for cardiogenic shock) but the answer cites I50.9 (heart
    failure unspecified — wrong code). Critic should flag warn."""
    cq = CompetencyQuestion(
        original_question="What's the cohort size for cardiogenic shock?",
        clinical_concepts=[
            ClinicalConcept(name="cardiogenic shock", concept_type="diagnosis"),
        ],
        return_type="text_and_table",
        scope="cohort",
        interpretation_summary="Cohort definition: ICD-10 R57.0 (cardiogenic shock).",
    )
    answer = AnswerResult(
        text_summary="Cardiogenic shock cohort (ICD-10 I50.9) has 4,127 hadm_ids in MIMIC-IV.",
        data_table=[{"icd_code": "I50.9", "n_hadm_ids": 4127}],
    )
    return cq, answer


def build_t2dm_snomed_mapping_cq() -> tuple[CompetencyQuestion, AnswerResult]:
    """§4c — SNOMED 73211009 (generic Diabetes mellitus) → ICD-10 E11.9
    (T2DM). The mapping is technically defensible per OMOP convention,
    so the critic typically returns ``severity=info``. We encode that as
    the expected verdict (catches if the critic over-flags)."""
    cq = CompetencyQuestion(
        original_question="Mortality rate for type 2 diabetes patients",
        clinical_concepts=[
            ClinicalConcept(name="type 2 diabetes", concept_type="diagnosis"),
        ],
        return_type="text_and_table",
        scope="cohort",
        interpretation_summary="Cohort: T2DM (ICD-10 E11.9 mapped from SNOMED 73211009).",
    )
    answer = AnswerResult(
        text_summary=(
            "In-hospital mortality for type 2 diabetes "
            "(ICD-10 E11.9, SNOMED 73211009) was 4.2% (n=8421)."
        ),
        data_table=[{"mortality_pct": 4.2, "n": 8421}],
    )
    return cq, answer


def build_clean_lactate_cq() -> tuple[CompetencyQuestion, AnswerResult]:
    """Known-good answer: 2.42 mmol/L matches the MIMIC sepsis cohort
    reference (mean=2.40, p50=1.8, p95=6.9). Critic should fire
    ``severity=info``, ``plausible=True``."""
    cq = CompetencyQuestion(
        original_question="What is the mean lactate in our sepsis cohort?",
        clinical_concepts=[
            ClinicalConcept(name="lactate", concept_type="biomarker", loinc_code="32693-4"),
        ],
        patient_filters=[
            PatientFilter(field="diagnosis", operator="contains", value="sepsis"),
        ],
        aggregation="mean",
        return_type="text_and_table",
        scope="cohort",
        interpretation_summary="Mean lactate across admissions with a sepsis diagnosis.",
    )
    answer = AnswerResult(
        text_summary="The mean lactate level in the sepsis cohort is 2.42 mmol/L.",
        data_table=[{"mean_value": 2.42}],
    )
    return cq, answer


def build_borderline_creatinine_cq() -> tuple[CompetencyQuestion, AnswerResult]:
    """§3b — creatinine value high enough to warrant a tool lookup but
    not impossible. 3.8 mg/dL in a CKD cohort is borderline (severe but
    not nonsensical). Critic should fire at least one tool call to
    triangulate against MIMIC distribution / reference range."""
    cq = CompetencyQuestion(
        original_question="What is the mean creatinine in our CKD cohort?",
        clinical_concepts=[
            ClinicalConcept(name="creatinine", concept_type="biomarker", loinc_code="2160-0"),
        ],
        patient_filters=[
            PatientFilter(field="diagnosis", operator="contains", value="chronic kidney disease"),
        ],
        aggregation="mean",
        return_type="text_and_table",
        scope="cohort",
        interpretation_summary="Mean serum creatinine across CKD admissions.",
    )
    answer = AnswerResult(
        text_summary="Mean creatinine in the CKD cohort is 3.8 mg/dL.",
        data_table=[{"mean_value": 3.8}],
    )
    return cq, answer


# ---------------------------------------------------------------------------
# Tier 1 / Tier 3 scenarios — natural-language questions + SQL-shape
# predicates the e2e tests assert against.
# ---------------------------------------------------------------------------


SMOKING_GUN_QUESTION = "What is the mean lactate in our sepsis cohort?"
"""The query that originally returned 7.99 mmol/L (LIKE-pollution) and
now should return ~2.42 mmol/L with ``di.icd_code IN (...)`` grounding.
Inc 9 smoking-gun regression."""

DIAGNOSIS_COUNT_QUESTION = "How many patients had sepsis?"
"""Diagnosis-count path; Inc 4 regression."""


# ICD-10-CM sepsis family. Used both as the mocked OMOPHub response in
# Tier 3 (so tests don't hit the network) AND as the expected substring
# set in Tier 1 SQL assertions.
SEPSIS_FAMILY_PREFIXES = ("A41", "R65", "A40", "A42")
"""ICD-10-CM sepsis-family prefixes that should appear in grounded
diagnosis filters / cohort definitions for any "sepsis" query."""


MOCK_OMOPHUB_SEPSIS_RESPONSE = {
    "status": "ok",
    "results": [
        {"code": "A41.9", "title": "Sepsis, unspecified organism", "confidence": 0.92},
        {"code": "R65.21", "title": "Severe sepsis with septic shock", "confidence": 0.81},
        {"code": "A40.9", "title": "Streptococcal sepsis, unspecified", "confidence": 0.71},
    ],
}
"""Canned ``icd_autocode`` response for "sepsis" — matches what real
OMOPHub returns. Tier 3 monkeypatches ``concept_resolver.icd_autocode``
to return this so tests are hermetic."""


MOCK_MIMIC_ITEMID_LACTATE_RESPONSE = {
    "status": "ok",
    "results": [
        {"itemid": 50813, "label": "Lactate", "table": "labevents", "loinc": "32693-4"},
        {"itemid": 52442, "label": "Lactate, ABG", "table": "labevents", "loinc": "2518-9"},
    ],
}
"""Canned ``mimic_itemid_search`` response for "lactate" — used by Tier 3
biomarker fallback test."""
