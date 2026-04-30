"""End-to-end verification of the LOINC-grounded biomarker fix.

Constructs the same CQ the decomposer would emit for "average creatinine for
patients over 65" — now with ``loinc_code='2160-0'`` — runs it through the
resolver, compiles the SQL, and queries BigQuery. The expected result is a
clinically sensible serum-creatinine mean (~1.2-1.5 mg/dL), not the
LIKE-pooled value of ~5 mg/dL we saw before the fix.

Also runs the un-grounded version (no loinc_code) for direct comparison —
that path still uses LIKE and returns the polluted value, demonstrating
the fix vs fallback behavior on real data.
"""

from pathlib import Path

from src.conversational.concept_resolver import ConceptResolver
from src.conversational.extractor import _BigQueryBackend
from src.conversational.models import (
    ClinicalConcept,
    CompetencyQuestion,
    PatientFilter,
)
from src.conversational.operations import get_default_registry
from src.conversational.sql_fastpath import compile_sql

MAPPINGS_DIR = Path(__file__).parent.parent / "data" / "mappings"


def build_cq(loinc_code: str | None) -> CompetencyQuestion:
    return CompetencyQuestion(
        original_question="What is the average creatinine for patients over 65?",
        clinical_concepts=[
            ClinicalConcept(
                name="creatinine",
                concept_type="biomarker",
                loinc_code=loinc_code,
            ),
        ],
        patient_filters=[
            PatientFilter(field="age", operator=">", value="65"),
        ],
        aggregation="mean",
        return_type="text",
        scope="cohort",
    )


def run_one(label: str, loinc_code: str | None) -> None:
    backend = _BigQueryBackend(project="mimic-485500")
    resolver = ConceptResolver(mappings_dir=MAPPINGS_DIR)
    cq = build_cq(loinc_code)

    resolved_itemids = None
    if loinc_code:
        biom = resolver.resolve_biomarker(cq.clinical_concepts[0])
        resolved_itemids = biom.itemids
        print(f"\n=== {label} ===")
        print(f"  LOINC: {loinc_code}  →  SNOMED: {biom.snomed_code}")
        print(f"  resolved itemids: {resolved_itemids}")
        print(f"  fallback_reason: {biom.fallback_reason!r}")
    else:
        print(f"\n=== {label} ===")
        print("  No LOINC supplied — falls back to LIKE")

    query = compile_sql(
        cq, backend, get_default_registry(),
        resolved_names=["creatinine"],
        resolved_itemids=resolved_itemids,
    )
    print(f"  SQL: {query.sql}")
    rows = backend.execute(query.sql, query.params)
    mean_value = rows[0][0] if rows else None
    print(f"  Mean creatinine over age 65: {mean_value:.3f} mg/dL")
    backend.close()


if __name__ == "__main__":
    run_one("Before-fix path (no LOINC, LIKE)", loinc_code=None)
    run_one("After-fix path (LOINC=2160-0, itemid IN)", loinc_code="2160-0")
