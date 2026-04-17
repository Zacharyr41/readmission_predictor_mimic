"""Comparison-axis operations — supplies SPARQL graph-pattern fragments.

Preserves the shape of the existing ``_COMPARISON_FIELD_MAP`` in
``reasoner.py``. Each axis contributes at most one clause attached to the
``Patient`` node and one attached to the ``HospitalAdmission`` node;
``reasoner.build_sparql`` splices them into the ``comparison_by_field``
template.

Adding a new comparison axis means registering one ``ComparisonOperation``
here, nothing else — the prompt section for comparison_axis is derived from
the registry at prompt-build time (Phase 3).
"""

from __future__ import annotations

from src.conversational.operations import (
    ComparisonOperation,
    OperationRegistry,
)


def register_default_comparisons(registry: OperationRegistry) -> None:
    """Register the six comparison axes the reasoner currently supports.

    Clauses copied verbatim from ``reasoner._COMPARISON_FIELD_MAP`` so the
    generated SPARQL is unchanged.
    """
    # Patient-node axes: GROUP BY on a patients-table column. The fast-path
    # compiler joins admissions → patients so these columns are in scope.
    registry.register(ComparisonOperation(
        name="gender",
        description="M vs F — grouped on the Patient node",
        patient_clause="mimic:hasGender ?group_value ;",
        sql_group_by="p.gender",
    ))
    registry.register(ComparisonOperation(
        name="age",
        description="patient anchor_age — grouped on the Patient node",
        patient_clause="mimic:hasAge ?group_value ;",
        sql_group_by="p.anchor_age",
    ))
    # Admission-node axes: GROUP BY on an admissions-table column (already in
    # scope via the cohort query's admission alias).
    registry.register(ComparisonOperation(
        name="readmitted_30d",
        description="readmitted within 30 days (0/1) — grouped on the admission",
        admission_clause="mimic:readmittedWithin30Days ?group_value ;",
        sql_group_by="rl.readmitted_30d",
    ))
    registry.register(ComparisonOperation(
        name="readmitted_60d",
        description="readmitted within 60 days (0/1) — grouped on the admission",
        admission_clause="mimic:readmittedWithin60Days ?group_value ;",
        sql_group_by="rl.readmitted_60d",
    ))
    registry.register(ComparisonOperation(
        name="admission_type",
        description="EMERGENCY / ELECTIVE / URGENT — grouped on the admission",
        admission_clause="mimic:hasAdmissionType ?group_value ;",
        sql_group_by="a.admission_type",
    ))
    registry.register(ComparisonOperation(
        name="discharge_location",
        description="HOME / SNF / HOSPICE / etc — grouped on the admission",
        admission_clause="mimic:hasDischargeLocation ?group_value ;",
        sql_group_by="a.discharge_location",
    ))
