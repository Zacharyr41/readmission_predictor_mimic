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


def _admission_type_description() -> str:
    """Comparison-axis description naming the REAL MIMIC-IV admission types.

    Drawn from the frozen schema-grounded categorical-domain artifact so the
    decomposer groups on the vocabulary the data actually uses (``EW EMER.``,
    ``EU OBSERVATION``, …) rather than stale MIMIC-III literals.
    """
    from src.similarity.categorical_domains import describe_domain

    examples = describe_domain("admission_type")
    tail = f"{examples} — " if examples else ""
    return f"{tail}grouped on the admission"


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
        description=_admission_type_description(),
        admission_clause="mimic:hasAdmissionType ?group_value ;",
        sql_group_by="a.admission_type",
    ))
    registry.register(ComparisonOperation(
        name="discharge_location",
        description="HOME / SNF / HOSPICE / etc — grouped on the admission",
        admission_clause="mimic:hasDischargeLocation ?group_value ;",
        sql_group_by="a.discharge_location",
    ))
    # Dynamic split-by-condition axis (SQL-fast-path only). Unlike the fixed-
    # column axes above, this one has NO ``sql_group_by`` and NO SPARQL clause:
    # the GROUP BY column is built at compile time from ``cq.split_condition``
    # as a ``CASE WHEN EXISTS(<sub-condition>) THEN 'yes' ELSE 'no' END`` (see
    # ``sql_fastpath._comparison_group_by_col``). It exists in the registry so
    # the decomposer prompt advertises it; the planner routes a
    # ``comparison_field='condition'`` CQ with a populated ``split_condition``
    # straight to the SQL fast-path, never to the graph, so the missing SPARQL
    # clause is never reached.
    registry.register(ComparisonOperation(
        name="condition",
        description=(
            "split the cohort by presence/absence of a sub-condition supplied "
            "in split_condition (e.g. a diagnosis like 'chronic anticoagulant "
            "use', or 'ventilation'); use comparison_field='condition' WITH a "
            "split_condition object — yields two groups, 'yes' and 'no'"
        ),
        sql_group_by=None,
    ))
