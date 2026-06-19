"""Unit tests for the query-routing decision log (``decision_log``).

The log exists so the routing decision — invisible until now — becomes
observable (``tail -f logs/routing_decisions.jsonl``) and the misrouting rate
becomes measurable (§8 "D0"). Each ``log_routing_decision`` call must append
exactly one JSON line carrying the chosen plan, the *reason* it was chosen, a
denormalized digest of the CQ, and a full replayable CQ dump — resolve its path
at call time, and never raise (a logging hiccup must not break the turn).
"""

from __future__ import annotations

import json
from pathlib import Path

from src.conversational.decision_log import log_routing_decision
from src.conversational.models import (
    ClinicalConcept,
    CompetencyQuestion,
    InterventionSpec,
    PatientFilter,
    TemporalConstraint,
)
from src.conversational.planner import QueryPlan, QueryPlanner, RoutingDecision


def _read_records(path: Path) -> list[dict]:
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    return [json.loads(ln) for ln in lines]


def _log(cq, *, log_path, turn_id="t0", cq_index=0, n_cqs=1, is_multi=False,
         question=None):
    decision = QueryPlanner().explain(cq)
    return log_routing_decision(
        cq, decision, turn_id=turn_id, cq_index=cq_index, n_cqs=n_cqs,
        is_multi=is_multi, question=question, log_path=log_path,
    )


def test_appends_one_record_with_plan_and_reason(tmp_path):
    log = tmp_path / "routing.jsonl"
    cq = CompetencyQuestion(
        original_question="average creatinine for patients over 65",
        clinical_concepts=[ClinicalConcept(name="creatinine", concept_type="biomarker")],
        aggregation="mean", scope="cohort",
    )
    rec = _log(cq, log_path=log, question="avg creatinine over 65")

    records = _read_records(log)
    assert len(records) == 1
    on_disk = records[0]
    assert on_disk == rec  # returned record matches what was written
    assert on_disk["kind"] == "routing_decision"
    assert on_disk["schema_version"] == "1"
    assert on_disk["plan"] == "sql_fast"
    assert on_disk["reason"] == "fallthrough_sql_fast"
    assert on_disk["rule"] == 14
    assert on_disk["detail"]
    assert on_disk["question"] == "avg creatinine over 65"
    assert on_disk["cq_question"] == "average creatinine for patients over 65"
    assert on_disk["turn_id"] == "t0"
    assert on_disk["timestamp"]


def test_digest_columns_are_denormalized(tmp_path):
    log = tmp_path / "routing.jsonl"
    cq = CompetencyQuestion(
        original_question="creatinine during ICU stay",
        clinical_concepts=[ClinicalConcept(name="creatinine", concept_type="biomarker")],
        patient_filters=[PatientFilter(field="age", operator=">", value="65")],
        temporal_constraints=[
            TemporalConstraint(relation="during", reference_event="ICU stay"),
        ],
        aggregation="mean", scope="cohort",
    )
    rec = _log(cq, log_path=log)

    # Part A: a window/anchor temporal constraint ("during ICU stay") is now
    # SQL-bound-able, so it falls through to the generic SQL leg instead of the
    # rule-4 veto. The digest still exposes that a temporal constraint was
    # present and which relation it carried.
    assert rec["plan"] == "sql_fast"
    assert rec["reason"] == "fallthrough_sql_fast"
    assert rec["rule"] == 14
    assert rec["scope"] == "cohort"
    assert rec["aggregation"] == "mean"
    assert rec["concept_count"] == 1
    assert rec["concept_types"] == ["biomarker"]
    assert rec["has_temporal"] is True
    assert rec["temporal_relations"] == ["during"]
    assert rec["n_filters"] == 1
    assert rec["split_condition_present"] is False
    assert rec["intervention_count"] == 0


def test_full_cq_dump_is_replayable(tmp_path):
    log = tmp_path / "routing.jsonl"
    cq = CompetencyQuestion(
        original_question="median lactate in sepsis",
        clinical_concepts=[ClinicalConcept(name="lactate", concept_type="biomarker")],
        aggregation="median", scope="cohort",
    )
    rec = _log(cq, log_path=log)

    # Reconstruct the CQ from the logged dump and confirm it re-routes identically.
    replayed = CompetencyQuestion.model_validate(rec["cq"])
    assert QueryPlanner().classify(replayed) == QueryPlan.GRAPH
    assert rec["plan"] == "graph"
    assert rec["reason"] == "aggregate_no_sql_fn"


def test_causal_fallthrough_flag(tmp_path):
    log = tmp_path / "routing.jsonl"
    # A causal-scoped CQ with a single intervention is degenerate (|I| < 2): it
    # drops through to the legacy dispatch rather than routing to CAUSAL.
    degenerate = CompetencyQuestion(
        original_question="effect of norepinephrine",
        clinical_concepts=[ClinicalConcept(name="lactate", concept_type="biomarker")],
        aggregation="mean", scope="causal_effect",
        intervention_set=[InterventionSpec(label="norepinephrine", kind="drug",
                                           rxnorm_ingredient="7512")],
    )
    rec = _log(degenerate, log_path=log)
    assert rec["plan"] != "causal"
    assert rec["had_causal_fallthrough"] is True

    # A plain cohort aggregate never "fell through" from causal.
    plain = CompetencyQuestion(
        original_question="mean creatinine",
        clinical_concepts=[ClinicalConcept(name="creatinine", concept_type="biomarker")],
        aggregation="mean", scope="cohort",
    )
    rec2 = log_routing_decision(
        plain, QueryPlanner().explain(plain), turn_id="t1", cq_index=0,
        n_cqs=1, is_multi=False, log_path=log,
    )
    assert rec2["had_causal_fallthrough"] is False


def test_appends_are_cumulative_and_keep_turn_grouping(tmp_path):
    log = tmp_path / "routing.jsonl"
    cq = CompetencyQuestion(
        original_question="q",
        clinical_concepts=[ClinicalConcept(name="creatinine", concept_type="biomarker")],
        aggregation="mean", scope="cohort",
    )
    _log(cq, log_path=log, turn_id="turnA", cq_index=0, n_cqs=2, is_multi=True)
    _log(cq, log_path=log, turn_id="turnA", cq_index=1, n_cqs=2, is_multi=True)
    records = _read_records(log)
    assert [r["turn_id"] for r in records] == ["turnA", "turnA"]
    assert [r["cq_index"] for r in records] == [0, 1]
    assert all(r["is_multi"] for r in records)


def test_resolves_path_from_env_at_call_time(tmp_path, monkeypatch):
    log = tmp_path / "from_env.jsonl"
    monkeypatch.setenv("NEUROGRAPH_ROUTING_LOG", str(log))
    cq = CompetencyQuestion(
        original_question="env q",
        clinical_concepts=[ClinicalConcept(name="creatinine", concept_type="biomarker")],
        aggregation="mean", scope="cohort",
    )
    _log(cq, log_path=None, question="env q")  # no explicit path → env wins
    records = _read_records(log)
    assert records[0]["question"] == "env q"


def test_never_raises_when_path_is_unwritable(tmp_path):
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("i am a file")
    doomed = blocker / "routing.jsonl"
    cq = CompetencyQuestion(
        original_question="q",
        clinical_concepts=[ClinicalConcept(name="creatinine", concept_type="biomarker")],
        aggregation="mean", scope="cohort",
    )
    rec = _log(cq, log_path=doomed)  # must not raise
    assert rec["plan"] == "sql_fast"
    assert not doomed.exists()


def test_never_raises_on_malformed_cq_or_decision(tmp_path):
    log = tmp_path / "routing.jsonl"
    # Bare objects have none of the expected attributes — the digest/dump steps
    # fail, but the function is total and still writes a minimal record.
    rec = log_routing_decision(
        object(), object(), turn_id="t", cq_index=0, n_cqs=1, is_multi=False,
        log_path=log,
    )
    assert rec["kind"] == "routing_decision"
    records = _read_records(log)
    assert len(records) == 1
