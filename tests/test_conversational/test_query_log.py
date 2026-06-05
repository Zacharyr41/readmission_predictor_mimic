"""Unit tests for the dashboard query-activity log (``query_log``).

The log exists so an operator can ``tail -f`` a file and watch chat queries
as the user runs them in the Streamlit UI. Each ``log_query_run`` call must
append exactly one JSON line summarizing the run (question, timing, status,
and a digest of the ``AnswerResult``), resolve its path at call time (so an
env override set before the app loads still takes effect), and never raise —
a logging hiccup must not break the user's query.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.conversational.models import AnswerResult, CriticVerdict, OutlierReport
from src.conversational.query_log import log_cohort_criteria, log_query_run


def _read_records(path: Path) -> list[dict]:
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    return [json.loads(ln) for ln in lines]


def test_appends_one_jsonl_record(tmp_path):
    log = tmp_path / "queries.jsonl"
    rec = log_query_run("How many sepsis patients?", duration_s=1.25, log_path=log)

    records = _read_records(log)
    assert len(records) == 1
    on_disk = records[0]
    assert on_disk == rec  # returned record matches what was written
    assert on_disk["question"] == "How many sepsis patients?"
    assert on_disk["duration_s"] == 1.25
    assert on_disk["status"] == "ok"
    assert on_disk["error"] is None
    assert "timestamp" in on_disk and on_disk["timestamp"]


def test_appends_are_cumulative(tmp_path):
    log = tmp_path / "queries.jsonl"
    log_query_run("q1", log_path=log)
    log_query_run("q2", log_path=log)
    records = _read_records(log)
    assert [r["question"] for r in records] == ["q1", "q2"]


def test_extracts_answer_digest(tmp_path):
    log = tmp_path / "queries.jsonl"
    answer = AnswerResult(
        text_summary="The mean lactate is 2.4 mmol/L.",
        data_table=[{"v": 1}, {"v": 2}, {"v": 3}],
        sparql_queries_used=["SELECT 1", "SELECT 2"],
        critic_verdict=CriticVerdict(
            plausible=True, severity="warn", concern="slightly high",
        ),
        outlier_report=OutlierReport(
            analyte="lactate",
            bound_low=0.0,
            bound_high=40.0,
            units="mmol/L",
            source="seed",
            method="biological_limits",
            n_removed=2,
            n_total=5,
            removed_rows=[{"valuenum": 1e6}, {"valuenum": 2e6}],
            value_with_outliers=400000.0,
            data_table_with_outliers=[{"Mean Value": 400000.0}],
        ),
    )
    rec = log_query_run("mean lactate?", answer=answer, log_path=log)

    assert rec["status"] == "ok"
    assert rec["n_sql"] == 2
    assert rec["n_rows"] == 3
    assert rec["critic_severity"] == "warn"
    assert rec["n_outliers_removed"] == 2
    assert "lactate" in rec["summary"]


def test_handles_answer_with_no_optional_fields(tmp_path):
    log = tmp_path / "queries.jsonl"
    rec = log_query_run("q", answer=AnswerResult(text_summary="ok"), log_path=log)
    assert rec["n_sql"] == 0
    assert rec["n_rows"] == 0
    assert rec["critic_severity"] is None
    assert rec["n_outliers_removed"] == 0


def test_records_error(tmp_path):
    log = tmp_path / "queries.jsonl"
    rec = log_query_run("boom q", duration_s=0.5, error="KeyError: x", log_path=log)
    assert rec["status"] == "error"
    assert rec["error"] == "KeyError: x"
    records = _read_records(log)
    assert records[0]["status"] == "error"


def test_long_summary_is_truncated(tmp_path):
    log = tmp_path / "queries.jsonl"
    answer = AnswerResult(text_summary="x" * 5000)
    rec = log_query_run("q", answer=answer, log_path=log)
    # Bounded so the log line stays scannable; exact cap is an impl detail
    # but it must be far smaller than the original.
    assert len(rec["summary"]) < 1000


def test_never_raises_when_path_is_unwritable(tmp_path):
    # Make the parent a *file* so ``mkdir(parents=True)`` on its child fails.
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("i am a file")
    doomed = blocker / "queries.jsonl"

    rec = log_query_run("q", log_path=doomed)  # must not raise
    assert rec["question"] == "q"
    assert not doomed.exists()


def test_resolves_path_from_env_at_call_time(tmp_path, monkeypatch):
    """The app reads the default path lazily, so an env override applied
    *after* import (as the dashboard tests do) must still be honored."""
    log = tmp_path / "from_env.jsonl"
    monkeypatch.setenv("NEUROGRAPH_QUERY_LOG", str(log))
    log_query_run("env q")  # no explicit log_path
    records = _read_records(log)
    assert records[0]["question"] == "env q"


# ---------------------------------------------------------------------------
# log_cohort_criteria — the cohort-definition audit line (plan II-D).
#
# Cohort selection must be reproducible from the activity log alone, so every
# prefilter and every trait's kind / direction (kernel) / weight / reference
# value lands on one JSONL line a clinician can ``tail -f``.
# ---------------------------------------------------------------------------


def _make_definition():
    from src.conversational.models import PatientFilter
    from src.similarity.models import CohortDefinition, TraitSpec

    return CohortDefinition(
        prefilters=[
            PatientFilter(field="admission_type", operator="=", value="EMERGENCY"),
        ],
        traits=[
            TraitSpec(
                name="age", source="sql", kind="quantitative",
                reference_value=68, direction="symmetric", weight=0.6,
            ),
            TraitSpec(
                name="creatinine_max", source="sql", kind="quantitative",
                reference_value=9.8, direction="higher_more_similar", weight=2.0,
            ),
        ],
        distance_threshold=0.35,
        top_k=30,
    )


def test_log_cohort_criteria_writes_full_criteria(tmp_path):
    log = tmp_path / "queries.jsonl"
    defn = _make_definition()
    rec = log_cohort_criteria(
        "Find emergency patients like a 68yo woman", defn, log_path=log,
    )

    records = _read_records(log)
    assert len(records) == 1
    on_disk = records[0]
    assert on_disk == rec  # returned record matches what was written
    assert on_disk["kind"] == "cohort_definition"
    assert on_disk["question"] == "Find emergency patients like a 68yo woman"
    assert on_disk["distance_threshold"] == 0.35
    assert on_disk["top_k"] == 30
    assert [f["field"] for f in on_disk["prefilters"]] == ["admission_type"]

    traits = {t["name"]: t for t in on_disk["traits"]}
    # Every trait's kernel-defining fields are auditable.
    assert traits["age"]["direction"] == "symmetric"
    assert traits["age"]["weight"] == 0.6
    assert traits["age"]["source"] == "sql"
    assert traits["creatinine_max"]["kind"] == "quantitative"
    assert traits["creatinine_max"]["direction"] == "higher_more_similar"
    assert traits["creatinine_max"]["reference_value"] == 9.8


def test_log_cohort_criteria_never_raises_on_malformed_definition(tmp_path):
    # A bare object has none of the expected attributes, so the digest step
    # fails — but the function is total and still writes a minimal record.
    log = tmp_path / "queries.jsonl"
    rec = log_cohort_criteria("q", object(), log_path=log)  # must not raise

    assert rec["kind"] == "cohort_definition"
    assert rec["question"] == "q"
    assert "traits" not in rec  # digest failed → only the minimal fields
    records = _read_records(log)
    assert len(records) == 1


def test_log_cohort_criteria_never_raises_when_path_unwritable(tmp_path):
    # Parent is a *file*, so ``mkdir(parents=True)`` on its child fails.
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("i am a file")
    doomed = blocker / "queries.jsonl"

    rec = log_cohort_criteria("q", _make_definition(), log_path=doomed)  # no raise
    assert rec["kind"] == "cohort_definition"
    assert not doomed.exists()
