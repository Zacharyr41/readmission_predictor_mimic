"""Tier 1 (cheap, ungated) — query logging is wired into the chat handler.

Guards against ``query_log`` being dead code: drives the *real* app.py
chat-input path with a fake pipeline (no Anthropic, no DuckDB) and asserts
that submitting a question appends a record to the query-activity log. The
log path is redirected to a tmp file via the ``NEUROGRAPH_QUERY_LOG`` env
var, which ``log_query_run`` resolves at call time.
"""

from __future__ import annotations

import json
from pathlib import Path

from streamlit.testing.v1 import AppTest

from src.conversational.models import AnswerResult

_APP = str(
    Path(__file__).resolve().parents[2] / "src" / "conversational" / "app.py"
)


class _FakePipeline:
    """Stand-in for ConversationalPipeline — returns a canned answer."""

    def __init__(self, answer: AnswerResult):
        self._answer = answer
        self.asked: list[str] = []

    def ask(self, question: str, progress_callback=None) -> AnswerResult:
        # Mirror the real pipeline: ``app.py`` passes ``progress_callback``;
        # the fake accepts and ignores it (the stage labels aren't asserted).
        self.asked.append(question)
        return self._answer

    def reset(self) -> None:  # used by the "New Conversation" button
        pass


def _read_records(path: Path) -> list[dict]:
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    return [json.loads(ln) for ln in lines]


def test_submitting_a_question_writes_a_query_log_record(tmp_path, monkeypatch):
    log = tmp_path / "dashboard_queries.jsonl"
    monkeypatch.setenv("NEUROGRAPH_QUERY_LOG", str(log))

    answer = AnswerResult(
        text_summary="412 sepsis patients.",
        sparql_queries_used=["SELECT COUNT(*) FROM diagnoses_icd"],
    )
    at = AppTest.from_file(_APP, default_timeout=60)
    at.session_state["pipeline"] = _FakePipeline(answer)
    at.run()
    assert not at.exception, at.exception

    at.chat_input[0].set_value("How many sepsis patients?").run()
    assert not at.exception, at.exception

    assert log.exists(), "no query log file written — logging is not wired in"
    records = _read_records(log)
    assert len(records) == 1, records
    rec = records[0]
    assert rec["question"] == "How many sepsis patients?"
    assert rec["status"] == "ok"
    assert rec["n_sql"] == 1
    assert rec["duration_s"] is not None
