"""Append-only activity log for dashboard chat queries.

Each chat query the user runs in the Streamlit UI appends one JSON line here
so an operator can ``tail -f`` the file and watch queries land in real time:

    tail -f logs/dashboard_queries.jsonl

The destination is ``$NEUROGRAPH_QUERY_LOG`` (default
``logs/dashboard_queries.jsonl``), resolved at call time so an override set
*after* the app module imports still takes effect. ``log_query_run`` is
deliberately total — it never raises, because a logging failure must not
break the query the user is waiting on.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_LOG_PATH = "logs/dashboard_queries.jsonl"
_SUMMARY_MAX_CHARS = 500


def _resolve_path(log_path: str | Path | None) -> Path:
    """Explicit ``log_path`` wins; otherwise read the env var at call time."""
    if log_path is not None:
        return Path(log_path)
    return Path(os.environ.get("NEUROGRAPH_QUERY_LOG", _DEFAULT_LOG_PATH))


def _digest_answer(answer: Any) -> dict[str, Any]:
    """Pull a compact, JSON-safe summary out of an ``AnswerResult``.

    Defensive by design: the answer is whatever the pipeline returned, so we
    read each attribute with ``getattr`` and tolerate ``None``/missing fields
    rather than assume a schema.
    """
    summary = (getattr(answer, "text_summary", None) or "")[:_SUMMARY_MAX_CHARS]

    verdict = getattr(answer, "critic_verdict", None)
    critic_severity = getattr(verdict, "severity", None) if verdict else None

    report = getattr(answer, "outlier_report", None)
    n_outliers_removed = int(getattr(report, "n_removed", 0) or 0) if report else 0

    return {
        "summary": summary,
        "n_sql": len(getattr(answer, "sparql_queries_used", None) or []),
        "n_rows": len(getattr(answer, "data_table", None) or []),
        "critic_severity": critic_severity,
        "n_outliers_removed": n_outliers_removed,
        "clarifying": bool(getattr(answer, "clarifying_question", None)),
    }


def _build_record(
    question: str,
    *,
    duration_s: float | None,
    answer: Any,
    error: str | None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "duration_s": round(duration_s, 3) if duration_s is not None else None,
        "status": "error" if error else "ok",
        "error": error,
        # Defaults so the schema is stable even when there's no answer.
        "summary": "",
        "n_sql": 0,
        "n_rows": 0,
        "critic_severity": None,
        "n_outliers_removed": 0,
        "clarifying": False,
    }
    if answer is not None:
        record.update(_digest_answer(answer))
    return record


def log_query_run(
    question: str,
    *,
    duration_s: float | None = None,
    answer: Any = None,
    error: str | None = None,
    log_path: str | Path | None = None,
) -> dict[str, Any]:
    """Append one JSON line describing a single dashboard query run.

    Returns the record dict (handy for tests and callers). Writes are
    best-effort: any I/O failure is logged at WARNING and swallowed so the
    UI keeps working.
    """
    record = _build_record(
        question, duration_s=duration_s, answer=answer, error=error,
    )

    # Mirror to the standard logger too, so it also shows up wherever the
    # process's logging is wired (console/stderr) — not just the JSONL file.
    logger.info(
        "query status=%s duration_s=%s n_sql=%s n_rows=%s q=%r",
        record["status"], record["duration_s"], record["n_sql"],
        record["n_rows"], question,
    )

    path = _resolve_path(log_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
    except Exception as exc:  # never break the UI on a logging failure
        logger.warning("query_log write to %s failed: %s", path, exc)

    return record
