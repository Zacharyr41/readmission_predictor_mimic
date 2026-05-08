"""Markdown report writer for the dashboard test suite.

Each test gets a ``Reporter`` instance via the ``reporter`` fixture in
``conftest.py``. Tests record the question, the SQL emitted, the answer
text, the critic verdict, and individual assertion outcomes. On test
teardown the reporter writes a self-contained markdown file under
``tests/dashboard/reports/<test_name>.md`` that's paste-able into chat,
Slack, or PR comments without further formatting.

The report is the qualitative-assessment artifact a human or agent uses
to confirm the dashboard is behaving — distinct from the pass/fail
boolean that pytest itself records.
"""

from __future__ import annotations

import datetime as _dt
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class _Assertion:
    name: str
    ok: bool
    detail: str = ""


@dataclass
class Reporter:
    """Collects qualitative-assessment material for one test, then writes
    a markdown file at teardown.

    The class is intentionally dumb: it just accumulates strings/objects
    in lists and renders them when ``write()`` is called. Callers should
    use the methods rather than mutating fields directly.
    """

    name: str
    tier: int
    output_dir: Path
    question: str | None = None
    answer: str | None = None
    sql_blocks: list[tuple[str, list[Any]]] = field(default_factory=list)
    verdict: dict | None = None
    assertions: list[_Assertion] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def add_question(self, q: str) -> None:
        self.question = q

    def add_answer(self, text: str | None) -> None:
        self.answer = text

    def add_sql(self, sql: str, params: list | None = None) -> None:
        self.sql_blocks.append((sql, list(params or [])))

    def add_verdict(self, verdict: Any) -> None:
        """Accepts a ``CriticVerdict``, a dict, or ``None``. Normalises to dict."""
        if verdict is None:
            self.verdict = None
            return
        if hasattr(verdict, "model_dump"):
            self.verdict = verdict.model_dump()
        elif isinstance(verdict, dict):
            self.verdict = verdict
        else:
            self.verdict = {"raw": repr(verdict)}

    def add_assertion(self, name: str, ok: bool, detail: str = "") -> None:
        self.assertions.append(_Assertion(name=name, ok=ok, detail=detail))

    def add_note(self, body: str) -> None:
        self.notes.append(body)

    def write(self, status: str, duration_s: float) -> Path:
        """Render the markdown file. Returns the path written.

        ``status`` should be one of ``"PASS"``, ``"FAIL"``, ``"ERROR"``,
        ``"SKIP"``. ``duration_s`` is the test wall time.
        """
        commit = _short_commit()
        date = _dt.datetime.now().isoformat(timespec="seconds")
        badge = {
            "PASS": "✅ PASS",
            "FAIL": "❌ FAIL",
            "ERROR": "⚠️ ERROR",
            "SKIP": "⏭️ SKIP",
        }.get(status, status)

        parts: list[str] = []
        parts.append(f"# Smoke Report: {self.name}")
        parts.append("")
        parts.append(
            f"**Status:** {badge} · **Tier:** {self.tier} · "
            f"**Duration:** {duration_s:.2f}s"
        )
        parts.append(f"**Date:** {date} · **Commit:** {commit}")
        parts.append("")
        if self.question:
            parts.append("## Question asked")
            parts.append("")
            parts.append(f"> {self.question}")
            parts.append("")
        if self.answer is not None:
            parts.append("## Answer text")
            parts.append("")
            parts.append(f"> {self.answer}")
            parts.append("")
        if self.sql_blocks:
            parts.append("## SQL emitted")
            parts.append("")
            for sql, params in self.sql_blocks:
                parts.append("```sql")
                parts.append(sql.strip())
                parts.append("```")
                parts.append("")
                if params:
                    parts.append(f"**Params:** `{params!r}`")
                    parts.append("")
        if self.verdict is not None:
            parts.append("## Critic verdict")
            parts.append("")
            v = self.verdict
            sev = v.get("severity")
            plausible = v.get("plausible")
            concern = v.get("concern")
            parts.append(f"- Severity: `{sev}`")
            parts.append(f"- Plausible: `{plausible}`")
            if concern:
                parts.append(f"- Concern: {concern}")
            tool_calls = v.get("tool_calls") or []
            if tool_calls:
                parts.append("- Tool calls:")
                for tc in tool_calls:
                    name = tc.get("name", "?")
                    tc_status = tc.get("status", "?")
                    n = tc.get("n_results")
                    parts.append(f"  - `{name}` status=`{tc_status}` n={n}")
            else:
                parts.append("- Tool calls: *(none)*")
            parts.append("")
        if self.assertions:
            parts.append("## Assertions")
            parts.append("")
            for a in self.assertions:
                mark = "✅" if a.ok else "❌"
                line = f"- {mark} {a.name}"
                if a.detail:
                    line += f" — {a.detail}"
                parts.append(line)
            parts.append("")
        if self.notes:
            parts.append("## Notes")
            parts.append("")
            for n in self.notes:
                parts.append(n)
                parts.append("")

        body = "\n".join(parts).rstrip() + "\n"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"{self.name}.md"
        path.write_text(body, encoding="utf-8")
        return path


def _short_commit() -> str:
    """Return short git SHA, or ``"unknown"`` if not in a repo."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        )
        return out.strip() or "unknown"
    except Exception:
        return "unknown"
