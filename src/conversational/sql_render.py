"""Inline bound parameter values back into a parameterized SQL template.

The fast-path compiler emits a ``?``-placeholder template plus a positional
``params`` list (``SqlFastpathQuery``); the executor binds them at run time —
DuckDB passes the list straight to ``.execute(sql, params)`` while BigQuery
rewrites each ``?`` to a typed ``@pN`` (see ``extractor._convert_params``).

For the "Query Details" expander we want the human-readable, copy-pasteable
statement *with the values filled in*, so a clinician sees what actually ran
(``LIKE 'I60%'``) rather than the placeholder (``LIKE ?``). This module is the
inverse of that binding, **for display only** — the rendered string is read,
never re-executed. It mirrors the executor's ``sql.split("?")`` positional
logic so placeholder N lines up with ``params[N]`` exactly as it did at bind
time. Values are rendered faithfully and copy-pasteably (strings single-quoted
with embedded quotes doubled); injection-safety is not a goal here because the
output is never run.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def render_sql_with_params(sql: str, params: Sequence[Any]) -> str:
    """Return ``sql`` with each ``?`` placeholder replaced by its bound value.

    Splits on ``?`` exactly like the BigQuery parameter converter
    (``extractor._convert_params``) so placeholder *N* maps to ``params[N]``.

    Raises
    ------
    ValueError
        If the number of ``?`` placeholders does not equal ``len(params)`` — a
        guard so a template and its params drifting out of sync fails loudly
        rather than rendering a misaligned (and misleading) statement.
    """
    segments = sql.split("?")
    n_placeholders = len(segments) - 1
    if n_placeholders != len(params):
        raise ValueError(
            f"placeholder/param count mismatch: {n_placeholders} '?' "
            f"placeholder(s) but {len(params)} param(s)"
        )
    out = [segments[0]]
    for value, tail in zip(params, segments[1:]):
        out.append(_render_value(value))
        out.append(tail)
    return "".join(out)


def _render_value(value: Any) -> str:
    """Render one bound value as a SQL literal (display only)."""
    if value is None:
        return "NULL"
    # bool is an int subclass — check it first so flags read TRUE/FALSE, not
    # 1/0 (a deliberate display divergence from the executor's INT64 binding).
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, tuple)):
        # An IN-list bound as a single param renders as ``(a, b, c)``.
        return "(" + ", ".join(_render_value(v) for v in value) + ")"
    # Strings and anything else (datetime, Decimal, …): single-quote and
    # escape embedded quotes by doubling, as SQL requires.
    return "'" + str(value).replace("'", "''") + "'"
