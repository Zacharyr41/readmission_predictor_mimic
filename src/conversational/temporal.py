"""Temporal-constraint classification and SQL bounding — the single source of
truth shared by the planner, the SQL fast-path compiler, and the graph
extractor.

Part A of the query-triage fix (``querytriagesystem.md`` §8). A
``TemporalConstraint`` is one of two kinds:

* **Window / anchor** — bounded by a *structural interval* the database already
  records: an ICU stay (``icustays.intime``…``icustays.outtime``) or a hospital
  admission (``admissions.admittime``…``admissions.dischtime``). "during the ICU
  stay", "in the first 24h of admission". These are expressible as a plain
  ``charttime`` ``WHERE`` bound, so they are eligible for the SQL fast-path.
* **Relational / Allen** — ``before``/``after`` an arbitrary clinical *event*
  ("before intubation"). The reference is not a structural interval, so it
  genuinely needs the graph's Allen-relation reasoning.

The discriminator (:func:`is_sql_window`) is *inferred* from ``reference_event``
— no model/prompt change, no LLM dependency — and the SQL bound
(:func:`temporal_where_predicates`) is generated here once. The graph extractor
delegates to the same generator, so a window constraint produces the *identical*
bound on both paths: SQL/graph parity holds by construction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from src.conversational.models import TemporalConstraint


# Concept types whose events carry a per-event timestamp the fast-path can bound
# to a structural-interval window. ``diagnosis``/``outcome`` are admission-level
# with no per-event time, and the extractor applies no temporal bound to them —
# so a temporally-constrained diagnosis/outcome CQ stays on the graph (routing
# it to SQL would silently drop the constraint and break SQL/graph parity).
# Shared by the planner (routing gate) and the compiler (defensive guard).
WINDOW_BOUNDABLE_CONCEPT_TYPES: frozenset[str] = frozenset({
    "biomarker", "vital", "drug", "microbiology",
})


@dataclass(frozen=True)
class _Anchor:
    """A structural interval a temporal constraint can be bounded against.

    ``keywords`` are matched as substrings against a lowercased
    ``reference_event``; the interval is ``[start_col, end_col]`` on ``table``,
    correlated to the event row by ``hadm_id``.
    """

    keywords: tuple[str, ...]
    table: str
    start_col: str
    end_col: str


# Order is load-bearing: ICU is checked first because it is the *more specific*
# anchor — "ICU admission" / "ICU stay" must resolve to the ICU interval
# (intime/outtime), not the hospital admission. Registry, not scattered
# literals, so new anchors are a one-line append (the escape hatch).
_ANCHORS: tuple[_Anchor, ...] = (
    _Anchor(("icu",), "icustays", "intime", "outtime"),
    _Anchor(
        ("admission", "admit", "hospital stay", "hospitalization"),
        "admissions",
        "admittime",
        "dischtime",
    ),
)


def _resolve_anchor(reference_event: str) -> _Anchor | None:
    """Return the structural-interval anchor a ``reference_event`` names, or
    ``None`` if it refers to an arbitrary clinical event (→ relational/Allen)."""
    ref = reference_event.lower()
    for anchor in _ANCHORS:
        if any(keyword in ref for keyword in anchor.keywords):
            return anchor
    return None


def parse_time_window(window: str) -> str:
    """Convert '48h', '7d', '30m' etc. to a SQL ``INTERVAL`` literal."""
    match = re.match(
        r"(\d+)\s*(h(?:ours?)?|d(?:ays?)?|m(?:in(?:utes?)?)?)",
        window.lower().strip(),
    )
    if not match:
        raise ValueError(f"Cannot parse time window: {window}")
    value = match.group(1)
    unit_char = match.group(2)[0]
    unit_map = {"h": "HOUR", "d": "DAY", "m": "MINUTE"}
    return f"INTERVAL {value} {unit_map[unit_char]}"


def is_sql_window(tc: TemporalConstraint) -> bool:
    """True iff ``tc`` references a structural interval (ICU stay or hospital
    admission) the fast-path can bound directly in SQL.

    This is the planner's window-vs-relational discriminator. A constraint that
    is *not* a window is relational/Allen and must stay on the graph path.
    """
    return _resolve_anchor(tc.reference_event) is not None


def temporal_where_predicates(
    constraints: list[TemporalConstraint],
    time_col: str,
    hadm_col: str,
    backend: Any,
) -> list[str]:
    """Build **bare** SQL ``WHERE`` predicates bounding ``time_col`` to each
    constraint's structural-anchor interval.

    Returns a list of predicate strings with *no* leading ``AND`` and *no*
    joining — each caller glues them in the way its surrounding SQL expects
    (the compiler list-joins with ``" AND "``; the extractor prefixes each with
    ``AND``). The predicates are parameterless (``INTERVAL`` literals are
    inlined), so they never perturb a query's positional parameter ordering.

    Constraints whose ``reference_event`` is not a recognized anchor (relational/
    Allen) — and ``within`` constraints with no ``time_window`` — emit nothing,
    exactly matching the extractor's prior behaviour.
    """
    parts: list[str] = []
    for tc in constraints:
        anchor = _resolve_anchor(tc.reference_event)
        if anchor is None:
            continue
        tbl = backend.table(anchor.table)
        prefix = f"EXISTS (SELECT 1 FROM {tbl} _win WHERE _win.hadm_id = {hadm_col}"
        start = f"_win.{anchor.start_col}"
        end = f"_win.{anchor.end_col}"

        if tc.relation == "during":
            parts.append(
                f"{prefix} AND {time_col} >= {start} AND {time_col} <= {end})"
            )
        elif tc.relation == "within" and tc.time_window:
            interval = parse_time_window(tc.time_window)
            parts.append(
                f"{prefix} AND {time_col} >= {start} "
                f"AND {time_col} <= {start} + {interval})"
            )
        elif tc.relation == "before":
            parts.append(f"{prefix} AND {time_col} < {start})")
        elif tc.relation == "after":
            parts.append(f"{prefix} AND {time_col} > {end})")
        # ``within`` without a time_window: no bound (matches prior behaviour).

    return parts
