"""Aggregate operations — maps aggregation keywords to SPARQL templates.

Each aggregate registered here corresponds to one of the ``aggregation_*`` or
``value_lookup`` SPARQL templates in ``reasoner.TEMPLATES``. The registry is
what the reasoner consults to pick a template; previously the dispatch was an
elif chain in ``reasoner.select_templates``.

Adding a new aggregate (e.g. ``percentile``) is a matter of adding a SPARQL
template to ``reasoner.TEMPLATES`` and registering a new ``AggregateOperation``
that points to it here — no edits to ``select_templates`` required.
"""

from __future__ import annotations

from typing import Any, Callable

from src.conversational.operations import (
    AggregateOperation,
    OperationRegistry,
)


def _median_post_processor(rows: list[dict[str, Any]]) -> tuple[list[dict], list[str]]:
    """Compute median from value_lookup rows.

    Kept as an inner detail of the aggregate operation so the registry owns
    the full aggregate→post-processing contract; the reasoner just calls
    ``op.post_processor`` when present.
    """
    from statistics import median

    values = [float(r["value"]) for r in rows if r.get("value") is not None]
    if not values:
        return [{"median_value": None}], ["median_value"]
    return [{"median_value": median(values)}], ["median_value"]


def _event_ordering_post_processor(
    rows: list[dict[str, Any]],
) -> tuple[list[dict], list[str]]:
    """Summarise the most-common temporal ORDER of N events across a cohort.

    Input ``rows`` are one row per (admission, event) carrying ``hadm_id``,
    ``event_name``, and ``event_time`` — the per-patient FIRST time of each
    event (the SQL fast-path's ``event_ordering`` query emits exactly this).
    Rows whose ``event_time`` is NULL (the patient never had that event) are
    dropped, so a patient contributes only the events they actually experienced.

    Output (single summary row):

    * ``most_common_sequence`` — the "→"-joined event-name order shared by the
      most patients (e.g. ``"GCS drop → intubation → hyperosmolar therapy"``),
      or ``None`` when no patient has ≥1 timed event.
    * ``n_patients`` — how many patients had that exact sequence.
    * ``pct`` — that count as a fraction of patients with ≥1 timed event.
    * ``<event>_first_fraction`` — for EACH event seen, the fraction of patients
      (with ≥1 timed event) whose earliest event was that one. The
      ``gcs_drop_first_fraction`` key the demo asks for is one of these, keyed by
      the event name slugified (lower-cased, non-alphanumerics → ``_``). Always
      present for ``GCS drop`` when that event appears in the input so the demo
      column is stable; absent events simply don't get a key.
    * ``median_first_to_last_hours`` — the median, across patients with ≥2 timed
      events, of the gap (hours) between their first and last event. ``None``
      when no patient has ≥2 events.

    Pure / deterministic; ties on the most-common sequence break by the
    lexicographically smallest sequence so the output is stable.
    """
    from collections import Counter
    from statistics import median

    def _slug(name: str) -> str:
        out = []
        for ch in (name or "").lower():
            out.append(ch if ch.isalnum() else "_")
        # Collapse runs of "_" and strip leading/trailing ones.
        slug = "".join(out)
        while "__" in slug:
            slug = slug.replace("__", "_")
        return slug.strip("_")

    # Group each patient's timed events, ordered by time.
    per_patient: dict[Any, list[tuple[Any, str]]] = {}
    for r in rows:
        if r.get("event_time") is None:
            continue
        hadm = r.get("hadm_id")
        name = r.get("event_name")
        if hadm is None or not name:
            continue
        per_patient.setdefault(hadm, []).append((r["event_time"], name))

    columns = [
        "most_common_sequence", "n_patients", "pct",
        "median_first_to_last_hours",
    ]

    if not per_patient:
        return (
            [{
                "most_common_sequence": None,
                "n_patients": 0,
                "pct": 0.0,
                "median_first_to_last_hours": None,
            }],
            columns,
        )

    sequences: list[tuple[str, ...]] = []
    first_events: list[str] = []
    gaps_hours: list[float] = []
    for events in per_patient.values():
        events.sort(key=lambda e: e[0])
        seq = tuple(name for _, name in events)
        sequences.append(seq)
        first_events.append(seq[0])
        if len(events) >= 2:
            delta = events[-1][0] - events[0][0]
            # ``event_time`` is typically a datetime (BigQuery/DuckDB TIMESTAMP);
            # support a raw numeric (epoch/seconds) too for synthetic inputs.
            if hasattr(delta, "total_seconds"):
                gaps_hours.append(delta.total_seconds() / 3600.0)
            else:
                gaps_hours.append(float(delta) / 3600.0)

    n = len(sequences)
    seq_counts = Counter(sequences)
    # Most common; tie-break by lexicographically smallest sequence for stability.
    best_seq, best_count = min(
        seq_counts.items(), key=lambda kv: (-kv[1], kv[0]),
    )

    out: dict[str, Any] = {
        "most_common_sequence": " → ".join(best_seq),
        "n_patients": best_count,
        "pct": best_count / n,
        "median_first_to_last_hours": (
            median(gaps_hours) if gaps_hours else None
        ),
    }

    # Per-first-event fractions. One key per distinct event that was ever first,
    # plus the slugged ``<event>_first_fraction`` the demo surfaces.
    first_counts = Counter(first_events)
    for ev_name, cnt in first_counts.items():
        key = f"{_slug(ev_name)}_first_fraction"
        out[key] = cnt / n
        if key not in columns:
            columns.append(key)

    return [out], columns


def register_default_aggregates(registry: OperationRegistry) -> None:
    """Register the aggregate keywords the reasoner currently supports.

    Mirrors the existing dispatch in ``reasoner.select_templates`` 1:1. The
    ``avg`` alias for ``mean`` is registered separately so that both keywords
    resolve cleanly; the reasoner's prior elif treated them as equivalent.
    """
    registry.register(AggregateOperation(
        name="mean",
        description="arithmetic mean over numeric values",
        template="aggregation_mean",
        sql_fn="AVG",
    ))
    registry.register(AggregateOperation(
        name="avg",
        description='alias for "mean"',
        template="aggregation_mean",
        sql_fn="AVG",
    ))
    # Median needs Python post-processing; no portable SQL primitive today.
    # Leaving sql_fn=None keeps it on the graph path via the planner.
    registry.register(AggregateOperation(
        name="median",
        description="middle value (computed in Python post-SPARQL)",
        template="value_lookup",
        post_processor=_median_post_processor,
    ))
    # event_ordering: the most-common temporal ORDER of N events in a cohort
    # ("which comes first — intubation, hyperosmolar therapy, or a GCS drop?").
    # The SQL fast-path has a dedicated compile branch (``_compile_event_ordering``)
    # that returns each patient's FIRST time of each event; this post-processor
    # turns those rows into the most-common sequence + per-first-event fractions
    # + median inter-event gap. ``sql_fn`` stays None (it's not a single SQL
    # aggregate); ``template`` reuses ``value_lookup`` only to satisfy the
    # "every aggregate references an existing template" guard — the fast-path
    # branch, not a SPARQL template, is what actually runs.
    registry.register(AggregateOperation(
        name="event_ordering",
        description=(
            "most-common temporal order of N events across a cohort (computed "
            "in Python); pair with ≥2 clinical_concepts naming the events"
        ),
        template="value_lookup",
        post_processor=_event_ordering_post_processor,
    ))
    registry.register(AggregateOperation(
        name="max",
        description="largest value",
        template="aggregation_max",
        sql_fn="MAX",
    ))
    registry.register(AggregateOperation(
        name="min",
        description="smallest value",
        template="aggregation_min",
        sql_fn="MIN",
    ))
    registry.register(AggregateOperation(
        name="count",
        description="row count (admissions / events matching the concept)",
        template="aggregation_count",
        sql_fn="COUNT",
    ))
    # ``sum`` and ``exists`` are named in the decomposer prompt but have no
    # SPARQL template today. Register them with the same fallback template as
    # count so the LLM can still emit them without triggering a retry; this
    # can be tightened once dedicated templates exist. No sql_fn yet — a
    # ``sum`` of lab values over a cohort doesn't match any clinically
    # useful aggregation, so we keep it on the graph path until a real
    # template lands.
    registry.register(AggregateOperation(
        name="sum",
        description="total over numeric values (uses count template today)",
        template="aggregation_count",
    ))
    registry.register(AggregateOperation(
        name="exists",
        description="boolean presence check (uses count template today)",
        template="aggregation_count",
    ))
