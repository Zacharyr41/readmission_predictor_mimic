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
