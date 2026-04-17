"""Operation registry for the conversational pipeline.

This module is the single source of truth for every "operation" the pipeline
supports — filters, aggregates, and comparison axes today. Before this, the
set of supported filter fields was defined in three places (prompt text,
decomposer's ``supported_fields`` literal, extractor's ``_get_filtered_hadm_ids``)
and drift between them was a silent failure mode. Now the prompt is generated
from the registry's self-description, the decomposer validates against
``registry.supported_names``, and the extractor compiles filters via the
registry — so there is exactly one place to add a new field.

Three operation kinds are defined today. Each is a concrete class with its own
``compile(...)`` signature — we could have flattened this into a single Protocol
per the original plan but the compile inputs and outputs differ enough per kind
that typed-per-kind classes are clearer to read and harder to misuse.

    FilterOperation       contributes SQL JOIN/WHERE fragments to the cohort query
    AggregateOperation    picks a SPARQL template + optional Python post-processing
    ComparisonOperation   supplies SPARQL graph-pattern fragments for GROUP BY

Future kinds (e.g. bucket/percentile/rate) plug into ``OperationRegistry`` the
same way — register, expose ``describe_for_prompt``, participate in the
prompt<->registry round-trip test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from src.conversational.models import CompetencyQuestion, PatientFilter


# ---------------------------------------------------------------------------
# Shared fragment / context types
# ---------------------------------------------------------------------------


@dataclass
class FilterCompileContext:
    """Everything a filter's ``compile`` needs from the extractor.

    ``backend`` is duck-typed — any object exposing ``table(name)``,
    ``ilike(col)``, and ``readmission_labels_expr()`` will do. That keeps the
    registry oblivious to DuckDB vs BigQuery vs whatever future backend we add.
    """

    backend: Any
    admission_alias: str = "a"
    patient_alias: str = "p"


@dataclass
class FilterFragment:
    """A chunk of SQL contributed by a single filter.

    The extractor concatenates fragments from every filter in the CQ: joins
    are emitted in order, where-predicates are AND-combined, params keep their
    positional correspondence with the ``?`` placeholders inside ``where``
    and the joins.

    ``needs_patients`` is a flag because multiple filters (age, gender) share
    the same patients JOIN — the extractor deduplicates by emitting a single
    ``JOIN patients p`` at the front if any fragment sets this.
    """

    joins: list[str] = field(default_factory=list)
    where: list[str] = field(default_factory=list)
    params: list[Any] = field(default_factory=list)
    needs_patients: bool = False


@dataclass
class AggregateFragment:
    """What an aggregate contributes to the reasoner's template selection.

    The reasoner today picks SPARQL templates by aggregation keyword; median
    also triggers a Python post-processing step on the returned rows.
    """

    template: str
    post_processor: Callable[[list[dict]], dict] | None = None


@dataclass
class ComparisonFragment:
    """SPARQL graph-pattern fragments for a comparison axis.

    Mirrors the existing ``_COMPARISON_FIELD_MAP`` in ``reasoner.py``: each
    axis contributes one clause attached to the Patient node and one to the
    HospitalAdmission node. Most axes contribute to only one of the two.
    """

    patient_clause: str = ""
    admission_clause: str = ""


# ---------------------------------------------------------------------------
# Operation base + per-kind classes
# ---------------------------------------------------------------------------


class Operation(Protocol):
    """Structural contract shared by every operation kind.

    ``describe_for_prompt`` is consumed by the prompt builder so the LLM sees
    exactly what the registry knows how to compile — no drift possible.
    """

    kind: str
    name: str
    description: str

    def describe_for_prompt(self) -> str: ...


# Concrete operators allowed on comparison-style filters (age, readmitted_*).
# These are SQL-safe only because they are held in a fixed set and interpolated
# directly into the predicate string rather than taken from user input.
_COMPARISON_OPERATORS: frozenset[str] = frozenset({">", "<", "=", ">=", "<="})


@dataclass
class FilterOperation:
    """Filter = contributes SQL fragments to the cohort-selection query.

    Most new filter fields can be expressed by constructing one of these at
    registration time and dropping it into ``operations_filters.py`` — the
    heavy lifting is in the compile closure.

    Attributes
    ----------
    name:
        The ``PatientFilter.field`` value this operation handles (e.g. ``"age"``).
    operators:
        The subset of operators the operation accepts. ``validate`` rejects
        filters with any other operator so the LLM gets a clean retry message.
    value_type:
        ``"scalar"`` (single str/int), ``"list"`` (list of strings — for ``in``),
        or ``"none"`` (no value, e.g. ``exists``).
    description:
        Human-readable one-line description shown in the prompt.
    compile_fn:
        Callable that returns a ``FilterFragment`` for a given filter + context.
        Kept as a plain callable so filter registration can be one function call
        per field in ``operations_filters.py`` — no subclassing needed.
    """

    name: str
    operators: frozenset[str]
    value_type: str  # "scalar" | "list" | "none"
    description: str
    compile_fn: Callable[[PatientFilter, FilterCompileContext], FilterFragment]
    kind: str = field(default="filter", init=False)

    def describe_for_prompt(self) -> str:
        ops = "|".join(sorted(self.operators)) if self.operators else "-"
        return f"  {self.name:<22} ({ops:<16}) {self.value_type:<8}  {self.description}"

    def validate(self, f: PatientFilter) -> list[str]:
        if f.field != self.name:
            return [f"filter field mismatch: expected {self.name!r}, got {f.field!r}"]
        if self.operators and f.operator not in self.operators:
            return [
                f"operator {f.operator!r} not supported for filter field "
                f"{self.name!r} (allowed: {sorted(self.operators)})"
            ]
        return []

    def compile(
        self, f: PatientFilter, ctx: FilterCompileContext
    ) -> FilterFragment:
        return self.compile_fn(f, ctx)


@dataclass
class AggregateOperation:
    """Aggregate = picks a SPARQL template (+ optional post-processing).

    Today aggregates have no operands beyond the concept they aggregate over,
    so the compile input is just the CQ. If we later add aggregates with
    parameters (e.g. ``percentile(0.95)``), this signature is the natural
    place to extend.

    Phase 7a: ``sql_fn`` lets the aggregate participate in the SQL fast-path.
    When set (e.g. ``"AVG"``, ``"MAX"``, ``"MIN"``, ``"COUNT"``) the planner
    routes single-concept non-temporal CQs using this aggregate around the
    graph entirely — one direct SQL aggregate call instead of extract +
    build_query_graph + SPARQL. Set to ``None`` for aggregates that require
    Python post-processing (median) or BigQuery-specific primitives that
    aren't portable (percentile).
    """

    name: str
    description: str
    template: str
    post_processor: Callable[[list[dict]], dict] | None = None
    sql_fn: str | None = None
    """SQL aggregate function name used by the fast-path. ``None`` means
    this aggregate must go through the graph/SPARQL path."""
    kind: str = field(default="aggregate", init=False)

    def describe_for_prompt(self) -> str:
        return f"  {self.name:<10}  {self.description}"

    def validate(self, cq: CompetencyQuestion) -> list[str]:
        if cq.aggregation != self.name:
            return [f"aggregate mismatch: expected {self.name!r}, got {cq.aggregation!r}"]
        return []

    def compile(self, cq: CompetencyQuestion) -> AggregateFragment:
        return AggregateFragment(
            template=self.template,
            post_processor=self.post_processor,
        )


@dataclass
class ComparisonOperation:
    """Comparison axis = supplies SPARQL graph-pattern fragments for GROUP BY.

    Preserves the shape of the existing ``_COMPARISON_FIELD_MAP`` 2-tuple
    (patient clause, admission clause) — most axes use only one of the two.

    Phase 7a: ``sql_group_by`` lets the axis participate in the SQL fast-path.
    It names the fully-qualified column (e.g. ``"p.gender"``,
    ``"a.admission_type"``) that the fast-path compiler emits in the
    ``GROUP BY`` clause. ``None`` means this axis is SPARQL-only today;
    a comparison CQ using such an axis is routed through the graph path.
    """

    name: str
    description: str
    patient_clause: str = ""
    admission_clause: str = ""
    sql_group_by: str | None = None
    """Fully-qualified SQL column for the SQL fast-path's ``GROUP BY``. ``None``
    means this axis isn't SQL-compilable yet."""
    kind: str = field(default="comparison_axis", init=False)

    def describe_for_prompt(self) -> str:
        return f"  {self.name:<22}  {self.description}"

    def validate(self, cq: CompetencyQuestion) -> list[str]:
        if cq.comparison_field != self.name:
            return [
                f"comparison_field mismatch: expected {self.name!r}, "
                f"got {cq.comparison_field!r}"
            ]
        return []

    def compile(self) -> ComparisonFragment:
        return ComparisonFragment(
            patient_clause=self.patient_clause,
            admission_clause=self.admission_clause,
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class OperationRegistry:
    """Central lookup for operations of any kind.

    Operations are keyed by ``(kind, name)`` so ``"gender"`` can exist as both
    a filter and a comparison axis without collision. ``register`` refuses
    silent overrides — a duplicate ``(kind, name)`` raises ``ValueError`` so
    that "shadow" registrations from a broken import order surface immediately.
    """

    def __init__(self) -> None:
        self._ops: dict[tuple[str, str], Any] = {}

    # -- mutation ----------------------------------------------------------

    def register(self, op: Any) -> None:
        """Register an operation. Refuses duplicates."""
        key = (op.kind, op.name)
        if key in self._ops:
            raise ValueError(
                f"Operation already registered: kind={op.kind!r} name={op.name!r}"
            )
        self._ops[key] = op

    # -- lookup ------------------------------------------------------------

    def get(self, kind: str, name: str) -> Any | None:
        return self._ops.get((kind, name))

    def require(self, kind: str, name: str) -> Any:
        """Like ``get`` but raises ``KeyError`` for unknown ops.

        Used by call sites that have already checked ``supported_names`` and
        want a loud failure if they get the contract wrong.
        """
        op = self.get(kind, name)
        if op is None:
            raise KeyError(f"No operation registered for kind={kind!r} name={name!r}")
        return op

    def supported_names(self, kind: str) -> frozenset[str]:
        """Return the set of registered operation names for ``kind``."""
        return frozenset(name for (k, name) in self._ops.keys() if k == kind)

    def iter_kind(self, kind: str) -> list[Any]:
        """All operations of one kind, sorted by name (stable ordering for prompts)."""
        return [
            self._ops[key]
            for key in sorted(self._ops.keys())
            if key[0] == kind
        ]

    # -- prompt integration ------------------------------------------------

    def describe_for_prompt(self, kind: str) -> str:
        """Render all operations of one kind as a prompt section body.

        The prompt builder calls this once per kind and drops the output into
        a labelled section. Empty kinds render as a single placeholder line so
        the section is never silently missing.
        """
        ops = self.iter_kind(kind)
        if not ops:
            return "  (none registered)"
        return "\n".join(op.describe_for_prompt() for op in ops)

    # -- compile-all -------------------------------------------------------

    def compile_filters(
        self,
        filters: list[PatientFilter],
        ctx: FilterCompileContext,
    ) -> FilterFragment:
        """Compile every filter in a CQ into a single combined fragment.

        Unknown filter fields are skipped silently — matching the current
        extractor behaviour (it logs a warning). Callers that want strict
        validation should use ``registry.validate_cq`` first (future).
        """
        joins: list[str] = []
        where: list[str] = []
        params: list[Any] = []
        needs_patients = False

        for f in filters:
            op = self.get("filter", f.field)
            if op is None:
                continue
            frag = op.compile(f, ctx)
            joins.extend(frag.joins)
            where.extend(frag.where)
            params.extend(frag.params)
            needs_patients = needs_patients or frag.needs_patients

        return FilterFragment(
            joins=joins,
            where=where,
            params=params,
            needs_patients=needs_patients,
        )


# ---------------------------------------------------------------------------
# Shared constant
# ---------------------------------------------------------------------------


COMPARISON_OPERATORS: frozenset[str] = _COMPARISON_OPERATORS
"""Re-exported so filter-seed modules don't have to reach into the private one."""


# ---------------------------------------------------------------------------
# Default registry factory
# ---------------------------------------------------------------------------


_default_registry: OperationRegistry | None = None


def get_default_registry() -> OperationRegistry:
    """Return the process-wide default registry, populated with all known ops.

    Lazy so registration side effects don't happen at import time (important
    because ``operations_filters`` imports the filter implementations which
    may import backend shims). Idempotent — subsequent calls return the same
    instance, so pipeline code can safely call this per request.

    Tests that want isolation should build their own ``OperationRegistry``
    and call the ``register_default_*`` functions directly.
    """
    global _default_registry
    if _default_registry is None:
        # Imported here to avoid a top-level circular import: operations_*
        # modules import from operations (for base classes); operations
        # transitively imports them here only when the factory is called.
        from src.conversational.operations_aggregates import (
            register_default_aggregates,
        )
        from src.conversational.operations_comparison import (
            register_default_comparisons,
        )
        from src.conversational.operations_filters import register_default_filters

        registry = OperationRegistry()
        register_default_filters(registry)
        register_default_aggregates(registry)
        register_default_comparisons(registry)
        _default_registry = registry
    return _default_registry
