"""Derived clinical-quantity formulas (shock index, anion gap, MELD, …).

A :class:`DerivedFormula` is an **N-operand** arithmetic expression over MIMIC
measurements plus a cohort threshold, used by the ``derived_value`` cohort
filter. The formula is sourced from PubMed (see
``concept_resolver._cached_clinical_formula_evidence``) and extracted by the
orchestrator; this module is the deterministic *representation* + the safe
**AST→SQL emitter** (there is no LLM here).

Two time-semantics, chosen by the formula:

* ``per_instant`` — operands co-measured at the same ``charttime`` (e.g. shock
  index = HR / SBP). Compiled as an N-alias same-charttime self-join, the
  expression evaluated per instant and reduced over the stay by
  ``stay_aggregate``. Requires all operands in one event table.
* ``per_stay`` — operands aggregated independently over the stay (e.g. anion
  gap = Na − (Cl + HCO₃); MELD's log-sum of components). Each operand is a
  scalar subquery and the expression combines the scalars.

The emitter whitelists arithmetic ops, **parameterizes every constant**, guards
division by zero, and resolves operand references only through the supplied
``col_for_ref`` callback — so the (LLM-extracted) AST can never inject SQL.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

_COMPARISON_OPERATORS = frozenset({">", "<", ">=", "<=", "="})
_STAY_AGGREGATES = frozenset({"max", "min", "avg"})
_TIME_SEMANTICS = frozenset({"per_instant", "per_stay"})
_TABLES = frozenset({"labevents", "chartevents"})
_BINARY_SYMBOL = {"+": "+", "-": "-", "*": "*", "/": "/"}
_SQL_FUNC = {"min": "LEAST", "max": "GREATEST"}


class FormulaError(ValueError):
    """Raised when a DerivedFormula or its AST is malformed or unsafe."""


@dataclass(frozen=True)
class Operand:
    """One measurement in a derived formula, grounded to MIMIC itemids.

    ``aggregate`` (per-stay reduction) is ignored under ``per_instant`` (the raw
    per-instant ``valuenum`` is used). ``guard_low``/``guard_high`` drop
    physiologically impossible readings (and, for a denominator, near-zero
    values that would explode a ratio)."""

    ref: str
    itemids: tuple[int, ...]
    table: str  # "labevents" | "chartevents"
    aggregate: str = "max"
    guard_low: float = 0.0
    guard_high: float = 1e9


@dataclass(frozen=True)
class DerivedFormula:
    operands: tuple[Operand, ...]
    expression: Any  # AST: ref(str) | number | {"op": str, "args": [node, ...]}
    operator: str
    threshold: float
    time_semantics: str = "per_instant"
    stay_aggregate: str = "max"
    # Provenance (PubMed PMIDs the definition was extracted from); informational.
    sources: tuple[str, ...] = field(default_factory=tuple)


def emit_expression(
    node: Any, col_for_ref: Callable[[str], tuple[str, list]],
) -> tuple[str, list]:
    """Emit ``(sql, params)`` for an arithmetic AST.

    ``col_for_ref(ref)`` returns the ``(sql, params)`` for an operand reference
    (a column for per_instant, a scalar subquery for per_stay). Params are
    appended in left-to-right SQL order so they line up with ``?`` placeholders.
    Constants are parameterized; ``/`` is NULLIF-guarded; only whitelisted ops
    are allowed (``+ - * / ln min max``).
    """
    if isinstance(node, bool):
        raise FormulaError("boolean is not a valid operand or constant")
    if isinstance(node, (int, float)):
        return "?", [float(node)]
    if isinstance(node, str):
        sql, params = col_for_ref(node)
        return sql, list(params)
    if not isinstance(node, dict) or "op" not in node or "args" not in node:
        raise FormulaError(f"invalid expression node: {node!r}")
    op = node["op"]
    args = node["args"]
    if not isinstance(args, list) or not args:
        raise FormulaError(f"op {op!r} requires a non-empty args list")

    parts: list[str] = []
    params: list = []
    for a in args:
        s, p = emit_expression(a, col_for_ref)
        parts.append(s)
        params.extend(p)

    if op in _BINARY_SYMBOL:
        if op in ("-", "/") and len(parts) != 2:
            raise FormulaError(f"op {op!r} requires exactly 2 args")
        if len(parts) < 2:
            raise FormulaError(f"op {op!r} requires at least 2 args")
        if op == "/":  # division-by-zero safe (NULL → excluded by threshold)
            return f"({parts[0]} / NULLIF({parts[1]}, 0))", params
        return "(" + f" {_BINARY_SYMBOL[op]} ".join(parts) + ")", params
    if op in _SQL_FUNC:  # min → LEAST, max → GREATEST
        if len(parts) < 2:
            raise FormulaError(f"op {op!r} requires at least 2 args")
        return f"{_SQL_FUNC[op]}({', '.join(parts)})", params
    if op == "ln":
        if len(parts) != 1:
            raise FormulaError("op 'ln' requires exactly 1 arg")
        return f"LN({parts[0]})", params
    raise FormulaError(f"unsupported op: {op!r}")


def _refs_in(node: Any) -> set[str]:
    if isinstance(node, str):
        return {node}
    if isinstance(node, dict):
        out: set[str] = set()
        for a in node.get("args", []):
            out |= _refs_in(a)
        return out
    return set()


def validate_formula(formula: DerivedFormula) -> None:
    """Raise :class:`FormulaError` if the formula is malformed or unsafe.

    Called before SQL emission so a bad PubMed extraction can never reach the
    database. Checks operand integrity, operator/aggregate/time-semantics
    vocabularies, that every expression ref is a declared operand, that
    ``per_instant`` uses a single event table, and (via a dry emit) that the AST
    contains only whitelisted ops at valid arities.
    """
    if not formula.operands:
        raise FormulaError("formula has no operands")
    if formula.operator not in _COMPARISON_OPERATORS:
        raise FormulaError(f"bad operator: {formula.operator!r}")
    if formula.time_semantics not in _TIME_SEMANTICS:
        raise FormulaError(f"bad time_semantics: {formula.time_semantics!r}")
    if formula.stay_aggregate not in _STAY_AGGREGATES:
        raise FormulaError(f"bad stay_aggregate: {formula.stay_aggregate!r}")
    refs = {o.ref for o in formula.operands}
    if len(refs) != len(formula.operands):
        raise FormulaError("duplicate operand refs")
    for o in formula.operands:
        if o.table not in _TABLES:
            raise FormulaError(f"bad operand table: {o.table!r}")
        if o.aggregate not in _STAY_AGGREGATES:
            raise FormulaError(f"bad operand aggregate: {o.aggregate!r}")
        if not o.itemids:
            raise FormulaError(f"operand {o.ref!r} has no itemids")
    missing = _refs_in(formula.expression) - refs
    if missing:
        raise FormulaError(f"expression references unknown operands: {sorted(missing)}")
    if formula.time_semantics == "per_instant":
        if len({o.table for o in formula.operands}) > 1:
            raise FormulaError(
                "per_instant requires all operands in one event table"
            )
    # Dry emit validates op whitelist + arities without touching a DB.
    emit_expression(formula.expression, lambda r: ("x", []))
