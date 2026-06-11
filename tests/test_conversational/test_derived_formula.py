"""Unit tests for the derived-formula AST→SQL emitter + validation.

The emitter is the deterministic core of the ``derived_value`` cohort filter:
it turns an N-operand arithmetic AST (extracted from PubMed) into safe,
parameterized SQL. These tests pin generality across operand counts and the
safety guarantees (whitelist, parameterized constants, NULLIF division,
ref validation) with no DB and no LLM.
"""

from __future__ import annotations

import pytest

from src.conversational.derived_formula import (
    DerivedFormula,
    FormulaError,
    Operand,
    emit_expression,
    validate_formula,
)


def _cols(mapping):
    """col_for_ref that maps a ref → (column_sql, no params)."""
    def col(ref):
        if ref not in mapping:
            raise FormulaError(f"unknown ref {ref!r}")
        return mapping[ref], []
    return col


class TestExpressionEmitter:
    def test_two_operand_ratio_is_nullif_guarded(self):
        sql, params = emit_expression(
            {"op": "/", "args": ["hr", "sbp"]},
            _cols({"hr": "op0.valuenum", "sbp": "op1.valuenum"}),
        )
        assert sql == "(op0.valuenum / NULLIF(op1.valuenum, 0))"
        assert params == []

    def test_three_operand_anion_gap_parenthesizes(self):
        # anion gap = Na - (Cl + HCO3)
        sql, _ = emit_expression(
            {"op": "-", "args": ["na", {"op": "+", "args": ["cl", "hco3"]}]},
            _cols({"na": "na.valuenum", "cl": "cl.valuenum", "hco3": "hco3.valuenum"}),
        )
        assert sql == "(na.valuenum - (cl.valuenum + hco3.valuenum))"

    def test_constant_is_parameterized(self):
        sql, params = emit_expression(
            {"op": "*", "args": [3.78, "bili"]}, _cols({"bili": "b.valuenum"}),
        )
        assert sql == "(? * b.valuenum)"
        assert params == [3.78]

    def test_ln_unary(self):
        sql, _ = emit_expression(
            {"op": "ln", "args": ["bili"]}, _cols({"bili": "b.valuenum"}),
        )
        assert sql == "LN(b.valuenum)"

    def test_min_max_map_to_least_greatest(self):
        sql, _ = emit_expression(
            {"op": "max", "args": ["a", 1]}, _cols({"a": "a.valuenum"}),
        )
        assert sql == "GREATEST(a.valuenum, ?)"

    def test_params_thread_in_sql_order_for_subquery_refs(self):
        """per_stay-style: each ref is a scalar subquery with its own params;
        params must appear in left-to-right SQL order."""
        def col(ref):
            return (f"(SELECT MAX(v) FROM t WHERE id IN (?) /*{ref}*/)", [ref])
        sql, params = emit_expression(
            {"op": "-", "args": ["na", {"op": "+", "args": ["cl", "hco3"]}]}, col,
        )
        # na first, then cl, then hco3 — matches the `?` order in the SQL.
        assert params == ["na", "cl", "hco3"]
        assert sql.index("na") < sql.index("cl") < sql.index("hco3")

    def test_rejects_unknown_op(self):
        with pytest.raises(FormulaError):
            emit_expression({"op": "pow", "args": ["a", 2]}, _cols({"a": "a"}))

    def test_rejects_bad_arity_for_minus(self):
        with pytest.raises(FormulaError):
            emit_expression(
                {"op": "-", "args": ["a", "b", "c"]},
                _cols({"a": "a", "b": "b", "c": "c"}),
            )

    def test_rejects_division_with_one_arg(self):
        with pytest.raises(FormulaError):
            emit_expression({"op": "/", "args": ["a"]}, _cols({"a": "a"}))

    def test_rejects_malformed_node(self):
        with pytest.raises(FormulaError):
            emit_expression({"not_an_op": 1}, _cols({}))

    def test_rejects_boolean_constant(self):
        with pytest.raises(FormulaError):
            emit_expression({"op": "*", "args": [True, "a"]}, _cols({"a": "a"}))


def _shock_index_formula(**overrides):
    base = dict(
        operands=(
            Operand(ref="hr", itemids=(220045,), table="chartevents",
                    guard_low=10, guard_high=300),
            Operand(ref="sbp", itemids=(220050, 220179), table="chartevents",
                    guard_low=30, guard_high=300),
        ),
        expression={"op": "/", "args": ["hr", "sbp"]},
        operator=">=", threshold=0.9,
        time_semantics="per_instant", stay_aggregate="max",
    )
    base.update(overrides)
    return DerivedFormula(**base)


class TestValidateFormula:
    def test_valid_shock_index_passes(self):
        validate_formula(_shock_index_formula())  # no raise

    def test_valid_three_operand_per_stay_passes(self):
        f = DerivedFormula(
            operands=(
                Operand(ref="na", itemids=(50983,), table="labevents"),
                Operand(ref="cl", itemids=(50902,), table="labevents"),
                Operand(ref="hco3", itemids=(50882,), table="labevents"),
            ),
            expression={"op": "-", "args": ["na", {"op": "+", "args": ["cl", "hco3"]}]},
            operator=">", threshold=12, time_semantics="per_stay",
        )
        validate_formula(f)

    def test_unknown_ref_rejected(self):
        with pytest.raises(FormulaError):
            validate_formula(_shock_index_formula(
                expression={"op": "/", "args": ["hr", "dbp"]}))

    def test_cross_table_per_instant_rejected(self):
        with pytest.raises(FormulaError):
            validate_formula(_shock_index_formula(operands=(
                Operand(ref="hr", itemids=(220045,), table="chartevents"),
                Operand(ref="sbp", itemids=(50820,), table="labevents"),
            )))

    def test_bad_operator_rejected(self):
        with pytest.raises(FormulaError):
            validate_formula(_shock_index_formula(operator="!="))

    def test_no_operands_rejected(self):
        with pytest.raises(FormulaError):
            validate_formula(_shock_index_formula(operands=()))

    def test_operand_without_itemids_rejected(self):
        with pytest.raises(FormulaError):
            validate_formula(_shock_index_formula(operands=(
                Operand(ref="hr", itemids=(), table="chartevents"),
                Operand(ref="sbp", itemids=(220050,), table="chartevents"),
            )))
