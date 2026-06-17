"""Unit tests for SQL parameter-value inlining (``render_sql_with_params``).

Pure-function tests: no database, no network, no LLM. They characterize how a
parameterized fast-path template (``?`` placeholders) plus its positional
``params`` list renders into the values-filled SQL the "Query Details" expander
shows the clinician. The renderer mirrors the executor's ``sql.split("?")``
positional logic (``extractor.py``) so the inlined positions match what was
actually bound.

The rendered SQL is display-only (read, never re-executed), so the priority is
faithful, copy-pasteable output — not injection-safety — but string values are
still single-quote-escaped so the result stays valid SQL.
"""

from __future__ import annotations

import datetime

import pytest

from src.conversational.sql_render import render_sql_with_params


def test_passthrough_when_no_placeholders():
    assert render_sql_with_params("SELECT 1", []) == "SELECT 1"


def test_int_param_inlined_as_bare_literal():
    out = render_sql_with_params(
        "SELECT * FROM labevents WHERE itemid = ?", [50912]
    )
    assert out == "SELECT * FROM labevents WHERE itemid = 50912"
    assert "?" not in out


def test_float_param_inlined_as_bare_literal():
    out = render_sql_with_params("SELECT * FROM t WHERE valuenum > ?", [5.0])
    assert out == "SELECT * FROM t WHERE valuenum > 5.0"


def test_string_param_is_single_quoted():
    out = render_sql_with_params(
        "SELECT * FROM diagnoses_icd WHERE icd_code LIKE ?", ["A41%"]
    )
    assert out == "SELECT * FROM diagnoses_icd WHERE icd_code LIKE 'A41%'"


def test_string_param_with_apostrophe_is_escaped():
    # SQL escapes a literal single quote by doubling it.
    out = render_sql_with_params("WHERE name LIKE ?", ["%O'Brien%"])
    assert out == "WHERE name LIKE '%O''Brien%'"


def test_none_param_renders_as_sql_null():
    out = render_sql_with_params("WHERE dod = ?", [None])
    assert out == "WHERE dod = NULL"
    assert "None" not in out


def test_true_renders_as_sql_true_not_one():
    # bool is an int subclass; render TRUE/FALSE (not 1/0) for readability.
    out = render_sql_with_params("SELECT ? AS flag", [True])
    assert out == "SELECT TRUE AS flag"
    assert "1" not in out


def test_false_renders_as_sql_false_not_zero():
    out = render_sql_with_params("SELECT ? AS flag", [False])
    assert out == "SELECT FALSE AS flag"
    assert "0" not in out


def test_list_param_renders_as_parenthesized_tuple():
    out = render_sql_with_params("WHERE itemid IN ?", [[50912, 50983, 51006]])
    assert out == "WHERE itemid IN (50912, 50983, 51006)"


def test_tuple_param_of_strings_each_quoted():
    out = render_sql_with_params("WHERE icd_code IN ?", [("I60", "I61")])
    assert out == "WHERE icd_code IN ('I60', 'I61')"


def test_multiple_mixed_params_inlined_in_order():
    out = render_sql_with_params(
        "SELECT * FROM labevents "
        "WHERE itemid = ? AND label = ? AND valuenum > ?",
        [50912, "creatinine", 1.5],
    )
    assert out == (
        "SELECT * FROM labevents "
        "WHERE itemid = 50912 AND label = 'creatinine' AND valuenum > 1.5"
    )


def test_question_mark_inside_a_string_value_is_preserved():
    # A literal "?" in a *value* must survive — it is inserted after the
    # template split, never re-processed as a placeholder.
    out = render_sql_with_params("WHERE note LIKE ?", ["a?b"])
    assert out == "WHERE note LIKE 'a?b'"


def test_datetime_param_falls_back_to_quoted_string():
    out = render_sql_with_params(
        "WHERE charttime < ?", [datetime.datetime(2150, 1, 1, 12, 0)]
    )
    assert out == "WHERE charttime < '2150-01-01 12:00:00'"


def test_too_few_params_raises_value_error():
    with pytest.raises(ValueError):
        render_sql_with_params("WHERE a = ? AND b = ?", [1])


def test_too_many_params_raises_value_error():
    with pytest.raises(ValueError):
        render_sql_with_params("WHERE a = ?", [1, 2])


def test_rendered_sql_property_degrades_to_template_on_mismatch():
    """The pure renderer raises on a placeholder/param mismatch, but the
    ``SqlFastpathQuery.rendered_sql`` property is display-only and must NEVER
    crash the live pipeline. On an inconsistent (sql, params) pair it falls
    back to the parameterized template instead of propagating ``ValueError``.
    (Real ``compile_sql`` output always matches; this guards the boundary.)"""
    from src.conversational.sql_fastpath import SqlFastpathQuery

    q = SqlFastpathQuery(
        sql="SELECT AVG(x) AS mean_value FROM t",  # 0 placeholders
        params=[1, 2, 3],                          # 3 params → mismatch
        columns=["mean_value"],
    )
    assert q.rendered_sql == q.sql  # template returned, no exception raised
