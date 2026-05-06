"""Tests for the cohort registry + name resolver (Phase H Tier D).

Two pieces under test:

* :func:`load_cohorts` — reads ``data/mappings/clinical_cohorts.json``,
  filters out ``_metadata``, returns a dict keyed by canonical cohort
  name.
* :func:`resolve_cohort_name` — maps a free-text medical phrase
  (e.g., "MI", "myocardial infarction", "heart attack") to the
  canonical cohort name (``"mi_acute"``). Case-insensitive,
  longest-substring-alias-wins so the registry can grow without
  ambiguity collisions.

The registry's ICD prefixes are stored in **dotted, human-readable**
form (e.g., ``"A41."``, ``"995.91"``). The runtime SQL builder
normalises by stripping dots before pattern-matching MIMIC's no-dot
storage (`tests/test_health_evidence/test_cohort_filter_sql.py`).
"""

from __future__ import annotations

import re

import pytest


def test_load_cohorts_returns_dict():
    from src.conversational.health_evidence.cohorts import load_cohorts
    cohorts = load_cohorts()
    assert isinstance(cohorts, dict)
    assert cohorts, "registry is empty"


def test_required_cohorts_present():
    from src.conversational.health_evidence.cohorts import load_cohorts
    cohorts = load_cohorts()
    for required in (
        "sepsis", "septic_shock", "aki", "mi_acute",
        "heart_failure", "hepatic_failure", "stroke_ischemic",
        "ards", "pneumonia", "covid_19",
    ):
        assert required in cohorts, f"missing cohort {required!r}"


def test_each_cohort_has_icd10_or_icd9():
    from src.conversational.health_evidence.cohorts import load_cohorts
    for name, defn in load_cohorts().items():
        if name.startswith("_"):
            continue
        assert defn.get("icd10_prefixes") or defn.get("icd9_prefixes"), (
            f"cohort {name!r} has no ICD prefixes"
        )


def test_cohort_names_snake_case():
    from src.conversational.health_evidence.cohorts import load_cohorts
    for name in load_cohorts():
        if name.startswith("_"):
            continue
        assert re.match(r"^[a-z][a-z0-9_]*$", name), (
            f"cohort name {name!r} is not snake_case"
        )


def test_metadata_filtered_from_load():
    """`_metadata` is allowed in the JSON but must NOT appear in the
    registry returned by load_cohorts() — callers shouldn't accidentally
    treat it as a cohort."""
    from src.conversational.health_evidence.cohorts import load_cohorts
    assert "_metadata" not in load_cohorts()


def test_all_is_reserved_name():
    """The cohort registry must never define 'all' — that's the
    reserved system name for the unstratified bucket."""
    from src.conversational.health_evidence.cohorts import load_cohorts
    assert "all" not in load_cohorts()


def test_each_cohort_has_aliases():
    """Aliases drive natural-phrase matching. Each cohort needs at
    least one (the prose form of the canonical name)."""
    from src.conversational.health_evidence.cohorts import load_cohorts
    for name, defn in load_cohorts().items():
        if name.startswith("_"):
            continue
        aliases = defn.get("aliases")
        assert isinstance(aliases, list) and aliases, (
            f"cohort {name!r} missing aliases"
        )
        assert all(isinstance(a, str) and a.strip() for a in aliases), (
            f"cohort {name!r} has empty/non-str aliases: {aliases!r}"
        )


def test_resolve_cohort_name_handles_canonical():
    from src.conversational.health_evidence.cohorts import resolve_cohort_name
    assert resolve_cohort_name("mi_acute") == "mi_acute"
    assert resolve_cohort_name("sepsis") == "sepsis"
    assert resolve_cohort_name("heart_failure") == "heart_failure"


def test_resolve_cohort_name_handles_aliases():
    """Critic passes 'MI' / 'myocardial infarction' / 'heart attack' —
    all three resolve to canonical 'mi_acute'."""
    from src.conversational.health_evidence.cohorts import resolve_cohort_name
    assert resolve_cohort_name("MI") == "mi_acute"
    assert resolve_cohort_name("myocardial infarction") == "mi_acute"
    assert resolve_cohort_name("heart attack") == "mi_acute"


def test_resolve_cohort_name_case_insensitive():
    from src.conversational.health_evidence.cohorts import resolve_cohort_name
    for phrase in ("CHF", "chf", "Chf", "Congestive Heart Failure"):
        assert resolve_cohort_name(phrase) == "heart_failure", (
            f"phrase {phrase!r} did not resolve"
        )


def test_resolve_cohort_name_covid_variants():
    """COVID has several common spellings — ensure the registry catches
    the lot, since users will type any of them."""
    from src.conversational.health_evidence.cohorts import resolve_cohort_name
    for phrase in ("COVID", "covid-19", "covid 19", "SARS-CoV-2", "coronavirus"):
        assert resolve_cohort_name(phrase) == "covid_19", phrase


def test_resolve_cohort_name_returns_none_for_unknown():
    from src.conversational.health_evidence.cohorts import resolve_cohort_name
    assert resolve_cohort_name("not_a_cohort") is None
    assert resolve_cohort_name("xyzzy") is None
    # Empty string is also not a match
    assert resolve_cohort_name("") is None


def test_resolve_cohort_name_longest_alias_wins():
    """When two aliases could substring-match, the longer / more
    specific alias wins. 'acute respiratory failure' must resolve to
    `respiratory_failure`, not `ards` (whose alias is 'acute
    respiratory distress syndrome' — longer, but the input doesn't
    contain it; what we're guarding here is that 'respiratory failure'
    isn't shadowed by some shorter alias of a different cohort)."""
    from src.conversational.health_evidence.cohorts import resolve_cohort_name
    assert resolve_cohort_name("acute respiratory failure") == "respiratory_failure"


def test_resolve_cohort_name_strips_punctuation():
    """Common cleanups: trailing punctuation, collapsed whitespace."""
    from src.conversational.health_evidence.cohorts import resolve_cohort_name
    assert resolve_cohort_name("MI.") == "mi_acute"
    assert resolve_cohort_name("  myocardial   infarction  ") == "mi_acute"


def test_load_cohorts_caches_or_reloads_consistently():
    """Two calls to load_cohorts() must return equivalent dicts. The
    implementation can cache or re-read; either is fine, but the
    behaviour must be deterministic."""
    from src.conversational.health_evidence.cohorts import load_cohorts
    a = load_cohorts()
    b = load_cohorts()
    assert a == b


def test_known_cohort_names_helper():
    """known_cohort_names() returns the full set of canonical names —
    used to populate the helpful-error hint in the tool when the
    caller passes an unrecognised cohort."""
    from src.conversational.health_evidence.cohorts import (
        known_cohort_names, load_cohorts,
    )
    names = known_cohort_names()
    assert isinstance(names, list)
    assert "sepsis" in names
    assert "mi_acute" in names
    # Returns canonical names only, not aliases
    assert "MI" not in names
    assert "myocardial infarction" not in names
    # Matches the registry exactly (modulo metadata)
    assert set(names) == set(load_cohorts().keys())


def test_normalize_icd_prefix_strips_dot_and_uppercases():
    """MIMIC stores ICD codes without dots and uppercase. Registry
    stores them dotted/mixed-case for human readability. The runtime
    SQL builder normalises before pattern-matching."""
    from src.conversational.health_evidence.cohorts import normalize_icd_prefix
    assert normalize_icd_prefix("A41.") == "A41"
    assert normalize_icd_prefix("R65.21") == "R6521"
    assert normalize_icd_prefix("995.91") == "99591"
    assert normalize_icd_prefix("i63.") == "I63"  # uppercases
    assert normalize_icd_prefix(" 410. ") == "410"  # strips spaces


# ---------------------------------------------------------------------------
# cohort_filter_sql — Inc 2
# ---------------------------------------------------------------------------


class TestCohortFilterSql:
    """SQL-builder for the named-cohort path. Returns a subquery that
    selects hadm_ids whose diagnoses_icd row matches any of the
    cohort's ICD prefixes (after dot-stripping). The helper is used
    by the on-the-fly compute path AND by the catalog generator."""

    def test_returns_select_subquery(self):
        from src.conversational.health_evidence.cohorts import cohort_filter_sql
        sql = cohort_filter_sql("sepsis")
        assert "SELECT" in sql.upper()
        assert "hadm_id" in sql
        assert "diagnoses_icd" in sql

    def test_uses_no_dot_form_in_like_patterns(self):
        """MIMIC stores codes without dots — the LIKE patterns must
        match that storage. Registry's 'A41.' becomes 'A41%' here."""
        from src.conversational.health_evidence.cohorts import cohort_filter_sql
        sql = cohort_filter_sql("sepsis")
        # The dotted form must not appear as a LIKE pattern.
        assert "LIKE 'A41.%'" not in sql
        assert "LIKE 'A41.'" not in sql
        # The no-dot prefix should appear.
        assert "A41%" in sql or "'A41'" in sql

    def test_handles_multiple_prefixes(self):
        from src.conversational.health_evidence.cohorts import cohort_filter_sql
        # mi_acute has icd10 prefixes I21. and I22. → I21%, I22%.
        sql = cohort_filter_sql("mi_acute")
        assert "I21%" in sql
        assert "I22%" in sql
        assert " OR " in sql

    def test_filters_on_both_icd_versions(self):
        """Cohorts with both ICD-9 and ICD-10 prefixes must filter
        each version separately, since mixing 'I21%' (ICD-10) with
        '410%' (ICD-9) on the same icd_code column would produce a
        cross-version match by accident."""
        from src.conversational.health_evidence.cohorts import cohort_filter_sql
        sql = cohort_filter_sql("mi_acute")
        assert "icd_version = 10" in sql or "icd_version=10" in sql
        assert "icd_version = 9" in sql or "icd_version=9" in sql

    def test_handles_icd10_only_cohort(self):
        """COVID has ICD-10 only (no ICD-9 prefixes). The SQL must
        still build cleanly and not crash on an empty ICD-9 list."""
        from src.conversational.health_evidence.cohorts import cohort_filter_sql
        sql = cohort_filter_sql("covid_19")
        assert "U071" in sql
        # No ICD-9 patterns since COVID didn't exist when ICD-9 was current.

    def test_unknown_cohort_raises_keyerror(self):
        from src.conversational.health_evidence.cohorts import cohort_filter_sql
        with pytest.raises(KeyError, match="not_a_cohort|unknown"):
            cohort_filter_sql("not_a_cohort")

    def test_executes_against_real_mimic(self):
        """End-to-end: the generated SQL must execute against the live
        MIMIC duckdb and return a non-trivial number of sepsis
        hadm_ids. Skipped when the duckdb file isn't present locally."""
        import duckdb
        from src.conversational.health_evidence.cohorts import cohort_filter_sql
        from pathlib import Path
        db_path = Path("data/processed/mimiciv.duckdb")
        if not db_path.exists():
            pytest.skip("MIMIC duckdb not present locally")
        con = duckdb.connect(str(db_path), read_only=True)
        sql = cohort_filter_sql("sepsis")
        n = con.execute(
            f"SELECT COUNT(DISTINCT hadm_id) FROM ({sql})"
        ).fetchone()[0]
        # MIMIC-IV has thousands of sepsis admissions.
        assert n > 1000, f"sepsis cohort returned only {n} hadm_ids"


# ---------------------------------------------------------------------------
# build_cohort_subquery_sql — the lower-level builder reused by Inc 4
# ---------------------------------------------------------------------------


class TestBuildCohortSubquerySql:
    """Lower-level SQL builder that takes raw prefix lists. Used by
    cohort_filter_sql AND by the tool's raw-prefix path (Inc 4). Has
    its own validation gate so callers passing untrusted input don't
    bypass security."""

    def test_builds_subquery_from_prefixes(self):
        from src.conversational.health_evidence.cohorts import (
            build_cohort_subquery_sql,
        )
        sql = build_cohort_subquery_sql(
            icd10_prefixes=["A41."], icd9_prefixes=["995.91"],
        )
        assert "A41%" in sql
        assert "99591%" in sql
        assert "icd_version = 10" in sql or "icd_version=10" in sql
        assert "icd_version = 9" in sql or "icd_version=9" in sql

    def test_rejects_invalid_icd_prefix(self):
        """SQL injection / charset check. The validated charset is
        ``[A-Za-z0-9.]`` — anything else means the caller is feeding
        untrusted data and we refuse before composing SQL."""
        from src.conversational.health_evidence.cohorts import (
            build_cohort_subquery_sql,
        )
        for bad in (
            "'; DROP TABLE labevents; --",
            "A41 OR 1=1",
            "A41' OR 'x'='x",
            "A41/*comment*/",
            "%41",   # leading wildcard rejected
            "_41",   # leading wildcard rejected
        ):
            with pytest.raises(ValueError, match="invalid"):
                build_cohort_subquery_sql(icd10_prefixes=[bad])

    def test_empty_prefix_lists_raise(self):
        """At least one ICD prefix must be supplied — empty filter
        would match every hadm_id."""
        from src.conversational.health_evidence.cohorts import (
            build_cohort_subquery_sql,
        )
        with pytest.raises(ValueError, match="prefix"):
            build_cohort_subquery_sql(icd10_prefixes=[], icd9_prefixes=[])
        with pytest.raises(ValueError, match="prefix"):
            build_cohort_subquery_sql(icd10_prefixes=None, icd9_prefixes=None)
