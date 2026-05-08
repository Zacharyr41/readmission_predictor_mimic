"""Pytest fixtures for the dashboard test suite.

Supplies:
- ``_clear_mcp_cache`` (autouse): clears the resolver's ``_cached_*``
  lru_caches between tests so cross-test pollution can't shadow a real
  failure (per the Streamlit testing guide §13).
- ``reporter``: per-test markdown report writer that records the
  qualitative-assessment artifact (Q → SQL → answer → verdict →
  assertions) and writes ``tests/dashboard/reports/<name>.md`` at
  teardown.
- ``at_dashboard``: ``AppTest.from_file("src/conversational/app.py")``
  with a generous timeout — Tier 1 only.
- ``real_pipeline``: live ``ConversationalPipeline`` against local
  DuckDB. Tier 1 only; skips with a clear message if the DuckDB file
  isn't present.

Reports are gitignored (see ``.gitignore``) — they're per-run artifacts,
not source.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from tests.dashboard.lib.reporter import Reporter


_REPORTS_DIR = Path(__file__).parent / "reports"


@pytest.fixture(autouse=True)
def _clear_mcp_cache():
    """Clear lru_cache on the resolver's cached helpers between tests.

    Streamlit's caching docs (and the testing guide §13 / issue #9139)
    flag that ``@st.cache_data`` / ``@st.cache_resource`` persist across
    AppTest instances within the same Python process. We have an
    analogous problem with ``functools.lru_cache`` on
    ``_cached_icd_autocode`` and ``_cached_mimic_itemid_search``: a real
    OMOPHub response cached by an earlier test could mask an
    ``unavailable`` test downstream. Clear before AND after to make
    each test fully hermetic.
    """
    from src.conversational import concept_resolver as cr

    cr._cached_icd_autocode.cache_clear()
    cr._cached_mimic_itemid_search.cache_clear()
    yield
    cr._cached_icd_autocode.cache_clear()
    cr._cached_mimic_itemid_search.cache_clear()


@pytest.fixture
def reporter(request):
    """Per-test ``Reporter`` that writes a markdown report on teardown.

    The fixture inspects the test's pass/fail/skip status via pytest's
    request.node.rep_call attribute (set by the pytest_runtest_makereport
    hook below). The report is named after the test function and lands
    under ``tests/dashboard/reports/``.
    """
    test_name = request.node.name
    # Tier inferred from the test file name (test_tier1_..., tier2_..., tier3_...).
    file_name = Path(request.fspath).stem
    if "tier1" in file_name:
        tier = 1
    elif "tier2" in file_name:
        tier = 2
    elif "tier3" in file_name:
        tier = 3
    else:
        tier = 0

    rep = Reporter(name=test_name, tier=tier, output_dir=_REPORTS_DIR)
    start = time.monotonic()
    try:
        yield rep
    finally:
        duration = time.monotonic() - start
        # Resolve test outcome via the makereport hook (below).
        result = getattr(request.node, "rep_call", None)
        if result is None:
            status = "ERROR"
        elif result.skipped:
            status = "SKIP"
        elif result.passed:
            status = "PASS"
        else:
            status = "FAIL"
        try:
            rep.write(status=status, duration_s=duration)
        except Exception as exc:  # noqa: BLE001
            # Reporter must not break the test itself; surface the failure.
            print(f"[reporter] write failed for {test_name}: {exc}")


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_makereport(item, call):
    """Stash the test's call-phase report on the item so the ``reporter``
    fixture can read pass/fail/skip on teardown."""
    outcome = yield
    rep = outcome.get_result()
    setattr(item, f"rep_{rep.when}", rep)


# ---------------------------------------------------------------------------
# Tier 1 fixtures — gated by RUN_LIVE_DASHBOARD=1 + real DuckDB
# ---------------------------------------------------------------------------


@pytest.fixture
def at_dashboard():
    """``AppTest`` instance pointing at the live dashboard script.

    Default timeout 120s — long enough for a real ``pipeline.ask`` call
    that does decompose → resolve (with OMOPHub) → compile → execute →
    answer → critique. Caller is responsible for ``.run()``.

    Tier 1 only: tests using this fixture must also gate on
    ``RUN_LIVE_DASHBOARD`` via ``pytest.mark.skipif``.
    """
    from streamlit.testing.v1 import AppTest

    return AppTest.from_file(
        "src/conversational/app.py", default_timeout=120,
    )


@pytest.fixture
def real_pipeline():
    """Construct a real ConversationalPipeline against local DuckDB.

    Skips with a clear message when the DuckDB file or required env
    vars aren't present, so a teammate running the suite without the
    full dataset gets a useful diagnostic rather than an opaque crash.
    """
    db_path = Path("data/processed/mimiciv.duckdb")
    if not db_path.exists():
        pytest.skip(f"DuckDB file not present at {db_path}")
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not set")

    from src.conversational.orchestrator import ConversationalPipeline

    return ConversationalPipeline(
        db_path=db_path,
        ontology_dir=Path("ontology/definition"),
        api_key=api_key,
        data_source="local",
        enable_critic=True,
    )
