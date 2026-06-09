"""Backend robustness suite: unseen-but-valid questions against **BigQuery**.

These tests drive the *real* ``ConversationalPipeline`` end-to-end against the
live ``physionet-data`` MIMIC-IV v3.1 datasets (billing project
``mimic-485500``) in **full production config** (critic + pre-validator +
disambiguation + outlier-screening ON), exactly as the live demo runs. They
exist to expose cases the curated test-set never covered: a *valid question*
(data is in MIMIC, the requested quantity is computable, the format is
supported) must yield a *valid answer* (no error, real rows, a finite numeric
result of plausible magnitude) — not a spurious "no data found".

Why BigQuery and not DuckDB: the demo runs on BigQuery, and the local DuckDB
load is partial (e.g. only ~9.4k of 546k admissions carry labs), so a DuckDB
run would mask real coverage bugs. These tests cost money (BigQuery scan +
Anthropic API) and are therefore **skipped by default** — they carry both the
``bigquery`` and ``live_llm`` markers, and the default ``addopts``
(``-m 'not bigquery_e2e and not live_llm'``) deselects them. Run explicitly:

    .venv/bin/python -m pytest tests/test_conversational/test_bq_unseen_questions.py -m bigquery -v

Each test's docstring records: the question, why it is valid, the path it
should take, and — for any case that initially failed — a BUG block
characterizing the root cause and the general (non-curated) fix.
"""

from __future__ import annotations

import math
import os
from decimal import Decimal
from pathlib import Path

import pytest
from dotenv import load_dotenv

pytestmark = [pytest.mark.bigquery, pytest.mark.live_llm]


# --------------------------------------------------------------------------- #
# Configuration / skip gating
# --------------------------------------------------------------------------- #

_BQ_PROJECT = os.environ.get("BIGQUERY_PROJECT", "mimic-485500")
# Unused for the BigQuery path (the DuckDB backend is never opened), but the
# constructor requires a path. Point at the real file when present.
_DB_PATH = Path("data/processed/mimiciv.duckdb")
_ONTOLOGY_DIR = Path("ontology/definition")

# Cohort cap for graph-path tests: the median/percentile/temporal questions
# build an in-memory graph from every matched admission's events, so the cohort
# is bounded to a fixed recent sample to keep BigQuery scan + extraction time
# (and Anthropic spend) predictable. Large enough that a robust statistic (the
# median) lands near the full-cohort reference; small enough to run in seconds.
_GRAPH_COHORT_CAP = 150

# Aggregate columns the answerer emits (see answerer._COLUMN_MAP). The oracle
# pulls the scalar result from these rather than guessing.
_AGG_COLUMNS = (
    "Mean Value", "Average", "Max Value", "Min Value", "Median Value",
    "Value", "Count",
)


def _load_demo_env() -> None:
    """Load the repo ``.env`` so the live MCP tools (OMOPHub: ``rxnorm_lookup``,
    ``icd_autocode``, ...) and BigQuery see their keys in ``os.environ`` —
    exactly as the Streamlit demo does via ``app.py``'s ``load_dotenv()``.
    Without it the production-config resolver silently degrades to LIKE-only
    fallbacks, masking the very grounding these tests exercise.

    Called from the credentials gate (i.e. only when a BigQuery test actually
    runs), **not** at module import: pytest imports every collected module —
    including this one when it is marker-deselected — so an import-time
    ``load_dotenv`` would leak ``.env`` (BIGQUERY_PROJECT, ADC creds, ...) into
    ``os.environ`` for the whole offline suite and flip env-sensitive code paths
    in unrelated tests. Explicit repo-root path avoids ``find_dotenv`` cwd
    ambiguity. ``override=False`` (the default) keeps a real shell env winning.
    """
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")


def _require_bigquery_credentials() -> None:
    """Skip cleanly when the keys / ADC needed for a live BigQuery + LLM run
    are absent, so a teammate without credentials gets a diagnostic, not a
    crash."""
    _load_demo_env()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")
    try:
        from google.cloud import bigquery  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"google-cloud-bigquery unavailable: {exc}")
    try:
        import google.auth

        google.auth.default()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"BigQuery ADC not configured: {exc}")


# --------------------------------------------------------------------------- #
# Ground-truth helper — direct BigQuery, bypasses the pipeline entirely
# --------------------------------------------------------------------------- #


def bq_scalar(sql: str) -> float | int | None:
    """Run a direct BigQuery query and return the single scalar in row 0,
    col 0. Used to cross-check the pipeline's number against a hand-written
    cohort query (the 'ground-truth' half of the hybrid oracle)."""
    from google.cloud import bigquery

    client = bigquery.Client(project=_BQ_PROJECT)
    rows = list(client.query(sql).result())
    if not rows:
        return None
    return list(rows[0].values())[0]


# --------------------------------------------------------------------------- #
# Oracle — the programmatic definition of "valid answer"
# --------------------------------------------------------------------------- #


def _leaf_answers(answer) -> list:
    """A comparison / multi-CQ answer carries its real results in
    ``sub_answers``; a single-CQ answer is its own leaf."""
    return list(answer.sub_answers) if answer.sub_answers else [answer]


def _as_finite_float(v) -> float | None:
    """Coerce a data_table cell to a finite float, or ``None`` if it is not a
    real number. ``Decimal`` is included deliberately: BigQuery's NUMERIC AVG
    arrives as ``Decimal`` on the graph path (the SQL fast-path returns float),
    and an oracle that only accepted ``int``/``float`` would mis-read a perfectly
    valid Decimal answer as 'no numeric data'. ``bool`` is excluded (it is an
    ``int`` subclass but never a measurement)."""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float, Decimal)):
        fv = float(v)
        return fv if math.isfinite(fv) else None
    return None


def _numeric_cells(leaf) -> list[float]:
    """Every finite numeric cell in a leaf's data_table (bools excluded)."""
    out: list[float] = []
    for row in (leaf.data_table or []):
        for v in row.values():
            fv = _as_finite_float(v)
            if fv is not None:
                out.append(fv)
    return out


def _aggregate_cells(leaf) -> list[float]:
    """Finite numeric cells found specifically under aggregate columns —
    the actual answer to an AVG/MAX/MIN/MEDIAN/COUNT question."""
    out: list[float] = []
    for row in (leaf.data_table or []):
        for col in _AGG_COLUMNS:
            fv = _as_finite_float(row.get(col))
            if fv is not None:
                out.append(fv)
    return out


def _any_query(answer) -> bool:
    """Did any node in the answer tree record an executed SQL/SPARQL query?"""
    if answer.sparql_queries_used:
        return True
    for sub in (answer.sub_answers or []):
        if _any_query(sub):
            return True
    return False


def assert_valid_answer(
    answer,
    *,
    min_groups: int = 1,
    value_predicate=None,
    require_query: bool = True,
) -> None:
    """Assert ``answer`` is a *valid answer* for a *valid question*.

    Hard checks (apply to every test):
      - not an error;
      - non-empty natural-language summary;
      - at least ``min_groups`` answered leaves, each carrying numeric data
        (so a spurious "no data found" fails here);
      - at least one query actually executed.

    Soft check (per test): ``value_predicate`` — a plausibility / magnitude
    bound on the aggregate value (sign, order of magnitude, or a tolerance
    band derived from a ``bq_scalar`` ground-truth query).
    """
    assert answer is not None, "pipeline returned None"
    assert answer.error is False, f"pipeline raised an error: {answer.text_summary!r}"
    assert answer.clarifying_question is None, (
        f"pipeline asked to clarify a valid question: {answer.clarifying_question!r}"
    )
    assert answer.text_summary and answer.text_summary.strip(), "empty summary"

    leaves = _leaf_answers(answer)
    assert len(leaves) >= min_groups, (
        f"expected >= {min_groups} answered groups, got {len(leaves)}: "
        f"{answer.text_summary!r}"
    )
    for i, leaf in enumerate(leaves):
        assert leaf.error is False, f"group {i} errored: {leaf.text_summary!r}"
        agg = _aggregate_cells(leaf)
        nums = agg or _numeric_cells(leaf)
        assert nums, (
            f"group {i} returned no numeric data (spurious 'no rows'?): "
            f"summary={leaf.text_summary!r} table={leaf.data_table!r}"
        )
        if value_predicate is not None:
            target = agg or nums
            assert any(value_predicate(v) for v in target), (
                f"group {i} aggregate values fail plausibility predicate: {target}"
            )

    if require_query:
        assert _any_query(answer), "no SQL/SPARQL query was recorded on the answer"


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def bq_pipeline():
    """Live ``ConversationalPipeline`` against BigQuery in full production
    config — the configuration the demo actually runs."""
    _require_bigquery_credentials()
    from src.conversational.orchestrator import ConversationalPipeline

    return ConversationalPipeline(
        db_path=_DB_PATH,
        ontology_dir=_ONTOLOGY_DIR,
        api_key=os.environ["ANTHROPIC_API_KEY"],
        data_source="bigquery",
        bigquery_project=_BQ_PROJECT,
        enable_critic=True,
        enable_pre_validator=True,
        enable_disambiguation=True,
        enable_outlier_screening=True,
    )


@pytest.fixture(scope="module")
def bq_pipeline_graph():
    """Live ``ConversationalPipeline`` against BigQuery in full production
    config, but with a **bounded cohort** for the graph DB path.

    Graph-path questions (median/percentile, temporal, raw time-series) build an
    in-memory clinical graph from the matched cohort's events. On the full MIMIC
    cohort that is unbounded scan + extraction time, so this variant injects an
    ``ExtractionConfig(max_admissions=...)`` to cap the cohort to a fixed recent
    sample (``cohort_strategy="recent"`` → ``ORDER BY admittime DESC LIMIT N``),
    holding cost/latency down while still exercising the real extraction →
    graph-build → SPARQL → answer path the demo runs."""
    _require_bigquery_credentials()
    from src.conversational.models import ExtractionConfig
    from src.conversational.orchestrator import ConversationalPipeline

    return ConversationalPipeline(
        db_path=_DB_PATH,
        ontology_dir=_ONTOLOGY_DIR,
        api_key=os.environ["ANTHROPIC_API_KEY"],
        data_source="bigquery",
        bigquery_project=_BQ_PROJECT,
        enable_critic=True,
        enable_pre_validator=True,
        enable_disambiguation=True,
        enable_outlier_screening=True,
        extraction_config=ExtractionConfig(max_admissions=_GRAPH_COHORT_CAP),
    )


@pytest.fixture(autouse=True)
def _reset_pipeline_state():
    """Keep each turn independent: clear conversation history and the
    resolver's MCP lru_caches before/after every test (mirrors the dashboard
    suite's hermeticity guard)."""
    from src.conversational import concept_resolver as cr

    cr._cached_icd_autocode.cache_clear()
    cr._cached_mimic_itemid_search.cache_clear()
    cr._cached_rxnorm_generics.cache_clear()
    yield
    cr._cached_icd_autocode.cache_clear()
    cr._cached_mimic_itemid_search.cache_clear()
    cr._cached_rxnorm_generics.cache_clear()


@pytest.fixture(autouse=True)
def _clear_history(request):
    """Clear the module-scoped pipelines' history before each test that uses
    one, so prior turns don't leak into decomposition."""
    for name in ("bq_pipeline", "bq_pipeline_graph"):
        if name in request.fixturenames:
            request.getfixturevalue(name).conversation_history.clear()
    yield


# --------------------------------------------------------------------------- #
# Iteration 1 — generic biomarker resolves to an unpopulated assay subtype
# --------------------------------------------------------------------------- #


def test_troponin_in_mi_patients_by_age(bq_pipeline):
    """Q: "What was the average troponin level among patients with a myocardial
    infarction above the age of 70 and separately, what was the average
    troponin level among patients with a myocardial infarction under and
    including the age of 70".

    Validity: troponin (MIMIC labevents) + myocardial-infarction diagnosis
    (diagnoses_icd) + anchor_age are all in MIMIC-IV; an average split by an
    age threshold is computable and answerable as text/table. → SQL fast-path
    (one aggregate per age group).

    BUG (iteration 1) — *generic biomarker resolves to an unpopulated assay
    subtype*:
      The decomposer prompt only teaches "troponin I → LOINC 10839-9"
      (prompts.py), so generic "troponin" grounds to LOINC 10839-9, which
      resolve_biomarker maps to itemids [51002, 52642]. In MIMIC-IV BigQuery,
      itemid 51002 has 0 rows and 52642 has 670; the populated assay is
      Troponin T (itemid 51003, 459,872 rows) under a *different* LOINC
      (6598-7). The fast-path emits ``WHERE l.itemid IN (51002,52642)``; once
      intersected with the MI + age cohort this yields ~0 rows → "no data
      found". resolve_biomarker treats "LOINC maps to *some* d_labitems row" as
      success — it has no awareness of actual ``labevents`` row coverage.
      General class: any analyte whose chosen LOINC subtype is sparse/empty in
      MIMIC while a sibling assay (same analyte) is the populated one.

    FIX (two general, composing fixes — neither troponin-specific):
      1. *Coverage-repair broadening* (orchestrator._run_sql_fastpath /
         _broaden_empty_biomarker): when an itemid-grounded biomarker aggregate
         returns no finite numeric cell (empty / all-NULL), recompile the SAME
         cohort once WITHOUT resolved_itemids, forcing the fast-path's existing
         ``d.label ILIKE '%<analyte>%'`` fallback. That matches every sibling
         assay by label (Troponin I AND the populated Troponin T), recovering
         data, and threads a visible "broadened to label family" warning to the
         user and the critic. General: fires for any analyte whose chosen LOINC
         subtype is empty but whose label family is populated. Label-based (not
         SNOMED-family) broadening is deliberate — SNOMED 105000003 pools
         troponin with CK-MB/myoglobin, the label token "troponin" does not.
      2. *Numeric-threshold split* (prompts.py decomposition): an explicit
         cutoff on a continuous field ("above 70" vs "under and including 70")
         now decomposes to Shape B — one ``scope:"cohort"`` sub-CQ per side with
         a ``patient_filters`` age entry (``>`` 70 / ``<=`` 70) — instead of a
         ``comparison_field:"age"`` that would GROUP BY raw age and lose the
         named cutoff. General: any same-metric split across a continuous
         threshold yields two filtered cohorts, not a per-value comparison axis.
      Both compose here: Fix B yields two age-split cohort CQs; each grounds
      generic troponin to the empty Troponin I itemids, and Fix A broadens each
      to the troponin label family to recover Troponin T. Verified end-to-end on
      live BigQuery (over-70 avg ≈ 1.15 ng/mL, ≤70 avg ≈ 1.34 ng/mL).
    """
    answer = bq_pipeline.ask(
        "What was the average troponin level among patients with a myocardial "
        "infarction above the age of 70 and separately, what was the average "
        "troponin level among patients with a myocardial infarction under and "
        "including the age of 70"
    )

    # Ground-truth (Troponin T = itemid 51003, the populated assay) for each
    # age group, MI cohort = ICD-9 410.x / ICD-10 I21.x,I22.x. Used as a loose
    # plausibility band, not an exact-equality assertion (the pipeline's cohort
    # construction may differ in join/dedup details).
    gt = bq_scalar(
        """
        SELECT AVG(l.valuenum)
        FROM `physionet-data.mimiciv_3_1_hosp.labevents` l
        JOIN `physionet-data.mimiciv_3_1_hosp.admissions` a ON l.hadm_id = a.hadm_id
        JOIN `physionet-data.mimiciv_3_1_hosp.patients`  p ON a.subject_id = p.subject_id
        JOIN `physionet-data.mimiciv_3_1_hosp.diagnoses_icd` d ON a.hadm_id = d.hadm_id
        WHERE l.itemid = 51003 AND l.valuenum IS NOT NULL
          AND ((d.icd_version = 10 AND (d.icd_code LIKE 'I21%' OR d.icd_code LIKE 'I22%'))
               OR (d.icd_version = 9 AND d.icd_code LIKE '410%'))
        """
    )
    assert gt is not None and gt > 0, "ground-truth troponin-T avg should be positive"
    upper = float(gt) * 10.0  # generous: assay/cohort differences allowed

    # Valid answer: two age groups, each with a positive, plausibly-bounded
    # average troponin (ng/mL range; generously < 10x the cohort-wide mean).
    assert_valid_answer(
        answer,
        min_groups=2,
        value_predicate=lambda v: 0 < v < max(upper, 50.0),
    )


# --------------------------------------------------------------------------- #
# Iteration 2 — vital-sign aggregate pools mixed-unit chartevents items
# --------------------------------------------------------------------------- #


def _plausible_body_temp(v: float) -> bool:
    """A coherent single-unit body temperature: plausible in Celsius
    (35-42 °C) OR in Fahrenheit (95-108 °F). A Celsius/Fahrenheit *pooled*
    average lands in the dead zone between the two bands (~43-94) and so fails
    both — that is exactly the unit-pooling signature this test screens for."""
    return (35.0 <= v <= 42.0) or (95.0 <= v <= 108.0)


def test_average_body_temperature_in_pneumonia(bq_pipeline):
    """Q: "What is the average body temperature of patients diagnosed with
    pneumonia".

    Validity: body temperature is charted in MIMIC-IV ``mimiciv_3_1_icu.
    chartevents`` and pneumonia is codeable in ``diagnoses_icd`` (ICD-9 480-486,
    ICD-10 J12-J18); a cohort average of a vital is computable and answerable as
    text/table. → SQL fast-path, vital aggregate (``concept_type: "vital"`` →
    ``chartevents`` JOIN ``d_items``).

    BUG (iteration 2, as actually observed) — *a blocked supplementary
    outlier-rows query aborts the whole vital answer*:
      The decomposer correctly routes temperature to the vital fast-path
      (``chartevents`` JOIN ``d_items``) and the MAIN screened aggregate runs
      fine. But ``_build_outlier_report`` (orchestrator.py ~1249) then issues a
      SECOND, companion query — ``outlier_rows_sql`` — to fetch the individual
      rows the biological-possibility screen removed. Over the ~330M-row
      ``chartevents`` table that companion query must full-scan ≈14.31 GiB,
      which the per-query cost validator (``_BigQueryBackend.execute`` →
      ``ValidatorBlockedQueryError``, extractor.py ~295) BLOCKS. That exception
      propagated uncaught and the orchestrator replaced the (already-computed)
      answer with the generic "please rephrase" error. General class: ANY vital
      aggregate (heart rate, temperature, BP, …) over the large ``chartevents``
      table trips this — the supplementary outlier-rows fetch is unbounded and a
      block/failure in it sinks the entire answer, contradicting the backend's
      own design intent (``blocked_queries`` is meant to surface a *warning*,
      not crash; extractor.py ~279-281).

      Secondary observation (a DISTINCT bug, deferred to a later iteration):
      vitals are never itemid-grounded (no ``resolve_vital``; orchestrator
      ~1127-1135), so the fast-path falls to ``d.label ILIKE '%temperature%'``
      (sql_fastpath.py ~389), which pools ``Temperature Fahrenheit`` (223761,
      n≈2.05M, 98.7 °F) with ``Temperature Celsius`` (223762, n≈395k, 37.1 °C).
      Here the screen's derived [10,47] °C envelope incidentally de-pools by
      discarding every °F reading as "impossible," so the recovered mean (≈37
      °C) is plausible — but it is biased to the Celsius minority and mislabels
      ~2M valid °F readings. That representativeness/unit-coherence bug is
      screened-off here and pursued under its own oracle separately.

    FIX (iteration 2 — graceful degradation, general to all vitals):
      Wrap the ``outlier_rows`` fetch in ``_build_outlier_report`` so a cost
      block (or any failure) of that supplementary query logs a warning and
      returns the outlier report WITHOUT per-row samples instead of raising —
      the main aggregate answer (which already carries n_removed / n_total /
      bounds) survives. No bytes are billed: the validator blocks on the
      dry-run estimate before execution. Verified: the question now returns a
      plausible body temperature (≈37 °C after the screen) instead of an error.
    """
    answer = bq_pipeline.ask(
        "What is the average body temperature of patients diagnosed with "
        "pneumonia"
    )

    # Ground-truth: the clean, single-unit Fahrenheit average (itemid 223761,
    # the dominant temperature item) over the same pneumonia cohort. This is the
    # valid answer's neighborhood; the buggy pooled value (≈85) is nowhere near
    # it and fails the plausibility band below.
    gt_f = bq_scalar(
        """
        SELECT AVG(c.valuenum)
        FROM `physionet-data.mimiciv_3_1_icu.chartevents` c
        JOIN `physionet-data.mimiciv_3_1_hosp.diagnoses_icd` d ON c.hadm_id = d.hadm_id
        WHERE c.itemid = 223761 AND c.valuenum IS NOT NULL
          AND (REGEXP_CONTAINS(d.icd_code, r'^J1[2-8]')
               OR REGEXP_CONTAINS(d.icd_code, r'^48[0-6]'))
        """
    )
    assert gt_f is not None and 95.0 <= float(gt_f) <= 108.0, (
        f"ground-truth Fahrenheit temp should be a plausible fever-ish "
        f"body temp, got {gt_f!r}"
    )

    # Valid answer: a coherent single-unit body temperature (≈98-99 °F or
    # ≈37 °C). A C/F-pooled ≈85 fails _plausible_body_temp → exposes the bug.
    assert_valid_answer(
        answer,
        min_groups=1,
        value_predicate=_plausible_body_temp,
    )


# --------------------------------------------------------------------------- #
# Iteration 3 — length-of-stay aggregate (metadata-only, no clinical concept)
# --------------------------------------------------------------------------- #


def test_average_length_of_stay_in_heart_failure(bq_pipeline):
    """Q: "What is the average length of stay for patients admitted with heart
    failure".

    Validity: length of stay is derivable in MIMIC-IV from
    ``admissions.dischtime - admissions.admittime`` (hospital LOS) or
    ``icustays.los`` (ICU LOS, already in days); heart failure is codeable in
    ``diagnoses_icd`` (ICD-9 428.x, ICD-10 I50.x); a cohort mean is computable
    and answerable as text/table. LOS is a *metadata* aggregate, not a clinical
    concept (prompts.py ~144: "Length of stay (LOS) is NOT a concept — omit
    clinical_concepts and use aggregation"). → SQL fast-path, no concept
    resolution.

    Expected path: decompose → ``aggregation: "mean"`` LOS over the HF cohort
    → SQL_FAST. Ground-truth neighborhood: hospital LOS ≈ 6.8 days
    (n≈80.6k admissions); ICU LOS ≈ 4.0 days (n≈36.6k stays). Either reading is
    a valid answer; the plausibility band below accepts both and rejects the
    classic failure modes (LOS reported in hours/seconds, a negative span, or a
    timestamp-subtraction artifact).

    OBSERVED (iteration 3 — PASS, with two known-limitation findings, NOT a
    correctness failure):

      1. ROUTING / LATENCY. The CQ carries no ``clinical_concepts`` (LOS is a
         metadata aggregate, per the prompt), so it does NOT take SQL_FAST as
         the "Expected path" line hoped. ``QueryPlanner.classify`` routes every
         concept-less CQ to GRAPH (planner.py:117-118), and ``compile_sql``
         *rejects* an empty-concept CQ outright (sql_fastpath.py:161-162). These
         are consistent BY DESIGN — planner.py:113-116 documents metadata-only
         CQs as deliberately kept on the graph path ("widen in a follow-up").
         Consequence: this question runs the full extract → build-graph → SPARQL
         sequence (~198 s on live BigQuery, with rdflib parser warnings) instead
         of a sub-second SQL ``AVG``. The answer is correct but the latency is
         demo-hostile. Classified as a KNOWN PERFORMANCE LIMITATION rather than
         fixed inline: a SQL-fast LOS branch is a *feature* with an
         underspecified-measure problem (the CQ says "average of <nothing>" — it
         never names LOS as the measure; the graph reasoner only *guesses* LOS),
         so a rushed port risks mis-aggregating sibling metadata questions like
         "average age of HF patients". Tracked in the final report's
         recommendations, not patched mid-loop.

      2. LOS-TYPE DEFAULT. For a concept-less aggregate the graph reasoner
         selects the ICU-LOS template (reasoner.py:466-468), so the system
         likely returns ICU LOS (~4 d), whereas "patients *admitted* with heart
         failure" reads most naturally as HOSPITAL LOS (~6.8 d). Both are "a
         length of stay", so this is a defensible default rather than a clear
         bug; the loose band below accepts either. Flagged for the report.

    Net: the oracle PASSES (a plausible LOS in days, a query executed, no
    no-data sentinel). The two findings above are limitations to document, not
    failures to fix in this iteration.
    """
    answer = bq_pipeline.ask(
        "What is the average length of stay for patients admitted with heart "
        "failure"
    )

    # Ground-truth: hospital LOS in DAYS over the HF cohort (the most common
    # reading). Used as a loose anchor; the predicate also admits the ICU-LOS
    # reading (~4 days).
    gt_los = bq_scalar(
        """
        SELECT AVG(TIMESTAMP_DIFF(a.dischtime, a.admittime, HOUR) / 24.0)
        FROM `physionet-data.mimiciv_3_1_hosp.admissions` a
        JOIN `physionet-data.mimiciv_3_1_hosp.diagnoses_icd` d ON a.hadm_id = d.hadm_id
        WHERE REGEXP_CONTAINS(d.icd_code, r'^I50')
           OR REGEXP_CONTAINS(d.icd_code, r'^428')
        """
    )
    assert gt_los is not None and 3.0 <= float(gt_los) <= 15.0, (
        f"ground-truth HF hospital LOS should be a few days, got {gt_los!r}"
    )

    # Valid answer: a plausible length of stay in DAYS (hospital ≈ 6.8 or ICU
    # ≈ 4.0). Rejects hours (~96-164), seconds, and negative spans.
    assert_valid_answer(
        answer,
        min_groups=1,
        value_predicate=lambda v: 0.5 < v < 30.0,
    )


# --------------------------------------------------------------------------- #
# Iteration 4 — drug cohort COUNT inflates event rows into a "patient" count
# --------------------------------------------------------------------------- #


def test_lasix_prescription_count_distinct_grain(bq_pipeline):
    """Q: "How many patients were prescribed Lasix".

    Validity: Lasix is the brand name for furosemide, one of the most common
    drugs in MIMIC-IV — ``furosemide`` appears on 447,330 prescription rows
    spanning 104,250 admissions / 54,534 distinct patients (the literal brand
    ``lasix`` is only 5 rows). A clinician asking about "Lasix" means
    furosemide; the count is computable and answerable as text.

    Expected path: ``concept_type: "drug"`` + ``aggregation: "count"`` →
    SQL_FAST → ``_compile_drug_aggregate``.

    Brand→generic grounding (cross-cutting with iteration 6): the decomposer
    emits the brand or the generic *non-deterministically* (a 6-run probe gave
    "Lasix" 5× and "furosemide" 1×). When it emits the literal brand, a naive
    ``pr.drug ILIKE '%lasix%'`` matches only the 5 brand-labelled rows and the
    count collapses. The iteration-6 fix grounds the brand to its RxNorm generic
    ingredient (``ConceptResolver.resolve_drug`` → ``%lasix% OR %furosemide%``),
    so this count is now stable on the full furosemide cohort regardless of which
    surface form the decomposer chose. (An earlier note here that "brand→generic
    is handled upstream; no fix needed" was wrong — it held only on the lucky run
    where the LLM volunteered the generic.)

    BUG (iteration 4 — confirmed via direct pipeline probe): the drug COUNT
    branch emitted ``SELECT COUNT(*) ... FROM prescriptions pr JOIN admissions``
    and answered **"447,325 patients were prescribed Lasix"**. That is the
    prescription-ROW count, not a patient or admission count — it exceeds the
    entire MIMIC-IV patient population (~364k), so it is impossible on its face.
    A patient on a furosemide drip accrues dozens of order rows; ``COUNT(*)``
    multiplies the cohort by that per-patient row fan-out (~8x here). The drug
    (and microbiology) branches were the only fast-path COUNT compilers using
    raw ``COUNT(*)``; the diagnosis (sql_fastpath.py:738) and outcome
    (sql_fastpath.py:886) branches already used ``COUNT(DISTINCT hadm_id)``.

    Why MIMIC data can't be blamed: the cohort is real and large; the query
    simply counts the wrong unit (rows, not admissions).

    FIX (general, consistency-restoring, no curation): align the drug and
    microbiology COUNT branches to the established distinct-admission grain
    (``COUNT(DISTINCT a.hadm_id)``), so a cohort COUNT reports admissions, not
    event rows. General across every drug/culture term, not a Lasix special-
    case. (A finer patient-vs-admission grain — and the fact that the answerer
    still labels an admission count "patients" — is a separate, system-wide
    refinement noted in the report; both diagnosis and outcome share it.)

    Oracle: the count must be a plausible distinct-cohort size — tens of
    thousands (54,534 patients .. 104,250 admissions), NOT the 447k row
    fan-out and NOT a single-digit ungrounded miss. The band 30k..150k admits
    either distinct grain and rejects both failure modes.
    """
    answer = bq_pipeline.ask("How many patients were prescribed Lasix")

    # Ground-truth anchors: the furosemide cohort at two distinct grains.
    gt_adm = bq_scalar(
        """
        SELECT COUNT(DISTINCT hadm_id)
        FROM `physionet-data.mimiciv_3_1_hosp.prescriptions`
        WHERE LOWER(drug) LIKE '%furosemide%' OR LOWER(drug) LIKE '%lasix%'
        """
    )
    assert gt_adm is not None and 80_000 < float(gt_adm) < 130_000, (
        f"ground-truth furosemide admission count should be ~104k, got {gt_adm!r}"
    )

    # Valid answer: a distinct-cohort count (admissions ~104k or patients ~54k),
    # NOT the ~447k prescription-row fan-out and NOT a single-digit ungrounded
    # miss. The drug COUNT branch must de-duplicate to a clinical unit.
    assert_valid_answer(
        answer,
        min_groups=1,
        value_predicate=lambda v: 30_000 <= v <= 150_000,
    )


# --------------------------------------------------------------------------- #
# Iteration 5 — outcome path answers a COUNT when the user asked for a RATE
# --------------------------------------------------------------------------- #


def _is_proportion_key(key: str) -> bool:
    """A rendered column that holds a proportion/rate (vs. a raw count)."""
    k = key.lower()
    return "proportion" in k or "fraction" in k or "rate" in k


def test_sepsis_in_hospital_mortality_rate(bq_pipeline):
    """Q: "What is the in-hospital mortality rate for patients with sepsis".

    Validity: in-hospital mortality is ``admissions.hospital_expire_flag``;
    sepsis is codeable in ``diagnoses_icd`` (ICD-10 A40/A41, R65.2; ICD-9 038.x,
    995.91/.92, 785.52). A cohort RATE = deaths / admissions is computable and
    answerable as text/table. Ground truth ≈ 0.195-0.20 and robust to the exact
    code set: narrow (A41 / 99591 / 99592) = 4,237 / 21,219 = 0.1997; broad
    (+ A40 / 038.x / R65.2 / 785.52) = 4,316 / 22,184 = 0.1946. → ``concept_type:
    "outcome"`` + a sepsis diagnosis filter → SQL_FAST → ``_compile_outcome_
    mortality``.

    BUG (iteration 5 — *the outcome compiler answers a COUNT when the user asked
    for a RATE*):
      ``_compile_outcome_mortality`` deliberately ignores ``cq.aggregation`` and
      always emits the grouped ``(expired, COUNT(DISTINCT hadm_id))`` shape —
      survivors and deaths as two raw counts. For "how many sepsis patients died"
      that is correct; for "what is the mortality RATE" it is the wrong
      *aggregate*. The structured ``data_table`` came back as
      ``[{Expired:0, Count:17641}, {Expired:1, Count:4363}]`` — no rate anywhere.
      The ≈0.198 the user asked for survived ONLY because the answerer LLM
      happened to divide 4363 / 22004 in prose. That is fragile (model- and
      phrasing-dependent, not deterministic) and invisible to every structured
      consumer: the critic's "mortality rate ∈ [0,1]" plausibility check
      (prompts.py:588) has no rate to range-check, and a table/visualization sees
      counts, not a proportion. Same class as iteration 4 (a COUNT compiler
      emitting the wrong unit) one level up — here the wrong *aggregate* (count
      instead of rate), not merely the wrong grain.

      Why MIMIC data can't be blamed: the sepsis cohort is real and large and the
      counts are correct; the pipeline simply never computes the ratio the
      question names.

    FIX (general, deterministic, no curation): ``_compile_outcome_mortality`` now
      also computes each outcome bucket's share of the cohort in-query —
      ``COUNT(DISTINCT hadm_id) / NULLIF(SUM(COUNT(DISTINCT hadm_id)) OVER (), 0)
      AS fraction`` — so the grouped shape becomes ``(expired, count, fraction)``.
      The ``expired = 1`` row's fraction IS the mortality rate; the ``expired = 0``
      row's is the survival rate. General across every outcome-rate question and
      cohort (no sepsis/mortality special-case), ontology-neutral, and free (a
      window over counts already computed — no second scan). The answerer now
      reports a rate it is *given* rather than one it must re-derive, and the
      critic can range-check a real proportion.

    Oracle: structural, not prose. The ``data_table`` must carry a proportion
      cell within 0.05 of the ground-truth sepsis mortality rate — NOT merely two
      counts. This FAILS on the pre-fix counts-only shape and PASSES once the rate
      is surfaced.
    """
    answer = bq_pipeline.ask(
        "What is the in-hospital mortality rate for patients with sepsis"
    )

    # Ground-truth mortality rate over a broad sepsis cohort. Loose anchor: the
    # pipeline's ICD grounding may differ slightly in the code set, so the
    # assertion below allows a 0.05 band rather than exact equality.
    gt_rate = bq_scalar(
        """
        SELECT SAFE_DIVIDE(
                 COUNT(DISTINCT IF(a.hospital_expire_flag = 1, a.hadm_id, NULL)),
                 COUNT(DISTINCT a.hadm_id))
        FROM `physionet-data.mimiciv_3_1_hosp.admissions` a
        JOIN `physionet-data.mimiciv_3_1_hosp.diagnoses_icd` d ON a.hadm_id = d.hadm_id
        WHERE REGEXP_CONTAINS(d.icd_code, r'^A4[01]')
           OR REGEXP_CONTAINS(d.icd_code, r'^038')
           OR d.icd_code IN ('99591', '99592', '78552', 'R652')
        """
    )
    assert gt_rate is not None and 0.15 <= float(gt_rate) <= 0.25, (
        f"ground-truth sepsis mortality rate should be ~0.20, got {gt_rate!r}"
    )

    # Hard validity: real rows, a query executed, no spurious error/clarify.
    assert_valid_answer(answer, min_groups=1)

    # The crux — a STRUCTURED rate, not counts-only. Scan every leaf's table for
    # a proportion-keyed cell in [0,1]; the survival share (≈0.80) is far from
    # the ground-truth mortality rate (≈0.20), so the 0.05 band selects exactly
    # the mortality proportion. Pre-fix the table has only {Expired, Count} and
    # this assertion fails — exposing the missing-rate bug.
    tables = [
        leaf.data_table or [] for leaf in (answer.sub_answers or [answer])
    ]
    prop_cells = [
        float(v)
        for table in tables
        for row in table
        for k, v in row.items()
        if _is_proportion_key(k)
        and isinstance(v, (int, float))
        and not isinstance(v, bool)
        and 0.0 <= float(v) <= 1.0
    ]
    assert prop_cells, (
        "no structured proportion/rate cell — the outcome answer carried only "
        f"counts, leaving the rate to LLM prose: tables={tables!r}"
    )
    assert any(abs(p - float(gt_rate)) <= 0.05 for p in prop_cells), (
        f"no structured proportion within 0.05 of the ground-truth sepsis "
        f"mortality rate {gt_rate}; got {prop_cells}"
    )


# --------------------------------------------------------------------------- #
# Iteration 6 — brand drug name never reaches its generic in MIMIC
# --------------------------------------------------------------------------- #


def test_coumadin_brand_resolves_to_generic_cohort(bq_pipeline):
    """Q: "How many patients were prescribed Coumadin".

    Validity: Coumadin is the brand name for warfarin. MIMIC-IV
    ``prescriptions`` stores the *generic*: ``warfarin`` spans 45,062 distinct
    admissions / 20,746 distinct patients (179,937 rows), while the literal
    brand ``coumadin`` appears on only 10 admissions. A clinician asking about
    "Coumadin" means warfarin; the count is computable and answerable as text.

    Expected path: ``concept_type: "drug"`` + ``aggregation: "count"`` →
    SQL_FAST → ``_compile_drug_aggregate`` (``COUNT(DISTINCT a.hadm_id)``).

    BUG (iteration 6 — *brand drug name never reaches its generic*): the
    decomposer emits the surface drug term non-deterministically — sometimes the
    brand ("Coumadin"), sometimes the generic. Concept resolution had no
    brand→generic step: ``ConceptResolver.resolve`` falls straight through to
    ``[concept.name]`` for a specific drug, so a "Coumadin" turn compiled
    ``pr.drug ILIKE '%coumadin%'`` and matched **10 admissions** — a ~4,500x
    undercount of the real 45,062 — while a "warfarin" turn answered correctly.
    The same defect zeroes out ``Tylenol`` (0 ``%tylenol%`` rows vs 778k
    ``%acetaminophen%``). It is a *general* brand-vs-generic mismatch: MIMIC's
    prescription vocabulary is generic-dominant, but users (and the LLM) speak
    in brands.

    Why MIMIC data can't be blamed: the warfarin cohort is real and large; the
    query just searched the brand string MIMIC almost never stores.

    FIX (general, ontology-grounded, no curation): add
    ``ConceptResolver.resolve_drug`` (wired in ``orchestrator`` beside the
    existing per-concept ``resolve``), which grounds a specific drug name to its
    RxNorm ingredient via the OMOPHub ``rxnorm_lookup`` MCP tool and *appends*
    the generic to the OR-match name list (``%coumadin% OR %warfarin%``). The
    ingredient is extracted from OMOPHub's SPL product records by stripping a
    uniform set of pharmaceutical salt counter-ions and dose-form tokens (a
    general chemistry/formulation normalization — "warfarin sodium" → "warfarin"
    — NOT a per-drug synonym table). Result is cached + frequency-ranked for
    determinism, and degrades to the literal name (with a visible fallback note)
    when OMOPHub has no coverage. General across every brand whose generic
    OMOPHub knows; the decomposer's brand-vs-generic coin flip no longer changes
    the answer.

    Oracle: the count must be the real warfarin cohort — tens of thousands
    (20,746 patients .. 45,062 admissions) — NOT the ~10-admission brand-literal
    collapse and NOT the 179,937-row fan-out. The band 12k..60k admits either
    distinct grain and rejects both failure modes.
    """
    answer = bq_pipeline.ask("How many patients were prescribed Coumadin")

    # Ground truth: the warfarin (= Coumadin generic) cohort, distinct grain.
    gt_adm = bq_scalar(
        """
        SELECT COUNT(DISTINCT hadm_id)
        FROM `physionet-data.mimiciv_3_1_hosp.prescriptions`
        WHERE LOWER(drug) LIKE '%warfarin%' OR LOWER(drug) LIKE '%coumadin%'
        """
    )
    assert gt_adm is not None and 35_000 < float(gt_adm) < 55_000, (
        f"ground-truth warfarin admission count should be ~45k, got {gt_adm!r}"
    )

    # Valid answer: a distinct-cohort count on the *generic* warfarin rows, not
    # the ~10-admission brand-literal miss and not the row fan-out. The brand
    # must have been grounded to its generic before SQL emission.
    assert_valid_answer(
        answer,
        min_groups=1,
        value_predicate=lambda v: 12_000 <= v <= 60_000,
    )


# --------------------------------------------------------------------------- #
# Iteration 7 — graph DB path (median routes off the SQL fast-path)
# --------------------------------------------------------------------------- #


def test_median_lactate_in_sepsis(bq_pipeline_graph):
    """Q: "What is the median lactate level in patients with sepsis".

    First coverage of the **graph DB path** (iterations 1-6 all stayed on the
    SQL fast-path). The planner routes any aggregate whose ``sql_fn is None`` to
    the graph path, and ``median`` is exactly that case
    (``operations_aggregates.py`` — there is no portable SQL median across
    DuckDB/BigQuery, so the pipeline materializes the cohort's lactate readings
    into an in-memory clinical graph and computes the quantile in Python via a
    SPARQL extraction). This question therefore exercises the *entire* graph
    stack the demo relies on — cohort extraction → graph build → SPARQL value
    pull → percentile answer — which the SQL-path tests never touch.

    Validity: sepsis is a large MIMIC-IV cohort (~22,184 admissions by the broad
    ICD set ``^A4[01] | ^038 | {99591,99592,78552,R652}``), of which 16,661 carry
    a serum lactate (itemid 50813, reported in mmol/L); the full-cohort median is
    1.9 mmol/L. The quantity is computable and answerable as a single scalar, so
    a spurious "no data" or a crash here would be a real graph-path bug.

    Bounded cohort: ``bq_pipeline_graph`` injects
    ``ExtractionConfig(max_admissions=150)`` so the graph is built from a fixed
    recent slice rather than all 22k admissions — bounding scan/extraction
    cost. The median is a robust statistic, so a recent 150-admission sample
    still lands in the clinically valid band (observed 2.3 mmol/L) near the
    full-cohort 1.9.

    Oracle (hybrid):
      - structural — ``assert_valid_answer`` (no error, real rows, a finite
        number, and a query actually executed, i.e. the graph path produced a
        SPARQL/extraction record), so a spurious "no rows" or graph crash fails;
      - plausibility — the median lactate must sit in the clinically valid band
        0.8 .. 6.0 mmol/L: above a near-zero (which would betray a units divide
        or a mis-read count) and below a mean-vs-median or mg/dL-scale blowup;
      - ground truth — the direct full-cohort median (~1.9 mmol/L, computed
        below) must be real and in mmol/L range, and the pipeline's bounded
        answer must fall within a generous multiplicative factor of it, catching
        any scale error the absolute band might miss.

    Status: PASSES on first authoring — the graph path handles this correctly.
    Recorded as positive coverage that the median/graph route is robust for a
    standard cohort-aggregate question (not every iteration must surface a bug;
    the loop also banks validated paths).
    """
    # Ground truth: full-cohort sepsis lactate median, direct from BigQuery.
    # Mirrors the validity probe's broad sepsis ICD set; itemid 50813 = lactate.
    gt_median = bq_scalar(
        """
        WITH sepsis AS (
            SELECT DISTINCT d.hadm_id
            FROM `physionet-data.mimiciv_3_1_hosp.diagnoses_icd` d
            WHERE REGEXP_CONTAINS(d.icd_code, r'^A4[01]')
               OR REGEXP_CONTAINS(d.icd_code, r'^038')
               OR d.icd_code IN ('99591','99592','78552','R652')
        )
        SELECT APPROX_QUANTILES(le.valuenum, 2)[OFFSET(1)]
        FROM `physionet-data.mimiciv_3_1_hosp.labevents` le
        JOIN sepsis s ON le.hadm_id = s.hadm_id
        WHERE le.itemid = 50813 AND le.valuenum IS NOT NULL
        """
    )
    assert gt_median is not None and 1.0 <= float(gt_median) <= 3.0, (
        f"ground-truth sepsis lactate median should be ~1.9 mmol/L, got "
        f"{gt_median!r}"
    )

    answer = bq_pipeline_graph.ask(
        "What is the median lactate level in patients with sepsis"
    )

    # Structural + absolute-plausibility band (clinically valid median lactate).
    assert_valid_answer(
        answer,
        min_groups=1,
        value_predicate=lambda v: 0.8 <= v <= 6.0,
    )

    # Ground-truth cross-check: the bounded-cohort median must track the
    # full-cohort reference within a generous factor (recent-slice sampling
    # moves a robust statistic only a little; a scale bug would move it a lot).
    ref = float(gt_median)
    medians = _aggregate_cells(answer) or _numeric_cells(answer)
    assert any(0.4 * ref <= m <= 3.0 * ref for m in medians), (
        f"graph-path median {medians} strayed from the full-cohort reference "
        f"{ref:.2f} mmol/L by more than the bounded-sampling tolerance"
    )


# --------------------------------------------------------------------------- #
# Iteration 8 — graph path resolves biomarkers by label-LIKE, not LOINC
# --------------------------------------------------------------------------- #


def test_mean_lactate_during_icu_is_not_ldh_polluted(bq_pipeline_graph):
    """Q: "What is the average lactate level during the ICU stay in patients
    with sepsis".

    Validity: serum lactate (LOINC-grounded itemid 50813, mmol/L) is densely
    recorded for septic ICU patients; an average over the ICU stay is a textbook
    sepsis-resuscitation question, computable and answerable as a single scalar.
    The clinically real cohort mean is low single digits of mmol/L (the robust
    full-cohort during-ICU *median* is 1.9; see ground truth below).

    Expected path: ``aggregation: "mean"`` would normally compile to the SQL
    fast-path (``AVG``), but the ``during ICU stay`` temporal constraint forces
    the **graph path** (``planner.classify`` routes any CQ with
    ``temporal_constraints`` to ``GRAPH`` before the aggregation check). The
    temporal reference contains "icu", so ``extractor._temporal_sql`` *does*
    honor it — this test deliberately isolates the biomarker-resolution defect,
    not the temporal one.

    FINDING (iteration 8 — defense-in-depth holds; the live answer is correct):
    in **full production config the pipeline answers this correctly** — the mean
    comes back ~3.06 mmol/L, a textbook septic-ICU lactate. That is the result
    under test and it is in-band. Getting there exercises two layers:

    1. *Latent resolution defect (confirmed, not a live wrong answer).* The graph
       extractor never got the LOINC grounding the SQL fast-path has:
       ``extractor._extract_biomarkers`` resolves the analyte with a raw label
       substring — ``d.label ILIKE '%lactate%'`` — not the LOINC→itemid index
       ``concept_resolver.resolve_biomarker`` builds. ``%lactate%`` matches *five*
       itemids in MIMIC-IV: serum **Lactate** (50813, mmol/L, ~2) **and four
       Lactate Dehydrogenase assays** (50954 / 51054 / 50843 / 51795, all
       **IU/L**, mean ~500). The extractor *does* over-pull all five (direct
       BigQuery: recent-150 sepsis during-ICU raw pooled mean ≈ 154 with ~220 LDH
       rows vs clean ≈ 3).
    2. *Why the answer is still right.* The LDH values (~500 IU/L) sit far outside
       serum lactate's biological limits, so the production **outlier-screening**
       stage strips them before the mean is taken — neutralising the
       unit-incompatible pooling and yielding the clean 3.06. This is genuine
       defense-in-depth: a resolution miss is caught downstream. (It would *not*
       save a same-unit, in-range cross-specimen pool; a glucose probe showed
       serum so dominates ``%glucose%`` that the pooled mean is statistically
       indistinguishable — i.e. no screening-proof live wrong answer was found
       here.)

    So this test pins the **end-to-end correctness** of a valid graph-path
    biomarker average and guards the screening safety-net: if outlier screening
    regressed, the pooled IU/L value (tens-to-hundreds) would breach the band and
    fail. The label-LIKE/LOINC divergence is logged as a **hardening
    recommendation** (align the graph extractor's biomarker resolution with the
    SQL path's ``resolve_biomarker`` LOINC→itemid grounding) — defense-in-depth,
    not a second screen — but is *not* a live defect to "fix" against a passing
    production-config oracle, so no speculative graph-extractor refactor is made.

    Harness bug this iteration actually surfaced & fixed: the live mean arrives as
    a BigQuery ``Decimal`` (NUMERIC AVG on the graph path; the SQL path returns
    float). The oracle's ``_numeric_cells`` / ``_aggregate_cells`` only accepted
    ``int``/``float`` and mis-read the valid 3.06 Decimal as "no numeric data".
    Fixed via ``_as_finite_float`` (``Decimal`` now coerced like any real number).

    Oracle: a cohort-mean serum lactate during ICU stay must sit in the
    clinically valid band 0.5 .. 25 mmol/L — above zero, and far below the
    LDH-polluted value (tens to low hundreds) a screening regression would
    produce. The wide upper bound (a true mean lactate is ~2-5; 25 is already
    incompatible with survival) accepts any real answer while unambiguously
    rejecting IU/L pollution.
    """
    # Ground truth: the clean serum-lactate during-ICU median (robust to the
    # in-itemid garbage-outlier the cohort also carries), proving the question
    # is valid and pinning the correct order of magnitude (~1.9 mmol/L).
    gt_clean_median = bq_scalar(
        """
        WITH sepsis AS (
            SELECT DISTINCT d.hadm_id
            FROM `physionet-data.mimiciv_3_1_hosp.diagnoses_icd` d
            WHERE REGEXP_CONTAINS(d.icd_code, r'^A4[01]')
               OR REGEXP_CONTAINS(d.icd_code, r'^038')
               OR d.icd_code IN ('99591','99592','78552','R652')
        )
        SELECT APPROX_QUANTILES(le.valuenum, 2)[OFFSET(1)]
        FROM `physionet-data.mimiciv_3_1_hosp.labevents` le
        JOIN sepsis s ON le.hadm_id = s.hadm_id
        WHERE le.itemid = 50813 AND le.valuenum IS NOT NULL
          AND EXISTS (
              SELECT 1 FROM `physionet-data.mimiciv_3_1_icu.icustays` icu
              WHERE icu.hadm_id = le.hadm_id
                AND le.charttime >= icu.intime AND le.charttime <= icu.outtime
          )
        """
    )
    assert gt_clean_median is not None and 1.0 <= float(gt_clean_median) <= 3.0, (
        f"ground-truth clean lactate median should be ~1.9 mmol/L, got "
        f"{gt_clean_median!r}"
    )

    answer = bq_pipeline_graph.ask(
        "What is the average lactate level during the ICU stay in patients "
        "with sepsis"
    )

    # The mean must be a plausible serum-lactate value (production returns
    # ~3.06), NOT the IU/L LDH pooling a screening regression would let through
    # (tens-to-hundreds). assert_valid_answer also rejects a spurious "no rows" /
    # graph crash, reads the BigQuery Decimal, and confirms a query executed.
    assert_valid_answer(
        answer,
        min_groups=1,
        value_predicate=lambda v: 0.5 <= v <= 25.0,
    )


# --------------------------------------------------------------------------- #
# Iteration 9 — "positive" culture status is dropped (counts cultures ordered,
# not cultures that grew an organism) → ~4x over-count
# --------------------------------------------------------------------------- #


def test_positive_blood_culture_counts_only_growth(bq_pipeline):
    """Q: "How many patients with sepsis had a positive blood culture?".

    Validity: blood-culture positivity is *the* microbiology question in sepsis.
    In MIMIC-IV ``microbiologyevents`` a culture is **positive** iff an organism
    was isolated — ``org_name IS NOT NULL`` (a negative / no-growth culture has
    the specimen row but a NULL ``org_name``). Over the sepsis cohort ~16.9k
    admissions had a blood culture *drawn* but only ~3.9k *grew* something
    (positivity ≈ 23%), so "positive" carries a real ~4.3x narrowing. Both the
    count and the cohort are unambiguous and computable as one scalar.

    Expected path: single ``microbiology`` concept + ``count`` aggregation, no
    temporal constraint → SQL fast-path ``_compile_microbiology_aggregate``
    (``COUNT(DISTINCT hadm_id)``).

    BUG (iteration 9 — the *result-status* qualifier "positive" is silently
    dropped, so the count is cultures **ordered**, not cultures that **grew**):
    the decomposer collapses "positive blood culture" to a bare ``microbiology``
    concept ``{name:"blood culture"}`` — its own few-shot example
    (``prompt_examples/single_cq/02_blood_culture_count.json``) literally taught
    "Count of admissions with at least one blood culture **event**". The
    compiler then emits only ``spec_type_desc ILIKE '%blood culture%'`` with **no
    ``org_name IS NOT NULL``** clause, so every *drawn* culture counts. Observed:
    the pipeline answers ~16,690 ("...had a positive blood culture") when the
    truth is ~3.9k — a confident ~4.3x over-statement, exactly the kind of wrong
    number a clinician would catch in a demo.

    Why MIMIC data can't be blamed: the positive cohort is large and real; the
    query just never applied the positivity predicate the schema makes trivial
    (organism isolated ⇔ positive).

    FIX (general, schema-grounded, no curation): give ``ClinicalConcept`` an
    optional ``culture_status`` ∈ {positive, negative}; teach the decomposer
    (corrected example + prompt rule) to set it from "positive"/"negative"/"grew"
    /"no growth" wording; and have ``_compile_microbiology_aggregate`` ground it
    to ``m.org_name IS NOT NULL`` (positive) / ``IS NULL`` (negative). This is the
    MIMIC definition of culture positivity, not a per-specimen synonym table, so
    it fixes "positive urine culture", "negative sputum culture", etc. at once.

    Oracle: the answer must track the **positive** ground truth (~3.9k), not the
    **ordered** count (~16.9k). The band [0.5x .. 2.0x] of the direct positive
    ground truth admits the true value under cohort-definition wobble while
    unambiguously rejecting the ~4.3x ordered over-count.
    """
    # Ground truth, direct from BigQuery: distinct sepsis admissions with a
    # POSITIVE blood culture (organism isolated) vs ANY blood culture drawn.
    _SEPSIS_CTE = """
        WITH sepsis AS (
            SELECT DISTINCT d.hadm_id
            FROM `physionet-data.mimiciv_3_1_hosp.diagnoses_icd` d
            WHERE REGEXP_CONTAINS(d.icd_code, r'^A4[01]')
               OR REGEXP_CONTAINS(d.icd_code, r'^038')
               OR d.icd_code IN ('99591','99592','78552','R652')
        )
    """
    gt_positive = bq_scalar(
        _SEPSIS_CTE
        + """
        SELECT COUNT(DISTINCT m.hadm_id)
        FROM `physionet-data.mimiciv_3_1_hosp.microbiologyevents` m
        JOIN sepsis s ON m.hadm_id = s.hadm_id
        WHERE LOWER(m.spec_type_desc) LIKE '%blood cult%'
          AND m.org_name IS NOT NULL
        """
    )
    gt_ordered = bq_scalar(
        _SEPSIS_CTE
        + """
        SELECT COUNT(DISTINCT m.hadm_id)
        FROM `physionet-data.mimiciv_3_1_hosp.microbiologyevents` m
        JOIN sepsis s ON m.hadm_id = s.hadm_id
        WHERE LOWER(m.spec_type_desc) LIKE '%blood cult%'
        """
    )
    assert gt_positive and gt_ordered, "ground-truth microbiology counts missing"
    # The question is only discriminating if "positive" really narrows the cohort.
    assert gt_ordered >= 3 * gt_positive, (
        f"expected a large positive-vs-ordered gap; got positive={gt_positive} "
        f"ordered={gt_ordered}"
    )

    answer = bq_pipeline.ask(
        "How many patients with sepsis had a positive blood culture?"
    )

    # The count must land near the POSITIVE ground truth, NOT the ~4.3x larger
    # ordered count. assert_valid_answer also rejects a spurious "no rows" and
    # confirms a query executed.
    lo, hi = 0.5 * gt_positive, 2.0 * gt_positive
    assert_valid_answer(
        answer,
        min_groups=1,
        value_predicate=lambda v: lo <= v <= hi,
    )
