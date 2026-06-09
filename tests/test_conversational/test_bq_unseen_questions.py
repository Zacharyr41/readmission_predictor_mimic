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
from pathlib import Path

import pytest

pytestmark = [pytest.mark.bigquery, pytest.mark.live_llm]


# --------------------------------------------------------------------------- #
# Configuration / skip gating
# --------------------------------------------------------------------------- #

_BQ_PROJECT = os.environ.get("BIGQUERY_PROJECT", "mimic-485500")
# Unused for the BigQuery path (the DuckDB backend is never opened), but the
# constructor requires a path. Point at the real file when present.
_DB_PATH = Path("data/processed/mimiciv.duckdb")
_ONTOLOGY_DIR = Path("ontology/definition")

# Aggregate columns the answerer emits (see answerer._COLUMN_MAP). The oracle
# pulls the scalar result from these rather than guessing.
_AGG_COLUMNS = (
    "Mean Value", "Average", "Max Value", "Min Value", "Median Value",
    "Value", "Count",
)


def _require_bigquery_credentials() -> None:
    """Skip cleanly when the keys / ADC needed for a live BigQuery + LLM run
    are absent, so a teammate without credentials gets a diagnostic, not a
    crash."""
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


def _numeric_cells(leaf) -> list[float]:
    """Every finite numeric cell in a leaf's data_table (bools excluded)."""
    out: list[float] = []
    for row in (leaf.data_table or []):
        for v in row.values():
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float)) and math.isfinite(v):
                out.append(float(v))
    return out


def _aggregate_cells(leaf) -> list[float]:
    """Finite numeric cells found specifically under aggregate columns —
    the actual answer to an AVG/MAX/MIN/MEDIAN/COUNT question."""
    out: list[float] = []
    for row in (leaf.data_table or []):
        for col in _AGG_COLUMNS:
            v = row.get(col)
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float)) and math.isfinite(v):
                out.append(float(v))
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


@pytest.fixture(autouse=True)
def _reset_pipeline_state():
    """Keep each turn independent: clear conversation history and the
    resolver's MCP lru_caches before/after every test (mirrors the dashboard
    suite's hermeticity guard)."""
    from src.conversational import concept_resolver as cr

    cr._cached_icd_autocode.cache_clear()
    cr._cached_mimic_itemid_search.cache_clear()
    yield
    cr._cached_icd_autocode.cache_clear()
    cr._cached_mimic_itemid_search.cache_clear()


@pytest.fixture(autouse=True)
def _clear_history(request):
    """Clear the module-scoped pipeline's history before each test that uses
    it, so prior turns don't leak into decomposition."""
    if "bq_pipeline" in request.fixturenames:
        pipe = request.getfixturevalue("bq_pipeline")
        pipe.conversation_history.clear()
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

    What we EXPECTED to fail (brand grounding) but didn't: the decomposer LLM
    self-substitutes the generic — it emits ``furosemide`` for "Lasix", so the
    compiled ``pr.drug ILIKE '%furosemide%'`` finds the full cohort, not the
    5 literal-"lasix" rows. Brand→generic is handled upstream; no fix needed
    there.

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
