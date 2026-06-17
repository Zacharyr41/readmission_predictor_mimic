"""Tests for the conversational data extractor."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pydantic import ValidationError

from src.conversational.extractor import (
    _BigQueryBackend,
    _DuckDBBackend,
    _fetch_admissions,
    _get_filtered_hadm_ids,
    extract,
    extract_bigquery,
)
from src.conversational.models import (
    ClinicalConcept,
    CompetencyQuestion,
    ExtractionConfig,
    PatientFilter,
    TemporalConstraint,
)


@pytest.fixture
def synthetic_db_path(tmp_path, synthetic_duckdb_with_events):
    """Return the file path to the synthetic DuckDB with all event tables."""
    synthetic_duckdb_with_events.close()
    return tmp_path / "test.duckdb"


class TestExtract:
    def test_biomarker_extraction(self, synthetic_db_path):
        cq = CompetencyQuestion(
            original_question="What are the creatinine values?",
            clinical_concepts=[
                ClinicalConcept(name="creatinine", concept_type="biomarker")
            ],
            scope="cohort",
        )
        result = extract(synthetic_db_path, cq)

        assert len(result.events["biomarker"]) == 3
        assert all(
            "creatinine" in e["label"].lower()
            for e in result.events["biomarker"]
        )
        assert len(result.patients) > 0
        assert len(result.admissions) > 0

    def test_biomarker_with_fluid_attribute(self, synthetic_db_path):
        """Attribute 'blood' filters biomarkers by d_labitems.fluid."""
        cq = CompetencyQuestion(
            original_question="serum creatinine",
            clinical_concepts=[
                ClinicalConcept(
                    name="creatinine", concept_type="biomarker",
                    attributes=["blood"],
                )
            ],
            scope="cohort",
        )
        result = extract(synthetic_db_path, cq)
        # All synthetic creatinine has fluid='Blood' → all 3 match
        assert len(result.events["biomarker"]) == 3

    def test_biomarker_attribute_filters_out_non_matching(self, synthetic_db_path):
        """Attribute that doesn't match any fluid returns empty."""
        cq = CompetencyQuestion(
            original_question="urine creatinine",
            clinical_concepts=[
                ClinicalConcept(
                    name="creatinine", concept_type="biomarker",
                    attributes=["urine"],
                )
            ],
            scope="cohort",
        )
        result = extract(synthetic_db_path, cq)
        # No synthetic creatinine has fluid='Urine'
        assert result.events.get("biomarker", []) == []

    def test_patient_filter_age(self, synthetic_db_path):
        cq = CompetencyQuestion(
            original_question="Creatinine for patients over 70",
            clinical_concepts=[
                ClinicalConcept(name="creatinine", concept_type="biomarker")
            ],
            patient_filters=[
                PatientFilter(field="age", operator=">", value="70")
            ],
            scope="cohort",
        )
        result = extract(synthetic_db_path, cq)

        # Patients over 70: subject 2 (72), subject 5 (80)
        assert len(result.events["biomarker"]) == 2
        subject_ids = {p["subject_id"] for p in result.patients}
        assert subject_ids == {2, 5}

    def test_temporal_constraint_within(self, synthetic_db_path):
        cq = CompetencyQuestion(
            original_question="Creatinine within 24 hours of ICU admission",
            clinical_concepts=[
                ClinicalConcept(name="creatinine", concept_type="biomarker")
            ],
            temporal_constraints=[
                TemporalConstraint(
                    relation="within",
                    reference_event="ICU admission",
                    time_window="24h",
                )
            ],
            scope="cohort",
        )
        result = extract(synthetic_db_path, cq)

        # Patient 1: charttime 20h after intime -> YES
        # Patient 2: charttime 24h after intime -> YES (inclusive)
        # Patient 5: charttime 34h after intime -> NO
        assert len(result.events["biomarker"]) == 2

    def test_empty_result(self, synthetic_db_path):
        cq = CompetencyQuestion(
            original_question="Show hemoglobin values",
            clinical_concepts=[
                ClinicalConcept(name="hemoglobin", concept_type="biomarker")
            ],
            scope="cohort",
        )
        result = extract(synthetic_db_path, cq)

        assert result.events.get("biomarker", []) == []

    def test_multiple_concepts(self, synthetic_db_path):
        cq = CompetencyQuestion(
            original_question="Show creatinine and heart rate",
            clinical_concepts=[
                ClinicalConcept(name="creatinine", concept_type="biomarker"),
                ClinicalConcept(name="heart rate", concept_type="vital"),
            ],
            scope="cohort",
        )
        result = extract(synthetic_db_path, cq)

        assert "biomarker" in result.events
        assert "vital" in result.events
        assert len(result.events["biomarker"]) == 3
        assert len(result.events["vital"]) == 3


# ---------------------------------------------------------------------------
# BigQuery backend tests (mocked — no real BQ needed)
# ---------------------------------------------------------------------------


def _make_mock_bq():
    """Create a mock google.cloud.bigquery module."""
    mock_bq = MagicMock()
    mock_bq.ScalarQueryParameter = MagicMock(
        side_effect=lambda n, t, v: (n, t, v)
    )
    # Array params surface as a distinguishable 4-tuple so tests can assert a
    # list-valued bind became ONE ArrayQueryParameter (not N scalar params).
    mock_bq.ArrayQueryParameter = MagicMock(
        side_effect=lambda n, t, v: ("array", n, t, list(v))
    )
    mock_bq.QueryJobConfig = MagicMock()
    return mock_bq


class TestBackendIlike:
    def test_duckdb_uses_ilike(self):
        assert "ILIKE" in _DuckDBBackend.ilike("col")

    def test_bigquery_uses_lower_like(self):
        result = _BigQueryBackend.ilike("col")
        assert "ILIKE" not in result
        assert "LOWER" in result
        assert "LIKE" in result


class TestBigQueryBackend:
    @patch("src.conversational.extractor._get_bigquery_module")
    def test_table_resolution(self, mock_get_bq):
        """Verify fully-qualified BigQuery table names."""
        mock_get_bq.return_value = _make_mock_bq()
        backend = _BigQueryBackend(project="test-project")

        assert backend.table("patients") == "`physionet-data.mimiciv_3_1_hosp.patients`"
        assert backend.table("icustays") == "`physionet-data.mimiciv_3_1_icu.icustays`"
        assert backend.table("labevents") == "`physionet-data.mimiciv_3_1_hosp.labevents`"
        assert backend.table("chartevents") == "`physionet-data.mimiciv_3_1_icu.chartevents`"
        backend.close()

    @patch("src.conversational.extractor._get_bigquery_module")
    def test_param_conversion(self, mock_get_bq):
        """Verify ? → @pN conversion and typed parameters."""
        mock_get_bq.return_value = _make_mock_bq()
        backend = _BigQueryBackend(project="test-project")

        sql = "SELECT * FROM t WHERE age > ? AND name ILIKE ?"
        converted_sql, bq_params = backend._convert_params(sql, [70, "%creatinine%"])

        assert "@p0" in converted_sql
        assert "@p1" in converted_sql
        assert "?" not in converted_sql
        assert bq_params[0] == ("p0", "INT64", 70)
        assert bq_params[1] == ("p1", "STRING", "%creatinine%")
        backend.close()

    @patch("src.conversational.extractor._get_bigquery_module")
    def test_list_param_becomes_single_array_query_parameter(self, mock_get_bq):
        """A list-valued bind compiles to ONE ArrayQueryParameter behind a single
        ``@pN`` (used as ``IN UNNEST(@pN)``). This is what lets an N-id IN-list be
        one array param instead of N scalar params — the latter blew past
        BigQuery's parameter limit and an O(n^2) per-? string rewrite on the
        ~199k-admission emergency cohort."""
        mock_get_bq.return_value = _make_mock_bq()
        backend = _BigQueryBackend(project="test-project")

        sql = "SELECT * FROM t WHERE hadm_id IN UNNEST(?)"
        converted_sql, bq_params = backend._convert_params(sql, [[1, 2, 3]])

        assert "UNNEST(@p0)" in converted_sql
        assert "?" not in converted_sql
        assert len(bq_params) == 1
        assert bq_params[0] == ("array", "p0", "INT64", [1, 2, 3])
        backend.close()

    @patch("src.conversational.extractor._get_bigquery_module")
    def test_mixed_scalar_and_array_params(self, mock_get_bq):
        """Scalars and a list bind side by side, each to its own ``@pN``."""
        mock_get_bq.return_value = _make_mock_bq()
        backend = _BigQueryBackend(project="test-project")

        sql = "SELECT * FROM t WHERE age > ? AND hadm_id IN UNNEST(?)"
        converted_sql, bq_params = backend._convert_params(sql, [70, [4, 5]])

        assert "@p0" in converted_sql
        assert "UNNEST(@p1)" in converted_sql
        assert "?" not in converted_sql
        assert bq_params[0] == ("p0", "INT64", 70)
        assert bq_params[1] == ("array", "p1", "INT64", [4, 5])
        backend.close()

    @patch("src.conversational.extractor._get_bigquery_module")
    def test_convert_params_maps_placeholders_left_to_right(self, mock_get_bq):
        """Pins the O(n) rewrite: N placeholders map to ``@p0``..``@p(N-1)`` in
        source order, each ``?`` replaced exactly once. (The prior implementation
        did a fresh ``sql.replace('?', ..., 1)`` per parameter — O(n^2) in the
        number of binds, which is what wedged the 199k-id cohort fetch.)"""
        mock_get_bq.return_value = _make_mock_bq()
        backend = _BigQueryBackend(project="test-project")

        n = 40
        sql = "SELECT " + ", ".join(["?"] * n)
        converted_sql, bq_params = backend._convert_params(sql, list(range(n)))

        assert "?" not in converted_sql
        assert len(bq_params) == n
        assert [p[0] for p in bq_params] == [f"p{i}" for i in range(n)]
        positions = [converted_sql.index(f"@p{i}") for i in range(n)]
        assert positions == sorted(positions)
        backend.close()

    @patch("src.conversational.extractor._get_bigquery_module")
    def test_extract_bigquery_generates_qualified_sql(self, mock_get_bq):
        """Full extract_bigquery call with mocked client verifies table names in SQL."""
        mock_bq = _make_mock_bq()
        mock_get_bq.return_value = mock_bq

        mock_client = mock_bq.Client.return_value
        mock_result = MagicMock()
        mock_result.result.return_value = []
        mock_client.query.return_value = mock_result

        cq = CompetencyQuestion(
            original_question="Show creatinine",
            clinical_concepts=[
                ClinicalConcept(name="creatinine", concept_type="biomarker")
            ],
            scope="cohort",
        )
        result = extract_bigquery(cq, project="test-project")

        # With no admissions returned, we get an empty result
        assert result.patients == []
        assert result.events == {}

        # Verify at least one query was sent to BigQuery
        assert mock_client.query.called
        first_sql = mock_client.query.call_args_list[0][0][0]
        assert "physionet-data.mimiciv_3_1_hosp.admissions" in first_sql


class TestBigQueryReconnect:
    """The long-lived BigQuery client's connection pool can hold a stale
    keep-alive socket (laptop sleep/wake, WiFi roam, Google idle-closing an
    idle connection). Reusing it raises the intermittent macOS
    ``SSL_ERROR_SYSCALL`` / ``[SYS] unknown error (_ssl.c:2427)``. google-api-core
    treats that as retryable and retries on the *same* poisoned session, so it
    never recovers — it just burns the full ~600s deadline (a ~20-min hang).

    These tests pin the fix: bound the per-attempt timeout AND, on a transport
    error, reconnect with a *fresh* client (new connection pool) so a
    stale-socket blip self-heals on retry instead of hanging.
    """

    @staticmethod
    def _fake_rows(*rows):
        out = []
        for r in rows:
            m = MagicMock()
            m.values.return_value = r
            out.append(m)
        return out

    @patch("src.conversational.extractor._get_bigquery_module")
    def test_query_uses_bounded_timeout_and_no_unbounded_retry(self, mock_get_bq):
        mock_bq = _make_mock_bq()
        mock_get_bq.return_value = mock_bq
        client = mock_bq.Client.return_value
        job = MagicMock()
        job.result.return_value = self._fake_rows((1,))
        client.query.return_value = job

        backend = _BigQueryBackend(project="test-project")
        backend.execute("SELECT 1", [])

        qkwargs = client.query.call_args.kwargs
        # a bounded per-request timeout is the core anti-hang guarantee
        assert qkwargs.get("timeout") is not None
        assert 0 < qkwargs["timeout"] < 600
        # google's own retry must be disabled so it cannot re-burn the ~600s
        # deadline on a poisoned session; our reconnect loop owns retries.
        assert qkwargs.get("retry") is None
        rkwargs = job.result.call_args.kwargs
        assert rkwargs.get("timeout") is not None
        assert rkwargs["timeout"] > 0
        backend.close()

    @patch("src.conversational.extractor._get_bigquery_module")
    def test_recovers_from_transient_transport_error(self, mock_get_bq):
        import ssl

        mock_bq = _make_mock_bq()
        mock_get_bq.return_value = mock_bq

        client1 = MagicMock()
        client2 = MagicMock()
        mock_bq.Client.side_effect = [client1, client2]
        client1.query.side_effect = ssl.SSLError(
            5, "[SYS] unknown error (_ssl.c:2427)"
        )
        job = MagicMock()
        job.result.return_value = self._fake_rows((42,))
        client2.query.return_value = job

        backend = _BigQueryBackend(project="test-project", reconnect_backoff=0.0)
        rows = backend.execute("SELECT 1", [])

        assert rows == [(42,)]
        # a fresh client (new connection pool) was built for the retry
        assert mock_bq.Client.call_count == 2
        # the poisoned client was closed
        assert client1.close.called

    @patch("src.conversational.extractor._get_bigquery_module")
    def test_gives_up_after_max_attempts_without_hanging(self, mock_get_bq):
        import ssl

        mock_bq = _make_mock_bq()
        mock_get_bq.return_value = mock_bq

        clients = [MagicMock() for _ in range(3)]
        for c in clients:
            c.query.side_effect = ssl.SSLError(5, "[SYS] unknown error")
        mock_bq.Client.side_effect = clients

        backend = _BigQueryBackend(
            project="test-project", max_attempts=3, reconnect_backoff=0.0
        )
        with pytest.raises(ssl.SSLError):
            backend.execute("SELECT 1", [])

        # exactly max_attempts attempts → bounded, no infinite loop
        total_query_calls = sum(c.query.call_count for c in clients)
        assert total_query_calls == 3
        assert mock_bq.Client.call_count == 3

    @patch("src.conversational.extractor._get_bigquery_module")
    def test_non_transport_error_propagates_without_retry(self, mock_get_bq):
        mock_bq = _make_mock_bq()
        mock_get_bq.return_value = mock_bq
        client = mock_bq.Client.return_value
        client.query.side_effect = ValueError("bad sql syntax")

        backend = _BigQueryBackend(project="test-project")
        with pytest.raises(ValueError):
            backend.execute("SELECT bad", [])

        # a genuine query error must surface immediately, not be masked by
        # pointless reconnect-and-retry
        assert client.query.call_count == 1
        assert mock_bq.Client.call_count == 1  # no reconnect
        backend.close()


# ---------------------------------------------------------------------------
# execute_with_columns — used by the corrected-query re-run path
# ---------------------------------------------------------------------------


def _fake_result(schema: list[str], rows: list[tuple]):
    """A stand-in for a BigQuery ``RowIterator``: carries ``.schema``
    (objects with ``.name``) and is iterable, yielding rows with
    ``.values()``."""
    res = MagicMock()
    fields = []
    for n in schema:
        f = MagicMock()
        f.name = n  # MagicMock(name=...) sets the repr name, not .name
        fields.append(f)
    res.schema = fields
    row_mocks = []
    for r in rows:
        m = MagicMock()
        m.values.return_value = r
        row_mocks.append(m)
    res.__iter__.return_value = iter(row_mocks)
    return res


class TestExecuteWithColumns:
    """``execute_with_columns`` returns rows AND the result column names so the
    corrected-query re-run can build dicts for the answerer (today's
    ``execute`` returns only ``list[tuple]``)."""

    def test_duckdb_returns_rows_and_columns(self, tmp_path):
        import duckdb

        p = tmp_path / "t.duckdb"
        con = duckdb.connect(str(p))
        con.execute("CREATE TABLE admissions (hadm_id INTEGER, flag INTEGER)")
        con.execute("INSERT INTO admissions VALUES (1, 1), (2, 0), (3, 0)")
        con.close()

        backend = _DuckDBBackend(p)
        rows, cols = backend.execute_with_columns(
            "SELECT flag AS group_value, COUNT(*) AS count "
            "FROM admissions GROUP BY flag ORDER BY flag",
            [],
        )
        assert cols == ["group_value", "count"]
        assert rows == [(0, 2), (1, 1)]
        backend.close()

    @patch("src.conversational.extractor._get_bigquery_module")
    def test_bigquery_returns_rows_and_columns(self, mock_get_bq):
        mock_bq = _make_mock_bq()
        mock_get_bq.return_value = mock_bq
        client = mock_bq.Client.return_value
        job = MagicMock()
        job.result.return_value = _fake_result(
            schema=["group_value", "count"],
            rows=[("yes", 3), ("no", 5)],
        )
        client.query.return_value = job

        backend = _BigQueryBackend(project="test-project")
        rows, cols = backend.execute_with_columns("SELECT 1", [])

        assert cols == ["group_value", "count"]
        assert rows == [("yes", 3), ("no", 5)]
        backend.close()

    @patch("src.conversational.extractor._get_bigquery_module")
    def test_execute_still_returns_rows_only(self, mock_get_bq):
        """Regression: the refactor to share the reconnect loop must not change
        ``execute``'s rows-only contract (the 9 extractor call sites rely on
        ``list[tuple]``). A result with no ``.schema`` must not crash."""
        mock_bq = _make_mock_bq()
        mock_get_bq.return_value = mock_bq
        client = mock_bq.Client.return_value
        job = MagicMock()
        job.result.return_value = TestBigQueryReconnect._fake_rows((42,))  # plain list, no .schema
        client.query.return_value = job

        backend = _BigQueryBackend(project="test-project")
        assert backend.execute("SELECT 1", []) == [(42,)]
        backend.close()


# ---------------------------------------------------------------------------
# ExtractionConfig
# ---------------------------------------------------------------------------


class TestExtractionConfig:
    def test_default_values(self):
        cfg = ExtractionConfig()
        assert cfg.batch_size == 2000
        assert cfg.cohort_strategy == "recent"

    def test_custom_values(self):
        cfg = ExtractionConfig(batch_size=500, cohort_strategy="random")
        assert cfg.batch_size == 500
        assert cfg.cohort_strategy == "random"

    def test_invalid_strategy_rejected(self):
        with pytest.raises(ValidationError):
            ExtractionConfig(cohort_strategy="invalid")

    def test_max_cohort_size_no_longer_exists(self):
        """Phase 2 removed the artificial 500-row cap. Attempting to construct
        an ExtractionConfig with ``max_cohort_size=...`` must raise, so any
        stale call site surfaces loudly at test time rather than silently
        passing the kwarg through without effect."""
        with pytest.raises(ValidationError):
            ExtractionConfig(max_cohort_size=100)


# ---------------------------------------------------------------------------
# Readmission filters
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_db_path_with_readmission(tmp_path, synthetic_duckdb_with_events):
    """Extend synthetic DuckDB with readmission_labels table."""
    conn = synthetic_duckdb_with_events
    conn.execute("""
        CREATE TABLE readmission_labels (
            subject_id INTEGER,
            hadm_id INTEGER,
            dischtime TIMESTAMP,
            next_admittime TIMESTAMP,
            days_to_readmission INTEGER,
            readmitted_30d INTEGER,
            readmitted_60d INTEGER
        )
    """)
    # Patient 1, hadm 101: readmitted within 30d (next admit 26 days later)
    # Patient 1, hadm 102: not readmitted
    # Patient 2, hadm 103: readmitted within 60d but not 30d
    # Patient 3, hadm 104: not readmitted
    # Patient 4, hadm 105: not readmitted
    # Patient 5, hadm 106: not readmitted
    conn.execute("""
        INSERT INTO readmission_labels VALUES
        (1, 101, '2150-01-20 14:00:00', '2150-02-10 10:00:00', 21, 1, 1),
        (1, 102, '2150-02-15 12:00:00', NULL, NULL, 0, 0),
        (2, 103, '2151-03-10 16:00:00', '2151-04-25 08:00:00', 46, 0, 1),
        (3, 104, '2152-05-25 10:00:00', NULL, NULL, 0, 0),
        (4, 105, '2150-07-05 11:00:00', NULL, NULL, 0, 0),
        (5, 106, '2151-04-20 08:00:00', NULL, NULL, 0, 0)
    """)
    conn.close()
    return tmp_path / "test.duckdb"


class TestReadmissionFilters:
    def test_readmitted_30d_filter(self, synthetic_db_path_with_readmission):
        backend = _DuckDBBackend(synthetic_db_path_with_readmission)
        filters = [PatientFilter(field="readmitted_30d", operator="=", value="1")]
        hadm_ids = _get_filtered_hadm_ids(backend, filters)
        backend.close()
        # Only hadm 101 has readmitted_30d=1
        assert set(hadm_ids) == {101}

    def test_readmitted_60d_filter(self, synthetic_db_path_with_readmission):
        backend = _DuckDBBackend(synthetic_db_path_with_readmission)
        filters = [PatientFilter(field="readmitted_60d", operator="=", value="1")]
        hadm_ids = _get_filtered_hadm_ids(backend, filters)
        backend.close()
        # hadm 101 and 103 have readmitted_60d=1
        assert set(hadm_ids) == {101, 103}

    def test_readmitted_not_readmitted(self, synthetic_db_path_with_readmission):
        backend = _DuckDBBackend(synthetic_db_path_with_readmission)
        filters = [PatientFilter(field="readmitted_30d", operator="=", value="0")]
        hadm_ids = _get_filtered_hadm_ids(backend, filters)
        backend.close()
        assert set(hadm_ids) == {102, 103, 104, 105, 106}

    def test_unknown_filter_skipped_gracefully(self, synthetic_db_path_with_readmission):
        backend = _DuckDBBackend(synthetic_db_path_with_readmission)
        filters = [PatientFilter(field="invented_field", operator="=", value="x")]
        hadm_ids = _get_filtered_hadm_ids(backend, filters)
        backend.close()
        # Unknown filter skipped → all 6 admissions returned
        assert len(hadm_ids) == 6


# ---------------------------------------------------------------------------
# Admission readmission labels in fetch
# ---------------------------------------------------------------------------


class TestFetchAdmissionsIncludesReadmission:
    def test_admissions_contain_readmission_labels(
        self, synthetic_db_path_with_readmission,
    ):
        """_fetch_admissions must include readmitted_30d/60d from readmission_labels.

        Regression: previously these were always defaulted to False by
        _augment_admission, making comparison SPARQL queries return empty.
        """
        backend = _DuckDBBackend(synthetic_db_path_with_readmission)
        admissions = _fetch_admissions(backend, [101, 102, 103])
        backend.close()

        by_hadm = {a["hadm_id"]: a for a in admissions}
        # hadm 101: readmitted_30d=1, readmitted_60d=1
        assert by_hadm[101]["readmitted_30d"] == 1
        assert by_hadm[101]["readmitted_60d"] == 1
        # hadm 102: not readmitted
        assert by_hadm[102]["readmitted_30d"] == 0
        assert by_hadm[102]["readmitted_60d"] == 0
        # hadm 103: readmitted within 60d only
        assert by_hadm[103]["readmitted_30d"] == 0
        assert by_hadm[103]["readmitted_60d"] == 1

    def test_admissions_without_readmission_table_computes_on_fly(
        self, synthetic_db_path,
    ):
        """When readmission_labels table doesn't exist, compute from admissions.

        Patient 1, hadm 101 (dischtime 2150-01-20) has a next admission
        hadm 102 (admittime 2150-02-10 = 21 days later) → readmitted_30d=1.
        """
        backend = _DuckDBBackend(synthetic_db_path)
        admissions = _fetch_admissions(backend, [101])
        backend.close()

        assert admissions[0]["readmitted_30d"] == 1
        assert admissions[0]["readmitted_60d"] == 1


# ---------------------------------------------------------------------------
# Cohort capping
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Uncapped cohort / SQL compatibility
# ---------------------------------------------------------------------------


class TestUncappedCohort:
    """The 500-row cap was Phase 2's primary target. The cohort query now
    returns every admission that matches the filters; batching happens
    downstream in the fetchers."""

    def test_no_filters_returns_all_admissions(
        self, synthetic_db_path_with_readmission,
    ):
        backend = _DuckDBBackend(synthetic_db_path_with_readmission)
        hadm_ids = _get_filtered_hadm_ids(backend, [], config=ExtractionConfig())
        backend.close()
        # Synthetic fixture has 6 admissions, all returned.
        assert len(hadm_ids) == 6

    def test_no_limit_in_emitted_sql(self, synthetic_db_path_with_readmission):
        """Structural guard: the generated SQL must not contain LIMIT. Captured
        by intercepting backend.execute to inspect the query string before any
        execution. If LIMIT sneaks back in during a refactor, this trips."""
        backend = _DuckDBBackend(synthetic_db_path_with_readmission)
        seen_sql: list[str] = []
        original_execute = backend.execute

        def spy(sql, params):
            seen_sql.append(sql)
            return original_execute(sql, params)

        backend.execute = spy  # type: ignore[method-assign]
        _get_filtered_hadm_ids(backend, [], config=ExtractionConfig())
        backend.close()
        assert seen_sql, "expected at least one SQL execution"
        assert not any("LIMIT" in s for s in seen_sql), (
            f"LIMIT found in emitted SQL:\n{seen_sql!r}"
        )

    def test_order_by_admittime_still_compatible(
        self, synthetic_db_path_with_readmission,
    ):
        """ORDER BY admittime must still reference a column visible after
        SELECT DISTINCT — regression from the pre-cap era (BigQuery rejected
        it otherwise). Executing on DuckDB fails loud if the subquery shape
        breaks."""
        backend = _DuckDBBackend(synthetic_db_path_with_readmission)
        hadm_ids = _get_filtered_hadm_ids(backend, [], config=ExtractionConfig())
        backend.close()
        # All 6 returned, ordered by admittime DESC.
        assert len(hadm_ids) == 6


class TestBatchedExtraction:
    """Phase 2 introduced ``batch_size`` to bound IN-clause width. The
    extractor splits hadm_ids into chunks of ``batch_size`` and accumulates
    across chunks; final result must not depend on ``batch_size``."""

    def test_iter_hadm_batches_splits_correctly(self):
        from src.conversational.extractor import _iter_hadm_batches

        assert list(_iter_hadm_batches([], 3)) == []
        assert list(_iter_hadm_batches([1, 2, 3], 3)) == [[1, 2, 3]]
        assert list(_iter_hadm_batches([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]
        assert list(_iter_hadm_batches([1], 100)) == [[1]]

    def test_biomarker_extraction_invariant_under_batch_size(
        self, synthetic_db_path,
    ):
        """Same CQ, two different batch sizes → same rows (order-insensitive).

        This is the central property: batching is a performance knob that
        must not change results.
        """
        cq = CompetencyQuestion(
            original_question="creatinine values",
            clinical_concepts=[
                ClinicalConcept(name="creatinine", concept_type="biomarker"),
            ],
            scope="cohort",
        )
        small_batch = extract(
            synthetic_db_path, cq, config=ExtractionConfig(batch_size=2),
        )
        big_batch = extract(
            synthetic_db_path, cq, config=ExtractionConfig(batch_size=1000),
        )

        # Compare event content by ID, not by list identity.
        def _ids(events: list[dict]) -> set:
            return {e["labevent_id"] for e in events}

        assert _ids(small_batch.events["biomarker"]) == _ids(
            big_batch.events["biomarker"]
        )

    def test_patient_dedup_across_batches(self, synthetic_db_path):
        """A patient with multiple admissions might appear in more than one
        batch. The extractor must dedup so each subject_id surfaces once."""
        cq = CompetencyQuestion(
            original_question="all biomarkers",
            clinical_concepts=[
                ClinicalConcept(name="creatinine", concept_type="biomarker"),
            ],
            scope="cohort",
        )
        # Patient 1 has hadm 101 and 102. With batch_size=1 those land in
        # separate batches — the patient fetcher must still return patient 1
        # exactly once.
        result = extract(synthetic_db_path, cq, config=ExtractionConfig(batch_size=1))
        subject_ids = [p["subject_id"] for p in result.patients]
        assert len(subject_ids) == len(set(subject_ids))

    def test_admission_count_unchanged_across_batch_sizes(self, synthetic_db_path):
        """Admissions are unique by hadm_id; batching must preserve count."""
        cq = CompetencyQuestion(
            original_question="all biomarkers",
            clinical_concepts=[
                ClinicalConcept(name="creatinine", concept_type="biomarker"),
            ],
            scope="cohort",
        )
        for batch_size in (1, 2, 3, 1000):
            result = extract(
                synthetic_db_path, cq, config=ExtractionConfig(batch_size=batch_size),
            )
            hadms = {a["hadm_id"] for a in result.admissions}
            assert len(hadms) == len(result.admissions), (
                f"admission duplication at batch_size={batch_size}"
            )


# ---------------------------------------------------------------------------
# Phase 7b — parallel batched extraction
# ---------------------------------------------------------------------------


class TestExtractionConfigParallelism:
    def test_default_max_concurrent_batches(self):
        cfg = ExtractionConfig()
        assert cfg.max_concurrent_batches == 8

    def test_custom_max_concurrent_batches(self):
        cfg = ExtractionConfig(max_concurrent_batches=16)
        assert cfg.max_concurrent_batches == 16

    def test_invalid_negative_rejected(self):
        """Zero or negative concurrency is meaningless and would deadlock the
        executor. Pydantic's ``extra="forbid"`` catches typos, but we also
        validate the value range explicitly."""
        with pytest.raises(ValidationError):
            ExtractionConfig(max_concurrent_batches=0)


class TestParallelBatchedExtraction:
    """Phase 7b: the per-batch event fetches are the graph-path's dominant
    cost. Running them in a ThreadPoolExecutor gives 10-30× speed-up on
    BigQuery (latency-bound). Correctness invariant: parallel result must
    equal sequential result."""

    def _run_with_concurrency(self, db_path, cq, *, max_concurrent_batches: int):
        cfg = ExtractionConfig(
            batch_size=2,
            max_concurrent_batches=max_concurrent_batches,
        )
        return extract(db_path, cq, config=cfg)

    def test_sequential_equals_parallel_no_concept(self, synthetic_db_path):
        """Structural fetches only (patients + admissions + ICU stays) —
        parallel batches must produce the same set as sequential."""
        cq = CompetencyQuestion(
            original_question="lab-free query",
            clinical_concepts=[
                ClinicalConcept(name="creatinine", concept_type="biomarker"),
            ],
            scope="cohort",
        )
        serial = self._run_with_concurrency(synthetic_db_path, cq, max_concurrent_batches=1)
        parallel = self._run_with_concurrency(synthetic_db_path, cq, max_concurrent_batches=8)

        assert sorted(p["subject_id"] for p in serial.patients) == sorted(
            p["subject_id"] for p in parallel.patients
        )
        assert sorted(a["hadm_id"] for a in serial.admissions) == sorted(
            a["hadm_id"] for a in parallel.admissions
        )
        assert sorted(s["stay_id"] for s in serial.icu_stays) == sorted(
            s["stay_id"] for s in parallel.icu_stays
        )

    def test_sequential_equals_parallel_biomarker_events(self, synthetic_db_path):
        """Event fetches run in parallel too — creatinine event IDs across
        batches must match between sequential and parallel."""
        cq = CompetencyQuestion(
            original_question="creatinine",
            clinical_concepts=[
                ClinicalConcept(name="creatinine", concept_type="biomarker"),
            ],
            scope="cohort",
        )
        serial = self._run_with_concurrency(synthetic_db_path, cq, max_concurrent_batches=1)
        parallel = self._run_with_concurrency(synthetic_db_path, cq, max_concurrent_batches=8)

        serial_ids = {e["labevent_id"] for e in serial.events.get("biomarker", [])}
        parallel_ids = {e["labevent_id"] for e in parallel.events.get("biomarker", [])}
        assert serial_ids == parallel_ids

    def test_patient_dedup_correct_under_parallelism(self, synthetic_db_path):
        """A patient with multiple admissions may land in different batches.
        Under parallel execution, the dedup map mutation must not lose or
        duplicate entries. ``batch_size=1`` forces maximum batch count."""
        cq = CompetencyQuestion(
            original_question="all biomarkers",
            clinical_concepts=[
                ClinicalConcept(name="creatinine", concept_type="biomarker"),
            ],
            scope="cohort",
        )
        cfg = ExtractionConfig(batch_size=1, max_concurrent_batches=8)
        result = extract(synthetic_db_path, cq, config=cfg)
        subject_ids = [p["subject_id"] for p in result.patients]
        assert len(subject_ids) == len(set(subject_ids)), (
            f"patient dedup failed under parallelism: {subject_ids}"
        )


class TestConceptResolverThreadSafety:
    """The resolver's lazy category-map and SCTID indices must be safe to
    hit from multiple threads concurrently. First call populates the cache;
    subsequent calls read. Race conditions here would corrupt the cache."""

    def test_concurrent_resolve_is_deterministic(self):
        from concurrent.futures import ThreadPoolExecutor
        from pathlib import Path

        from src.conversational.concept_resolver import ConceptResolver

        mappings = Path(__file__).parent.parent.parent / "data" / "mappings"
        if not mappings.exists():
            pytest.skip("data/mappings/ not in this repo checkout")

        resolver = ConceptResolver(mappings_dir=mappings)
        concept = ClinicalConcept(name="antibiotics", concept_type="drug")

        # Hammer the resolver from 16 threads; every call must return the
        # same list (deterministic) without raising.
        with ThreadPoolExecutor(max_workers=16) as pool:
            results = list(pool.map(lambda _: resolver.resolve(concept), range(64)))

        baseline = results[0]
        assert len(baseline) > 5  # curated antibiotics category
        for r in results:
            assert r == baseline
