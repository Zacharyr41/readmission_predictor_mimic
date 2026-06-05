"""Template-based SQL extraction driven by CompetencyQuestion.

Supports two backends:
- DuckDB (local): ``extract(db_path, cq)``
- BigQuery (remote): ``extract_bigquery(cq, project=...)``
"""

from __future__ import annotations

import logging
import re
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

import duckdb

from src.conversational.models import (
    ClinicalConcept,
    CompetencyQuestion,
    ExtractionConfig,
    ExtractionResult,
    PatientFilter,
    TemporalConstraint,
)

if TYPE_CHECKING:
    from google.cloud import bigquery

    from src.conversational.concept_resolver import ConceptResolver

logger = logging.getLogger(__name__)

# BigQuery fully-qualified table mapping (physionet-data MIMIC-IV v3.1)
_BQ_SOURCE = "physionet-data"
_BQ_TABLES = {
    "patients": f"{_BQ_SOURCE}.mimiciv_3_1_hosp.patients",
    "admissions": f"{_BQ_SOURCE}.mimiciv_3_1_hosp.admissions",
    "icustays": f"{_BQ_SOURCE}.mimiciv_3_1_icu.icustays",
    "labevents": f"{_BQ_SOURCE}.mimiciv_3_1_hosp.labevents",
    "d_labitems": f"{_BQ_SOURCE}.mimiciv_3_1_hosp.d_labitems",
    "chartevents": f"{_BQ_SOURCE}.mimiciv_3_1_icu.chartevents",
    "d_items": f"{_BQ_SOURCE}.mimiciv_3_1_icu.d_items",
    "prescriptions": f"{_BQ_SOURCE}.mimiciv_3_1_hosp.prescriptions",
    "diagnoses_icd": f"{_BQ_SOURCE}.mimiciv_3_1_hosp.diagnoses_icd",
    "d_icd_diagnoses": f"{_BQ_SOURCE}.mimiciv_3_1_hosp.d_icd_diagnoses",
    "microbiologyevents": f"{_BQ_SOURCE}.mimiciv_3_1_hosp.microbiologyevents",
}


def _get_bigquery_module():
    """Lazy import of google.cloud.bigquery (patchable for tests)."""
    from google.cloud import bigquery as bq

    return bq


# Exception *type names* that signal a transport/connection failure where the
# right recovery is a fresh client (new connection pool), not more polling on
# the same poisoned session. OSError subclasses — ssl.SSLError, socket errors,
# builtin ConnectionError/TimeoutError, and requests' RequestException (which
# subclasses OSError) — are matched by isinstance below; this name set covers
# the google-api-core wrappers that are *not* OSError subclasses.
_TRANSIENT_TRANSPORT_ERROR_NAMES = frozenset({
    "RetryError",
    "ServiceUnavailable",
    "ServerError",
    "InternalServerError",
    "BadGateway",
    "GatewayTimeout",
    "TooManyRequests",
    "ConnectionError",
    "SSLError",
    "Timeout",
    "ReadTimeout",
    "ConnectTimeout",
    "ChunkedEncodingError",
    "ProtocolError",
    "MaxRetryError",
})


def _is_transient_transport_error(exc: BaseException) -> bool:
    """True when ``exc`` (or any error in its ``__cause__`` / ``__context__``
    chain) is a network/transport failure where reconnecting with a fresh
    BigQuery client is the correct recovery.

    Targets the intermittent macOS ``SSL_ERROR_SYSCALL`` / ``[SYS] unknown
    error (_ssl.c:2427)`` raised when the client reuses a stale keep-alive
    socket, plus the google-api-core 5xx / retry-deadline wrappers. Genuine
    query errors (BadRequest, NotFound, Forbidden, SQL syntax) are deliberately
    *not* matched, so a real failure surfaces immediately instead of being
    masked by pointless retries.
    """
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, OSError):
            return True
        if type(cur).__name__ in _TRANSIENT_TRANSPORT_ERROR_NAMES:
            return True
        cur = cur.__cause__ or cur.__context__
    return False


def _bq_array_element_type(values) -> str:
    """BigQuery element type for an array parameter, inferred from the first
    element. Our array binds are hadm_id lists (INT64); empty → INT64 default."""
    if not values:
        return "INT64"
    first = values[0]
    if isinstance(first, float):
        return "FLOAT64"
    if isinstance(first, int):  # bool is an int subclass — INT64 domain
        return "INT64"
    return "STRING"


# ---------------------------------------------------------------------------
# Backend abstraction
# ---------------------------------------------------------------------------


class _DuckDBBackend:
    """Thin wrapper around a read-only DuckDB connection.

    Phase 7b: supports concurrent ``execute`` from multiple threads by
    giving each thread its own cursor via ``self._conn.cursor()``. DuckDB
    connections are thread-safe for opening cursors but NOT for direct
    concurrent ``execute`` calls on the same connection object — without
    this, parallel extraction deadlocks on DuckDB's internal mutex.
    """

    def __init__(self, db_path: Path) -> None:
        self._conn = duckdb.connect(str(db_path), read_only=True)
        # Thread-local cursor storage. Each worker thread gets its own
        # cursor on first ``execute`` call; cursors are created lazily so
        # single-threaded callers pay no cost.
        self._local = threading.local()

    def _cursor(self):
        # Lazy-init the thread-local storage so test subclasses that bypass
        # __init__ (e.g. to wrap an existing connection) still work.
        local = getattr(self, "_local", None)
        if local is None:
            local = threading.local()
            self._local = local
        cur = getattr(local, "cursor", None)
        if cur is None:
            cur = self._conn.cursor()
            local.cursor = cur
        return cur

    def table(self, name: str) -> str:
        return name

    def execute(self, sql: str, params: list) -> list[tuple]:
        return self._cursor().execute(sql, params).fetchall()

    def execute_tolerant(self, sql: str, params: list) -> list[tuple]:
        """No validator on DuckDB (local; no cost). Passthrough to execute()."""
        return self.execute(sql, params)

    @staticmethod
    def ilike(column: str) -> str:
        """Case-insensitive LIKE expression (DuckDB supports ILIKE)."""
        return f"{column} ILIKE ?"

    @staticmethod
    def random_fn() -> str:
        return "RANDOM()"

    def readmission_labels_expr(self) -> str:
        """Return a table expression for readmission labels.

        Uses the materialized table if it exists, otherwise computes on the fly.
        Route through ``self._cursor()`` so concurrent threads don't collide.
        """
        try:
            self._cursor().execute("SELECT 1 FROM readmission_labels LIMIT 0")
            return "readmission_labels"
        except Exception:
            return self._readmission_cte()

    def _readmission_cte(self) -> str:
        t = self.table
        return f"""(
            SELECT
                hadm_id,
                CASE WHEN LEAD(admittime) OVER (
                         PARTITION BY subject_id ORDER BY admittime
                     ) <= dischtime + INTERVAL 30 DAY
                     THEN 1 ELSE 0 END AS readmitted_30d,
                CASE WHEN LEAD(admittime) OVER (
                         PARTITION BY subject_id ORDER BY admittime
                     ) <= dischtime + INTERVAL 60 DAY
                     THEN 1 ELSE 0 END AS readmitted_60d
            FROM {t('admissions')}
        )"""

    def close(self) -> None:
        self._conn.close()


class ValidatorBlockedQueryError(Exception):
    """Raised by ``_BigQueryBackend.execute`` when the injected pre-execution
    validator returns a ``block`` verdict.

    The orchestrator (or extraction wrappers) catch this and convert it
    into a structured warning on the AnswerResult — the goal is to skip
    the BigQuery scan without crashing the turn.
    """

    def __init__(self, verdict, sql: str = "") -> None:
        self.verdict = verdict
        self.sql = sql
        super().__init__(
            f"pre-execution validator blocked query: {getattr(verdict, 'concern', '?')}"
        )


class _BigQueryBackend:
    """Queries physionet-data MIMIC-IV datasets on BigQuery directly.

    Phase E hardening:
    - Optional ``validator`` callable runs before every ``execute()``;
      blocks raise ``ValidatorBlockedQueryError``. The orchestrator
      injects ``validate_sql_deterministic`` bound to the bq-validator
      MCP client.
    - ``maximum_bytes_billed`` kill-switch is set on every job config
      (defaults to 10 GiB) — fails the query at no charge if exceeded.
      Belt-and-suspenders for the case where the validator misjudges.
    """

    # A list-valued bind compiles to a single ArrayQueryParameter
    # (``IN UNNEST(@p)``), so callers can filter by a large id set with one
    # parameter instead of one-per-id. See ``_convert_params`` / the cohort
    # feature-fetch (``src.similarity.run._in_list_clause``).
    supports_array_params = True

    DEFAULT_MAX_BYTES_BILLED = 10 * 1024**3  # 10 GiB
    # Bounded per-attempt timeouts + reconnect retry so a stale-socket SSL blip
    # fails fast and self-heals, instead of burning google-api-core's ~600s
    # default retry deadline on a poisoned session (a ~20-min hang). Tunable.
    # Job *creation* (the POST to /jobs) is where the observed SSL hang lived,
    # so it gets a short bound — that's the lever that kills the ~20-min hang.
    DEFAULT_QUERY_TIMEOUT = 60.0      # seconds for the job-creation HTTP request
    # Job *completion* must be generous: a heavy cohort feature-fetch (labevents
    # join over the candidate pool) can legitimately run a few minutes. Too
    # tight a bound would falsely treat a healthy slow query as a stuck socket
    # and retry it. 600s comfortably covers real execution time.
    DEFAULT_RESULT_TIMEOUT = 600.0    # seconds to wait for the job to finish
    DEFAULT_MAX_ATTEMPTS = 3
    DEFAULT_RECONNECT_BACKOFF = 0.5   # base seconds; exponential per retry

    def __init__(
        self,
        project: str | None = None,
        *,
        validator=None,
        max_bytes_billed: int | None = DEFAULT_MAX_BYTES_BILLED,
        query_timeout: float = DEFAULT_QUERY_TIMEOUT,
        result_timeout: float = DEFAULT_RESULT_TIMEOUT,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        reconnect_backoff: float = DEFAULT_RECONNECT_BACKOFF,
    ) -> None:
        bq = _get_bigquery_module()
        self._project = project
        self._bq = bq
        self._client: bigquery.Client = bq.Client(project=project)
        self._validator = validator
        self._max_bytes_billed = max_bytes_billed
        self._query_timeout = query_timeout
        self._result_timeout = result_timeout
        self._max_attempts = max_attempts
        self._reconnect_backoff = reconnect_backoff
        # Per-session log of pre-execution blocks. The orchestrator drains
        # this after extraction to surface a structured warning to the user.
        self.blocked_queries: list[dict] = []

    def table(self, name: str) -> str:
        return f"`{_BQ_TABLES[name]}`"

    def execute(self, sql: str, params: list) -> list[tuple]:
        if self._validator is not None:
            try:
                verdict = self._validator(sql, params)
            except Exception:  # noqa: BLE001
                # Validator infrastructure failure: log and proceed.
                # Cost-of-validation insurance.
                verdict = None
            if verdict is not None and getattr(verdict, "verdict", None) == "block":
                raise ValidatorBlockedQueryError(verdict, sql=sql)
        bq_sql, bq_params = self._convert_params(sql, params)
        config = self._bq.QueryJobConfig(
            query_parameters=bq_params,
            maximum_bytes_billed=self._max_bytes_billed,
        )
        return self._run_with_reconnect(bq_sql, config)

    def _run_with_reconnect(self, bq_sql: str, config) -> list[tuple]:
        """Run the query with a bounded per-attempt timeout, reconnecting with
        a fresh client on a transport error.

        The long-lived client's connection pool can hold a stale keep-alive
        socket; reusing it raises the intermittent macOS SSL_ERROR_SYSCALL
        (``[SYS] unknown error``). google-api-core would retry that on the same
        poisoned session until its ~600s deadline elapses (a ~20-min hang), so
        we disable its retry and own the loop here: on a transport error we
        build a fresh client (new pool) and try again, which lets the blip
        self-heal. Genuine query errors are re-raised immediately.
        """
        last_exc: BaseException | None = None
        for attempt in range(self._max_attempts):
            try:
                job = self._client.query(
                    bq_sql,
                    job_config=config,
                    timeout=self._query_timeout,
                    retry=None,
                )
                rows = job.result(timeout=self._result_timeout)
                return [tuple(row.values()) for row in rows]
            except Exception as exc:  # noqa: BLE001
                if not _is_transient_transport_error(exc):
                    raise
                last_exc = exc
                is_last = attempt + 1 >= self._max_attempts
                logger.warning(
                    "BigQuery transport error (attempt %d/%d): %s%s",
                    attempt + 1,
                    self._max_attempts,
                    exc,
                    "" if is_last else " — reconnecting with a fresh client",
                )
                if is_last:
                    break
                self._reconnect()
                if self._reconnect_backoff:
                    time.sleep(self._reconnect_backoff * (2 ** attempt))
        assert last_exc is not None
        raise last_exc

    def _reconnect(self) -> None:
        """Discard the current client (its pool may hold a dead keep-alive
        socket) and build a fresh one. The new connection pool is what lets a
        transient SSL/transport failure self-heal on the next attempt."""
        try:
            self._client.close()
        except Exception:  # noqa: BLE001 — best-effort; the socket may be dead
            pass
        self._client = self._bq.Client(project=self._project)

    def execute_tolerant(self, sql: str, params: list) -> list[tuple]:
        """Like ``execute`` but on a pre-execution block, log to
        ``blocked_queries`` and return ``[]`` instead of raising.

        Used by extractor.py call sites where a single blocked extraction
        step should not abort the entire turn — the caller continues with
        an empty data slice and the orchestrator surfaces a structured
        warning."""
        try:
            return self.execute(sql, params)
        except ValidatorBlockedQueryError as exc:
            self.blocked_queries.append({
                "sql_preview": sql[:200],
                "concern": getattr(exc.verdict, "concern", None),
                "stage": getattr(exc.verdict, "reference_used", None),
                "estimated_usd": getattr(exc.verdict, "estimated_usd", None),
                "bytes_processed": getattr(exc.verdict, "bytes_processed", None),
            })
            return []

    @staticmethod
    def ilike(column: str) -> str:
        """Case-insensitive LIKE for BigQuery (no ILIKE support)."""
        return f"LOWER({column}) LIKE LOWER(?)"

    @staticmethod
    def random_fn() -> str:
        return "RAND()"

    def readmission_labels_expr(self) -> str:
        """Compute readmission labels on the fly using a window function."""
        t = self.table
        return f"""(
            SELECT
                hadm_id,
                CASE WHEN LEAD(admittime) OVER (
                         PARTITION BY subject_id ORDER BY admittime
                     ) <= DATETIME_ADD(dischtime, INTERVAL 30 DAY)
                     THEN 1 ELSE 0 END AS readmitted_30d,
                CASE WHEN LEAD(admittime) OVER (
                         PARTITION BY subject_id ORDER BY admittime
                     ) <= DATETIME_ADD(dischtime, INTERVAL 60 DAY)
                     THEN 1 ELSE 0 END AS readmitted_60d
            FROM {t('admissions')}
        )"""

    def _convert_params(
        self, sql: str, params: list
    ) -> tuple[str, list]:
        """Replace each ``?`` with ``@p0, @p1, …`` and build typed parameters.

        Single left-to-right pass (split once on ``?``), so the cost is
        O(len(sql) + n_params) rather than the O(n²) of a fresh
        ``str.replace`` per parameter — the latter wedged the ~199k-id cohort
        feature-fetch. A list/tuple value binds as one ``ArrayQueryParameter``
        (used as ``IN UNNEST(@pN)``), keeping an N-id membership test to a
        single parameter instead of N scalars (which also exceeds BigQuery's
        parameter limit).
        """
        segments = sql.split("?")
        n = min(len(params), len(segments) - 1)
        bq_params = [self._build_param(f"p{i}", params[i]) for i in range(n)]
        out = [segments[0]]
        for i in range(n):
            out.append(f"@p{i}")
            out.append(segments[i + 1])
        # More ``?`` than params (shouldn't happen): leave the extras literal.
        for seg in segments[n + 1:]:
            out.append("?")
            out.append(seg)
        return "".join(out), bq_params

    def _build_param(self, name: str, val):
        """One typed BigQuery parameter. List/tuple → ArrayQueryParameter."""
        if isinstance(val, (list, tuple)):
            return self._bq.ArrayQueryParameter(
                name, _bq_array_element_type(val), list(val)
            )
        if isinstance(val, int):  # incl. bool, as before
            return self._bq.ScalarQueryParameter(name, "INT64", val)
        if isinstance(val, float):
            return self._bq.ScalarQueryParameter(name, "FLOAT64", val)
        return self._bq.ScalarQueryParameter(name, "STRING", str(val))

    def close(self) -> None:
        self._client.close()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract(
    db_path: Path,
    cq: CompetencyQuestion,
    config: ExtractionConfig | None = None,
    resolver: ConceptResolver | None = None,
) -> ExtractionResult:
    """Extract MIMIC-IV data from a local DuckDB database.

    Opens a read-only connection, builds template SQL from the structured
    question fields, and returns patients/admissions/ICU stays/events.
    """
    backend = _DuckDBBackend(db_path)
    try:
        return _extract(backend, cq, config=config, resolver=resolver)
    finally:
        backend.close()


def extract_bigquery(
    cq: CompetencyQuestion,
    project: str | None = None,
    config: ExtractionConfig | None = None,
    resolver: ConceptResolver | None = None,
) -> ExtractionResult:
    """Extract MIMIC-IV data directly from BigQuery.

    Queries the ``physionet-data`` MIMIC-IV datasets. Requires
    Application Default Credentials (``gcloud auth application-default login``).
    """
    backend = _BigQueryBackend(project)
    try:
        return _extract(backend, cq, config=config, resolver=resolver)
    finally:
        backend.close()


# ---------------------------------------------------------------------------
# Core extraction logic (backend-agnostic)
# ---------------------------------------------------------------------------


def _extract(
    backend: _DuckDBBackend | _BigQueryBackend,
    cq: CompetencyQuestion,
    config: ExtractionConfig | None = None,
    resolver: ConceptResolver | None = None,
) -> ExtractionResult:
    """Extract patient/admission/ICU/event rows for a CompetencyQuestion.

    Flow:
      1. Cohort query → ``hadm_ids`` (single round-trip, Phase 2; no LIMIT).
      2. Chunk ``hadm_ids`` into ``batch_size`` batches.
      3. Pre-resolve every clinical concept once (category → concrete names).
      4. Per batch run structural fetches + per-concept event fetches. Phase
         7b: batches run concurrently via a ThreadPoolExecutor
         (``max_concurrent_batches`` workers) since BigQuery and DuckDB are
         both fine with modest concurrency. Result merging is sequential so
         dedup maps are never touched from multiple threads.
    """
    cfg = config or ExtractionConfig()
    hadm_ids = _get_filtered_hadm_ids(backend, cq.patient_filters, config=cfg)
    if not hadm_ids:
        return ExtractionResult()

    batch_size = cfg.batch_size
    max_workers = cfg.max_concurrent_batches

    # Pre-resolve concept name expansion once — batch-independent. Resolver
    # state is initialised here on the main thread before workers read it,
    # so the lazy ``_load_category_map`` cache is populated without racing.
    resolved_concepts: list[tuple[str, ClinicalConcept]] = []
    for concept in cq.clinical_concepts:
        names = resolver.resolve(concept) if resolver else [concept.name]
        for name in names:
            resolved_concept = (
                ClinicalConcept(
                    name=name,
                    concept_type=concept.concept_type,
                    attributes=concept.attributes,
                )
                if name != concept.name
                else concept
            )
            resolved_concepts.append((concept.concept_type, resolved_concept))

    batches = list(_iter_hadm_batches(hadm_ids, batch_size))
    # Per-batch worker: runs all queries for its batch, returns a dict the
    # main thread merges. No shared mutable state inside the worker.
    def _run_batch(batch: list[int]) -> dict:
        out: dict = {
            "patients": _fetch_patients(backend, batch),
            "admissions": _fetch_admissions(backend, batch),
            "icu_stays": _fetch_icu_stays(backend, batch),
            "events": {},
        }
        for concept_type, rc in resolved_concepts:
            rows = _extract_concept(backend, rc, batch, cq.temporal_constraints)
            if rows:
                out["events"].setdefault(concept_type, []).extend(rows)
        return out

    # One ThreadPoolExecutor per call. BigQuery's Client and DuckDB's
    # Connection are thread-safe for concurrent query execution in the
    # patterns we use here (no shared cursor; each execute() holds its own).
    patients_by_subject: dict = {}
    admissions: list[dict] = []
    icu_stays: list[dict] = []
    events: dict[str, list[dict]] = {}

    if max_workers == 1 or len(batches) <= 1:
        # Sequential fallback — preserves pre-7b behaviour for debugging.
        batch_results = (_run_batch(b) for b in batches)
    else:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            batch_results = list(pool.map(_run_batch, batches))

    # Merge sequentially on the main thread — dedup maps are never
    # concurrently mutated.
    for out in batch_results:
        for p in out["patients"]:
            patients_by_subject.setdefault(p["subject_id"], p)
        admissions.extend(out["admissions"])
        icu_stays.extend(out["icu_stays"])
        for kind, rows in out["events"].items():
            events.setdefault(kind, []).extend(rows)

    return ExtractionResult(
        patients=list(patients_by_subject.values()),
        admissions=admissions,
        icu_stays=icu_stays,
        events=events,
    )


def merge_extractions(results: list[ExtractionResult]) -> ExtractionResult:
    """Union-with-dedup over multiple ExtractionResults.

    Phase 4.5: when a big question decomposes into multiple CompetencyQuestions,
    each is run through the extractor independently, and the resulting
    ExtractionResults are merged into one before the RDF graph is built. This
    way exactly ONE graph is constructed for the whole turn — much cheaper
    than building N graphs, and cross-CQ relationships (e.g. "this patient's
    labs and this patient's readmission flag live in the same graph") are
    preserved automatically.

    Dedup strategy per bucket:
      patients           → by ``subject_id``
      admissions         → by ``hadm_id``
      icu_stays          → by ``stay_id``
      events[biomarker]  → by ``labevent_id``     (MIMIC PK)
      events[microbio…]  → by ``microevent_id``   (MIMIC PK)
      events[vital]      → composite ``(stay_id, itemid, charttime)``
      events[drug]       → composite ``(hadm_id, drug, starttime)``
      events[diagnosis]  → composite ``(hadm_id, seq_num)``
      events[other]      → per-object identity (pessimistic fallback; no crash)

    The merged result is a fresh ExtractionResult whose lists do not alias
    the inputs. Mutating it does not affect any source.
    """
    if not results:
        return ExtractionResult()

    patients_by_id: dict = {}
    admissions_by_hadm: dict = {}
    icu_by_stay: dict = {}
    events_by_kind: dict[str, dict] = {}

    for r in results:
        for p in r.patients:
            patients_by_id.setdefault(p.get("subject_id"), p)
        for a in r.admissions:
            admissions_by_hadm.setdefault(a.get("hadm_id"), a)
        for s in r.icu_stays:
            icu_by_stay.setdefault(s.get("stay_id"), s)
        for kind, rows in r.events.items():
            bucket = events_by_kind.setdefault(kind, {})
            key_fn = _EVENT_DEDUP_KEYS.get(kind)
            for row in rows:
                if key_fn is None:
                    # Unknown event kind — keep each object distinct (fallback
                    # that prefers not-losing-rows over perfect dedup).
                    bucket[id(row)] = row
                else:
                    key = key_fn(row)
                    bucket.setdefault(key, row)

    return ExtractionResult(
        patients=list(patients_by_id.values()),
        admissions=list(admissions_by_hadm.values()),
        icu_stays=list(icu_by_stay.values()),
        events={kind: list(b.values()) for kind, b in events_by_kind.items()},
    )


_EVENT_DEDUP_KEYS: dict[str, "callable"] = {  # type: ignore[valid-type]
    # Primary-key types (single field).
    "biomarker": lambda r: r.get("labevent_id"),
    "microbiology": lambda r: r.get("microevent_id"),
    # Composite-key types (no single PK available from MIMIC tables).
    "vital": lambda r: (r.get("stay_id"), r.get("itemid"), r.get("charttime")),
    "drug": lambda r: (r.get("hadm_id"), r.get("drug"), r.get("starttime")),
    "diagnosis": lambda r: (r.get("hadm_id"), r.get("seq_num")),
}
"""Per-event-kind dedup-key extractors. Must match the primary keys the
MIMIC-IV source tables expose; adding a new event kind means adding an entry
here (and writing the corresponding extractor)."""


def _iter_hadm_batches(hadm_ids: list[int], batch_size: int):
    """Yield successive slices of ``hadm_ids`` of length ``batch_size``.

    The primary reason batching exists is to bound the width of the
    ``hadm_id IN (...)`` clause sent to the database (BigQuery caps at ~10k
    positional params, DuckDB has no hard limit but very wide IN lists slow
    planning). Final result shape must not depend on the batch size —
    that's exercised by ``test_biomarker_extraction_invariant_under_batch_size``.
    """
    for i in range(0, len(hadm_ids), batch_size):
        yield hadm_ids[i : i + batch_size]


# ---------------------------------------------------------------------------
# Patient filtering
# ---------------------------------------------------------------------------


def _get_filtered_hadm_ids(
    backend: _DuckDBBackend | _BigQueryBackend,
    filters: list[PatientFilter],
    config: ExtractionConfig | None = None,
) -> list[int]:
    """Return admission IDs matching all patient filters.

    Per-filter SQL is produced by the ``OperationRegistry`` (see
    ``src/conversational/operations.py``); this function owns only the
    outer SQL shell — patients JOIN prepend, ORDER BY, LIMIT. Unknown filter
    fields are skipped inside ``compile_filters`` (matching prior behaviour);
    the decomposer is responsible for catching them earlier and retrying.
    """
    from src.conversational.operations import (
        FilterCompileContext,
        get_default_registry,
    )

    cfg = config or ExtractionConfig()
    t = backend.table
    registry = get_default_registry()
    ctx = FilterCompileContext(backend=backend)

    # Warn about unknown fields — the legacy path logged this per field.
    supported = registry.supported_names("filter")
    for f in filters:
        if f.field not in supported:
            logger.warning("Ignoring unsupported patient filter field: %s", f.field)

    frag = registry.compile_filters(filters, ctx)

    joins: list[str] = list(frag.joins)
    if frag.needs_patients:
        joins.insert(0, f"JOIN {t('patients')} p ON a.subject_id = p.subject_id")

    join_sql = " ".join(joins)
    where_clause = f" WHERE {' AND '.join(frag.where)}" if frag.where else ""

    order = "admittime DESC" if cfg.cohort_strategy == "recent" else backend.random_fn()
    # Use a subquery so ORDER BY references a column visible after SELECT DISTINCT
    # (BigQuery rejects ORDER BY on columns not in the DISTINCT select list).
    # Phase 2: no LIMIT — downstream fetchers batch via cfg.batch_size.
    inner = (
        f"SELECT DISTINCT a.hadm_id, a.admittime"
        f" FROM {t('admissions')} a {join_sql}{where_clause}"
    )
    sql = f"SELECT hadm_id FROM ({inner}) sub ORDER BY {order}"
    return [r[0] for r in backend.execute_tolerant(sql, frag.params)]


# ---------------------------------------------------------------------------
# Temporal constraint helpers
# ---------------------------------------------------------------------------


def _parse_time_window(window: str) -> str:
    """Convert '48h', '7d', '30m' etc. to an INTERVAL literal."""
    match = re.match(
        r"(\d+)\s*(h(?:ours?)?|d(?:ays?)?|m(?:in(?:utes?)?)?)",
        window.lower().strip(),
    )
    if not match:
        raise ValueError(f"Cannot parse time window: {window}")
    value = match.group(1)
    unit_char = match.group(2)[0]
    unit_map = {"h": "HOUR", "d": "DAY", "m": "MINUTE"}
    return f"INTERVAL {value} {unit_map[unit_char]}"


def _temporal_sql(
    constraints: list[TemporalConstraint],
    time_col: str,
    hadm_col: str,
    backend: _DuckDBBackend | _BigQueryBackend,
) -> str:
    """Build SQL fragment for temporal constraints (empty string if none)."""
    icu_tbl = backend.table("icustays")
    parts: list[str] = []
    for tc in constraints:
        ref = tc.reference_event.lower()
        if "icu" not in ref:
            continue

        exists_prefix = (
            f"AND EXISTS (SELECT 1 FROM {icu_tbl} _icu "
            f"WHERE _icu.hadm_id = {hadm_col}"
        )

        if tc.relation == "during":
            parts.append(
                f"{exists_prefix} "
                f"AND {time_col} >= _icu.intime "
                f"AND {time_col} <= _icu.outtime)"
            )
        elif tc.relation == "within" and tc.time_window:
            interval = _parse_time_window(tc.time_window)
            parts.append(
                f"{exists_prefix} "
                f"AND {time_col} >= _icu.intime "
                f"AND {time_col} <= _icu.intime + {interval})"
            )
        elif tc.relation == "before":
            parts.append(f"{exists_prefix} AND {time_col} < _icu.intime)")
        elif tc.relation == "after":
            parts.append(f"{exists_prefix} AND {time_col} > _icu.outtime)")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# SQL helpers
# ---------------------------------------------------------------------------


def _in_clause(values: list[int]) -> tuple[str, list[int]]:
    """Return (placeholders, params) for an IN clause."""
    return ", ".join(["?"] * len(values)), list(values)


# ---------------------------------------------------------------------------
# Concept extraction dispatch
# ---------------------------------------------------------------------------


def _extract_concept(
    backend: _DuckDBBackend | _BigQueryBackend,
    concept: ClinicalConcept,
    hadm_ids: list[int],
    temporal: list[TemporalConstraint],
) -> list[dict]:
    """Dispatch to the appropriate handler by concept_type."""
    handlers = {
        "biomarker": _extract_biomarkers,
        "vital": _extract_vitals,
        "drug": _extract_drugs,
        "diagnosis": _extract_diagnoses,
        "microbiology": _extract_microbiology,
    }
    handler = handlers.get(concept.concept_type)
    if handler is None:
        return []
    if concept.concept_type == "biomarker":
        return handler(backend, concept, hadm_ids, temporal)
    return handler(backend, concept.name, hadm_ids, temporal)


# ---------------------------------------------------------------------------
# Per-concept-type extractors
# ---------------------------------------------------------------------------


def _extract_biomarkers(
    backend: _DuckDBBackend | _BigQueryBackend,
    concept: ClinicalConcept,
    hadm_ids: list[int],
    temporal: list[TemporalConstraint],
) -> list[dict]:
    t = backend.table
    name = concept.name
    ph, params = _in_clause(hadm_ids)
    tc_sql = _temporal_sql(temporal, "l.charttime", "l.hadm_id", backend)

    # Optional attribute filtering (e.g. fluid="Blood" for serum)
    attr_sql = ""
    attr_params: list[str] = []
    if concept.attributes:
        attr_clauses = [backend.ilike("d.fluid") for _ in concept.attributes]
        attr_sql = f"AND ({' OR '.join(attr_clauses)})"
        attr_params = [f"%{a}%" for a in concept.attributes]

    sql = f"""
        SELECT l.labevent_id, l.subject_id, l.hadm_id, l.itemid,
               l.charttime, d.label, d.fluid, d.category,
               l.valuenum, l.valueuom, l.ref_range_lower, l.ref_range_upper
        FROM {t('labevents')} l
        JOIN {t('d_labitems')} d ON l.itemid = d.itemid
        WHERE {backend.ilike('d.label')}
          AND l.valuenum IS NOT NULL
          AND l.hadm_id IN ({ph})
          {attr_sql}
          {tc_sql}
        ORDER BY l.charttime
    """
    cols = [
        "labevent_id", "subject_id", "hadm_id", "itemid", "charttime",
        "label", "fluid", "category", "valuenum", "valueuom",
        "ref_range_lower", "ref_range_upper",
    ]
    return [dict(zip(cols, r)) for r in backend.execute_tolerant(
        sql, [f"%{name}%"] + params + attr_params,
    )]


def _extract_vitals(
    backend: _DuckDBBackend | _BigQueryBackend,
    name: str,
    hadm_ids: list[int],
    temporal: list[TemporalConstraint],
) -> list[dict]:
    t = backend.table
    ph, params = _in_clause(hadm_ids)
    tc_sql = _temporal_sql(temporal, "c.charttime", "c.hadm_id", backend)
    sql = f"""
        SELECT c.stay_id, c.subject_id, c.hadm_id, c.itemid,
               c.charttime, d.label, d.category, c.valuenum
        FROM {t('chartevents')} c
        JOIN {t('d_items')} d ON c.itemid = d.itemid
        WHERE {backend.ilike('d.label')}
          AND c.valuenum IS NOT NULL
          AND c.hadm_id IN ({ph})
          {tc_sql}
        ORDER BY c.charttime
    """
    cols = [
        "stay_id", "subject_id", "hadm_id", "itemid",
        "charttime", "label", "category", "valuenum",
    ]
    return [dict(zip(cols, r)) for r in backend.execute_tolerant(sql, [f"%{name}%"] + params)]


def _extract_drugs(
    backend: _DuckDBBackend | _BigQueryBackend,
    name: str,
    hadm_ids: list[int],
    temporal: list[TemporalConstraint],
) -> list[dict]:
    t = backend.table
    ph, params = _in_clause(hadm_ids)
    tc_sql = _temporal_sql(temporal, "pr.starttime", "pr.hadm_id", backend)
    sql = f"""
        SELECT pr.hadm_id, pr.subject_id, pr.drug, pr.starttime,
               pr.stoptime, pr.dose_val_rx, pr.dose_unit_rx, pr.route
        FROM {t('prescriptions')} pr
        WHERE {backend.ilike('pr.drug')}
          AND pr.hadm_id IN ({ph})
          {tc_sql}
        ORDER BY pr.starttime
    """
    cols = [
        "hadm_id", "subject_id", "drug", "starttime",
        "stoptime", "dose_val_rx", "dose_unit_rx", "route",
    ]
    return [dict(zip(cols, r)) for r in backend.execute_tolerant(sql, [f"%{name}%"] + params)]


def _extract_diagnoses(
    backend: _DuckDBBackend | _BigQueryBackend,
    name: str,
    hadm_ids: list[int],
    temporal: list[TemporalConstraint] | None = None,
) -> list[dict]:
    t = backend.table
    ph, params = _in_clause(hadm_ids)
    sql = f"""
        SELECT di.hadm_id, di.subject_id, di.seq_num, di.icd_code,
               di.icd_version, dd.long_title
        FROM {t('diagnoses_icd')} di
        LEFT JOIN {t('d_icd_diagnoses')} dd
          ON di.icd_code = dd.icd_code AND di.icd_version = dd.icd_version
        WHERE ({backend.ilike('dd.long_title')} OR di.icd_code LIKE ?)
          AND di.hadm_id IN ({ph})
        ORDER BY di.seq_num
    """
    cols = [
        "hadm_id", "subject_id", "seq_num", "icd_code",
        "icd_version", "long_title",
    ]
    return [
        dict(zip(cols, r))
        for r in backend.execute_tolerant(sql, [f"%{name}%", f"{name}%"] + params)
    ]


def _extract_microbiology(
    backend: _DuckDBBackend | _BigQueryBackend,
    name: str,
    hadm_ids: list[int],
    temporal: list[TemporalConstraint],
) -> list[dict]:
    t = backend.table
    ph, params = _in_clause(hadm_ids)
    tc_sql = _temporal_sql(temporal, "m.charttime", "m.hadm_id", backend)
    sql = f"""
        SELECT m.microevent_id, m.subject_id, m.hadm_id, m.charttime,
               m.spec_type_desc, m.org_name
        FROM {t('microbiologyevents')} m
        WHERE ({backend.ilike('m.spec_type_desc')} OR {backend.ilike('m.org_name')})
          AND m.hadm_id IN ({ph})
          {tc_sql}
        ORDER BY m.charttime
    """
    cols = [
        "microevent_id", "subject_id", "hadm_id", "charttime",
        "spec_type_desc", "org_name",
    ]
    return [
        dict(zip(cols, r))
        for r in backend.execute_tolerant(sql, [f"%{name}%", f"%{name}%"] + params)
    ]


# ---------------------------------------------------------------------------
# Supporting context fetchers
# ---------------------------------------------------------------------------


def _fetch_patients(
    backend: _DuckDBBackend | _BigQueryBackend, hadm_ids: list[int]
) -> list[dict]:
    t = backend.table
    ph, params = _in_clause(hadm_ids)
    sql = f"""
        SELECT DISTINCT p.subject_id, p.gender, p.anchor_age
        FROM {t('patients')} p
        JOIN {t('admissions')} a ON p.subject_id = a.subject_id
        WHERE a.hadm_id IN ({ph})
        ORDER BY p.subject_id
    """
    cols = ["subject_id", "gender", "anchor_age"]
    return [dict(zip(cols, r)) for r in backend.execute_tolerant(sql, params)]


def _fetch_admissions(
    backend: _DuckDBBackend | _BigQueryBackend, hadm_ids: list[int]
) -> list[dict]:
    t = backend.table
    ph, params = _in_clause(hadm_ids)

    rl_expr = backend.readmission_labels_expr()
    sql = f"""
        SELECT a.hadm_id, a.subject_id, a.admittime, a.dischtime,
               a.admission_type, a.discharge_location,
               a.hospital_expire_flag,
               COALESCE(rl.readmitted_30d, 0) AS readmitted_30d,
               COALESCE(rl.readmitted_60d, 0) AS readmitted_60d
        FROM {t('admissions')} a
        LEFT JOIN {rl_expr} rl ON a.hadm_id = rl.hadm_id
        WHERE a.hadm_id IN ({ph})
        ORDER BY a.admittime
    """
    cols = [
        "hadm_id", "subject_id", "admittime", "dischtime",
        "admission_type", "discharge_location", "hospital_expire_flag",
        "readmitted_30d", "readmitted_60d",
    ]
    return [dict(zip(cols, r)) for r in backend.execute_tolerant(sql, params)]


def _fetch_icu_stays(
    backend: _DuckDBBackend | _BigQueryBackend, hadm_ids: list[int]
) -> list[dict]:
    t = backend.table
    ph, params = _in_clause(hadm_ids)
    sql = f"""
        SELECT stay_id, hadm_id, subject_id, intime, outtime, los
        FROM {t('icustays')}
        WHERE hadm_id IN ({ph})
        ORDER BY intime
    """
    cols = ["stay_id", "hadm_id", "subject_id", "intime", "outtime", "los"]
    return [dict(zip(cols, r)) for r in backend.execute_tolerant(sql, params)]
