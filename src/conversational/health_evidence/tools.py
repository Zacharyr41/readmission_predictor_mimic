"""External evidence tools available to the EvidenceAgent.

Tools wrap external APIs and offline data registries (PubMed via NCBI
E-utilities, MIMIC offline distribution stats, LOINC catalog). The agent
invokes them via Anthropic's tool-use mechanism when its caller's
in-prompt knowledge isn't enough.

**Privacy boundary:** only model-generated query strings (and IDs the
model already knows) are passed to tools. Raw row data, PHI, and
admission-level details never leave the system. Verify in code review
that no new tool added here breaks this invariant.

**Tool-result envelope:** every tool function returns either::

    {"status": "ok", "results": [...]}

or::

    {"status": "unavailable", "error": "<message>"}

Tool functions never raise. The EvidenceAgent's loop relies on this
contract to keep failure handling simple.

**Result-size budget:** ``_MAX_TOOL_RESULT_BYTES`` (4 KB) caps the JSON-
serialized size of any tool result inserted into the agent's message
history, so the context doesn't explode on a verbose response.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared configuration
# ---------------------------------------------------------------------------


_NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_HTTP_TIMEOUT_SECONDS = 8.0  # leave headroom under agent's 30s budget
_MAX_RESULTS_CEILING = 5
_MAX_TOOL_RESULT_BYTES = 4096
_PER_RECORD_TITLE_BUDGET = 250

_LOINC_CODE_PATTERN = re.compile(r"^\d{1,7}-\d$")

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_MIMIC_DISTRIBUTIONS_PATH = (
    _REPO_ROOT / "data" / "processed" / "lab_distributions.json"
)
_DEFAULT_LOINC_CATALOG_PATH = (
    _REPO_ROOT / "data" / "ontology_cache" / "loinc_reference_ranges.json"
)


# ---------------------------------------------------------------------------
# Helpers (shared across tools)
# ---------------------------------------------------------------------------


def _api_key_param() -> dict[str, str]:
    """NCBI raises rate limits when an API key is supplied. Optional;
    deployments without one get 3 req/s, with one get 10 req/s."""
    key = os.environ.get("NCBI_API_KEY")
    return {"api_key": key} if key else {}


def _truncate_title(s: str) -> str:
    if len(s) <= _PER_RECORD_TITLE_BUDGET:
        return s
    return s[: _PER_RECORD_TITLE_BUDGET - 1] + "…"


def _enforce_size_budget(payload: dict) -> dict:
    """Ensure JSON-serialized payload fits in ``_MAX_TOOL_RESULT_BYTES``.
    Drops trailing records until under budget; if even one record is too big,
    aggressively truncates its title with a sentinel suffix."""
    serialized = json.dumps(payload).encode("utf-8")
    if len(serialized) <= _MAX_TOOL_RESULT_BYTES:
        return payload
    results = list(payload.get("results", []))
    while results:
        size = len(json.dumps({**payload, "results": results}).encode("utf-8"))
        if size <= _MAX_TOOL_RESULT_BYTES:
            break
        last = results[-1]
        title = last.get("title", "")
        if len(title) > 60:
            last["title"] = title[:60] + "…[truncated]"
            continue
        results.pop()
    return {**payload, "results": results}


# ---------------------------------------------------------------------------
# Shared MCP client cache + helper (used by all MCP-backed tools below).
# ---------------------------------------------------------------------------


# Lazy-cached MCP clients keyed by an opaque name. One persistent client
# per logical server; reused across calls so we don't pay subprocess /
# HTTP-handshake cost on every lookup. The pubmed backend keeps its own
# entries here under "pubmed:mcp_anthropic" / "pubmed:mcp_self_host" for
# the legacy 3-backend dispatch.
_MCP_CLIENTS: dict[str, Any] = {}

# Backward-compat alias for the original pubmed-only cache. Existing
# tests in test_pubmed_backends.py monkeypatch this name directly.
_PUBMED_MCP_CLIENTS = _MCP_CLIENTS


def _get_mcp_client(name: str, config_factory):
    """Lazy-fetch or create-and-cache an MCP client.

    ``config_factory``: zero-arg callable returning either an
    :class:`McpServerConfig` (caller wants to talk to a real server) or
    ``None`` (configuration unavailable — e.g., binary not on PATH,
    required env var missing). When the factory returns None, this
    function returns None so the calling tool can return a clean
    ``unavailable`` envelope WITHOUT raising.

    Each tool function should pass a stable ``name`` (e.g. "hermes",
    "omophub") so subsequent calls reuse the same persistent client.
    """
    client = _MCP_CLIENTS.get(name)
    if client is not None:
        return client
    try:
        config = config_factory()
    except Exception as exc:  # noqa: BLE001
        logger.warning("MCP config for %s failed: %s", name, exc)
        return None
    if config is None:
        return None
    try:
        from src.conversational.health_evidence.mcp_client import McpClient

        client = McpClient(config)
        _MCP_CLIENTS[name] = client
        return client
    except Exception as exc:  # noqa: BLE001
        logger.warning("MCP client init for %s failed: %s", name, exc)
        return None


# ---------------------------------------------------------------------------
# pubmed_search: 3-backend dispatch
# ---------------------------------------------------------------------------


_ANTHROPIC_PUBMED_MCP_URL = "https://pubmed.mcp.claude.com/mcp"


def pubmed_search(query: str, max_results: int = 5) -> dict[str, Any]:
    """Search PubMed. Backend selected by ``PUBMED_BACKEND`` env var:

    - ``"direct"`` (default) — calls NCBI E-utilities via ``requests``.
      No external MCP. PHI-safe (only the query string egresses).
    - ``"mcp_anthropic"`` — calls Anthropic's hosted PubMed MCP at
      https://pubmed.mcp.claude.com/mcp via streamable HTTP. PHI-safe
      (only the query string). NOT ZDR-eligible per Anthropic's
      Dec 2025 BAA, so do NOT use for any flow that has previously
      handled MIMIC content in the same conversation.
    - ``"mcp_self_host"`` — calls a self-hosted PubMed MCP at the URL
      in ``PUBMED_MCP_URL``. The audit-grade option for production.

    Returns the project's standard envelope on success / failure. Never
    raises.
    """
    backend = os.environ.get("PUBMED_BACKEND", "direct").lower()
    if backend == "mcp_anthropic":
        return _pubmed_via_mcp(
            query, max_results, backend_name="mcp_anthropic",
            url=_ANTHROPIC_PUBMED_MCP_URL,
        )
    if backend == "mcp_self_host":
        url = os.environ.get("PUBMED_MCP_URL")
        if not url:
            return {
                "status": "unavailable",
                "error": "PUBMED_BACKEND=mcp_self_host but PUBMED_MCP_URL is unset",
            }
        return _pubmed_via_mcp(
            query, max_results, backend_name="mcp_self_host", url=url,
        )
    return _pubmed_direct(query, max_results)


def _pubmed_via_mcp(
    query: str, max_results: int, *, backend_name: str, url: str,
) -> dict[str, Any]:
    """Call a remote PubMed MCP server. Lazy-builds the client on first
    use and caches it for the process lifetime."""
    try:
        retmax = max(1, min(int(max_results), _MAX_RESULTS_CEILING))
    except (TypeError, ValueError):
        retmax = 5

    def _config():
        from src.conversational.health_evidence.mcp_client import McpServerConfig

        return McpServerConfig(name=backend_name, transport="http", url=url)

    client = _get_mcp_client(backend_name, _config)
    if client is None:
        return {
            "status": "unavailable",
            "error": f"pubmed MCP client init failed for {backend_name}",
        }

    envelope = client.call_tool(
        "search",  # Anthropic-hosted PubMed exposes "search" + "fetch"
        {"query": query, "max_results": retmax},
        timeout=_HTTP_TIMEOUT_SECONDS,
    )
    if envelope.get("status") != "ok":
        return envelope

    # Normalize MCP results into our standard PubMed envelope shape.
    raw_results = envelope.get("results") or []
    normalized = []
    for r in raw_results[:retmax]:
        if not isinstance(r, dict):
            continue
        pmid = str(r.get("pmid") or r.get("id") or "")
        if not pmid:
            continue
        normalized.append({
            "pmid": pmid,
            "title": _truncate_title(str(r.get("title") or "")),
            "source": str(r.get("source") or r.get("journal") or ""),
            "pubdate": str(r.get("pubdate") or r.get("date") or ""),
            "url": str(
                r.get("url")
                or f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            ),
        })
    return _enforce_size_budget({"status": "ok", "results": normalized})


def _pubmed_direct(query: str, max_results: int = 5) -> dict[str, Any]:
    """Direct NCBI E-utilities backend (the v1 implementation).

    Returns ``{"status": "ok", "results": [{pmid, title, source, pubdate, url}, ...]}``
    on success, ``{"status": "unavailable", "error": str}`` on any failure.
    Never raises.
    """
    try:
        retmax = max(1, min(int(max_results), _MAX_RESULTS_CEILING))
    except (TypeError, ValueError):
        retmax = 5

    base_params: dict[str, Any] = {
        "db": "pubmed",
        "retmode": "json",
        **_api_key_param(),
    }

    # Step 1: esearch — get list of PMIDs.
    try:
        esearch_resp = requests.get(
            f"{_NCBI_BASE}/esearch.fcgi",
            params={**base_params, "term": query, "retmax": str(retmax)},
            timeout=_HTTP_TIMEOUT_SECONDS,
        )
        esearch_resp.raise_for_status()
        esearch_data = esearch_resp.json()
        idlist = (
            esearch_data.get("esearchresult", {}).get("idlist") or []
        )
    except requests.RequestException as exc:
        logger.warning("pubmed_search esearch failed: %s", exc)
        return {"status": "unavailable", "error": str(exc)}
    except (ValueError, KeyError, TypeError) as exc:
        logger.warning("pubmed_search esearch malformed: %s", exc)
        return {"status": "unavailable", "error": f"malformed esearch: {exc}"}

    if not idlist:
        return {"status": "ok", "results": []}

    # Step 2: esummary — fetch metadata for each PMID.
    try:
        esummary_resp = requests.get(
            f"{_NCBI_BASE}/esummary.fcgi",
            params={**base_params, "id": ",".join(idlist)},
            timeout=_HTTP_TIMEOUT_SECONDS,
        )
        esummary_resp.raise_for_status()
        esummary_data = esummary_resp.json()
        result_map = esummary_data.get("result")
        if not isinstance(result_map, dict):
            raise ValueError("missing 'result' key in esummary response")
    except requests.RequestException as exc:
        logger.warning("pubmed_search esummary failed: %s", exc)
        return {"status": "unavailable", "error": str(exc)}
    except (ValueError, KeyError, TypeError) as exc:
        logger.warning("pubmed_search esummary malformed: %s", exc)
        return {"status": "unavailable", "error": f"malformed esummary: {exc}"}

    records = []
    for pmid in idlist:
        meta = result_map.get(pmid) or {}
        records.append({
            "pmid": pmid,
            "title": _truncate_title(str(meta.get("title") or "")),
            "source": str(meta.get("source") or ""),
            "pubdate": str(meta.get("pubdate") or ""),
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        })

    payload = {"status": "ok", "results": records}
    return _enforce_size_budget(payload)


# ---------------------------------------------------------------------------
# mimic_distribution_lookup: offline cohort-stats registry
# ---------------------------------------------------------------------------


# Module-level so tests can monkeypatch.
MIMIC_DISTRIBUTIONS_PATH: Path = _DEFAULT_MIMIC_DISTRIBUTIONS_PATH


# On-the-fly cohort compute (Phase H Tier D — Inc 4). When the catalog
# doesn't have the requested (itemid, cohort) pair, the tool opens the
# MIMIC duckdb read-only and computes stats fresh. Module-level so tests
# can monkeypatch alongside MIMIC_DISTRIBUTIONS_PATH.
_DEFAULT_MIMIC_COMPUTE_DUCKDB_PATH = (
    _REPO_ROOT / "data" / "processed" / "mimiciv.duckdb"
)
MIMIC_COMPUTE_DUCKDB_PATH: Path = _DEFAULT_MIMIC_COMPUTE_DUCKDB_PATH

# Min-n threshold for on-the-fly compute. Same value the generator uses
# for its cohort-stratified buckets, so cache hits and compute hits are
# directly comparable.
_COMPUTE_MIN_N: int = 30


def mimic_distribution_lookup(
    itemid: int,
    cohort: str | None = None,
    icd10_prefixes: list[str] | None = None,
    icd9_prefixes: list[str] | None = None,
) -> dict[str, Any]:
    """Look up the empirical MIMIC distribution for a given lab/chart itemid.

    Reads from ``MIMIC_DISTRIBUTIONS_PATH``. Two schemas are accepted:

    * **Legacy flat** — ``{itemid: {n, mean, p50, p95, units}}``. The
      single record is treated as the unstratified bucket;
      ``cohort=None`` returns it tagged ``cohort="all"``,
      ``source="catalog"``.
    * **Nested (Tier D)** — ``{itemid: {cohort_name: {n, mean, p50,
      p95, units}}}``. ``cohort=None`` returns the ``"all"`` bucket;
      ``cohort="<name_or_alias>"`` resolves via the cohort registry
      and returns the stratified bucket.

    The ``cohort`` parameter accepts canonical cohort names (``"sepsis"``,
    ``"mi_acute"``) AND natural medical phrases (``"MI"``,
    ``"myocardial infarction"``, ``"heart attack"``) — they all
    resolve to the canonical record via
    :func:`src.conversational.health_evidence.cohorts.resolve_cohort_name`.
    The reserved name ``"all"`` always means the unstratified bucket.

    Result records carry ``itemid``, ``cohort``, ``source`` ("catalog"
    or "computed"), and the standard ``n``/``mean``/``p50``/``p95``/
    ``units`` stats. Returns ``unavailable`` on miss / file absent /
    malformed file / unknown cohort. Never raises.
    """
    try:
        itemid_int = int(itemid)
    except (TypeError, ValueError):
        return {"status": "unavailable", "error": f"invalid itemid: {itemid!r}"}

    # Tier D path 1: raw ICD prefixes — skip catalog entirely. Caller
    # is asking about an arbitrary cohort that may or may not have a
    # name. Compute on the fly. ``cohort=`` is ignored if also passed.
    has_raw_prefixes = bool(icd10_prefixes) or bool(icd9_prefixes)
    if has_raw_prefixes:
        stats, err = _compute_cohort_stats_on_the_fly(
            itemid_int,
            icd10_prefixes=icd10_prefixes,
            icd9_prefixes=icd9_prefixes,
        )
        if stats is None:
            return {"status": "unavailable", "error": err or "compute failed"}
        echoed = list(icd10_prefixes or []) + list(icd9_prefixes or [])
        return _stats_to_envelope(
            itemid_int, "custom", "computed", stats,
            icd_prefixes=echoed,
        )

    # Catalog read for the named-cohort and default paths.
    path = MIMIC_DISTRIBUTIONS_PATH
    if not path.exists():
        return {
            "status": "unavailable",
            "error": f"distribution registry not found at {path}",
        }
    try:
        registry = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        logger.warning("mimic_distribution_lookup read failed: %s", exc)
        return {
            "status": "unavailable",
            "error": f"registry read error: {exc}",
        }
    if not isinstance(registry, dict):
        return {
            "status": "unavailable",
            "error": "registry is not a JSON object",
        }
    candidate = registry.get(str(itemid_int))
    if not isinstance(candidate, dict):
        return {
            "status": "unavailable",
            "error": f"itemid {itemid_int} not in registry",
        }
    catalog_record: dict = candidate

    # Detect schema shape. Flat record has stat fields directly
    # (``n``, ``mean``, …); nested record's values are themselves dicts
    # (``{"all": {...}, "sepsis": {...}}``). Use ``"n"`` as the marker
    # since every legitimate flat record carries it.
    is_flat = bool(catalog_record) and "n" in catalog_record and not (
        isinstance(catalog_record.get("all"), dict)
        and "n" in (catalog_record.get("all") or {})
    )

    if is_flat:
        # Legacy flat schema — only the unstratified bucket is cached.
        if cohort is None or (
            isinstance(cohort, str) and cohort.strip().lower() == "all"
        ):
            return _stats_to_envelope(
                itemid_int, "all", "catalog", catalog_record,
            )
        # Cohort requested but flat catalog can't serve it → try L2.
        target = _resolve_target_cohort(cohort)
        if target is None or target == "all":
            return {
                "status": "unavailable",
                "error": _unknown_cohort_error(cohort),
            }
        return _l2_compute_named_cohort(itemid_int, target)

    # Nested schema (Tier D).
    requested = cohort  # original phrase for error messages
    target = _resolve_target_cohort(requested)
    if target is None:
        return {
            "status": "unavailable",
            "error": _unknown_cohort_error(requested),
        }

    stats = catalog_record.get(target)
    if isinstance(stats, dict):
        return _stats_to_envelope(itemid_int, target, "catalog", stats)

    # Catalog miss for a known cohort → L2 compute. Only attempt this
    # for non-``all`` cohorts (``all`` should always be in the catalog;
    # if it's not, the catalog is broken — don't paper over it).
    if target == "all":
        return {
            "status": "unavailable",
            "error": (
                f"itemid {itemid_int} catalog entry missing 'all' bucket"
            ),
        }
    return _l2_compute_named_cohort(itemid_int, target)


def _l2_compute_named_cohort(
    itemid: int, canonical_cohort_name: str,
) -> dict:
    """Resolve a registered cohort name to its ICD prefixes and run
    on-the-fly compute. Returns the standard tool envelope (ok or
    unavailable)."""
    from src.conversational.health_evidence.cohorts import load_cohorts

    cohorts = load_cohorts()
    defn = cohorts.get(canonical_cohort_name)
    if defn is None:
        # Should be unreachable (caller already resolved the name),
        # but defensive — never raise from a tool.
        return {
            "status": "unavailable",
            "error": f"cohort {canonical_cohort_name!r} not in registry",
        }
    stats, err = _compute_cohort_stats_on_the_fly(
        itemid,
        icd10_prefixes=defn.get("icd10_prefixes"),
        icd9_prefixes=defn.get("icd9_prefixes"),
    )
    if stats is None:
        return {"status": "unavailable", "error": err or "compute failed"}
    return _stats_to_envelope(itemid, canonical_cohort_name, "computed", stats)


def _resolve_target_cohort(cohort: str | None) -> str | None:
    """Map the caller's ``cohort=`` argument to a target bucket name.

    - ``None`` → ``"all"`` (default unstratified bucket).
    - ``"all"`` (any case) → ``"all"`` (reserved).
    - Any other phrase → resolved to canonical name via the registry's
      alias matcher; returns ``None`` if no match (the tool turns that
      into an "unknown cohort" error)."""
    if cohort is None:
        return "all"
    from src.conversational.health_evidence.cohorts import resolve_cohort_name
    if cohort.strip().lower() == "all":
        return "all"
    return resolve_cohort_name(cohort)


def _unknown_cohort_error(requested: str | None) -> str:
    """Build the helpful error message for an unrecognised cohort.
    Lists known canonical names and points at the icd_prefixes escape
    hatch."""
    from src.conversational.health_evidence.cohorts import known_cohort_names
    names = sorted(known_cohort_names())
    return (
        f"could not resolve cohort {requested!r}. "
        f"Known names: {names}. Each has aliases — try natural medical "
        "phrases like 'sepsis', 'myocardial infarction', 'heart failure'. "
        "For an arbitrary cohort, pass icd10_prefixes=[...] and/or "
        "icd9_prefixes=[...]."
    )


def _stats_to_envelope(
    itemid: int, cohort: str, source: str, stats: dict,
    *, icd_prefixes: list[str] | None = None,
) -> dict:
    """Pack a stats dict into the standard tool envelope. ``source`` is
    ``"catalog"`` for catalog hits, ``"computed"`` for on-the-fly
    results. When ``icd_prefixes`` is supplied (raw-prefix lookups),
    they are echoed back in the result record for audit-trail purposes.
    """
    record: dict[str, Any] = {
        "itemid": itemid,
        "cohort": cohort,
        "source": source,
        "n": stats.get("n"),
        "mean": stats.get("mean"),
        "p50": stats.get("p50"),
        "p95": stats.get("p95"),
        "units": stats.get("units"),
    }
    if icd_prefixes is not None:
        record["icd_prefixes"] = list(icd_prefixes)
    return _enforce_size_budget({"status": "ok", "results": [record]})


# ---------------------------------------------------------------------------
# On-the-fly cohort compute (Phase H Tier D — Inc 4)
# ---------------------------------------------------------------------------


# BigQuery fully-qualified MIMIC table identifiers. Mirrors
# ``extractor._BQ_TABLES`` so both the extractor and the cohort
# compute helper hit the same physionet-data dataset.
_BQ_DIAGNOSES_ICD = "`physionet-data.mimiciv_3_1_hosp.diagnoses_icd`"
_BQ_LABEVENTS = "`physionet-data.mimiciv_3_1_hosp.labevents`"
_BQ_CHARTEVENTS = "`physionet-data.mimiciv_3_1_icu.chartevents`"
_BQ_D_LABITEMS = "`physionet-data.mimiciv_3_1_hosp.d_labitems`"
_BQ_D_ITEMS = "`physionet-data.mimiciv_3_1_icu.d_items`"

# Path to the labitem→{loinc, label} mapping used for LOINC enrichment
# in mimic_itemid_search. Generator script (build_phase_h_catalogs.py)
# already loads this file; we share the constant so the two stay in sync.
_DEFAULT_LABITEM_TO_SNOMED_PATH = (
    _REPO_ROOT / "data" / "mappings" / "labitem_to_snomed.json"
)
LABITEM_TO_SNOMED_PATH: Path = _DEFAULT_LABITEM_TO_SNOMED_PATH

# Charset whitelist for analyte-search queries. Letters, digits, ASCII
# space, and basic punctuation that appears in lab/chart labels
# (period, comma, hyphen, slash). Anything else (LIKE wildcards %/_,
# quotes, semicolons, parentheses) is rejected before SQL composition
# as a security gate.
_VALID_SEARCH_QUERY_RE = re.compile(r"^[A-Za-z0-9 .,\-/]+$")
_MIN_SEARCH_QUERY_CHARS = 2
_DEFAULT_SEARCH_MAX_RESULTS = 5
_SEARCH_MAX_RESULTS_CEILING = 25


def _get_bigquery_module():
    """Lazy import of ``google.cloud.bigquery`` (patchable for tests).
    Mirrors ``extractor._get_bigquery_module`` so tests can mock with
    the same recipe."""
    from google.cloud import bigquery as bq

    return bq


def _compute_cohort_stats_on_the_fly(
    itemid: int,
    *,
    icd10_prefixes: list[str] | None,
    icd9_prefixes: list[str] | None,
    min_n: int = _COMPUTE_MIN_N,
) -> tuple[dict | None, str | None]:
    """Backend-aware on-the-fly compute. Returns ``(stats, error)``:

    - On success: ``({"n", "mean", "p50", "p95", "units"}, None)``.
    - On failure: ``(None, "<error message>")``.

    Backend is chosen by the ``DATA_SOURCE`` env var (``"local"`` /
    ``"bigquery"``) — same dispatch the rest of the conversational
    pipeline uses. The ``BIGQUERY_PROJECT`` env var is required when
    ``DATA_SOURCE="bigquery"``.

    Both paths share the ``build_cohort_subquery_sql`` validator —
    invalid ICD prefixes are rejected before any backend is opened.
    """
    data_source = os.environ.get("DATA_SOURCE", "local").lower()
    if data_source == "bigquery":
        return _compute_via_bigquery(
            itemid,
            icd10_prefixes=icd10_prefixes,
            icd9_prefixes=icd9_prefixes,
            min_n=min_n,
        )
    return _compute_via_duckdb(
        itemid,
        icd10_prefixes=icd10_prefixes,
        icd9_prefixes=icd9_prefixes,
        min_n=min_n,
    )


def _compute_via_duckdb(
    itemid: int,
    *,
    icd10_prefixes: list[str] | None,
    icd9_prefixes: list[str] | None,
    min_n: int,
) -> tuple[dict | None, str | None]:
    """Local DuckDB compute path. Returns ``(stats, error)``."""
    from src.conversational.health_evidence.cohorts import (
        build_cohort_subquery_sql,
    )

    if not MIMIC_COMPUTE_DUCKDB_PATH.exists():
        return None, (
            f"compute backend (MIMIC duckdb) not available at "
            f"{MIMIC_COMPUTE_DUCKDB_PATH}"
        )
    try:
        cohort_subquery = build_cohort_subquery_sql(
            icd10_prefixes=icd10_prefixes,
            icd9_prefixes=icd9_prefixes,
        )
    except ValueError as exc:
        return None, f"invalid prefix: {exc}"

    sql = f"""
        WITH cohort_hadms AS ({cohort_subquery}),
             vals AS (
                 SELECT valuenum, valueuom FROM labevents
                 WHERE itemid = ? AND valuenum IS NOT NULL
                   AND hadm_id IN (SELECT hadm_id FROM cohort_hadms)
                 UNION ALL
                 SELECT valuenum, valueuom FROM chartevents
                 WHERE itemid = ? AND valuenum IS NOT NULL
                   AND hadm_id IN (SELECT hadm_id FROM cohort_hadms)
             )
        SELECT COUNT(*) AS n,
               AVG(valuenum) AS mean,
               quantile_cont(valuenum, 0.5) AS p50,
               quantile_cont(valuenum, 0.95) AS p95,
               mode() WITHIN GROUP (ORDER BY valueuom) AS units
        FROM vals
    """
    try:
        import duckdb  # local import — kept off the module top because
        # health_evidence is otherwise duckdb-free in v1.

        con = duckdb.connect(str(MIMIC_COMPUTE_DUCKDB_PATH), read_only=True)
        try:
            row = con.execute(sql, [itemid, itemid]).fetchone()
        finally:
            con.close()
    except Exception as exc:  # noqa: BLE001 — never raise to caller
        logger.warning(
            "on-the-fly mimic_distribution_lookup duckdb failed: %s", exc,
        )
        return None, f"duckdb compute failed: {exc}"

    if row is None or row[0] is None or int(row[0]) < min_n:
        n = int(row[0]) if row and row[0] is not None else 0
        return None, (
            f"insufficient cohort data for itemid {itemid}: "
            f"n={n} (min {min_n})"
        )
    return {
        "n": int(row[0]),
        "mean": float(row[1]) if row[1] is not None else None,
        "p50": float(row[2]) if row[2] is not None else None,
        "p95": float(row[3]) if row[3] is not None else None,
        "units": str(row[4]) if row[4] is not None else "",
    }, None


def _compute_via_bigquery(
    itemid: int,
    *,
    icd10_prefixes: list[str] | None,
    icd9_prefixes: list[str] | None,
    min_n: int,
) -> tuple[dict | None, str | None]:
    """BigQuery compute path. Returns ``(stats, error)``."""
    from src.conversational.health_evidence.cohorts import (
        build_cohort_subquery_sql,
    )

    project = os.environ.get("BIGQUERY_PROJECT")
    if not project:
        return None, (
            "DATA_SOURCE=bigquery but BIGQUERY_PROJECT is unset — "
            "cannot run BigQuery compute"
        )
    try:
        cohort_subquery = build_cohort_subquery_sql(
            icd10_prefixes=icd10_prefixes,
            icd9_prefixes=icd9_prefixes,
            diagnoses_icd_table=_BQ_DIAGNOSES_ICD,
        )
    except ValueError as exc:
        return None, f"invalid prefix: {exc}"

    # BigQuery uses APPROX_QUANTILES instead of quantile_cont; ARRAY[2]
    # gives the median, ARRAY[19] gives p95 (when num_buckets=20).
    sql = f"""
        WITH cohort_hadms AS ({cohort_subquery}),
             vals AS (
                 SELECT valuenum, valueuom FROM {_BQ_LABEVENTS}
                 WHERE itemid = @itemid AND valuenum IS NOT NULL
                   AND hadm_id IN (SELECT hadm_id FROM cohort_hadms)
                 UNION ALL
                 SELECT valuenum, valueuom FROM {_BQ_CHARTEVENTS}
                 WHERE itemid = @itemid AND valuenum IS NOT NULL
                   AND hadm_id IN (SELECT hadm_id FROM cohort_hadms)
             )
        SELECT COUNT(*) AS n,
               AVG(valuenum) AS mean,
               APPROX_QUANTILES(valuenum, 100)[OFFSET(50)] AS p50,
               APPROX_QUANTILES(valuenum, 100)[OFFSET(95)] AS p95,
               APPROX_TOP_COUNT(valueuom, 1)[OFFSET(0)].value AS units
        FROM vals
    """
    try:
        bq = _get_bigquery_module()
        client = bq.Client(project=project)
        # ScalarQueryParameter — keep the @itemid placeholder safe even
        # though itemid is already an int. Tests mock the whole SDK so
        # the param-shape isn't enforced; production runs use the SDK's
        # real binding.
        try:
            params = [bq.ScalarQueryParameter("itemid", "INT64", itemid)]
            job_config = bq.QueryJobConfig(query_parameters=params)
            job = client.query(sql, job_config=job_config)
        except (AttributeError, TypeError):
            # Fallback for mocks that don't implement
            # QueryJobConfig/ScalarQueryParameter — just substitute the
            # itemid as a literal (safe since we already int()'d it).
            job = client.query(sql.replace("@itemid", str(int(itemid))))
        rows = list(job.result())
    except Exception as exc:  # noqa: BLE001 — never raise
        logger.warning(
            "on-the-fly mimic_distribution_lookup bigquery failed: %s", exc,
        )
        return None, f"bigquery compute failed: {exc}"

    if not rows:
        return None, f"itemid {itemid} returned no rows from bigquery"
    row = rows[0]
    n_val = getattr(row, "n", None)
    if n_val is None or int(n_val) < min_n:
        return None, (
            f"insufficient cohort data for itemid {itemid}: "
            f"n={int(n_val) if n_val is not None else 0} (min {min_n})"
        )
    return {
        "n": int(n_val),
        "mean": (
            float(row.mean) if getattr(row, "mean", None) is not None else None
        ),
        "p50": (
            float(row.p50) if getattr(row, "p50", None) is not None else None
        ),
        "p95": (
            float(row.p95) if getattr(row, "p95", None) is not None else None
        ),
        "units": (
            str(row.units) if getattr(row, "units", None) is not None else ""
        ),
    }, None


# ---------------------------------------------------------------------------
# loinc_reference_range: published-normal lookup
# ---------------------------------------------------------------------------


# Module-level so tests can monkeypatch.
LOINC_CATALOG_PATH: Path = _DEFAULT_LOINC_CATALOG_PATH


# ---------------------------------------------------------------------------
# Phase G — additional MCP-backed source-of-truth tools.
#
# Each tool follows the same shape:
#   1. Build (or reuse) an McpClient via _get_mcp_client; return the
#      "unavailable" envelope cleanly when configuration is missing.
#   2. Call the upstream MCP tool.
#   3. Normalise upstream results into a stable per-tool envelope shape
#      so callers don't have to handle server-specific field names.
#
# All five are PHI-safe egress: they only ever see model-generated query
# strings and ontology codes — never row data.
# ---------------------------------------------------------------------------


# --- snomed_search via Hermes (local stdio) --------------------------------


def _hermes_config():
    """Build a Hermes server config when the binary + DB are configured.

    Returns ``None`` when ``HERMES_MCP_COMMAND`` resolves to a binary
    that's not on PATH — the calling tool then returns ``unavailable``."""
    import shutil

    from src.conversational.health_evidence.mcp_client import McpServerConfig

    command = os.environ.get("HERMES_MCP_COMMAND", "hermes")
    db_path = os.environ.get(
        "HERMES_MCP_DB", "/var/lib/hermes/snomed.db",
    )
    if not shutil.which(command):
        return None
    return McpServerConfig(
        name="hermes", transport="stdio",
        command=command, args=["--db", db_path, "mcp"],
    )


def snomed_search(term: str, max_results: int = 10) -> dict[str, Any]:
    """Search SNOMED CT for concepts matching ``term`` via the Hermes
    MCP server.

    Returns ``{"status": "ok", "results": [{concept_id, preferred_term,
    fully_specified_name, semantic_tag}, ...]}`` on success, or the
    standard ``unavailable`` envelope when the Hermes binary is not
    installed or the call fails. Never raises.
    """
    client = _get_mcp_client("hermes", _hermes_config)
    if client is None:
        return {
            "status": "unavailable",
            "error": "Hermes MCP not configured (set HERMES_MCP_COMMAND)",
        }
    envelope = client.call_tool(
        "search",
        {"term": term, "max_results": max_results},
        timeout=_HTTP_TIMEOUT_SECONDS,
    )
    if envelope.get("status") != "ok":
        return envelope
    raw = envelope.get("results") or []
    normalized = []
    for r in raw[:max_results]:
        if not isinstance(r, dict):
            continue
        concept_id = str(r.get("concept_id") or r.get("id") or "")
        if not concept_id:
            continue
        normalized.append({
            "concept_id": concept_id,
            "preferred_term": str(
                r.get("preferred_term") or r.get("term") or r.get("name") or ""
            ),
            "fully_specified_name": str(
                r.get("fsn") or r.get("fully_specified_name") or ""
            ),
            "semantic_tag": str(r.get("semantic_tag") or ""),
        })
    return _enforce_size_budget({"status": "ok", "results": normalized})


# --- snomed_expand_ecl via Hermes (local stdio) ---------------------------


def snomed_expand_ecl(expression: str, max_results: int = 200) -> dict[str, Any]:
    """Expand a SNOMED CT Expression Constraint Language (ECL) expression.

    ECL is SNOMED's native query language for selecting concept sets —
    e.g. ``<<73211009 |Diabetes mellitus|`` returns Diabetes mellitus
    and all its descendants. Use this when you need a structured cohort
    definition rather than free-text search.

    Returns ``{"status": "ok", "results": [{concept_id, preferred_term,
    fully_specified_name}, ...]}`` on success, or ``unavailable`` when
    Hermes is not installed / the ECL expression is malformed. Never
    raises.
    """
    client = _get_mcp_client("hermes", _hermes_config)
    if client is None:
        return {
            "status": "unavailable",
            "error": "Hermes MCP not configured (set HERMES_MCP_COMMAND)",
        }
    envelope = client.call_tool(
        "expand_ecl",
        {"expression": expression, "max_results": max_results},
        timeout=_HTTP_TIMEOUT_SECONDS,
    )
    if envelope.get("status") != "ok":
        return envelope
    raw = envelope.get("results") or []
    normalized = []
    for r in raw[:max_results]:
        if not isinstance(r, dict):
            continue
        concept_id = str(r.get("concept_id") or r.get("id") or "")
        if not concept_id:
            continue
        normalized.append({
            "concept_id": concept_id,
            "preferred_term": str(
                r.get("preferred_term") or r.get("term") or r.get("name") or ""
            ),
            "fully_specified_name": str(
                r.get("fsn") or r.get("fully_specified_name") or ""
            ),
        })
    return _enforce_size_budget({"status": "ok", "results": normalized})


# --- rxnorm_lookup via OMOPHub (HTTP) --------------------------------------


def _omophub_config():
    from src.conversational.health_evidence.mcp_client import McpServerConfig

    url = os.environ.get("OMOPHUB_MCP_URL")
    if not url:
        return None
    return McpServerConfig(name="omophub", transport="http", url=url)


def rxnorm_lookup(drug_name: str, max_results: int = 5) -> dict[str, Any]:
    """Look up an RxNorm concept (RXCUI) for a drug name via the OMOPHub
    MCP server.

    Returns ``{"status": "ok", "results": [{rxcui, name, tty,
    vocabulary}, ...]}``. Returns ``unavailable`` when ``OMOPHUB_MCP_URL``
    is not set or the call fails.

    Note on auth: OMOPHub requires ``OMOPHUB_API_KEY`` for production
    use; the MCP server is expected to read it from the env passed to
    the subprocess. The HTTP transport here assumes the caller has
    configured the URL with auth (e.g., a managed gateway) appropriately.
    """
    client = _get_mcp_client("omophub", _omophub_config)
    if client is None:
        return {
            "status": "unavailable",
            "error": "OMOPHub MCP not configured (set OMOPHUB_MCP_URL)",
        }
    envelope = client.call_tool(
        "search",
        {"query": drug_name, "vocabulary": "RxNorm",
         "max_results": max_results},
        timeout=_HTTP_TIMEOUT_SECONDS,
    )
    if envelope.get("status") != "ok":
        return envelope
    raw = envelope.get("results") or []
    normalized = []
    for r in raw[:max_results]:
        if not isinstance(r, dict):
            continue
        rxcui = str(r.get("rxcui") or r.get("concept_code") or r.get("id") or "")
        if not rxcui:
            continue
        normalized.append({
            "rxcui": rxcui,
            "name": str(r.get("name") or r.get("concept_name") or ""),
            "tty": str(r.get("tty") or r.get("term_type") or ""),
            "vocabulary": str(r.get("vocabulary") or "RxNorm"),
        })
    return _enforce_size_budget({"status": "ok", "results": normalized})


# --- code_map via OMOPHub (HTTP) -------------------------------------------


def code_map(
    *,
    source_vocabulary: str,
    source_code: str,
    target_vocabulary: str,
    max_results: int = 25,
) -> dict[str, Any]:
    """Map a code from one vocabulary to another via OMOPHub.

    Common cross-vocab use cases:
    - ICD-10 ↔ SNOMED  ("E11.9" → "44054006" Diabetes mellitus type 2)
    - ICD-10 ↔ RxNorm  (treats-relationship pivot)
    - LOINC ↔ SNOMED   (specimen / analyte mapping)

    Returns ``{"status": "ok", "results": [{source_code,
    source_vocabulary, target_code, target_vocabulary, target_name,
    relationship}, ...]}``. Returns ``unavailable`` when OMOPHUB_MCP_URL
    is not set or the call fails.
    """
    client = _get_mcp_client("omophub", _omophub_config)
    if client is None:
        return {
            "status": "unavailable",
            "error": "OMOPHub MCP not configured (set OMOPHUB_MCP_URL)",
        }
    envelope = client.call_tool(
        "map_code",
        {
            "source_vocabulary": source_vocabulary,
            "source_code": source_code,
            "target_vocabulary": target_vocabulary,
            "max_results": max_results,
        },
        timeout=_HTTP_TIMEOUT_SECONDS,
    )
    if envelope.get("status") != "ok":
        return envelope
    raw = envelope.get("results") or []
    normalized = []
    for r in raw[:max_results]:
        if not isinstance(r, dict):
            continue
        target_code = str(
            r.get("target_code")
            or r.get("target_concept_code")
            or ""
        )
        if not target_code:
            continue
        normalized.append({
            "source_code": str(
                r.get("source_code")
                or r.get("source_concept_code")
                or source_code
            ),
            "source_vocabulary": str(
                r.get("source_vocabulary")
                or r.get("source_vocabulary_id")
                or source_vocabulary
            ),
            "target_code": target_code,
            "target_vocabulary": str(
                r.get("target_vocabulary")
                or r.get("target_vocabulary_id")
                or target_vocabulary
            ),
            "target_name": str(
                r.get("target_name") or r.get("target_concept_name") or ""
            ),
            "relationship": str(
                r.get("relationship") or r.get("relationship_id") or ""
            ),
        })
    return _enforce_size_budget({"status": "ok", "results": normalized})


# --- trials_search via cyanheads ClinicalTrials MCP (stdio) ----------------


def _clinicaltrials_config():
    """Build a ClinicalTrials MCP server config.

    Defaults to ``bunx clinicaltrialsgov-mcp-server@latest`` per the user
    research; falls back to ``npx`` when bun is unavailable. Returns
    ``None`` when neither launcher is on PATH.
    """
    import shutil

    from src.conversational.health_evidence.mcp_client import McpServerConfig

    explicit = os.environ.get("CLINICALTRIALS_MCP_COMMAND")
    if explicit:
        parts = explicit.split()
        if not parts:
            return None
        if not shutil.which(parts[0]):
            return None
        return McpServerConfig(
            name="clinicaltrials", transport="stdio",
            command=parts[0], args=parts[1:],
        )
    for launcher in ("bunx", "npx"):
        if shutil.which(launcher):
            args = (
                ["-y", "clinicaltrialsgov-mcp-server@latest"]
                if launcher == "npx"
                else ["clinicaltrialsgov-mcp-server@latest"]
            )
            return McpServerConfig(
                name="clinicaltrials", transport="stdio",
                command=launcher, args=args,
            )
    return None


def trials_search(query: str, max_results: int = 10) -> dict[str, Any]:
    """Search ClinicalTrials.gov for studies matching ``query``.

    Returns ``{"status": "ok", "results": [{nct_id, brief_title, status,
    conditions, phase}, ...]}``. Returns ``unavailable`` when the MCP
    server is not configured or the call fails.
    """
    client = _get_mcp_client("clinicaltrials", _clinicaltrials_config)
    if client is None:
        return {
            "status": "unavailable",
            "error": (
                "ClinicalTrials MCP not configured (need bunx/npx on PATH "
                "or CLINICALTRIALS_MCP_COMMAND)"
            ),
        }
    # cyanheads MCP exposes its tools with a ``clinicaltrials_`` prefix
    # (e.g. ``clinicaltrials_search_studies``); the un-prefixed name in
    # earlier code was wrong against the actual server.
    envelope = client.call_tool(
        "clinicaltrials_search_studies",
        {"query": query, "max_results": max_results},
        timeout=_HTTP_TIMEOUT_SECONDS,
    )
    if envelope.get("status") != "ok":
        return envelope
    raw = envelope.get("results") or []
    normalized: list[dict] = []
    for r in raw:
        if not isinstance(r, dict):
            continue
        # Real cyanheads server returns ``[{"text": "<markdown>"}]`` — a
        # human-readable summary, not structured records. Parse out
        # NCT entries when we see the text shape.
        text_blob = r.get("text")
        if isinstance(text_blob, str) and "NCT" in text_blob:
            normalized.extend(_parse_clinicaltrials_text(text_blob, max_results))
            continue
        # Legacy/hypothetical shape: structured record fields.
        nct_id = str(r.get("nct_id") or r.get("nctId") or "")
        if not nct_id:
            continue
        cond = r.get("conditions") or r.get("condition") or []
        if isinstance(cond, str):
            cond = [cond]
        normalized.append({
            "nct_id": nct_id,
            "brief_title": str(r.get("brief_title") or r.get("briefTitle") or ""),
            "status": str(r.get("status") or r.get("overall_status") or ""),
            "conditions": [str(c) for c in cond if c],
            "phase": str(r.get("phase") or ""),
        })
        if len(normalized) >= max_results:
            break
    return _enforce_size_budget({"status": "ok", "results": normalized[:max_results]})


_CT_HEADER_RE = re.compile(
    r"^- \*\*(NCT\d+)\*\*: (.+?) \[([A-Z_]+)\]\s*$",
)
_CT_DETAIL_RE = re.compile(
    r"^\s+(PHASE\d+|NA)\s*\|\s*N=(\d+|NA)\s*\|\s*([^|]+?)\s*\|\s*(.+?)\s*$",
)


def _parse_clinicaltrials_text(text: str, max_results: int) -> list[dict]:
    """Extract structured records from cyanheads' markdown summary.

    Per-study format:
        - **NCT06968559**: <title> [RECRUITING]
          PHASE3 | N=478 | <sponsor> | <conditions>

    Some studies omit the detail line; we capture what's there and leave
    missing fields empty rather than fabricate them. Returns up to
    ``max_results`` entries."""
    records: list[dict] = []
    lines = text.splitlines()
    current: dict | None = None
    for line in lines:
        m = _CT_HEADER_RE.match(line)
        if m:
            if current:
                records.append(current)
                if len(records) >= max_results:
                    break
            current = {
                "nct_id": m.group(1),
                "brief_title": _truncate_title(m.group(2)),
                "status": m.group(3),
                "conditions": [],
                "phase": "",
            }
            continue
        if current is None:
            continue
        d = _CT_DETAIL_RE.match(line)
        if d:
            phase = d.group(1)
            if phase == "NA":
                phase = ""
            current["phase"] = phase
            conds = [c.strip() for c in d.group(4).split(",") if c.strip()]
            current["conditions"] = conds
    if current and len(records) < max_results:
        records.append(current)
    return records[:max_results]


# --- openfda_drug_label via OpenFDA MCP (stdio) ----------------------------


def _openfda_config():
    import shutil

    from src.conversational.health_evidence.mcp_client import McpServerConfig

    explicit = os.environ.get("OPENFDA_MCP_COMMAND")
    if explicit:
        parts = explicit.split()
        if not parts or not shutil.which(parts[0]):
            return None
        env = None
        api_key = os.environ.get("OPENFDA_API_KEY")
        if api_key:
            env = {"OPENFDA_API_KEY": api_key}
        return McpServerConfig(
            name="openfda", transport="stdio",
            command=parts[0], args=parts[1:], env=env,
        )
    if shutil.which("npx"):
        env = None
        api_key = os.environ.get("OPENFDA_API_KEY")
        if api_key:
            env = {"OPENFDA_API_KEY": api_key}
        return McpServerConfig(
            name="openfda", transport="stdio",
            command="npx",
            args=["-y", "openfda-mcp-server"],
            env=env,
        )
    return None


def openfda_drug_label(drug_name: str) -> dict[str, Any]:
    """Look up an OpenFDA drug label for ``drug_name``.

    Returns ``{"status": "ok", "results": [{brand_name, generic_name,
    indications_and_usage, warnings}, ...]}`` on success, or the standard
    ``unavailable`` envelope.
    """
    client = _get_mcp_client("openfda", _openfda_config)
    if client is None:
        return {
            "status": "unavailable",
            "error": (
                "OpenFDA MCP not configured (need npx on PATH or "
                "OPENFDA_MCP_COMMAND)"
            ),
        }
    envelope = client.call_tool(
        "search_drug_labels",
        {"drug_name": drug_name},
        timeout=_HTTP_TIMEOUT_SECONDS,
    )
    if envelope.get("status") != "ok":
        return envelope
    raw = envelope.get("results") or []
    # The npm openfda-mcp-server wraps records as ``[{"success": true,
    # "data": [<records>]}]``. Unwrap when we see that shape; otherwise
    # treat ``raw`` as a direct list of records (legacy / hypothetical).
    flat_records: list[dict] = []
    for r in raw:
        if not isinstance(r, dict):
            continue
        if "data" in r and isinstance(r["data"], list):
            flat_records.extend(d for d in r["data"] if isinstance(d, dict))
        else:
            flat_records.append(r)

    normalized = []
    for r in flat_records[:5]:
        # OpenFDA labels can be either snake_case (legacy/hypothetical
        # shape used in tests) or camelCase (real npm server). Some
        # fields nest as lists; collapse to first entry.
        def _first(*fields):
            for f in fields:
                v = r.get(f)
                if isinstance(v, list) and v:
                    return str(v[0])
                if v is not None and v != "":
                    return str(v)
            return ""

        brand = _first("brand_name", "brandName")
        generic = _first("generic_name", "genericName")
        if not brand and not generic:
            continue
        normalized.append({
            "brand_name": brand,
            "generic_name": generic,
            "indications_and_usage": _truncate_title(
                _first("indications_and_usage", "indications"),
            ),
            "warnings": _truncate_title(_first("warnings")),
        })
    return _enforce_size_budget({"status": "ok", "results": normalized})


# --- icd_lookup via self-hosted ICD MCP (HTTP) -----------------------------


def _icd_config():
    from src.conversational.health_evidence.mcp_client import McpServerConfig

    url = os.environ.get("ICD_MCP_URL")
    if not url:
        return None
    return McpServerConfig(name="icd", transport="http", url=url)


def icd_lookup(query: str, version: str = "10", max_results: int = 10) -> dict[str, Any]:
    """Look up ICD codes matching ``query`` via a self-hosted ICD MCP.

    ``version`` is "10" or "11". Returns ``{"status": "ok", "results":
    [{code, title, version, chapter}, ...]}``. Returns ``unavailable``
    when ``ICD_MCP_URL`` is not set.
    """
    client = _get_mcp_client("icd", _icd_config)
    if client is None:
        return {
            "status": "unavailable",
            "error": "ICD MCP not configured (set ICD_MCP_URL)",
        }
    envelope = client.call_tool(
        "lookup",
        {"query": query, "version": str(version), "max_results": max_results},
        timeout=_HTTP_TIMEOUT_SECONDS,
    )
    if envelope.get("status") != "ok":
        return envelope
    raw = envelope.get("results") or []
    normalized = []
    for r in raw[:max_results]:
        if not isinstance(r, dict):
            continue
        code = str(r.get("code") or r.get("icd_code") or "")
        if not code:
            continue
        normalized.append({
            "code": code,
            "title": _truncate_title(str(
                r.get("title") or r.get("description") or ""
            )),
            "version": str(r.get("version") or version),
            "chapter": str(r.get("chapter") or ""),
        })
    return _enforce_size_budget({"status": "ok", "results": normalized})


# --- icd_autocode via self-hosted ICD MCP (HTTP) --------------------------


def icd_autocode(
    text: str,
    version: str = "10",
    max_results: int = 10,
) -> dict[str, Any]:
    """Suggest ICD-10/11 codes for free-text clinical narrative.

    The autocoding tool returns ranked candidates with relevance/confidence
    scores — useful when the orchestrator needs to translate clinical
    prose (a problem-list bullet, an admission note phrase) into structured
    ICD codes for downstream filtering or billing-side reasoning.

    Returns ``{"status": "ok", "results": [{code, title, version,
    confidence}, ...]}``. ``confidence`` is None when the upstream did
    not score the candidate. Returns ``unavailable`` when ``ICD_MCP_URL``
    is not set.
    """
    client = _get_mcp_client("icd", _icd_config)
    if client is None:
        return {
            "status": "unavailable",
            "error": "ICD MCP not configured (set ICD_MCP_URL)",
        }
    envelope = client.call_tool(
        "autocode",
        {"text": text, "version": str(version), "max_results": max_results},
        timeout=_HTTP_TIMEOUT_SECONDS,
    )
    if envelope.get("status") != "ok":
        return envelope
    raw = envelope.get("results") or []
    normalized = []
    for r in raw[:max_results]:
        if not isinstance(r, dict):
            continue
        code = str(r.get("code") or r.get("icd_code") or "")
        if not code:
            continue
        # Confidence/score: keep as float when present; never synthesise.
        conf = r.get("confidence")
        if conf is None:
            conf = r.get("score")
        if conf is not None:
            try:
                conf = float(conf)
            except (TypeError, ValueError):
                conf = None
        normalized.append({
            "code": code,
            "title": _truncate_title(str(
                r.get("title") or r.get("description") or ""
            )),
            "version": str(r.get("version") or version),
            "confidence": conf,
        })
    return _enforce_size_budget({"status": "ok", "results": normalized})


def loinc_reference_range(loinc_code: str) -> dict[str, Any]:
    """Look up the published reference range for a LOINC code.

    Reads from ``LOINC_CATALOG_PATH`` — a JSON dict keyed by canonical
    LOINC code (``"NNNNN-N"``) mapping to ``{low, high, units}``. Returns
    ``{"status": "ok", "results": [{...}]}`` on hit, ``unavailable`` envelope
    on miss / file absent / malformed code. Never raises.
    """
    if not isinstance(loinc_code, str) or not _LOINC_CODE_PATTERN.match(loinc_code):
        return {
            "status": "unavailable",
            "error": f"invalid LOINC code: {loinc_code!r}",
        }

    path = LOINC_CATALOG_PATH
    if not path.exists():
        return {
            "status": "unavailable",
            "error": f"LOINC catalog not found at {path}",
        }
    try:
        catalog = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        logger.warning("loinc_reference_range read failed: %s", exc)
        return {"status": "unavailable", "error": f"catalog read error: {exc}"}
    if not isinstance(catalog, dict):
        return {
            "status": "unavailable",
            "error": "catalog is not a JSON object",
        }
    record = catalog.get(loinc_code)
    if record is None or not isinstance(record, dict):
        return {
            "status": "unavailable",
            "error": f"LOINC {loinc_code} not in catalog",
        }
    payload = {
        "status": "ok",
        "results": [
            {
                "loinc_code": loinc_code,
                "low": record.get("low"),
                "high": record.get("high"),
                "units": record.get("units"),
            }
        ],
    }
    return _enforce_size_budget(payload)


# ---------------------------------------------------------------------------
# mimic_itemid_search: free-text analyte → MIMIC itemid lookup
# (Phase H follow-up — fills the gap surfaced by the procalcitonin smoke).
# ---------------------------------------------------------------------------


def _validate_search_query(query: str) -> str:
    """Charset gate for the user-facing analyte search query.

    Returns the trimmed query on success; raises ``ValueError`` if the
    input is None / empty / too short / contains characters outside
    the safe set. SQL injection payloads, LIKE wildcards (% / _), and
    classic quote-based escapes are all rejected here BEFORE any
    backend connection is opened."""
    if not isinstance(query, str):
        raise ValueError(f"query must be a string, got {type(query).__name__}")
    s = query.strip()
    if len(s) < _MIN_SEARCH_QUERY_CHARS:
        raise ValueError(
            f"query too short: {s!r} (need at least "
            f"{_MIN_SEARCH_QUERY_CHARS} non-whitespace characters)"
        )
    if not _VALID_SEARCH_QUERY_RE.match(s):
        raise ValueError(
            f"invalid query {query!r}: must match [A-Za-z0-9 .,\\-/]+ "
            f"(no SQL wildcards / quotes / semicolons)"
        )
    return s


# Lazy-cached mapping from itemid → LOINC code, populated from
# data/mappings/labitem_to_snomed.json on first call. Only labevents
# itemids are LOINC-coded; chart items don't appear in this mapping.
# Tests can monkeypatch ``LABITEM_TO_SNOMED_PATH`` to point at a
# fixture mapping; the cache is keyed on the path so test fixtures
# don't bleed into one another.
_ITEMID_TO_LOINC_CACHE: dict[str, dict[int, str]] = {}


def _load_itemid_to_loinc() -> dict[int, str]:
    """Load + cache the itemid → LOINC mapping. Cache is keyed on the
    current ``LABITEM_TO_SNOMED_PATH`` so monkeypatching in tests
    works as expected (different paths get separate caches)."""
    cache_key = str(LABITEM_TO_SNOMED_PATH)
    cached = _ITEMID_TO_LOINC_CACHE.get(cache_key)
    if cached is not None:
        return cached
    out: dict[int, str] = {}
    path = LABITEM_TO_SNOMED_PATH
    if not path.exists():
        _ITEMID_TO_LOINC_CACHE[cache_key] = out
        return out
    try:
        raw = json.loads(path.read_text())
    except (OSError, ValueError):
        _ITEMID_TO_LOINC_CACHE[cache_key] = out
        return out
    if not isinstance(raw, dict):
        _ITEMID_TO_LOINC_CACHE[cache_key] = out
        return out
    for itemid_str, rec in raw.items():
        if itemid_str == "_metadata":
            continue
        if not isinstance(rec, dict):
            continue
        loinc = rec.get("loinc")
        if not loinc:
            continue
        try:
            out[int(itemid_str)] = str(loinc)
        except (TypeError, ValueError):
            continue
    _ITEMID_TO_LOINC_CACHE[cache_key] = out
    return out


def _match_label(label: str, query: str) -> str:
    """Classify how a row matches the query — drives the result's
    ``match`` field. Used by tests + the model when it picks among
    candidates. Inputs are already normalised (both lowercase)."""
    label_low = (label or "").lower()
    query_low = (query or "").lower()
    if label_low == query_low:
        return "exact"
    if label_low.startswith(query_low):
        return "prefix"
    return "substring"


def mimic_itemid_search(
    query: str,
    max_results: int = _DEFAULT_SEARCH_MAX_RESULTS,
) -> dict[str, Any]:
    """Search MIMIC's ``d_labitems`` and ``d_items`` for itemids whose
    label matches a free-text analyte/measurement name.

    Returns a list of ranked candidates (exact > prefix > substring),
    each carrying ``itemid``, ``label``, ``table`` (``"labevents"``
    or ``"chartevents"``), optional ``fluid`` (lab items only),
    ``category``, and (when available) the LOINC code from
    ``data/mappings/labitem_to_snomed.json``.

    **Empty results are NOT a failure.** When the analyte isn't in
    MIMIC at all (procalcitonin is genuinely absent in MIMIC-IV), the
    tool returns ``{status: "ok", results: []}`` so the model knows
    to pivot to PubMed rather than retry with a guess.
    ``status="unavailable"`` is reserved for backend failures
    (duckdb missing, BigQuery auth, etc.).

    Backend dispatch follows the same ``DATA_SOURCE`` env var as the
    on-the-fly cohort compute — local DuckDB or BigQuery against
    physionet-data.mimiciv_3_1_*.

    PHI-safe: returns reference data only (itemid, label, fluid,
    category, LOINC). Never returns row-level patient values.
    """
    try:
        n = int(max_results)
    except (TypeError, ValueError):
        n = _DEFAULT_SEARCH_MAX_RESULTS
    n = max(1, min(n, _SEARCH_MAX_RESULTS_CEILING))

    try:
        validated = _validate_search_query(query)
    except ValueError as exc:
        return {"status": "unavailable", "error": f"invalid query: {exc}"}

    data_source = os.environ.get("DATA_SOURCE", "local").lower()
    if data_source == "bigquery":
        rows, err = _search_itemid_via_bigquery(validated, max_results=n)
    else:
        rows, err = _search_itemid_via_duckdb(validated, max_results=n)
    if err is not None:
        return {"status": "unavailable", "error": err}

    # LOINC enrichment for lab items.
    loinc_index = _load_itemid_to_loinc()

    results: list[dict[str, Any]] = []
    for row in rows:
        itemid = int(row["itemid"])
        rec: dict[str, Any] = {
            "itemid": itemid,
            "label": row["label"] or "",
            "table": row["source_table"],
            "category": row.get("category") or "",
            "match": _match_label(row["label"] or "", validated),
        }
        if row["source_table"] == "labevents":
            rec["fluid"] = row.get("fluid") or ""
            loinc = loinc_index.get(itemid)
            if loinc:
                rec["loinc"] = loinc
        results.append(rec)

    return _enforce_size_budget({"status": "ok", "results": results})


def _search_itemid_via_duckdb(
    query: str, *, max_results: int,
) -> tuple[list[dict], str | None]:
    """Local DuckDB path. Returns ``(rows, error)``. Rows are dicts
    with keys: itemid, label, fluid, category, source_table,
    match_rank."""
    if not MIMIC_COMPUTE_DUCKDB_PATH.exists():
        return [], (
            f"compute backend (MIMIC duckdb) not available at "
            f"{MIMIC_COMPUTE_DUCKDB_PATH}"
        )
    sql = """
        WITH unioned AS (
            SELECT itemid, label, fluid, category,
                   'labevents' AS source_table,
                   CASE
                       WHEN LOWER(label) = LOWER(?) THEN 0
                       WHEN LOWER(label) LIKE LOWER(?) || '%' THEN 1
                       ELSE 2
                   END AS match_rank
            FROM d_labitems
            WHERE LOWER(label) LIKE '%' || LOWER(?) || '%'
            UNION ALL
            SELECT itemid, label, NULL AS fluid, category,
                   'chartevents' AS source_table,
                   CASE
                       WHEN LOWER(label) = LOWER(?) THEN 0
                       WHEN LOWER(label) LIKE LOWER(?) || '%' THEN 1
                       ELSE 2
                   END AS match_rank
            FROM d_items
            WHERE LOWER(label) LIKE '%' || LOWER(?) || '%'
        )
        SELECT itemid, label, fluid, category, source_table, match_rank
        FROM unioned
        ORDER BY match_rank, length(label), label
        LIMIT ?
    """
    try:
        import duckdb

        con = duckdb.connect(str(MIMIC_COMPUTE_DUCKDB_PATH), read_only=True)
        try:
            raw = con.execute(
                sql,
                [query, query, query, query, query, query, max_results],
            ).fetchall()
        finally:
            con.close()
    except Exception as exc:  # noqa: BLE001 — never raise to caller
        logger.warning("mimic_itemid_search duckdb failed: %s", exc)
        return [], f"duckdb search failed: {exc}"

    rows = [
        {
            "itemid": r[0],
            "label": r[1] or "",
            "fluid": r[2],
            "category": r[3],
            "source_table": r[4],
            "match_rank": int(r[5]),
        }
        for r in raw
    ]
    return rows, None


def _search_itemid_via_bigquery(
    query: str, *, max_results: int,
) -> tuple[list[dict], str | None]:
    """BigQuery path. Returns ``(rows, error)``."""
    project = os.environ.get("BIGQUERY_PROJECT")
    if not project:
        return [], (
            "DATA_SOURCE=bigquery but BIGQUERY_PROJECT is unset — "
            "cannot run BigQuery itemid search"
        )
    sql = f"""
        SELECT itemid, label, fluid, category,
               'labevents' AS source_table,
               CASE
                   WHEN LOWER(label) = LOWER(@q) THEN 0
                   WHEN STARTS_WITH(LOWER(label), LOWER(@q)) THEN 1
                   ELSE 2
               END AS match_rank
        FROM {_BQ_D_LABITEMS}
        WHERE STRPOS(LOWER(label), LOWER(@q)) > 0
        UNION ALL
        SELECT itemid, label, NULL AS fluid, category,
               'chartevents' AS source_table,
               CASE
                   WHEN LOWER(label) = LOWER(@q) THEN 0
                   WHEN STARTS_WITH(LOWER(label), LOWER(@q)) THEN 1
                   ELSE 2
               END AS match_rank
        FROM {_BQ_D_ITEMS}
        WHERE STRPOS(LOWER(label), LOWER(@q)) > 0
        ORDER BY match_rank, LENGTH(label), label
        LIMIT @n
    """
    try:
        bq = _get_bigquery_module()
        client = bq.Client(project=project)
        try:
            params = [
                bq.ScalarQueryParameter("q", "STRING", query),
                bq.ScalarQueryParameter("n", "INT64", max_results),
            ]
            job_config = bq.QueryJobConfig(query_parameters=params)
            job = client.query(sql, job_config=job_config)
        except (AttributeError, TypeError):
            # Mocks may not implement QueryJobConfig — fall back to
            # interpolated values (safe since both are validated).
            interpolated = (
                sql.replace("@q", f"'{query}'").replace("@n", str(int(max_results)))
            )
            job = client.query(interpolated)
        result_rows = list(job.result())
    except Exception as exc:  # noqa: BLE001
        logger.warning("mimic_itemid_search bigquery failed: %s", exc)
        return [], f"bigquery search failed: {exc}"

    rows = []
    for r in result_rows:
        rows.append({
            "itemid": int(getattr(r, "itemid", 0)),
            "label": getattr(r, "label", "") or "",
            "fluid": getattr(r, "fluid", None),
            "category": getattr(r, "category", None),
            "source_table": str(getattr(r, "source_table", "")),
            "match_rank": int(getattr(r, "match_rank", 2)),
        })
    return rows, None
