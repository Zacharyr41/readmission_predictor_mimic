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


def mimic_distribution_lookup(itemid: int) -> dict[str, Any]:
    """Look up the empirical MIMIC distribution for a given lab/chart itemid.

    Reads from ``MIMIC_DISTRIBUTIONS_PATH`` — a JSON dict keyed by stringified
    itemid mapping to ``{n, mean, p50, p95, units}``. Returns
    ``{"status": "ok", "results": [{...}]}`` on hit, ``unavailable`` envelope
    on miss / file absent / malformed file. Never raises.
    """
    try:
        itemid_int = int(itemid)
    except (TypeError, ValueError):
        return {"status": "unavailable", "error": f"invalid itemid: {itemid!r}"}

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
        return {"status": "unavailable", "error": f"registry read error: {exc}"}
    if not isinstance(registry, dict):
        return {
            "status": "unavailable",
            "error": "registry is not a JSON object",
        }
    record = registry.get(str(itemid_int))
    if record is None or not isinstance(record, dict):
        return {
            "status": "unavailable",
            "error": f"itemid {itemid_int} not in registry",
        }
    payload = {
        "status": "ok",
        "results": [
            {
                "itemid": itemid_int,
                "n": record.get("n"),
                "mean": record.get("mean"),
                "p50": record.get("p50"),
                "p95": record.get("p95"),
                "units": record.get("units"),
            }
        ],
    }
    return _enforce_size_budget(payload)


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
    envelope = client.call_tool(
        "search_studies",
        {"query": query, "max_results": max_results},
        timeout=_HTTP_TIMEOUT_SECONDS,
    )
    if envelope.get("status") != "ok":
        return envelope
    raw = envelope.get("results") or []
    normalized = []
    for r in raw[:max_results]:
        if not isinstance(r, dict):
            continue
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
    return _enforce_size_budget({"status": "ok", "results": normalized})


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
    normalized = []
    for r in raw[:5]:
        if not isinstance(r, dict):
            continue
        # OpenFDA labels often nest fields as lists; collapse to first
        # entry's text to keep the envelope compact.
        def _first(field):
            v = r.get(field)
            if isinstance(v, list) and v:
                return str(v[0])
            return str(v) if v is not None else ""

        brand = _first("brand_name") or _first("openfda")
        generic = _first("generic_name")
        if not brand and not generic:
            continue
        normalized.append({
            "brand_name": brand,
            "generic_name": generic,
            "indications_and_usage": _truncate_title(_first("indications_and_usage")),
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
