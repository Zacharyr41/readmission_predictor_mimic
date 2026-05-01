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
# pubmed_search: NCBI E-utilities wrapper (moved verbatim from critic_tools.py)
# ---------------------------------------------------------------------------


def pubmed_search(query: str, max_results: int = 5) -> dict[str, Any]:
    """Search PubMed via NCBI E-utilities (esearch + esummary).

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
