"""bq-validator MCP server entry point.

Run as::

    python -m mcp_servers.bq_validator.server   # (when installed as a package)

or directly::

    python /path/to/mcp/bq_validator/server.py

Configuration via env vars:
- ``BQ_VALIDATOR_PROJECT`` — GCP project for dry-run billing (optional;
  uses the ADC default if unset).
- ``BQ_VALIDATOR_ALLOWED_DATASETS`` — comma-separated dataset allowlist;
  defaults to the physionet-data.mimiciv_3_1_* set.
- ``BQ_VALIDATOR_USD_PER_TIB`` — on-demand price; defaults to 6.25.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import sqlglot
from fastmcp import FastMCP
from sqlglot import exp

logger = logging.getLogger(__name__)

mcp = FastMCP("bq-validator")


_DEFAULT_ALLOWED_DATASETS = {
    "physionet-data.mimiciv_3_1_hosp",
    "physionet-data.mimiciv_3_1_icu",
    "physionet-data.mimiciv_3_1_derived",
    "physionet-data.mimic_iv_demo",
    "physionet-data.mimiciv_hosp",
    "physionet-data.mimiciv_icu",
    "physionet-data.mimiciv_derived",
}


def _allowed_datasets() -> set[str]:
    raw = os.environ.get("BQ_VALIDATOR_ALLOWED_DATASETS")
    if not raw:
        return set(_DEFAULT_ALLOWED_DATASETS)
    return {s.strip() for s in raw.split(",") if s.strip()}


def _usd_per_tib() -> float:
    try:
        return float(os.environ.get("BQ_VALIDATOR_USD_PER_TIB", "6.25"))
    except ValueError:
        return 6.25


_bq_client = None


def _bq():
    """Lazy BigQuery client. Imported here so the server module can be
    imported in test environments without GCP credentials."""
    global _bq_client
    if _bq_client is not None:
        return _bq_client
    from google.cloud import bigquery

    project = os.environ.get("BQ_VALIDATOR_PROJECT") or None
    _bq_client = bigquery.Client(project=project)
    return _bq_client


@mcp.tool
def validate_sql(
    sql: str,
    params: list | None = None,
    max_usd: float = 0.50,
    max_bytes: int = 10 * 1024**3,  # 10 GiB hard kill
) -> str:
    """Pre-execution SQL validator. Returns a JSON-serialised verdict.

    Stages: parse → policy → scope → BQ dry_run → cost gate.

    The verdict is wrapped in the project's envelope shape::

        {"status": "ok", "results": [<verdict>]}

    on success / clean rejection, or::

        {"status": "unavailable", "error": "..."}

    on infrastructure failure (e.g. dry_run API unreachable). The verdict
    itself carries ``ok`` (boolean) and ``stage`` for the caller to
    interpret.
    """
    verdict = _run_stages(sql, params or [], max_usd, max_bytes)
    return json.dumps({"status": "ok", "results": [verdict]})


def _run_stages(
    sql: str,
    params: list,
    max_usd: float,
    max_bytes: int,
) -> dict[str, Any]:
    # 1. Parse
    try:
        tree = sqlglot.parse_one(sql, dialect="bigquery")
    except sqlglot.errors.ParseError as exc:
        return {
            "ok": False, "stage": "parse",
            "error": str(exc),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False, "stage": "parse",
            "error": f"sqlglot raised: {exc}",
        }

    # 2. Read-only policy
    if not isinstance(tree, (exp.Select, exp.Union, exp.With)):
        return {
            "ok": False, "stage": "policy",
            "error": (
                f"Only SELECT/WITH/UNION permitted; got "
                f"{type(tree).__name__}"
            ),
        }

    # 3. Dataset scope
    allowed = _allowed_datasets()
    for tbl in tree.find_all(exp.Table):
        catalog = tbl.catalog or "physionet-data"
        db = tbl.db or ""
        fq = f"{catalog}.{db}".rstrip(".")
        # Empty fq means a CTE reference (no project/db) — allowed.
        if not db:
            continue
        if fq not in allowed:
            return {
                "ok": False, "stage": "scope",
                "error": f"Table {tbl} ({fq}) outside allowlist",
            }

    # 4. BigQuery dry_run
    try:
        from google.cloud import bigquery

        cfg = bigquery.QueryJobConfig(
            dry_run=True,
            use_query_cache=False,
            query_parameters=_to_bq_params(params),
        )
        job = _bq().query(sql, job_config=cfg)
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False, "stage": "dry_run",
            "error": str(exc),
        }

    # 5. Cost gate
    bytes_processed = job.total_bytes_processed or 0
    cost = (bytes_processed / 1024**4) * _usd_per_tib()
    too_big = bytes_processed > max_bytes
    too_expensive = cost > max_usd

    if too_big or too_expensive:
        return {
            "ok": False,
            "stage": "bytes" if too_big else "cost",
            "bytes_processed": bytes_processed,
            "estimated_usd": round(cost, 4),
            "max_usd": max_usd,
            "max_bytes": max_bytes,
            "error": (
                f"Estimated {bytes_processed / 1024**3:.2f} GiB "
                f"(${cost:.4f}) exceeds limit"
            ),
        }

    return {
        "ok": True,
        "stage": "ok",
        "bytes_processed": bytes_processed,
        "estimated_usd": round(cost, 4),
        "referenced_tables": [
            f"{t.project}.{t.dataset_id}.{t.table_id}"
            for t in (job.referenced_tables or [])
        ],
    }


def _to_bq_params(params: list) -> list:
    """Convert a positional ``[?, ?, ?]`` parameter list to BigQuery
    QueryParameter objects.

    The orchestrator passes positional params; we map them to
    ScalarQueryParameter with a positional binding via ``None`` name.
    Type inference is best-effort — int → INT64, float → FLOAT64,
    everything else → STRING. Dry-run only checks shape, so over-broad
    typing is fine.
    """
    from google.cloud import bigquery

    out: list = []
    for p in params:
        if isinstance(p, bool):
            type_ = "BOOL"
        elif isinstance(p, int):
            type_ = "INT64"
        elif isinstance(p, float):
            type_ = "FLOAT64"
        else:
            type_ = "STRING"
            p = str(p)
        out.append(bigquery.ScalarQueryParameter(None, type_, p))
    return out


if __name__ == "__main__":
    mcp.run(transport="stdio")
