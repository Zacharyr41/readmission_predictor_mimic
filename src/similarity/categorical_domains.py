"""Frozen schema-grounded value sets for the cohort's categorical columns.

Sibling of :mod:`src.similarity.reference_ranges`. Where reference-ranges freeze
the *numeric scale* a quantitative trait's Gower distance is normalized against,
this module freezes the *legal value set* (domain) of each categorical column —
``admission_type``, ``gender`` — into a committed artifact,
``data/mappings/similarity_categorical_domains.json``.

Why this exists: field *names* are schema-grounded (the ``OperationRegistry``),
but categorical *values* used to be hardcoded MIMIC-III literals
(``EMERGENCY``/``ELECTIVE``/``URGENT``) scattered across the cohort prompt, the
contextual one-hot scorer, and the causal covariates. MIMIC-IV uses a different
vocabulary (``EW EMER.``, ``DIRECT EMER.``, …), so those literals silently
matched nothing — a prefilter on ``admission_type = 'EMERGENCY'`` returned an
empty pool ("0 of 0"). This artifact is the single source of truth that:

* teaches the cohort-definition LLM the *real* vocabulary (the prompt's worked
  example + the feature catalog's category list draw from it), and
* backs a validate-and-repair guard that rejects an out-of-domain value and
  re-prompts the model.

Two halves (mirroring reference_ranges):

* :func:`compute_categorical_domains` (builder) runs a ``GROUP BY`` per column
  and returns the distinct non-null values **most-common first**, with counts.
* :func:`load_categorical_domains` (loader) reads the committed JSON into the
  ``{column: (value, ...)}`` mapping the prompt/guard consume. It never raises
  on a missing / malformed file (returns ``{}``); a caller that finds no domain
  for a column simply skips grounding that column rather than crashing.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Repo-root-relative default. ``__file__`` is ``src/similarity/categorical_domains.py``
# so three ``.parent`` hops land at the repository root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CATEGORICAL_DOMAINS_PATH = (
    _REPO_ROOT / "data" / "mappings" / "similarity_categorical_domains.json"
)

# Module-level so tests can monkeypatch the artifact location.
CATEGORICAL_DOMAINS_PATH: Path = _DEFAULT_CATEGORICAL_DOMAINS_PATH


# ``{column: (table, human description)}``. Add a column with one line — the same
# spirit as ``reference_ranges._FEATURE_QUERIES``. The description is reused by
# the feature catalog / prompt grounding so it reads naturally to the LLM.
_CATEGORICAL_COLUMNS: dict[str, tuple[str, str]] = {
    "admission_type": ("admissions", "admission type"),
    "gender": ("patients", "recorded sex"),
}


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def compute_categorical_domains(
    backend: Any,
    columns: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Compute the distinct value domain of each categorical column.

    ``backend`` is any object exposing ``execute(sql, params) -> list[tuple]``
    (the production ``_DuckDBBackend`` / ``_BigQueryBackend`` and the test
    wrappers all satisfy this). ``columns`` restricts the set of columns built;
    ``None`` means every column in :data:`_CATEGORICAL_COLUMNS`.

    Returns ``{column: {"values": [...most-common first...], "counts": {v: n},
    "n": total non-null}}``. Values are derived from the data — whatever
    distinct non-null literals the column holds — so the frozen domain always
    matches the live schema.
    """
    names = list(columns) if columns is not None else list(_CATEGORICAL_COLUMNS)

    domains: dict[str, dict[str, Any]] = {}
    for name in names:
        spec = _CATEGORICAL_COLUMNS.get(name)
        if spec is None:
            logger.warning("no categorical column registered for %r; skipping", name)
            continue
        table, _desc = spec
        # ``name``/``table`` are code-controlled identifiers from the registry
        # above (never user input), so interpolating them is injection-safe and
        # sidesteps the fact that a column/table name can't be a bound param.
        sql = (
            f"SELECT {name} AS v, COUNT(*) AS n "
            f"FROM {table} WHERE {name} IS NOT NULL "
            f"GROUP BY {name} ORDER BY n DESC, v ASC"
        )
        try:
            rows = backend.execute(sql, [])
        except Exception as exc:  # pragma: no cover - backend-specific failures
            logger.warning("categorical-domain query for %r failed: %s", name, exc)
            continue
        values: list[str] = []
        counts: dict[str, int] = {}
        total = 0
        for v, n in rows:
            if v is None:
                continue
            v_str = str(v)
            values.append(v_str)
            counts[v_str] = int(n)
            total += int(n)
        if not values:
            logger.debug("column %r has no non-null values; skipping", name)
            continue
        domains[name] = {"values": values, "counts": counts, "n": total}
    return domains


def build_artifact(
    backend: Any,
    *,
    source: str = "all_admissions",
    columns: list[str] | None = None,
) -> dict[str, Any]:
    """Assemble the full frozen categorical-domain artifact (metadata + domains)."""
    domains = compute_categorical_domains(backend, columns=columns)
    return {
        "version": "1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "domains": domains,
    }


def write_artifact(artifact: dict[str, Any], path: Path | None = None) -> Path:
    """Write the artifact JSON to ``path`` (default ``CATEGORICAL_DOMAINS_PATH``)."""
    target = Path(path) if path is not None else CATEGORICAL_DOMAINS_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(artifact, indent=2, sort_keys=True))
    return target


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_categorical_domains(path: Path | None = None) -> dict[str, tuple[str, ...]]:
    """Load the frozen domains into ``{column: (value, ...)}`` (most-common first).

    Never raises: a missing / unreadable / malformed artifact yields ``{}`` so
    callers degrade gracefully (prompt grounding / validation simply skip a
    column with no known domain rather than crashing).
    """
    target = Path(path) if path is not None else CATEGORICAL_DOMAINS_PATH
    if not target.exists():
        logger.debug("categorical-domains artifact not found at %s", target)
        return {}
    try:
        doc = json.loads(target.read_text())
    except (OSError, ValueError) as exc:
        logger.warning("categorical-domains artifact unreadable (%s): %s", target, exc)
        return {}
    if not isinstance(doc, dict):
        return {}
    domains = doc.get("domains")
    if not isinstance(domains, dict):
        return {}
    out: dict[str, tuple[str, ...]] = {}
    for name, rec in domains.items():
        if not isinstance(rec, dict):
            continue
        values = rec.get("values")
        if not isinstance(values, list):
            continue
        out[name] = tuple(str(v) for v in values)
    return out


def describe_domain(
    column: str, *, limit: int = 4, path: Path | None = None
) -> str | None:
    """A short, quoted list of a column's most-common real values, for prompts.

    Returns e.g. ``'"EW EMER.", "EU OBSERVATION", "OBSERVATION ADMIT", "URGENT", …'``
    (most-common first, truncated at ``limit``), or ``None`` when the artifact
    has no domain for the column — so a caller can keep a generic description
    rather than naming any literals it can't verify against the schema.
    """
    values = load_categorical_domains(path).get(column)
    if not values:
        return None
    shown = ", ".join(f'"{v}"' for v in values[:limit])
    return shown + (", …" if len(values) > limit else "")
