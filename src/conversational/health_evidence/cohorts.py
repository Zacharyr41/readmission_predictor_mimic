"""Cohort registry + name resolver for the Phase H Tier-D distribution
catalog.

The user types medical terminology in chat ("lactate in sepsis
patients"); the critic LLM is what passes ``cohort=`` to
:func:`mimic_distribution_lookup`. This module does the translation
from "natural medical phrase" to "canonical cohort name + ICD code
prefixes" so the SQL builder downstream has unambiguous inputs.

Architecture notes:

* The registry lives in ``data/mappings/clinical_cohorts.json``. ICD
  prefixes are stored in **dotted** form (``"A41."``, ``"995.91"``)
  because that's how clinicians + ICD references write them. MIMIC's
  ``diagnoses_icd`` table stores codes WITHOUT dots — see
  :func:`normalize_icd_prefix` and the SQL builder for the runtime
  conversion.
* :func:`resolve_cohort_name` accepts a free-text phrase and returns
  the canonical cohort name (or ``None``). Resolution is case-
  insensitive, alias-aware, and longest-substring-match-wins so the
  registry can grow without ambiguity collisions.
* The registry name ``"all"`` is reserved for the unstratified bucket
  emitted by the catalog; user data must never define it. The name
  ``"custom"`` is reserved for raw-prefix lookups (no registry
  involvement).
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Final


_REGISTRY_PATH: Final[Path] = (
    Path(__file__).resolve().parents[3]
    / "data" / "mappings" / "clinical_cohorts.json"
)

_RESERVED_NAMES: Final[frozenset[str]] = frozenset({"all", "custom"})

# Punctuation we strip from incoming phrases before alias matching. Keep
# letters/digits/spaces so multi-word aliases ("heart attack") still
# match. Hyphen is handled specially — kept in COVID variants
# ("covid-19", "sars-cov-2") because it's part of the term.
_PUNCT_TO_DROP = re.compile(r"[.,;:!?()\[\]{}'\"`]")
_WHITESPACE_COLLAPSE = re.compile(r"\s+")

# ICD code charset — letters, digits, dot only. Dots are stripped at
# normalisation time, but we accept them in the input for human
# readability. Anything outside this charset (whitespace, quotes,
# semicolons, SQL wildcards, parentheses, etc.) is rejected before SQL
# composition as a security gate.
_VALID_ICD_PREFIX_RE = re.compile(r"^[A-Za-z0-9.]+$")


@lru_cache(maxsize=1)
def load_cohorts() -> dict[str, dict]:
    """Load the cohort registry from disk and cache it.

    Returns a dict mapping canonical cohort name → ``{icd10_prefixes,
    icd9_prefixes, aliases}``. The ``_metadata`` block in the JSON file
    is filtered out so callers can iterate the dict without special-
    casing it.

    The ``lru_cache`` is process-lifetime; tests that mutate the file
    on disk should call ``load_cohorts.cache_clear()`` after writing."""
    raw = json.loads(_REGISTRY_PATH.read_text())
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def known_cohort_names() -> list[str]:
    """Return the list of canonical cohort names. Used to populate
    helpful errors when a caller passes an unrecognised cohort."""
    return list(load_cohorts().keys())


def normalize_icd_prefix(prefix: str) -> str:
    """Convert a dotted ICD prefix (``"A41."`` / ``"995.91"``) into the
    no-dot form MIMIC stores (``"A41"`` / ``"99591"``).

    Also uppercases — ICD-10 letter codes are stored uppercase in MIMIC.
    No further validation here; the SQL builder is responsible for
    rejecting prefixes that don't match the ICD charset
    (see ``cohort_filter_sql``)."""
    return prefix.replace(".", "").upper().strip()


def _normalize_phrase(phrase: str) -> str:
    """Lowercase, strip drop-punctuation, and collapse internal
    whitespace. Keeps hyphen so "covid-19" and "sars-cov-2" still
    match. The output is what we substring-search against aliases.

    Returns empty string if the input is None / empty / whitespace."""
    if not phrase:
        return ""
    s = phrase.lower()
    s = _PUNCT_TO_DROP.sub("", s)
    s = _WHITESPACE_COLLAPSE.sub(" ", s).strip()
    return s


def resolve_cohort_name(phrase: str) -> str | None:
    """Map a free-text medical phrase to a canonical cohort name.

    Resolution order:

    1. Exact match against any canonical name (after normalisation).
    2. Exact match against any alias (after normalisation).
    3. Substring match against aliases — longest matching alias wins.
       This handles input like "patients with myocardial infarction"
       where the alias ``"myocardial infarction"`` is a substring.

    Returns the canonical name on hit, or ``None`` when nothing
    matches. Case-insensitive throughout.

    Note on "longest wins": when "respiratory failure" matches both
    ``acute respiratory failure`` (`respiratory_failure`'s alias) and
    ``acute respiratory distress syndrome`` (`ards`'s alias), the
    longer alias substring of the input wins. This is what makes
    "acute respiratory failure" deterministically resolve to
    `respiratory_failure` even though both cohorts have "acute
    respiratory" in their alias text."""
    norm = _normalize_phrase(phrase)
    if not norm:
        return None

    cohorts = load_cohorts()

    # Tier 1: exact canonical-name match.
    if norm in cohorts:
        return norm

    # Tier 2: exact alias match (after normalisation on both sides).
    for name, defn in cohorts.items():
        for alias in defn.get("aliases") or []:
            if _normalize_phrase(alias) == norm:
                return name

    # Tier 3: substring match — longest matching alias wins.
    best_match: tuple[int, str] | None = None  # (alias_length, canonical_name)
    for name, defn in cohorts.items():
        for alias in defn.get("aliases") or []:
            alias_norm = _normalize_phrase(alias)
            if not alias_norm:
                continue
            if alias_norm in norm:
                # Tie-break by alias length so more specific aliases win.
                key = (len(alias_norm), name)
                if best_match is None or key > (best_match[0], best_match[1]):
                    best_match = (len(alias_norm), name)
    return best_match[1] if best_match else None


# ---------------------------------------------------------------------------
# SQL builders — used by the catalog generator AND the on-the-fly path
# ---------------------------------------------------------------------------


def _validate_icd_prefix(prefix: str) -> str:
    """Reject prefixes outside the ICD charset before composing SQL.

    The ICD code character set is letters + digits, with optional dot
    separators. Anything else (whitespace, quotes, semicolons, SQL
    wildcards ``%`` / ``_``, parentheses, etc.) is a sign the caller is
    feeding untrusted input and we MUST refuse before string-
    interpolating into SQL.

    Returns the validated (still dotted) prefix on success; raises
    :class:`ValueError` otherwise."""
    if not isinstance(prefix, str):
        raise ValueError(f"invalid ICD prefix (not a string): {prefix!r}")
    s = prefix.strip()
    if not s:
        raise ValueError("invalid ICD prefix (empty)")
    if not _VALID_ICD_PREFIX_RE.match(s):
        raise ValueError(
            f"invalid ICD prefix {prefix!r} (must match [A-Za-z0-9.])"
        )
    return s


def build_cohort_subquery_sql(
    *,
    icd10_prefixes: list[str] | None = None,
    icd9_prefixes: list[str] | None = None,
    diagnoses_icd_table: str = "diagnoses_icd",
) -> str:
    """Build a ``SELECT hadm_id FROM <diagnoses_icd_table> WHERE …``
    subquery matching any hadm_id whose ``icd_code`` starts with one
    of the supplied prefixes (per ICD version).

    MIMIC stores ICD codes WITHOUT dots. Prefixes are dot-stripped via
    :func:`normalize_icd_prefix` before being used in LIKE patterns.

    ``diagnoses_icd_table`` selects which storage the subquery targets:

    - ``"diagnoses_icd"`` (default) — local DuckDB.
    - ``"`physionet-data.mimiciv_3_1_hosp.diagnoses_icd`"`` (or similar
      backtick-quoted FQN) — BigQuery. The FQN is interpolated as-is
      so callers must pass it in the correct quoting for their dialect.

    All prefixes are validated against the ICD charset; any invalid
    prefix aborts the call before SQL is composed. The
    ``diagnoses_icd_table`` argument is itself validated to contain
    only safe identifier characters (letters, digits, dots, dashes,
    underscores, backticks) so it can't smuggle SQL injection. At
    least one non-empty prefix list must be supplied — otherwise the
    resulting subquery would match every hadm_id, which is surely
    not what the caller wanted.
    """
    icd10 = list(icd10_prefixes or [])
    icd9 = list(icd9_prefixes or [])
    if not icd10 and not icd9:
        raise ValueError(
            "at least one icd10_prefixes or icd9_prefixes entry required",
        )

    # Validate the diagnoses table identifier — accepts both bare names
    # ('diagnoses_icd') and BigQuery-style FQNs
    # ('`physionet-data.mimiciv_3_1_hosp.diagnoses_icd`'). Reject
    # anything outside the safe-identifier charset.
    if not re.match(r"^[A-Za-z0-9_`.\-]+$", diagnoses_icd_table):
        raise ValueError(
            f"invalid diagnoses_icd table identifier: {diagnoses_icd_table!r}"
        )

    # Validate all prefixes BEFORE composing any SQL fragment. A single
    # invalid prefix aborts the whole call.
    for p in icd10:
        _validate_icd_prefix(p)
    for p in icd9:
        _validate_icd_prefix(p)

    or_clauses: list[str] = []
    if icd10:
        like10 = " OR ".join(
            f"icd_code LIKE '{normalize_icd_prefix(p)}%'" for p in icd10
        )
        or_clauses.append(f"(icd_version = 10 AND ({like10}))")
    if icd9:
        like9 = " OR ".join(
            f"icd_code LIKE '{normalize_icd_prefix(p)}%'" for p in icd9
        )
        or_clauses.append(f"(icd_version = 9 AND ({like9}))")

    where = " OR ".join(or_clauses)
    return f"SELECT DISTINCT hadm_id FROM {diagnoses_icd_table} WHERE {where}"


def cohort_filter_sql(cohort_name: str) -> str:
    """Build the SQL subquery that selects hadm_ids belonging to a
    registered cohort. Wraps :func:`build_cohort_subquery_sql` with the
    registry lookup.

    Raises :class:`KeyError` when ``cohort_name`` is not in the
    registry (callers should pre-resolve aliases via
    :func:`resolve_cohort_name` before calling — this builder works
    on canonical names only)."""
    cohorts = load_cohorts()
    defn = cohorts.get(cohort_name)
    if defn is None:
        raise KeyError(
            f"unknown cohort {cohort_name!r}; "
            f"known: {sorted(cohorts.keys())}"
        )
    return build_cohort_subquery_sql(
        icd10_prefixes=defn.get("icd10_prefixes") or [],
        icd9_prefixes=defn.get("icd9_prefixes") or [],
    )
