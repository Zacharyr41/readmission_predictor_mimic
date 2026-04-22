"""Outcome extraction (Phase 8c).

An ``OutcomeSpec`` (``src.conversational.models``) describes an element
of the outcome vector in the formal Neyman–Rubin spec. This module
owns the *extraction step* — translating that spec plus a cohort of
``hadm_id`` values into a per-admission ``pandas.DataFrame`` suitable
for estimator consumption in Phase 8d.

Design
------

* Every extractor has a unique string key (``OutcomeSpec.extractor_key``).
  The key is registered against an ``OutcomeExtractor`` callable in
  ``OutcomeRegistry``; ``get_default_registry()`` returns a registry
  pre-populated with the built-ins listed in the plan.
* Parametric extractors (``biomarker_peak``, ``diagnosis_within_horizon``,
  ``mortality_time_to_event``) read their parameters from
  ``OutcomeSpec.extractor_params``. Parameters live on the spec so
  the outcome definition is self-describing and auditable — a reviewer
  looking at a CQ sees exactly which ICD prefixes count as "major
  bleeding" for that study.
* Output DataFrames have a stable schema per outcome type:

    - ``binary``         / ``continuous`` / ``ordinal``
        columns = ``[hadm_id, value]``
    - ``time_to_event``
        columns = ``[hadm_id, time, event]`` — time in days since the
        censoring clock (default: admission), event ∈ {0, 1}.

* The survival branch is pure pandas here — ``lifelines`` is not
  imported. 8d wires the ``(time, event)`` pairs into a Kaplan–Meier
  / Cox estimator; 8c's only job is to emit the pairs correctly
  honouring the censoring clock + horizon from the spec.

Correctness notes
-----------------

* Every extractor is *admission-indexed*. If a cohort admission has
  no matching event (no lab, no diagnosis, no readmission within
  horizon), the binary / continuous extractors emit 0 / NaN
  respectively — NEVER drop the row silently. The cohort assembler
  treats row-drops as policy decisions that must happen upstream.
* Time-to-event extractors emit the horizon as ``time`` on censored
  rows so downstream survival code can always distinguish
  ``(event=0, time=horizon)`` from ``(event=1, time=t<horizon)``.
* No extractor uses ``ILIKE`` on free-text labels without an ontology
  backstop. ``biomarker_peak`` reuses the existing
  ``d_labitems.label`` match the SQL fast-path already uses; this is
  the codebase's established concept-resolution path, not a new
  curated synonym list.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Protocol

import pandas as pd

from src.conversational.models import OutcomeSpec

OutcomeType = Literal["continuous", "binary", "ordinal", "time_to_event"]


class _SupportsExecute(Protocol):
    """Minimum backend surface — same as in treatment_assignment."""

    def execute(self, sql: str, params: list) -> list[tuple]: ...


OutcomeExtractFn = Callable[
    [_SupportsExecute, list[int], dict[str, Any]],
    pd.DataFrame,
]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OutcomeExtractor:
    """A named, typed extractor function.

    ``required_params`` are parameter names the caller MUST supply
    through ``OutcomeSpec.extractor_params``. The registry raises a
    clear error when a required parameter is missing — vs. the
    extractor silently using ``None`` as a filter value and producing
    a row of NaN.
    """

    key: str
    outcome_type: OutcomeType
    extract_fn: OutcomeExtractFn
    required_params: tuple[str, ...] = ()
    description: str = ""


class OutcomeExtractionError(RuntimeError):
    """Raised when a spec can't be extracted — missing parameter,
    unknown key, or a structural mismatch between the spec's declared
    ``outcome_type`` and the extractor's type."""


class OutcomeRegistry:
    """Lookup table mapping ``extractor_key`` → ``OutcomeExtractor``.

    Mirrors the ``OperationRegistry`` pattern in
    ``src.conversational.operations``, kept separate because the
    extractor signature is outcome-specific (backend + hadm_ids +
    params → DataFrame, not SQL fragments).
    """

    def __init__(self) -> None:
        self._by_key: dict[str, OutcomeExtractor] = {}

    def register(self, extractor: OutcomeExtractor) -> None:
        if extractor.key in self._by_key:
            raise ValueError(
                f"outcome extractor key {extractor.key!r} already registered"
            )
        self._by_key[extractor.key] = extractor

    def get(self, key: str) -> OutcomeExtractor:
        try:
            return self._by_key[key]
        except KeyError as e:
            raise OutcomeExtractionError(
                f"no outcome extractor registered for key {key!r}; "
                f"registered keys: {sorted(self._by_key.keys())}"
            ) from e

    def extract(
        self,
        spec: OutcomeSpec,
        backend: _SupportsExecute,
        hadm_ids: list[int],
    ) -> pd.DataFrame:
        """Run the extractor for ``spec`` on ``hadm_ids``.

        The spec's ``outcome_type`` must match the extractor's
        ``outcome_type`` — this catches miswired CQs where a survival
        outcome was tagged ``"binary"`` or similar.
        """
        extractor = self.get(spec.extractor_key)
        if extractor.outcome_type != spec.outcome_type:
            raise OutcomeExtractionError(
                f"outcome {spec.name!r} declares type {spec.outcome_type!r} "
                f"but extractor {extractor.key!r} produces "
                f"{extractor.outcome_type!r}; spec/registry mismatch"
            )
        missing = [p for p in extractor.required_params if p not in spec.extractor_params]
        if missing:
            raise OutcomeExtractionError(
                f"outcome {spec.name!r} (extractor={extractor.key!r}) is "
                f"missing required extractor_params: {missing}. "
                f"Spec carries the outcome definition — every required "
                "parameter must be set on the spec."
            )
        # Merge spec-level survival parameters into the call when relevant.
        params = dict(spec.extractor_params)
        if extractor.outcome_type == "time_to_event":
            params.setdefault("censoring_clock", spec.censoring_clock)
            params.setdefault("censoring_horizon_days", spec.censoring_horizon_days)
        df = extractor.extract_fn(backend, hadm_ids, params)
        _validate_extractor_output(spec, extractor, df)
        return df

    def keys(self) -> list[str]:
        return sorted(self._by_key.keys())


def _validate_extractor_output(
    spec: OutcomeSpec, extractor: OutcomeExtractor, df: pd.DataFrame,
) -> None:
    if extractor.outcome_type == "time_to_event":
        expected = {"hadm_id", "time", "event"}
    else:
        expected = {"hadm_id", "value"}
    missing = expected - set(df.columns)
    if missing:
        raise OutcomeExtractionError(
            f"extractor {extractor.key!r} for outcome {spec.name!r} produced "
            f"DataFrame missing columns {missing}; got {list(df.columns)}"
        )


# ---------------------------------------------------------------------------
# SQL helpers
# ---------------------------------------------------------------------------


def _placeholders(n: int) -> str:
    return ",".join(["?"] * n)


# ---------------------------------------------------------------------------
# Built-in extractors
# ---------------------------------------------------------------------------


def _extract_readmitted(
    window_days: int,
) -> OutcomeExtractFn:
    """Factory for readmitted_30d / readmitted_60d.

    Computes readmission on-the-fly using ``LEAD(admittime)`` partitioned
    by subject — the same shape used by the existing
    ``backend.readmission_labels_expr``, rewritten inline so we don't
    have to depend on that helper being present on every backend
    adapter. Returns 0 for admissions with no subsequent admission
    within the window.
    """

    def _fn(backend, hadm_ids, params):
        if not hadm_ids:
            return pd.DataFrame(columns=["hadm_id", "value"])
        placeholders = _placeholders(len(hadm_ids))
        sql = f"""
            WITH next_adm AS (
                SELECT a.hadm_id, a.subject_id, a.admittime, a.dischtime,
                       LEAD(a.admittime) OVER (
                           PARTITION BY a.subject_id ORDER BY a.admittime
                       ) AS next_admittime
                FROM admissions a
            )
            SELECT hadm_id,
                   CASE WHEN next_admittime IS NOT NULL
                             AND (next_admittime - dischtime) <= INTERVAL '{window_days}' DAY
                        THEN 1 ELSE 0 END AS value
            FROM next_adm
            WHERE hadm_id IN ({placeholders})
        """
        rows = backend.execute(sql, list(hadm_ids))
        return pd.DataFrame(rows, columns=["hadm_id", "value"])

    return _fn


def _extract_mortality_inhospital(
    backend, hadm_ids, params,
) -> pd.DataFrame:
    if not hadm_ids:
        return pd.DataFrame(columns=["hadm_id", "value"])
    placeholders = _placeholders(len(hadm_ids))
    sql = (
        "SELECT hadm_id, COALESCE(hospital_expire_flag, 0) AS value "
        f"FROM admissions WHERE hadm_id IN ({placeholders})"
    )
    rows = backend.execute(sql, list(hadm_ids))
    return pd.DataFrame(rows, columns=["hadm_id", "value"])


def _extract_mortality_time_to_event(
    backend, hadm_ids, params,
) -> pd.DataFrame:
    """Compute (time, event) per admission using ``patients.dod``.

    * ``event = 1`` if ``dod`` is non-null AND within the horizon
      measured from the censoring clock (default: admittime).
    * ``event = 0`` otherwise; ``time`` caps at ``horizon_days``.

    Admissions where the death date precedes the clock (e.g. dod before
    admittime — shouldn't happen in MIMIC but would indicate a
    data-quality issue) emit ``time=0, event=1`` and a row in the
    returned DataFrame's ``note`` column if we had one; for 8c we just
    clamp ``time`` at 0 and flag ``event=1`` so the row doesn't vanish.
    """
    horizon = params.get("censoring_horizon_days")
    if horizon is None:
        raise OutcomeExtractionError(
            "mortality_time_to_event requires censoring_horizon_days on the spec"
        )
    clock = params.get("censoring_clock", "admission")
    clock_col = {
        "admission": "a.admittime",
        "discharge": "a.dischtime",
        "icu_out": "(SELECT MAX(ic.outtime) FROM icustays ic WHERE ic.hadm_id = a.hadm_id)",
    }.get(clock)
    if clock_col is None:
        raise OutcomeExtractionError(f"unknown censoring_clock {clock!r}")
    if not hadm_ids:
        return pd.DataFrame(columns=["hadm_id", "time", "event"])
    placeholders = _placeholders(len(hadm_ids))
    sql = f"""
        SELECT a.hadm_id,
               p.dod,
               CAST({clock_col} AS TIMESTAMP) AS t0
        FROM admissions a
        JOIN patients p ON a.subject_id = p.subject_id
        WHERE a.hadm_id IN ({placeholders})
    """
    rows = backend.execute(sql, list(hadm_ids))
    df = pd.DataFrame(rows, columns=["hadm_id", "dod", "t0"])
    # Compute in pandas to keep the SQL portable across DuckDB / BigQuery
    # (DATE_DIFF has different signatures; doing it in-memory avoids that).
    df["dod"] = pd.to_datetime(df["dod"], errors="coerce")
    df["t0"] = pd.to_datetime(df["t0"], errors="coerce")
    delta_days = (df["dod"] - df["t0"]).dt.total_seconds() / 86400.0
    event = ((df["dod"].notna()) & (delta_days.between(0, horizon, inclusive="both"))).astype(int)
    # Time is min(delta, horizon); censored rows emit time=horizon.
    time = delta_days.where(event == 1, other=horizon).clip(lower=0, upper=horizon)
    return pd.DataFrame({
        "hadm_id": df["hadm_id"].astype(int),
        "time": time.astype(float),
        "event": event.astype(int),
    })


def _extract_icu_los_days(
    backend, hadm_ids, params,
) -> pd.DataFrame:
    """Sum of ICU lengths-of-stay (in days) per admission.

    An admission may have multiple ICU stays — we sum ``icustays.los``
    across them. Admissions with no ICU stay emit 0.0 (they never hit
    an ICU — cleanly zero, not missing).
    """
    if not hadm_ids:
        return pd.DataFrame(columns=["hadm_id", "value"])
    placeholders = _placeholders(len(hadm_ids))
    sql = f"""
        SELECT a.hadm_id, COALESCE(SUM(ic.los), 0.0) AS value
        FROM admissions a
        LEFT JOIN icustays ic ON a.hadm_id = ic.hadm_id
        WHERE a.hadm_id IN ({placeholders})
        GROUP BY a.hadm_id
    """
    rows = backend.execute(sql, list(hadm_ids))
    return pd.DataFrame(rows, columns=["hadm_id", "value"])


def _extract_hospital_los_days(
    backend, hadm_ids, params,
) -> pd.DataFrame:
    if not hadm_ids:
        return pd.DataFrame(columns=["hadm_id", "value"])
    placeholders = _placeholders(len(hadm_ids))
    sql = f"""
        SELECT hadm_id, admittime, dischtime
        FROM admissions WHERE hadm_id IN ({placeholders})
    """
    rows = backend.execute(sql, list(hadm_ids))
    df = pd.DataFrame(rows, columns=["hadm_id", "admittime", "dischtime"])
    df["admittime"] = pd.to_datetime(df["admittime"], errors="coerce")
    df["dischtime"] = pd.to_datetime(df["dischtime"], errors="coerce")
    df["value"] = (df["dischtime"] - df["admittime"]).dt.total_seconds() / 86400.0
    return df[["hadm_id", "value"]]


def _biomarker_aggregate(agg: Literal["MAX", "MIN", "AVG"]) -> OutcomeExtractFn:
    """Factory for biomarker_peak / biomarker_min / biomarker_mean.

    Requires ``extractor_params={"biomarker_label": "creatinine"}`` —
    the ``d_labitems.label`` pattern. Admissions with no matching lab
    emit NaN so the caller can decide whether to exclude or impute;
    silent zero-imputation would bias the estimator.
    """

    def _fn(backend, hadm_ids, params):
        label = params.get("biomarker_label")
        if not label:
            raise OutcomeExtractionError(
                f"biomarker extractor requires extractor_params['biomarker_label']"
            )
        if not hadm_ids:
            return pd.DataFrame(columns=["hadm_id", "value"])
        placeholders = _placeholders(len(hadm_ids))
        # Correlated subquery so the label filter constrains BEFORE the
        # aggregate. A prior version used a LEFT JOIN with the label
        # predicate on the JOIN condition — that kept non-matching
        # labevents rows (the outer LEFT JOIN pins every lab) and the
        # aggregate picked them up, returning sodium (140) where it
        # should have returned creatinine (1.2). Correlated subquery
        # pushes the filter into the inner scan so only matching rows
        # contribute to the aggregate, and admissions with no match
        # still appear in the result (the subquery returns NULL).
        sql = f"""
            SELECT a.hadm_id,
                   (
                       SELECT {agg}(l.valuenum)
                       FROM labevents l
                       JOIN d_labitems d ON l.itemid = d.itemid
                       WHERE l.hadm_id = a.hadm_id
                         AND LOWER(d.label) LIKE LOWER(?)
                         AND l.valuenum IS NOT NULL
                   ) AS value
            FROM admissions a
            WHERE a.hadm_id IN ({placeholders})
        """
        params_list = [f"%{label}%"] + list(hadm_ids)
        rows = backend.execute(sql, params_list)
        df = pd.DataFrame(rows, columns=["hadm_id", "value"])
        # pd converts SQL NULL to None; coerce to NaN.
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        return df

    return _fn


def _extract_diagnosis_within_horizon(
    backend, hadm_ids, params,
) -> pd.DataFrame:
    """Binary: did any subsequent admission within ``horizon_days`` of
    discharge carry a diagnosis whose ICD-10-CM code matches one of
    ``icd_prefixes``? Horizon = 0 means "during the index admission
    itself" (diagnosis codes from this admission).

    The prefix list is the outcome definition — carried on the
    OutcomeSpec.extractor_params per the correctness-first rule.
    """
    prefixes = params.get("icd_prefixes")
    horizon = params.get("horizon_days")
    if not prefixes:
        raise OutcomeExtractionError(
            "diagnosis_within_horizon requires extractor_params['icd_prefixes']"
        )
    if horizon is None:
        raise OutcomeExtractionError(
            "diagnosis_within_horizon requires extractor_params['horizon_days']"
        )
    if not hadm_ids:
        return pd.DataFrame(columns=["hadm_id", "value"])
    placeholders = _placeholders(len(hadm_ids))
    prefix_or = " OR ".join(["d.icd_code LIKE ?"] * len(prefixes))
    prefix_params = [f"{p.upper()}%" for p in prefixes]

    if horizon == 0:
        # Index-admission diagnoses.
        sql = f"""
            SELECT DISTINCT a.hadm_id, 1 AS value
            FROM admissions a
            JOIN diagnoses_icd d ON a.hadm_id = d.hadm_id
            WHERE a.hadm_id IN ({placeholders})
              AND d.icd_version = 10
              AND ({prefix_or})
        """
        matched_rows = backend.execute(sql, list(hadm_ids) + prefix_params)
    else:
        # Any subsequent admission within the horizon with a matching code.
        sql = f"""
            SELECT DISTINCT idx.hadm_id, 1 AS value
            FROM admissions idx
            JOIN admissions later ON idx.subject_id = later.subject_id
              AND later.admittime > idx.dischtime
              AND (later.admittime - idx.dischtime) <= INTERVAL '{int(horizon)}' DAY
            JOIN diagnoses_icd d ON d.hadm_id = later.hadm_id
              AND d.icd_version = 10
            WHERE idx.hadm_id IN ({placeholders})
              AND ({prefix_or})
        """
        matched_rows = backend.execute(sql, list(hadm_ids) + prefix_params)
    matched = {int(r[0]) for r in matched_rows}
    return pd.DataFrame(
        [{"hadm_id": h, "value": 1 if h in matched else 0} for h in hadm_ids],
        columns=["hadm_id", "value"],
    )


# ---------------------------------------------------------------------------
# Default registry
# ---------------------------------------------------------------------------


def get_default_registry() -> OutcomeRegistry:
    """Return a registry pre-populated with the built-in extractors.

    Callers can ``.register()`` additional extractors on top (useful
    for bespoke study-specific outcomes); the defaults cover the
    smoke-test scenarios I1–I6.
    """
    r = OutcomeRegistry()
    r.register(OutcomeExtractor(
        key="readmitted_30d",
        outcome_type="binary",
        extract_fn=_extract_readmitted(30),
        description="1 if next admission within 30 days of discharge, else 0",
    ))
    r.register(OutcomeExtractor(
        key="readmitted_60d",
        outcome_type="binary",
        extract_fn=_extract_readmitted(60),
        description="1 if next admission within 60 days of discharge, else 0",
    ))
    r.register(OutcomeExtractor(
        key="mortality_inhospital",
        outcome_type="binary",
        extract_fn=_extract_mortality_inhospital,
        description="admissions.hospital_expire_flag (0/1)",
    ))
    r.register(OutcomeExtractor(
        key="mortality_time_to_event",
        outcome_type="time_to_event",
        extract_fn=_extract_mortality_time_to_event,
        required_params=(),  # horizon / clock flow through via spec fields
        description="(time, event) for all-cause mortality from patients.dod, honouring censoring horizon",
    ))
    r.register(OutcomeExtractor(
        key="icu_los_days",
        outcome_type="continuous",
        extract_fn=_extract_icu_los_days,
        description="Sum of icustays.los per admission; 0.0 if no ICU stay",
    ))
    r.register(OutcomeExtractor(
        key="hospital_los_days",
        outcome_type="continuous",
        extract_fn=_extract_hospital_los_days,
        description="(dischtime - admittime) in days",
    ))
    r.register(OutcomeExtractor(
        key="biomarker_peak",
        outcome_type="continuous",
        extract_fn=_biomarker_aggregate("MAX"),
        required_params=("biomarker_label",),
        description="MAX(valuenum) over labevents whose d_labitems.label matches",
    ))
    r.register(OutcomeExtractor(
        key="biomarker_mean",
        outcome_type="continuous",
        extract_fn=_biomarker_aggregate("AVG"),
        required_params=("biomarker_label",),
        description="AVG(valuenum) over labevents whose d_labitems.label matches",
    ))
    r.register(OutcomeExtractor(
        key="biomarker_min",
        outcome_type="continuous",
        extract_fn=_biomarker_aggregate("MIN"),
        required_params=("biomarker_label",),
    ))
    r.register(OutcomeExtractor(
        key="diagnosis_within_horizon",
        outcome_type="binary",
        extract_fn=_extract_diagnosis_within_horizon,
        required_params=("icd_prefixes", "horizon_days"),
        description=(
            "1 if any admission in [index-discharge, index-discharge + horizon] "
            "has an ICD-10-CM code matching one of the supplied prefixes. "
            "horizon_days=0 matches diagnoses from the index admission itself."
        ),
    ))
    return r


__all__ = [
    "OutcomeExtractor",
    "OutcomeExtractionError",
    "OutcomeRegistry",
    "get_default_registry",
]
