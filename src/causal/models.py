"""Output-side pydantic models for the causal-inference pipeline.

Input-side specs (``InterventionSpec``, ``OutcomeSpec``,
``AggregationSpec``) are part of the ``CompetencyQuestion`` extension
surface and live in ``src.conversational.models`` so the decomposer +
planner can construct them without importing from this package.

This module owns the result-side contract: everything ``run_causal``
returns must conform to these models. The names and structure match the
formal spec (§7):

  * ``CausalEffectResult.mu_c``       → §7.1   μ_c(x) per intervention.
  * ``CausalEffectResult.mu_c_k``     → §7.2   μ_{c,k}(x) per (intervention, outcome).
  * ``CausalEffectResult.tau_cc_prime``→ §7.3  τ_{c,c'}(x) pairwise CATE.
  * ``CausalEffectResult.ranking``    → §7.4   intervention ranking.
  * ``CausalEffectResult.diagnostics``→ §7.5   overlap / balance / positivity / …
  * ``CausalEffectResult.mode`` +
    ``assumption_ledger``             → §7.6   causal vs. associative.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class UncertaintyInterval(BaseModel):
    """A point estimate with a (1-α) uncertainty interval.

    Shape is deliberately minimal: ``point`` is the estimator's best
    estimate, ``lower``/``upper`` are the interval bounds. The interval
    kind (confidence / credible / prediction) is recorded on the
    enclosing ``CausalEffectResult`` via ``uncertainty_kind`` — we don't
    repeat it per datum to keep the payload compact.
    """

    model_config = {"extra": "forbid"}

    point: float
    lower: float
    upper: float

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.point, self.lower, self.upper)


class DiagnosticReport(BaseModel):
    """Spec §7.5 diagnostics.

    Phase 8a emits an empty-but-well-typed report; phase 8h fills each
    field with real computations (overlap KDE summary, balance SMDs,
    positivity bound for the query point x, extrapolation flag, per-arm
    missingness).
    """

    model_config = {"extra": "forbid"}

    overlap: dict[str, float] = Field(default_factory=dict)
    balance: dict[str, float] = Field(default_factory=dict)
    positivity: dict[str, float] = Field(default_factory=dict)
    extrapolation_flag: bool = False
    missingness: dict[str, float] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class AssumptionClaim(BaseModel):
    """One row of the assumption ledger (spec §§5, 7.6).

    ``name`` is a short identifier (``"consistency"``, ``"ignorability"``,
    ``"positivity"``, ``"sutva"``). ``status`` is one of:

      * ``"declared"`` — caller asserted the assumption; not checked.
      * ``"passed"``   — diagnostic check confirmed (phase 8h only).
      * ``"failed"``   — diagnostic check rejected — forces associative mode.

    ``detail`` carries the diagnostic evidence (min propensity, max SMD,
    etc.) so the UI can surface why an assumption failed.
    """

    model_config = {"extra": "forbid"}

    name: Literal["consistency", "ignorability", "positivity", "sutva"]
    status: Literal["declared", "passed", "failed"]
    detail: str = ""


class CausalEffectResult(BaseModel):
    """Spec §7 output contract.

    Keys in ``mu_c``, ``mu_c_k``, ``tau_cc_prime`` are string-encoded
    rather than tuple-keyed because pydantic + JSON serialization
    requires string keys. Encoding convention:

      * ``mu_c[c_label]``                → intervention label.
      * ``mu_c_k["<c_label>|<k_label>"]`` → "<intervention>|<outcome>".
      * ``tau_cc_prime["<c>|<c_prime>"]`` → ordered pair "<c>|<c_prime>".

    ``ranking`` is a list of intervention labels ordered best (rank 1)
    to worst by the composite scalar μ_c.
    """

    model_config = {"extra": "forbid"}

    mu_c: dict[str, UncertaintyInterval]
    mu_c_k: dict[str, UncertaintyInterval] = Field(default_factory=dict)
    tau_cc_prime: dict[str, UncertaintyInterval] = Field(default_factory=dict)
    ranking: list[str] = Field(default_factory=list)
    diagnostics: DiagnosticReport = Field(default_factory=DiagnosticReport)
    mode: Literal["causal", "associative"] = "associative"
    assumption_ledger: list[AssumptionClaim] = Field(default_factory=list)
    uncertainty_kind: Literal["confidence", "credible", "prediction"] = "confidence"
    alpha: float = 0.05
    # Phase 8a — stub results carry this flag so the orchestrator + UI
    # can surface "not a real estimate yet" to the user during the demo.
    is_stub: bool = False
