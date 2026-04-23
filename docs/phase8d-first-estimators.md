# Phase 8d: First Real Estimators (T/S/X-learner + Bootstrap)

Phase 8d replaces the 8a NaN stub at `src/causal/run.py` with a real
causal-inference dispatch. After 8d, `run_causal(cq, backend)` returns
a populated `CausalEffectResult` with bootstrap confidence intervals
— `is_stub=False`, real μ_c / μ_{c,k} / τ_{c,c'}, and an overlap
diagnostic.

- Plan file: `/Users/zacharyrothstein/.claude/plans/vivid-knitting-forest.md`
- Phases 8a/8b/8c (prior): `52f65e0` / `b9f91d2` / `c47ad76`
- Phase 8d commits (this PR): `ff167c5` (RED) → `42cbe1e` (scaffold) →
  `ce030a8` (T + bootstrap) → `2dca600` (S + X) → `34f7075` (run_causal)

## Overview

A ``CompetencyQuestion`` with ``scope='causal_effect'`` now flows
through five stages in sequence:

```
CompetencyQuestion
   │
   │   src/causal/run.py::run_causal            (entry, dispatch)
   ▼
_validate_causal_cq                              src/causal/run.py:61-92
   │
   ▼
AggregationSpec + OutcomeSpec guards             src/causal/run.py:147-170
  (SurvivalNotYetSupported, AggregationNotYetSupported)
   │
   ▼
EstimatorRegistry lookup                         src/causal/estimators/registry.py:23-105
  (require + check_outcome_type)
   │
   ▼
build_cohort_frame                               src/causal/cohort.py:100-240
  (or cohort_frame= injection for tests)
   │
   ▼
BootstrapRunner (once per outcome)               src/causal/estimators/base.py:120-307
  ├─ stratified resample                         base.py:202-223
  ├─ estimator fit/predict per replicate         metalearners.py
  ├─ quantile CIs (paired τ)                     base.py:258-278
  └─ always-fit propensity + overlap             _propensity.py
   │
   ▼
merged CausalEffectResult                        src/causal/models.py:88-117
  (mu_c, mu_c_k, tau_cc_prime, ranking, diagnostics, ledger)
```

## The five locked decisions

All five are recorded in the plan at
`/Users/zacharyrothstein/.claude/plans/vivid-knitting-forest.md`.
Implementation pointers follow.

### 1. Stratified bootstrap by arm

Bootstrap resamples preserve per-arm row counts exactly — rare arms
don't get wiped out in a replicate. Implemented at
`src/causal/estimators/base.py::BootstrapRunner._stratified_resample`
(lines ~202-223). Regression-tested by
`tests/test_causal/test_bootstrap.py::TestStratifiedResample::test_per_arm_sizes_preserved`.

### 2. `query_patient_covariates=None` ⇒ ATE = average CATE

When no query patient is supplied, `run_causal` returns the
population estimand by predicting μ_c on every cohort row and
averaging. This differs from "predict at mean X" when the base
learner is non-linear (XGBoost). Implemented at
`src/causal/estimators/base.py::BootstrapRunner._target_X`
(lines ~233-239) and used in `run()` (lines ~248-254).

### 3. Always-fit propensity for the overlap diagnostic

`BootstrapRunner` always fits a multinomial propensity on the full
cohort as its last step and populates `DiagnosticReport.overlap`
with `arm_{c}_{min,max}_propensity` for every arm — regardless of
whether the selected estimator family consumes propensity
internally. Keeps the diagnostic shape consistent across
estimators so 8h + the UI don't branch on `estimator_family`.
Implemented at `src/causal/estimators/_propensity.py::fit_propensity`
and `overlap_from_propensity`, invoked from `BootstrapRunner.run`
(lines ~286-297).

### 4. XGBoost for continuous base learner

Continuous outcomes fit via `xgboost.XGBRegressor(n_estimators=200)`
— already a repo dep through `src/prediction/model.py`. Binary and
ordinal use `sklearn.linear_model.LogisticRegression`. The dispatch
lives in `src/causal/estimators/_base_learners.py::make_base_learner`.
Survival outcomes raise `SurvivalNotYetSupported` here (Phase 8g).

### 5. Plan file under `~/.claude/plans/`

Plan persists at
`/Users/zacharyrothstein/.claude/plans/vivid-knitting-forest.md`
and is cited from every module docstring in
`src/causal/estimators/`. The decision-number references (`decision #1`
..  `decision #4`) appear in module docstrings and in `BootstrapRunner`'s
class docstring so a reader of the code can trace back without
needing the PR description.

## Estimator registry

`EstimatorRegistry` mirrors `OperationRegistry` at
`src/conversational/operations.py:264-328` — same
`register` / `get` / `require` / `describe_for_prompt` trio, same
duplicate-rejection discipline. Single-string key (only one kind of
entry).

The three built-in estimators:

| Key | Class | Outcome types | Propensity | Source |
|---|---|---|---|---|
| `"t_learner"` (default) | `TLearnerAdapter` | binary, continuous, ordinal | external | hand-rolled |
| `"s_learner"` | `SLearnerAdapter` | binary, continuous, ordinal | external | hand-rolled |
| `"x_learner"` | `XLearnerAdapter` | binary, continuous, ordinal | built-in | wraps `econml.metalearners.XLearner` |

Default registry: `src/causal/estimators/registry.py::get_default_registry`.
Picking the default happens via `CompetencyQuestion.estimator_family`
(default `"t_learner"` per `src/conversational/models.py:212`).

**Why hand-roll T/S but econml-wrap X?** The T-learner recipe is
literally "one outcome regressor per arm" and the S-learner recipe is
"one outcome regressor with treatment one-hot appended to X" — both
need zero econml ceremony. The X-learner, on the other hand, involves
per-arm outcome regressors + cross-imputed pseudo-outcomes +
propensity-weighted combination, and econml's implementation is
correct for C ≥ 2 out of the box. Matches the plan's "econml backbone
from 8d onwards" decision without dragging ceremony into the trivial
cases.

## What 8d does *not* do

These all land in later sub-phases. 8d rejects them with typed errors
citing the next phase:

| Capability | Error | Next phase |
|---|---|---|
| Multi-outcome composite (weighted_sum / dominant / utility) | `AggregationNotYetSupported` ("Phase 8f") | 8f |
| Survival outcomes (`OutcomeSpec.outcome_type='time_to_event'`) | `SurvivalNotYetSupported` ("Phase 8g") | 8g |
| AIPW / TMLE / Causal Forest / BART | n/a — just absent from the registry | 8g |
| Asymptotic / Bayesian uncertainty | n/a — `BootstrapRunner` is the only runner | 8g |
| Balance / positivity / extrapolation / missingness diagnostics | `DiagnosticReport.notes` says "arrive in 8h" | 8h |
| Mode downgrade on failed assumption | `mode` stays `"associative"`, ledger stays `"declared"` | 8h |
| LLM prompt surface + Streamlit UX | n/a | 8i |

## Example

Binary outcome, binary intervention — the I3 shape:

```python
from src.causal.run import run_causal
from src.conversational.models import (
    AggregationSpec, CompetencyQuestion, InterventionSpec, OutcomeSpec,
)

# Labels must match the cohort's intervention_labels. When run_causal
# builds the cohort via the DuckDB path, the resolver + assigner
# populate labels from InterventionSpec.label.
cq = CompetencyQuestion(
    original_question="Effect of tPA on 30-day readmission",
    scope="causal_effect",
    intervention_set=[
        InterventionSpec(label="tPA", kind="drug", rxnorm_ingredient="8410"),
        InterventionSpec(
            label="no_tPA", kind="drug", rxnorm_ingredient="8410", is_control=True,
        ),
    ],
    outcome_vector=[
        OutcomeSpec(name="readmitted_30d", outcome_type="binary",
                    extractor_key="readmitted_30d"),
    ],
    aggregation_spec=AggregationSpec(kind="identity"),
    estimator_family="t_learner",
    uncertainty_reps=200,  # B for bootstrap (plan default)
    random_state=0,
)

result = run_causal(cq, backend=duckdb_backend)

# mu_c[label] — per-arm predicted outcome at the cohort-average X.
print(result.mu_c["tPA"].point, result.mu_c["tPA"].lower, result.mu_c["tPA"].upper)
# mu_c_k[f"{label}|{outcome}"] — per-arm × per-outcome.
print(result.mu_c_k["tPA|readmitted_30d"])
# tau_cc_prime[lex-ordered-key] — pairwise treatment effect with paired CI.
print(result.tau_cc_prime["no_tPA|tPA"])
# DiagnosticReport.overlap — per-arm min/max propensity (decision #3).
print(result.diagnostics.overlap)
# Not a stub anymore.
assert result.is_stub is False
# Ledger still "declared" — 8h flips to "passed"/"failed".
print([(a.name, a.status) for a in result.assumption_ledger])
```

For tests or callers that already have a `CohortFrame` (e.g., hand-
built synthetic cohorts for correctness testing), pass it directly
and skip cohort assembly:

```python
from tests.test_causal.conftest import make_synthetic_cohort_frame

cohort = make_synthetic_cohort_frame(
    n_per_arm=120, n_arms=2, ate=2.0, seed=0,
)
result = run_causal(cq, backend=None, cohort_frame=cohort)
```

## Test matrix

Ten new test files under `tests/test_causal/` landed in the RED
commit (`ff167c5`). All go green by the end of 8d.

| Test file | Covers | Flips which xfail? |
|---|---|---|
| `test_estimator_registry.py` | `EstimatorRegistry` plumbing, `check_outcome_type`, default registry | — |
| `test_bootstrap.py` | reproducibility, stratified-resample size preservation, wider-CI-at-smaller-B, paired τ narrower than naive | — |
| `test_t_learner_recovers_known_ate.py` | T-learner recovers programmed ATE=2.0 within bootstrap SE; DGP-bias sanity check | — |
| `test_s_learner_multi_arm.py` | C=3 S-learner: non-NaN μ_c, τ matrix keys, ranking length | — |
| `test_x_learner_propensity_diagnostic.py` | X-learner overlap keys + value range | — |
| `test_run_causal_end_to_end_binary.py` | `run_causal` binary outcome end-to-end via `cohort_frame=` | — |
| `test_run_causal_end_to_end_nary.py` | C=3 end-to-end; covers the intent of the removed I5 xfail | was `test_run_causal_returns_real_n_arms_for_c_equals_3` |
| `test_run_causal_per_outcome_breakdown.py` | n=3 scalar outcomes; covers the scalar-outcome intent of the remaining-xfailed I6 | superseded by scope |
| `test_run_causal_rejects_survival_outcome.py` | guard rail: `time_to_event` → `SurvivalNotYetSupported` (cites 8g) | — |
| `test_run_causal_rejects_nonidentity_aggregation.py` | guard rail: `weighted_sum` → `AggregationNotYetSupported` (cites 8f) | — |

Remaining xfails after 8d:

- `tests/test_conversational/test_intervention_effect.py::test_run_causal_returns_real_per_outcome_breakdown_for_n_equals_3`
  — structurally 8g (survival) + 8f (aggregation); flips when both land.
- `tests/test_conversational/test_intervention_effect.py::test_positivity_failure_triggers_associative_mode`
  — 8h diagnostic + mode downgrade.

Full suite counts before/after 8d:

| | passed | skipped | xfailed |
|---|---|---|---|
| Before 8d | 1310 | 3 | 3 |
| After 8d | 1343 | 3 | 2 |

Net +33 passed, −1 xfailed, zero regressions.

## Running the suite

```
# Phase 8d new tests only
.venv/bin/python -m pytest tests/test_causal -v

# Intervention-effect spec tests (where the remaining xfails live)
.venv/bin/python -m pytest tests/test_conversational/test_intervention_effect.py -v

# Full repo
.venv/bin/python -m pytest tests/
```
