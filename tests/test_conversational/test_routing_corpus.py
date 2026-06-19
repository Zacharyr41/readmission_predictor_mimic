"""Routing-corpus measurement — the "measure" half of §8 "D0".

The labeled corpus under ``fixtures/routing_corpus/`` tags real clinical
questions with the route they *should* take. These tests turn that into a
**measurable misrouting rate** and a **regression gate**, using a dual scheme
(a bare rate is directional-blind — it can't tell a fix from a fresh break):

1. ``test_every_corpus_entry_classifies_cleanly`` — health: each CQ validates
   and classifies to a real plan without raising.
2. ``test_no_unexpected_route_flips`` — the hard gate: the live route must equal
   the committed ``current_plan`` snapshot, so ANY change at the choke point
   (in either direction) fails loudly. This is the "D0 changes no routing
   behavior" contract. A deliberate routing change (Directions A/B/C) must
   re-baseline ``current_plan`` (re-run ``scripts/seed_routing_corpus.py``) in
   the same PR, making the routing diff explicit and reviewable.
3. ``test_misrouting_rate_within_baseline`` — the trend ratchet + dashboard:
   count how many questions route somewhere other than ``desired_plan``, keep it
   at/below a pinned ceiling, and print a per-reason / per-tag breakdown.
"""

from __future__ import annotations

from collections import Counter

import pytest

from src.conversational.models import CompetencyQuestion
from src.conversational.planner import QueryPlan, QueryPlanner
from tests.test_conversational.conftest import load_routing_corpus

# Ratchet: the number of corpus questions allowed to route to something other
# than their ``desired_plan``. After Direction A (split the temporal veto) the
# three window-style temporal cases now fast-path correctly; the one remaining
# known misroute is the metadata-only LOS question (rule 5, awaiting Direction
# C). Lower this further as later directions land and re-baseline current_plan
# with the seeder.
MAX_KNOWN_MISROUTES = 1

_VALID_PLANS = {p.value for p in QueryPlan}


@pytest.mark.parametrize("case", load_routing_corpus())
def test_every_corpus_entry_classifies_cleanly(case):
    cq = CompetencyQuestion.model_validate(case["cq"])
    plan = QueryPlanner().classify(cq)
    assert isinstance(plan, QueryPlan)
    assert case["desired_plan"] in _VALID_PLANS, case["name"]
    assert case["current_plan"] in _VALID_PLANS, case["name"]


@pytest.mark.parametrize("case", load_routing_corpus())
def test_no_unexpected_route_flips(case):
    cq = CompetencyQuestion.model_validate(case["cq"])
    actual = QueryPlanner().classify(cq).value
    assert actual == case["current_plan"], (
        f"route flip for {case['name']!r}: committed current_plan="
        f"{case['current_plan']!r} but the planner now returns {actual!r}. "
        "If this change is intentional, re-run scripts/seed_routing_corpus.py "
        "to re-baseline and review the resulting fixture diff."
    )


def test_misrouting_rate_within_baseline():
    corpus = [p.values[0] for p in load_routing_corpus()]
    assert corpus, "routing corpus is empty — run scripts/seed_routing_corpus.py"
    planner = QueryPlanner()

    misrouted: list[tuple[str, str, str, str]] = []
    plan_counter: Counter[str] = Counter()
    reason_counter: Counter[str] = Counter()
    tag_counter: Counter[str] = Counter()
    for case in corpus:
        cq = CompetencyQuestion.model_validate(case["cq"])
        decision = planner.explain(cq)
        plan_counter[decision.plan.value] += 1
        if decision.plan.value != case["desired_plan"]:
            misrouted.append(
                (case["name"], case["desired_plan"], decision.plan.value,
                 decision.reason.value)
            )
            reason_counter[decision.reason.value] += 1
            for tag in case.get("tags", []):
                tag_counter[tag] += 1

    n = len(corpus)
    rate = len(misrouted) / n
    report = [
        "",
        f"=== routing corpus: {n} questions ===",
        f"plan distribution : {dict(sorted(plan_counter.items()))}",
        f"misrouting rate   : {rate:.3f} ({len(misrouted)}/{n})",
        f"misroute reasons  : {dict(sorted(reason_counter.items()))}",
        f"misroute tags     : {dict(sorted(tag_counter.items()))}",
        "misrouted (name: desired -> actual [reason]):",
        *[f"  - {name}: {desired} -> {actual} [{reason}]"
          for name, desired, actual, reason in misrouted],
    ]
    print("\n".join(report))

    assert len(misrouted) <= MAX_KNOWN_MISROUTES, (
        f"{len(misrouted)} corpus questions misroute, above the "
        f"{MAX_KNOWN_MISROUTES} known. New/regressed misroutes:\n"
        + "\n".join(f"  {m[0]}: {m[1]} -> {m[2]} [{m[3]}]" for m in misrouted)
    )
