#!/usr/bin/env python
"""Audit query-routing decisions — the "observe" half of §8 "D0".

Two modes:

**Offline** (default) — read the decision log written by
``decision_log.log_routing_decision`` (``$NEUROGRAPH_ROUTING_LOG`` or
``logs/routing_decisions.jsonl``) and print where real traffic is going: the
plan distribution, the reason/rule histogram (which §4.1 rule fired how often),
per-turn route mixes (a single turn can split across SQL + graph), and the count
of degenerate-causal fall-throughs. Pure stdlib; no LLM, no network::

    .venv/bin/python scripts/routing_audit.py
    .venv/bin/python scripts/routing_audit.py --log /tmp/routing_demo.jsonl

**Live** (``--live``, gated on ``ANTHROPIC_API_KEY``) — run each corpus *question*
through the real decomposer + planner and compare to its ``desired_plan``,
measuring *decomposer-induced* misrouting (§6.2) that the offline CQ-level corpus
can't see. Hits the Anthropic API (costs tokens); never run in CI::

    .venv/bin/python scripts/routing_audit.py --live
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG = REPO_ROOT / "logs" / "routing_decisions.jsonl"
CORPUS_DIR = (
    REPO_ROOT / "tests" / "test_conversational" / "fixtures" / "routing_corpus"
)
# Record shapes this tool understands. Lines tagged otherwise are skipped rather
# than crashing the audit (the full-CQ dump can drift across schema versions).
KNOWN_SCHEMA_VERSIONS = {"1"}


# ---------------------------------------------------------------------------
# Offline: analyze the decision log
# ---------------------------------------------------------------------------


def _resolve_log_path(arg: str | None) -> Path:
    if arg:
        return Path(arg)
    env = os.environ.get("NEUROGRAPH_ROUTING_LOG")
    return Path(env) if env else DEFAULT_LOG


def read_log(path: Path) -> tuple[list[dict], int]:
    """Return (routing-decision records, count of skipped lines)."""
    records: list[dict] = []
    skipped = 0
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            if rec.get("kind") != "routing_decision":
                continue
            if rec.get("schema_version") not in KNOWN_SCHEMA_VERSIONS:
                skipped += 1
                continue
            records.append(rec)
    return records, skipped


def summarize_log(records: list[dict], skipped: int) -> None:
    if not records:
        print("no routing-decision records found.")
        if skipped:
            print(f"(skipped {skipped} unparseable / unknown-schema lines)")
        return

    n = len(records)
    plan_counter = Counter(r.get("plan") for r in records)
    reason_counter = Counter(r.get("reason") for r in records)
    rule_counter = Counter(r.get("rule") for r in records)
    scope_counter = Counter(r.get("scope") for r in records)
    fallthrough = sum(1 for r in records if r.get("had_causal_fallthrough"))

    by_turn: dict[str, list[str]] = defaultdict(list)
    for r in records:
        by_turn[r.get("turn_id")].append(r.get("plan"))
    multi_turns = {t: p for t, p in by_turn.items() if len(p) > 1}
    mixed_turns = {t: p for t, p in multi_turns.items() if len(set(p)) > 1}

    def _fmt(counter: Counter) -> str:
        return ", ".join(f"{k}={v}" for k, v in counter.most_common())

    print(f"=== routing decision log: {n} decisions across {len(by_turn)} turns ===")
    print(f"plan distribution : {_fmt(plan_counter)}")
    print(f"reason histogram  : {_fmt(reason_counter)}")
    rules = sorted(rule_counter.items(), key=lambda kv: (kv[0] is None, kv[0]))
    print("rule histogram    : " + ", ".join(f"rule{k}={v}" for k, v in rules))
    print(f"scope histogram   : {_fmt(scope_counter)}")
    print(f"causal fall-throughs (|I|<2): {fallthrough}")
    print(
        f"multi-CQ turns    : {len(multi_turns)} "
        f"(of which mixed-route: {len(mixed_turns)})"
    )
    for turn, plans in list(mixed_turns.items())[:5]:
        print(f"  - turn {turn}: {plans}")
    if skipped:
        print(f"(skipped {skipped} unparseable / unknown-schema lines)")


# ---------------------------------------------------------------------------
# Live: end-to-end decomposer + planner over the corpus
# ---------------------------------------------------------------------------


def run_live_corpus(corpus_dir: Path) -> int:
    import anthropic

    from src.conversational.decomposer import decompose_question
    from src.conversational.planner import QueryPlanner

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        try:
            from config.settings import Settings

            api_key = Settings().anthropic_api_key
        except Exception:
            api_key = None
    if not api_key:
        print("ANTHROPIC_API_KEY not set — skipping live corpus run.")
        return 1

    cases = [json.loads(p.read_text()) for p in sorted(corpus_dir.glob("*.json"))]
    if not cases:
        print(f"no corpus fixtures in {corpus_dir}")
        return 1

    client = anthropic.Anthropic(api_key=api_key)
    planner = QueryPlanner()

    misrouted: list[tuple[str, str, list[str]]] = []
    errors: list[tuple[str, str]] = []
    for case in cases:
        question = case["question"]
        desired = case["desired_plan"]
        try:
            decomp = decompose_question(client, question)
            plans = [planner.classify(cq).value for cq in decomp.competency_questions]
        except Exception as exc:  # an LLM hiccup shouldn't abort the whole audit
            errors.append((case["name"], type(exc).__name__))
            print(f"[ERROR ] {case['name']}: {type(exc).__name__}")
            continue
        matched = desired in plans
        print(
            f"[{'ok' if matched else 'MISROUTE':8s}] {case['name']}: "
            f"desired={desired} got={plans}"
        )
        if not matched:
            misrouted.append((case["name"], desired, plans))

    n = len(cases) - len(errors)
    print()
    print(f"=== live end-to-end: {n} questions decomposed ===")
    if n:
        rate = len(misrouted) / n
        print(f"end-to-end misrouting rate: {rate:.3f} ({len(misrouted)}/{n})")
    if errors:
        print(f"decompose errors ({len(errors)}): {errors}")
    return 0


# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Audit query-routing decisions.")
    ap.add_argument(
        "--log",
        help="path to routing_decisions.jsonl "
        "(default: $NEUROGRAPH_ROUTING_LOG or logs/routing_decisions.jsonl)",
    )
    ap.add_argument(
        "--live",
        action="store_true",
        help="run corpus questions through the real decomposer+planner "
        "(needs ANTHROPIC_API_KEY; costs tokens)",
    )
    ap.add_argument("--corpus-dir", help="corpus fixture dir for --live")
    args = ap.parse_args(argv)

    if args.live:
        corpus_dir = Path(args.corpus_dir) if args.corpus_dir else CORPUS_DIR
        return run_live_corpus(corpus_dir)

    path = _resolve_log_path(args.log)
    if not path.exists():
        print(
            f"routing log not found: {path}\n"
            "Run questions through ConversationalPipeline.ask() first, or point "
            "--log at an existing file."
        )
        return 1
    print(f"reading {path}")
    records, skipped = read_log(path)
    summarize_log(records, skipped)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
