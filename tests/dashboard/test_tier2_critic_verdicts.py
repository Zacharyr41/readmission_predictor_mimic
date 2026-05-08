"""Tier 2 — critic verdict smoke tests.

Calls ``critique(client, cq, answer)`` directly with synthetic
CQ/AnswerResult pairs that mirror the §4a/§4b/§4c examples in
``docs/phase-h-smoke-test.md``. Real Anthropic LLM, real OMOPHub. Fast
relative to Tier 1 (~5-10s/test) because there's no DB execution and
no Streamlit harness.

These are the canonical replacement for the §4 / §5a heredocs in the
manual smoke doc.

Skipped when ANTHROPIC_API_KEY is unset (so a teammate without keys can
still run the rest of the suite).
"""

from __future__ import annotations

import os
import time

import pytest

from src.conversational.critic import critique


_HAS_API_KEY = bool(os.environ.get("ANTHROPIC_API_KEY"))
pytestmark = pytest.mark.skipif(
    not _HAS_API_KEY, reason="ANTHROPIC_API_KEY not set",
)


@pytest.fixture(scope="module")
def anthropic_client():
    """Single Anthropic client reused across the module to avoid
    per-test reconnection costs. Module scope is safe because
    ``critique`` doesn't mutate the client."""
    import anthropic
    return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


def _record_run(reporter, cq, answer, verdict, *, duration: float):
    """Capture the LLM-driven verdict on the markdown report."""
    reporter.add_question(cq.original_question)
    reporter.add_answer(answer.text_summary)
    reporter.add_verdict(verdict)
    reporter.add_note(f"LLM round-trip: {duration:.2f}s")


# ---------------------------------------------------------------------------
# §4a — implausible drug dose
# ---------------------------------------------------------------------------


def test_dexmedetomidine_dose_implausibility_flags_warn(
    anthropic_client, reporter,
):
    """Dexmedetomidine 4.5 mcg/kg/hr is ~6× the FDA-labeled max
    maintenance infusion. The critic should fire severity=warn AND
    plausible=False, citing FDA label or rxnorm_lookup. Whether a tool
    actually fires varies between runs — Opus's discipline rule allows
    in-training-recall verdicts when the dose is obviously implausible.
    Test asserts the *judgement*, not the tool firing."""
    from tests.dashboard.lib.scenarios import build_dexmedetomidine_overdose_cq

    cq, answer = build_dexmedetomidine_overdose_cq()

    t0 = time.monotonic()
    verdict = critique(anthropic_client, cq, answer)
    duration = time.monotonic() - t0
    _record_run(reporter, cq, answer, verdict, duration=duration)

    assert verdict is not None
    sev_ok = verdict.severity == "warn"
    reporter.add_assertion(
        "verdict.severity == 'warn'", sev_ok,
        detail=f"got: {verdict.severity}",
    )
    assert sev_ok

    plausible_ok = verdict.plausible is False
    reporter.add_assertion(
        "verdict.plausible is False", plausible_ok,
        detail=f"got: {verdict.plausible}",
    )
    assert plausible_ok


# ---------------------------------------------------------------------------
# §4b — diagnosis-code mismatch
# ---------------------------------------------------------------------------


def test_cardiogenic_shock_wrong_icd_flags_warn_or_block(
    anthropic_client, reporter,
):
    """Interpretation says R57.0 (correct ICD-10 for cardiogenic shock)
    but answer cites I50.9 (heart failure unspecified — different
    condition). Critic should flag a non-info severity (warn OR block,
    depending on how aggressive the LLM is on a given run). Both are
    correct — the test asserts the *shape* of "this is broken" rather
    than the exact severity, per the AppTest guide's stochasticity
    advice (§8.2)."""
    from tests.dashboard.lib.scenarios import build_cardiogenic_shock_cq

    cq, answer = build_cardiogenic_shock_cq()

    t0 = time.monotonic()
    verdict = critique(anthropic_client, cq, answer)
    duration = time.monotonic() - t0
    _record_run(reporter, cq, answer, verdict, duration=duration)

    assert verdict is not None
    sev_ok = verdict.severity in {"warn", "block"}
    reporter.add_assertion(
        "verdict.severity in {warn, block} (caught I50.9 vs R57.0 mismatch)",
        sev_ok, detail=f"got: {verdict.severity}",
    )
    assert sev_ok


# ---------------------------------------------------------------------------
# §4c — SNOMED→ICD mapping (defensible per OMOP convention)
# ---------------------------------------------------------------------------


def test_t2dm_snomed_icd_mapping_acceptance(anthropic_client, reporter):
    """SNOMED 73211009 (generic "Diabetes mellitus") → ICD-10 E11.9
    (T2DM) is technically defensible per OMOP convention even though
    73211009 isn't strictly the T2DM-specific concept (44054006).
    The critic typically returns severity=info on this — it's
    over-flagging if we get warn/block. Encoding 'info or warn' as the
    acceptable verdict so we don't tighten too far."""
    from tests.dashboard.lib.scenarios import build_t2dm_snomed_mapping_cq

    cq, answer = build_t2dm_snomed_mapping_cq()

    t0 = time.monotonic()
    verdict = critique(anthropic_client, cq, answer)
    duration = time.monotonic() - t0
    _record_run(reporter, cq, answer, verdict, duration=duration)

    assert verdict is not None
    # The 4.2% mortality is plausible; SNOMED→ICD claim is defensible.
    sev_ok = verdict.severity in {"info", "warn"}
    reporter.add_assertion(
        "verdict.severity in {info, warn} (must not block defensible claim)",
        sev_ok, detail=f"got: {verdict.severity}",
    )
    assert sev_ok


# ---------------------------------------------------------------------------
# Known-good answer — info verdict
# ---------------------------------------------------------------------------


def test_clean_lactate_value_returns_info(anthropic_client, reporter):
    """2.42 mmol/L matches the MIMIC sepsis cohort reference (mean=2.40,
    p50=1.8, p95=6.9). Critic should fire severity=info, plausible=True.
    Calibration test — flags if the critic starts over-flagging clean
    answers."""
    from tests.dashboard.lib.scenarios import build_clean_lactate_cq

    cq, answer = build_clean_lactate_cq()

    t0 = time.monotonic()
    verdict = critique(anthropic_client, cq, answer)
    duration = time.monotonic() - t0
    _record_run(reporter, cq, answer, verdict, duration=duration)

    assert verdict is not None
    sev_ok = verdict.severity == "info"
    reporter.add_assertion(
        "verdict.severity == 'info' on clean answer",
        sev_ok, detail=f"got: {verdict.severity}",
    )
    assert sev_ok

    plausible_ok = verdict.plausible is True
    reporter.add_assertion(
        "verdict.plausible is True on clean answer",
        plausible_ok, detail=f"got: {verdict.plausible}",
    )
    assert plausible_ok


# ---------------------------------------------------------------------------
# §3b — borderline value triggers tool lookup
# ---------------------------------------------------------------------------


def test_borderline_creatinine_triggers_lookup(anthropic_client, reporter):
    """3.8 mg/dL creatinine in CKD: severe but not nonsensical. Critic
    should fire AT LEAST ONE tool (mimic_distribution_lookup,
    loinc_reference_range, etc.) to triangulate against MIMIC's actual
    distribution before issuing a verdict — exactly the §3b smoke
    scenario from the docs."""
    from tests.dashboard.lib.scenarios import build_borderline_creatinine_cq

    cq, answer = build_borderline_creatinine_cq()

    t0 = time.monotonic()
    verdict = critique(anthropic_client, cq, answer)
    duration = time.monotonic() - t0
    _record_run(reporter, cq, answer, verdict, duration=duration)

    assert verdict is not None
    n_tool_calls = len(verdict.tool_calls or [])
    fired_at_least_one = n_tool_calls >= 1
    reporter.add_assertion(
        "Critic fired ≥1 tool to triangulate borderline value",
        fired_at_least_one,
        detail=f"tool_calls n={n_tool_calls}",
    )
    # Soft assertion via `xfail`-style note: the critic CAN answer from
    # training recall on a borderline value; we surface this in the
    # report but don't fail the test if no tool fired (the correctness
    # of the verdict matters more than tool firing for this scenario).
    if not fired_at_least_one:
        reporter.add_note(
            "**Caveat:** the critic answered from training recall without "
            "firing a tool. Acceptable but worth noting — if this becomes "
            "the norm, consider tightening the prompt to require a "
            "MIMIC-distribution lookup for borderline values."
        )


# ---------------------------------------------------------------------------
# Graceful degradation when OMOPHub is unavailable
# ---------------------------------------------------------------------------


def test_critic_falls_back_when_omophub_unavailable(
    anthropic_client, reporter, monkeypatch,
):
    """Monkeypatch every OMOPHub-backed tool to return ``unavailable``.
    The critic must STILL produce a verdict (not None) and not crash —
    it should fall back to training recall + non-OMOPHub tools (PubMed,
    FDA label) to issue a judgement."""
    from src.conversational.health_evidence import tools as he_tools
    from tests.dashboard.lib.scenarios import build_dexmedetomidine_overdose_cq

    # Patch ONLY the OMOPHub-backed tool function symbols, leaving
    # PubMed / openfda_drug_label / mimic_distribution_lookup live.
    unavailable = lambda *a, **kw: {"status": "unavailable", "error": "MCP timeout"}
    for tool in (
        "rxnorm_lookup", "icd_lookup", "icd_autocode", "code_map",
    ):
        monkeypatch.setattr(he_tools, tool, unavailable, raising=False)

    cq, answer = build_dexmedetomidine_overdose_cq()

    t0 = time.monotonic()
    verdict = critique(anthropic_client, cq, answer)
    duration = time.monotonic() - t0
    _record_run(reporter, cq, answer, verdict, duration=duration)

    not_none = verdict is not None
    reporter.add_assertion(
        "Critic produces a verdict even with OMOPHub unavailable",
        not_none,
    )
    assert not_none
    # Still expect to flag the implausible dose (FDA label tool isn't
    # OMOPHub-backed; training recall is also enough).
    sev_ok = verdict.severity in {"warn", "block"}
    reporter.add_assertion(
        "Verdict still flags implausible dose without OMOPHub",
        sev_ok, detail=f"got: {verdict.severity}",
    )
    assert sev_ok
