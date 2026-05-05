"""Real-MCP smoke for the externally-grounded critic (Phase H).

Probes which optional Phase-G MCPs are configured in the environment and
runs one synthetic critic invocation per configured backend. For each
configured backend, prints whether the backend was actually exercised by
the model, the envelope status, the latency, and any citation that
landed in the verdict.

Usage:
    .venv/bin/python scripts/critic_evidence_smoke.py

The script never raises on a missing backend — it prints "skipped (env
var X not set)" and moves on. Exit code 0 when at least one configured
backend was exercised and returned ``status=ok``; non-zero otherwise so
this can be wired into CI later.

Backends probed (env vars that toggle them):
- pubmed_search        — always configured (default backend hits NCBI)
- loinc_reference_range— offline JSON; OK when data/ontology_cache file present
- mimic_distribution_lookup — offline JSON; OK when data/processed file present
- snomed_search        — HERMES_MCP_COMMAND
- rxnorm_lookup        — OMOPHUB_MCP_URL (+ OMOPHUB_API_KEY)
- icd_lookup           — ICD_MCP_URL
- openfda_drug_label   — OPENFDA_MCP_COMMAND or npx on PATH
- trials_search        — CLINICALTRIALS_MCP_COMMAND or bunx/npx on PATH
"""

from __future__ import annotations

import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import anthropic

from src.conversational.critic import critique
from src.conversational.health_evidence.tools import (
    LOINC_CATALOG_PATH,
    MIMIC_DISTRIBUTIONS_PATH,
)
from src.conversational.models import (
    AnswerResult,
    ClinicalConcept,
    CompetencyQuestion,
)


@dataclass
class BackendProbe:
    """One synthetic scenario designed to encourage the model to invoke a
    specific tool. The cq + answer pair is crafted so the natural critic
    move is to call the named tool — but the model is free to skip."""

    tool: str
    label: str
    cq: CompetencyQuestion
    answer: AnswerResult
    skip_reason: str | None = None


def _cq(question: str) -> CompetencyQuestion:
    return CompetencyQuestion(
        original_question=question,
        clinical_concepts=[ClinicalConcept(name="probe", concept_type="biomarker")],
        return_type="text_and_table",
        scope="cohort",
    )


def _answer(text: str) -> AnswerResult:
    return AnswerResult(text_summary=text)


def _check_skip(tool: str) -> str | None:
    """Return a reason string when the backend is not configured, else None."""
    if tool == "loinc_reference_range":
        return None if LOINC_CATALOG_PATH.exists() else (
            f"LOINC catalog missing at {LOINC_CATALOG_PATH}"
        )
    if tool == "mimic_distribution_lookup":
        return None if MIMIC_DISTRIBUTIONS_PATH.exists() else (
            f"MIMIC distributions missing at {MIMIC_DISTRIBUTIONS_PATH}"
        )
    if tool == "snomed_search":
        cmd = os.environ.get("HERMES_MCP_COMMAND", "hermes")
        return None if shutil.which(cmd) else (
            f"Hermes binary not on PATH (HERMES_MCP_COMMAND={cmd})"
        )
    if tool == "rxnorm_lookup":
        return None if os.environ.get("OMOPHUB_MCP_URL") else "OMOPHUB_MCP_URL not set"
    if tool == "icd_lookup":
        return None if os.environ.get("ICD_MCP_URL") else "ICD_MCP_URL not set"
    if tool == "openfda_drug_label":
        cmd = os.environ.get("OPENFDA_MCP_COMMAND")
        if cmd and shutil.which(cmd.split()[0]):
            return None
        if shutil.which("npx"):
            return None
        return "OpenFDA MCP not configured (need OPENFDA_MCP_COMMAND or npx on PATH)"
    if tool == "trials_search":
        cmd = os.environ.get("CLINICALTRIALS_MCP_COMMAND")
        if cmd and shutil.which(cmd.split()[0]):
            return None
        if shutil.which("bunx") or shutil.which("npx"):
            return None
        return "ClinicalTrials MCP not configured (need bunx or npx on PATH)"
    if tool == "pubmed_search":
        return None  # default backend always works (NCBI public API)
    return f"unknown tool {tool!r}"


def _build_probes() -> list[BackendProbe]:
    """One probe per tool. Each scenario's text is shaped to make the
    relevant tool a natural choice, but the model decides whether to
    actually invoke it."""
    return [
        BackendProbe(
            "pubmed_search", "PubMed (NCBI E-utilities)",
            _cq("typical lactate range in septic shock literature"),
            _answer("Mean lactate 7.2 mmol/L in our septic shock cohort."),
        ),
        BackendProbe(
            "loinc_reference_range", "LOINC catalog (offline)",
            _cq("procalcitonin range"),
            _answer("Mean procalcitonin 1.2 ng/mL (LOINC 33747-0)."),
        ),
        BackendProbe(
            "mimic_distribution_lookup", "MIMIC distribution (offline)",
            _cq("creatinine in AKI cohort"),
            _answer(
                "Mean serum creatinine 3.1 mg/dL "
                "(itemid 50912) in our AKI cohort."
            ),
        ),
        BackendProbe(
            "snomed_search", "SNOMED via Hermes",
            _cq("septic shock definition"),
            _answer(
                "Cohort of septic shock patients (SNOMED 76571007) "
                "had median lactate 4.1 mmol/L."
            ),
        ),
        BackendProbe(
            "rxnorm_lookup", "RxNorm via OMOPHub",
            _cq("norepinephrine dose typical"),
            _answer("Mean norepinephrine dose 0.4 mcg/kg/min."),
        ),
        BackendProbe(
            "icd_lookup", "ICD via self-hosted MCP",
            _cq("heart failure cohort"),
            _answer("ICD-10 I50.9 (heart failure unspecified) cohort, n=412."),
        ),
        BackendProbe(
            "openfda_drug_label", "OpenFDA via npx MCP",
            _cq("levophed indication"),
            _answer("Levophed administered in 312 patients."),
        ),
        BackendProbe(
            "trials_search", "ClinicalTrials.gov via npx MCP",
            _cq("recent sepsis trials"),
            _answer("Patients enrolled in ongoing sepsis trials, n=12."),
        ),
    ]


def _run_one(client: anthropic.Anthropic, probe: BackendProbe) -> dict:
    """Run one critique() call and report what happened."""
    t0 = time.time()
    verdict = critique(client, probe.cq, probe.answer)
    elapsed = time.time() - t0

    invoked = False
    invoked_status = "n/a"
    if verdict is not None and verdict.tool_calls:
        for tc in verdict.tool_calls:
            if tc.get("name") == probe.tool:
                invoked = True
                invoked_status = tc.get("status", "?")
                break

    citation = None
    if verdict is not None and verdict.cited_sources:
        for s in verdict.cited_sources:
            # Match either {source: ...} or {type: pubmed} for legacy shape.
            srcs = (s.get("source"), s.get("type"))
            registry = {
                "pubmed_search": ("pubmed",),
                "loinc_reference_range": ("loinc",),
                "mimic_distribution_lookup": ("mimic_distribution",),
                "snomed_search": ("snomed",),
                "rxnorm_lookup": ("rxnorm",),
                "icd_lookup": ("icd",),
                "openfda_drug_label": ("openfda",),
                "trials_search": ("clinicaltrials",),
            }.get(probe.tool, ())
            if any(s_part in registry for s_part in srcs if s_part):
                citation = s
                break

    return {
        "tool": probe.tool,
        "label": probe.label,
        "elapsed_sec": round(elapsed, 2),
        "verdict_returned": verdict is not None,
        "invoked": invoked,
        "invoked_status": invoked_status,
        "cited": citation,
        "severity": verdict.severity if verdict else None,
    }


def main() -> int:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        env_path = Path(".env")
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("ANTHROPIC_API_KEY="):
                    api_key = line.split("=", 1)[1].strip()
                    break
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set in env or .env", file=sys.stderr)
        return 2

    client = anthropic.Anthropic(api_key=api_key)
    probes = _build_probes()

    print(f"Phase H critic-evidence smoke — {len(probes)} probes\n")
    print(f"{'BACKEND':<32} {'STATUS':<14} {'LATENCY':<10} {'INVOKED':<8} CITATION")
    print("-" * 95)

    any_ok = False
    for probe in probes:
        skip_reason = _check_skip(probe.tool)
        if skip_reason:
            print(f"{probe.label:<32} {'skipped':<14} {'-':<10} {'-':<8} ({skip_reason})")
            continue
        result = _run_one(client, probe)
        invoked_str = "YES" if result["invoked"] else "no"
        if result["invoked"] and result["invoked_status"] == "ok":
            any_ok = True
        cite_str = ""
        if result["cited"]:
            c = result["cited"]
            cite_str = (
                f"{c.get('source', c.get('type', '?'))}/"
                f"{c.get('id', c.get('pmid', '?'))}"
            )
        print(
            f"{result['label']:<32} "
            f"{result['invoked_status']:<14} "
            f"{result['elapsed_sec']:<10} "
            f"{invoked_str:<8} "
            f"{cite_str}"
        )

    print()
    print(
        "Exit 0: at least one configured backend was invoked and returned ok."
        if any_ok else
        "Exit 1: no configured backend was actually exercised."
    )
    return 0 if any_ok else 1


if __name__ == "__main__":
    sys.exit(main())
