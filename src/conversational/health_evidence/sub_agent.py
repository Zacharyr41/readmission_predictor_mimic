"""HealthSourceOfTruthAgent — biomedical-grounding sub-agent.

Wraps :class:`EvidenceAgent` with a focused system prompt that forces
the model to emit a structured JSON object describing claims, evidence,
and unresolved questions. Designed to be called by the orchestrator,
critic, or clinical_consult helpers when they need cross-MCP biomedical
grounding (PubMed + RxNorm + SNOMED + ClinicalTrials + OpenFDA + ICD).

Per the user's research, the sub-agent has:
- Its own (sub-)agent abstraction so the orchestrator can encapsulate
  multi-MCP lookups in one call.
- A strict JSON output schema with citations, so the caller can
  programmatically inspect what was claimed and what evidence backs it.
- The same anti-hallucination filter as the critic: only sources
  actually returned by tool calls survive into the final ``findings``.

PHI-safety invariant: this agent NEVER receives MIMIC content. The
orchestrator only passes the question + decomposer interpretation to
``consult()`` — never row data, hadm_ids, or subject_ids. A test in
``test_sub_agent.py`` enforces this at CI time.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import anthropic
from pydantic import ValidationError

from src.conversational.health_evidence.agent import EvidenceAgent
from src.conversational.health_evidence.tool_defs import (
    ALL_TOOL_DEFS,
    TOOL_DISPATCH,
)
from src.conversational.models import (
    Evidence,
    HealthAnswer,
    HealthFinding,
)

logger = logging.getLogger(__name__)


_SUB_AGENT_MODEL = "claude-sonnet-4-6"
_MAX_TOKENS = 1500  # bigger than critic — sub-agent emits richer JSON
_DEFAULT_TIMEOUT = 45.0  # multiple tool calls + a long-form synthesis
_MAX_ITERATIONS = 5


HEALTH_SOURCE_OF_TRUTH_SYSTEM_PROMPT = """\
You are the Health Source-of-Truth agent. The orchestrator delegates
biomedical fact-finding to you when its parent agent needs verified
information about drugs, diagnoses, codes, trials, FDA labels, or
literature.

OPERATING PRINCIPLES
1. Never speculate. If a fact is not retrievable from a connected tool,
   include it under ``unresolved`` and mark the finding ``status:
   "unverified"`` or ``"conflicting"``.
2. Always cite. Every claim must reference at least one evidence record
   tied to a specific tool call.
3. Resolve ambiguity through SNOMED/UMLS first (when available);
   otherwise PubMed search context.
4. Prefer specialized tools: LOINC reference ranges for lab values;
   PubMed for literature evidence; mimic_distribution_lookup for
   empirical population statistics.
5. Parallel-call independent lookups where possible.
6. Anti-hallucination guard: do NOT invent identifiers (PMIDs, LOINC
   codes, etc.). Only cite sources that a tool call actually returned
   this turn.

OUTPUT SCHEMA (return ONLY valid JSON of this shape — no fenced code
block, no surrounding prose)

{
  "query": "<the original question, verbatim>",
  "answer_summary": "<≤3 sentences summarising what you found>",
  "findings": [
    {
      "claim": "<one specific claim>",
      "evidence": [
        {
          "source": "pubmed|snomed|rxnorm|loinc|mimic_distribution|openfda|clinicaltrials|icd",
          "id": "<canonical identifier from the source registry>",
          "tool": "<tool name you called>",
          "snippet": "<≤300 chars from the tool result>"
        }
      ],
      "confidence": "high|medium|low",
      "status": "verified|unverified|conflicting"
    }
  ],
  "unresolved": ["<question you could not answer>", ...],
  "tools_called": [{"name": "<tool>", "count": <int>}, ...]
}

If you cannot answer at all, still return a valid JSON object — set
``answer_summary`` to a brief explanation of why you could not.
"""


class HealthSourceOfTruthAgent:
    """Run a focused biomedical lookup. Returns a structured HealthAnswer.

    The agent owns its own EvidenceAgent (with all source-of-truth
    tools available). Failures are silent — ``consult`` returns ``None``
    rather than raising. Callers should treat ``None`` as "no
    grounding available, decide whether to proceed without it."
    """

    def __init__(
        self,
        client: anthropic.Anthropic,
        *,
        timeout: float = _DEFAULT_TIMEOUT,
        max_iterations: int = _MAX_ITERATIONS,
        max_tokens: int = _MAX_TOKENS,
    ) -> None:
        self._agent = EvidenceAgent(
            client,
            model=_SUB_AGENT_MODEL,
            max_tokens=max_tokens,
            max_iterations=max_iterations,
            timeout=timeout,
            tools=ALL_TOOL_DEFS,
            tool_dispatch=TOOL_DISPATCH,
        )

    def consult(
        self,
        question: str,
        *,
        context: dict | None = None,
    ) -> HealthAnswer | None:
        """Delegate to the sub-agent for biomedical grounding.

        ``question`` should be a focused biomedical question (drug
        identity, normal lab range, trial existence, etc.).
        ``context`` is an optional dict of additional non-PHI hints
        the caller wants to pass through (e.g., decomposer
        interpretation, suspected concept names).

        Returns ``None`` on any failure (API error, malformed JSON,
        schema validation). Never raises.

        PHI safety: do NOT pass row-level data, hadm_ids, or subject_ids
        in ``context``. Only safe fields are accepted (free-text strings,
        concept names, ontology codes).
        """
        try:
            user_msg = self._build_user_msg(question, context or {})
            result = self._agent.consult(
                HEALTH_SOURCE_OF_TRUTH_SYSTEM_PROMPT, user_msg,
            )
            if result.parsed_json is None:
                return None
            payload = dict(result.parsed_json)
            # Filter findings' evidence against observed citations.
            payload["findings"] = self._filter_findings(
                payload.get("findings"), result.observed_citations,
            )
            try:
                return HealthAnswer(**payload)
            except ValidationError as exc:
                logger.info("HealthSourceOfTruthAgent schema fail: %s", exc)
                return None
        except Exception as exc:  # noqa: BLE001
            logger.info(
                "HealthSourceOfTruthAgent failed: %s (%s)",
                exc, type(exc).__name__,
            )
            return None

    @staticmethod
    def _build_user_msg(question: str, context: dict) -> str:
        parts = [f"## Question\n{question}"]
        if context:
            parts.append("## Context\n```json\n" + json.dumps(
                context, default=str, indent=2,
            ) + "\n```")
        parts.append("Respond with the JSON object only, per the system schema.")
        return "\n\n".join(parts)

    @staticmethod
    def _filter_findings(
        raw: Any, observed_citations: list,
    ) -> list[dict]:
        """Drop evidence entries whose ``(source, id)`` pair was not
        observed in any tool call. Anti-hallucination guard.

        Findings whose evidence is fully filtered are kept (with empty
        evidence) but downgraded to ``status="unverified"`` and
        ``confidence="low"``.
        """
        if not isinstance(raw, list):
            return []
        observed_keys: set[tuple[str, str]] = {
            (c.source, c.id) for c in observed_citations
        }
        cleaned: list[dict] = []
        for f in raw:
            if not isinstance(f, dict):
                continue
            evidence_in = f.get("evidence") or []
            if not isinstance(evidence_in, list):
                evidence_in = []
            evidence_out: list[dict] = []
            for e in evidence_in:
                if not isinstance(e, dict):
                    continue
                src = str(e.get("source") or "")
                id_ = str(e.get("id") or "")
                if (src, id_) in observed_keys:
                    evidence_out.append(e)
            new_f = dict(f)
            new_f["evidence"] = evidence_out
            if not evidence_out:
                new_f["status"] = "unverified"
                new_f["confidence"] = "low"
            cleaned.append(new_f)
        return cleaned
