"""Pydantic data models for the health-evidence sub-system.

These describe the contract between the ``EvidenceAgent`` and its callers
(critic, sql_validator, clinical_consult). The agent owns the tool-use
loop and citation tracking; callers parse ``parsed_json`` into their own
verdict types and use ``filter_claimed_citations`` to drop hallucinated
references.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


CitationSource = Literal["pubmed", "loinc", "mimic_distribution"]


class Citation(BaseModel):
    """One observed citation — a record returned by a tool call this turn.

    ``source`` is the registry the record came from; ``id`` is the canonical
    identifier within that registry (PMID for pubmed, LOINC code for loinc,
    MIMIC ``itemid`` for mimic_distribution). ``title`` and ``url`` are
    convenience fields for downstream rendering."""

    source: CitationSource
    id: str
    title: str | None = None
    url: str | None = None


class ToolCall(BaseModel):
    """Telemetry record of one tool invocation.

    Used by callers (and by tests) to inspect what the agent did this turn
    without re-parsing the message history."""

    name: str
    input: dict
    status: Literal["ok", "unavailable"]
    n_results: int = 0


class EvidenceResult(BaseModel):
    """Outcome of one ``EvidenceAgent.consult`` invocation.

    ``final_text`` is whatever the model emitted on the end_turn iteration
    (empty string on any failure). ``parsed_json`` is the first JSON object
    extracted from ``final_text`` (None if no parseable JSON or if the agent
    failed). ``observed_citations`` is the deduplicated set of citations
    actually returned by tool calls this turn — the anti-hallucination
    ground truth. ``tool_calls`` is the per-invocation telemetry in order.

    The agent NEVER raises; on any failure (API error, malformed JSON,
    timeout, etc.) it returns an EvidenceResult with empty fields and
    ``parsed_json=None`` so callers can detect the failure cleanly.
    """

    final_text: str = ""
    parsed_json: dict | None = None
    observed_citations: list[Citation] = []
    tool_calls: list[ToolCall] = []

    def filter_claimed_citations(
        self, claimed: list[dict] | None,
    ) -> list[dict] | None:
        """Drop entries from ``claimed`` whose ``(source, id)`` pair was not
        observed in any tool call this turn.

        Anti-hallucination guard: the model can propose citations in its
        JSON verdict, but only ones backed by an observed tool result are
        surfaced to the user.

        Returns ``None`` (not ``[]``) when the input is missing or all
        entries are filtered out, so the no-citations and empty-list states
        remain distinguishable. Accepts both the new ``{source, id}`` shape
        and the legacy critic ``{type, pmid}`` shape for backward compat
        with the existing CriticVerdict.cited_sources schema.
        """
        if not isinstance(claimed, list):
            return None
        observed_keys: set[tuple[str, str]] = {
            (c.source, c.id) for c in self.observed_citations
        }
        keep: list[dict] = []
        for item in claimed:
            if not isinstance(item, dict):
                continue
            source = str(
                item.get("source")
                or item.get("type")
                or "pubmed"
            )
            id_val = str(
                item.get("id")
                or item.get("pmid")
                or item.get("loinc_code")
                or ""
            )
            if id_val and (source, id_val) in observed_keys:
                keep.append(item)
        return keep or None
