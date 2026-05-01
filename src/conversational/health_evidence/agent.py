"""EvidenceAgent — reusable Anthropic tool-use loop with citation tracking.

Owns the agentic loop (iteration cap, tool dispatch, observed-citations
tracking, mutation discipline, graceful failure). Callers (critic,
sql_validator, clinical_consult) prepare a system prompt + user prompt
and parse the resulting ``EvidenceResult.parsed_json`` into their own
verdict types.

This is the consolidation of the loop previously inlined in
``critic.critique`` (lines 99-192 of the pre-refactor file). All callers
that need tool-using LLM judgment go through this class.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import anthropic

from src.conversational.health_evidence.models import (
    Citation,
    EvidenceResult,
    ToolCall,
)
from src.conversational.health_evidence.tool_defs import (
    ALL_TOOL_DEFS,
    TOOL_DISPATCH,
)

logger = logging.getLogger(__name__)


_DEFAULT_MODEL = "claude-sonnet-4-6"
_DEFAULT_MAX_TOKENS = 600
_DEFAULT_TIMEOUT_SECONDS = 30.0
_DEFAULT_MAX_ITERATIONS = 3
_MAX_TOOL_RESULT_BYTES = 4096
_RAW_TEXT_TRUNCATE_FOR_LOG = 500


class EvidenceAgent:
    """Wraps Anthropic's tool-use loop into a single ``consult`` call.

    Per-instance configuration: model, max_tokens, max_iterations, timeout,
    available tools and dispatch. Per-call configuration: system + user
    prompts. Stateless across calls — each ``consult`` is independent.
    """

    def __init__(
        self,
        client: anthropic.Anthropic,
        *,
        model: str = _DEFAULT_MODEL,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        max_iterations: int = _DEFAULT_MAX_ITERATIONS,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
        tools: list[dict[str, Any]] | None = None,
        tool_dispatch: dict[str, Any] | None = None,
    ) -> None:
        self._client = client
        self._model = model
        self._max_tokens = max_tokens
        self._max_iterations = max_iterations
        self._timeout = timeout
        self._tools = list(tools) if tools is not None else list(ALL_TOOL_DEFS)
        # Snapshot the dispatch dict at construction time so callers can't
        # surprise us mid-call by mutating the global registry.
        self._tool_dispatch = dict(
            tool_dispatch if tool_dispatch is not None else TOOL_DISPATCH
        )

    def consult(self, system_prompt: str, user_prompt: str) -> EvidenceResult:
        """Run one tool-using consultation.

        ``system_prompt`` is wrapped with ``cache_control: ephemeral``
        internally so callers don't repeat the ceremony.

        Returns an :class:`EvidenceResult`. NEVER raises — on any failure
        (API error, malformed JSON, timeout, dispatch error) returns an
        EvidenceResult with empty fields and ``parsed_json=None``.
        """
        try:
            return self._run(system_prompt, user_prompt)
        except Exception as exc:  # noqa: BLE001 — never raise to callers
            logger.warning(
                "EvidenceAgent.consult failed; returning empty result: %s (%s)",
                exc, type(exc).__name__,
            )
            return EvidenceResult()

    # -- internal --------------------------------------------------------

    def _run(self, system_prompt: str, user_prompt: str) -> EvidenceResult:
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": user_prompt},
        ]
        observed: list[Citation] = []
        observed_keys: set[tuple[str, str]] = set()
        tool_calls: list[ToolCall] = []

        system_blocks = [
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            },
        ]

        for iteration in range(self._max_iterations + 1):
            extra: dict[str, Any] = {}
            if iteration == self._max_iterations:
                # Cap iter — force end_turn, no more tool calls.
                extra["tool_choice"] = {"type": "none"}
            # Pass a snapshot copy so post-call mutation of the loop's
            # ``messages`` list doesn't retroactively change earlier
            # call_args (matters for test assertions and any caller that
            # might inspect history later). Per-iteration cost is trivial.
            response = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                timeout=self._timeout,
                system=system_blocks,
                tools=self._tools,
                messages=list(messages),
                **extra,
            )

            stop_reason = getattr(response, "stop_reason", None)
            if stop_reason == "tool_use":
                # Mutate ``messages`` ONLY when we'll continue — so the
                # final messages.create call's args remain clean for tests
                # that inspect call_args.kwargs["messages"].
                messages.append(
                    {"role": "assistant", "content": response.content}
                )
                tool_result_blocks = []
                for block in response.content:
                    if getattr(block, "type", None) != "tool_use":
                        continue
                    name = getattr(block, "name", "")
                    tool_input = getattr(block, "input", {}) or {}
                    result = self._safe_call_tool(name, tool_input)
                    self._record_observations(
                        name, result, observed, observed_keys,
                    )
                    tool_calls.append(self._record_tool_call(
                        name, tool_input, result,
                    ))
                    tool_result_blocks.append({
                        "type": "tool_result",
                        "tool_use_id": getattr(block, "id", "tu_unknown"),
                        "content": _truncate_tool_result(result),
                    })
                messages.append(
                    {"role": "user", "content": tool_result_blocks}
                )
                continue

            # Final turn (end_turn or any non-tool_use stop_reason).
            text = _extract_final_text(response.content)
            parsed = _extract_json(text)
            return EvidenceResult(
                final_text=text,
                parsed_json=parsed,
                observed_citations=observed,
                tool_calls=tool_calls,
            )

        # Loop fell through the cap without producing a final response.
        # Defensive: tool_choice="none" on cap iter should force end_turn.
        logger.warning(
            "EvidenceAgent loop exhausted (%d iterations) without a final response",
            self._max_iterations + 1,
        )
        return EvidenceResult(
            observed_citations=observed,
            tool_calls=tool_calls,
        )

    def _safe_call_tool(self, name: str, tool_input: dict) -> dict:
        """Look up + call a tool. Returns an envelope dict; never raises."""
        if name not in self._tool_dispatch:
            return {
                "status": "unavailable",
                "error": f"unknown tool: {name!r}",
            }
        fn = self._tool_dispatch[name]
        if not callable(fn):
            return {
                "status": "unavailable",
                "error": f"tool {name!r} is not callable",
            }
        try:
            return fn(**tool_input)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "tool %r raised (should be impossible per envelope contract): %s",
                name, exc,
            )
            return {"status": "unavailable", "error": str(exc)}

    @staticmethod
    def _record_observations(
        tool_name: str,
        result: dict,
        observed: list[Citation],
        observed_keys: set[tuple[str, str]],
    ) -> None:
        """Append observed citations from a tool result to ``observed`` /
        ``observed_keys``, deduplicating by ``(source, id)``."""
        if not isinstance(result, dict) or result.get("status") != "ok":
            return
        for rec in result.get("results", []) or []:
            if not isinstance(rec, dict):
                continue
            if tool_name == "pubmed_search":
                pmid = str(rec.get("pmid") or "")
                if not pmid:
                    continue
                key = ("pubmed", pmid)
                if key in observed_keys:
                    continue
                observed_keys.add(key)
                observed.append(Citation(
                    source="pubmed",
                    id=pmid,
                    title=rec.get("title") or None,
                    url=rec.get("url") or None,
                ))
            elif tool_name == "loinc_reference_range":
                code = str(rec.get("loinc_code") or "")
                if not code:
                    continue
                key = ("loinc", code)
                if key in observed_keys:
                    continue
                observed_keys.add(key)
                observed.append(Citation(
                    source="loinc",
                    id=code,
                    title=None,
                    url=None,
                ))
            elif tool_name == "mimic_distribution_lookup":
                itemid = rec.get("itemid")
                if itemid is None:
                    continue
                id_str = str(itemid)
                key = ("mimic_distribution", id_str)
                if key in observed_keys:
                    continue
                observed_keys.add(key)
                observed.append(Citation(
                    source="mimic_distribution",
                    id=id_str,
                    title=None,
                    url=None,
                ))

    @staticmethod
    def _record_tool_call(
        name: str, tool_input: dict, result: dict,
    ) -> ToolCall:
        status = "ok" if (
            isinstance(result, dict) and result.get("status") == "ok"
        ) else "unavailable"
        n_results = 0
        if status == "ok":
            results = result.get("results") or []
            if isinstance(results, list):
                n_results = len(results)
        return ToolCall(
            name=name,
            input=dict(tool_input),
            status=status,
            n_results=n_results,
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _truncate_tool_result(result: dict) -> str:
    """Serialize a tool result and truncate to ``_MAX_TOOL_RESULT_BYTES``.

    Defense-in-depth: tools also enforce their own size budgets."""
    serialized = json.dumps(result, ensure_ascii=False)
    encoded = serialized.encode("utf-8")
    if len(encoded) <= _MAX_TOOL_RESULT_BYTES:
        return serialized
    truncated = encoded[: _MAX_TOOL_RESULT_BYTES - 32].decode(
        "utf-8", errors="ignore",
    )
    return truncated + '..."truncated":true}'


def _extract_final_text(content_blocks) -> str:
    """First text block's text from a response, or empty string."""
    for block in content_blocks or []:
        if getattr(block, "type", None) == "text":
            return getattr(block, "text", "") or ""
    if content_blocks:
        text = getattr(content_blocks[0], "text", None)
        if isinstance(text, str):
            return text
    return ""


def _extract_json(text: str) -> dict[str, Any] | None:
    """Pull the first JSON object out of a model response, or return None.

    Tries fenced ```json blocks first, then a bare object match. Same
    pattern as the existing critic._extract_json — kept here to keep the
    health_evidence package import-self-contained.
    """
    fenced = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if fenced:
        candidate = fenced.group(1).strip()
    else:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        candidate = match.group(0)
    try:
        result = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if not isinstance(result, dict):
        return None
    return result
