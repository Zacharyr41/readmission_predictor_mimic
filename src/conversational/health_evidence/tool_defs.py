"""Anthropic tool-definition dicts and dispatch registry for the
EvidenceAgent.

Adding a new tool: define the function in :mod:`.tools` (envelope-contract
compliant — see that module's docstring), add a ``*_TOOL_DEF`` here, append
it to ``ALL_TOOL_DEFS``, and add an entry to ``TOOL_DISPATCH`` keyed on
the tool name.
"""

from __future__ import annotations

from typing import Any

from src.conversational.health_evidence.tools import (
    loinc_reference_range,
    mimic_distribution_lookup,
    pubmed_search,
)


PUBMED_SEARCH_TOOL_DEF: dict[str, Any] = {
    "name": "pubmed_search",
    "description": (
        "Search PubMed for clinical literature relevant to a plausibility "
        "judgment, disambiguation, or contextualization decision. Use this "
        "when your prompt's reference table or model recall isn't enough "
        "and you need population-level evidence (analyte distributions in "
        "specific cohorts, canonical naming, etc.). Returns up to 5 records: "
        "each has a PMID, title, source/journal info, and URL. On failure "
        "returns a {status: 'unavailable'} envelope."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "PubMed query string. Be specific: include analyte, "
                    "cohort/condition, and what aspect (population mean, "
                    "mortality, distribution) when relevant."
                ),
            },
            "max_results": {
                "type": "integer",
                "description": "Records to return (default 5; max 5).",
                "default": 5,
                "maximum": 5,
            },
        },
        "required": ["query"],
    },
}


MIMIC_DISTRIBUTION_TOOL_DEF: dict[str, Any] = {
    "name": "mimic_distribution_lookup",
    "description": (
        "Look up the empirical distribution of a MIMIC laboratory or vital "
        "by ``itemid``. Returns aggregate statistics (n, mean, median, p95, "
        "units) computed offline from the full MIMIC cohort. Use to verify "
        "that an aggregate value (e.g. 'mean creatinine 1.4 mg/dL') is in "
        "line with the population. Returns {status: 'unavailable'} if the "
        "distribution registry is missing or the itemid is not present."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "itemid": {
                "type": "integer",
                "description": (
                    "MIMIC labitem or chartitem id (the value joined to "
                    "``labevents.itemid`` / ``chartevents.itemid``)."
                ),
            },
        },
        "required": ["itemid"],
    },
}


LOINC_REFERENCE_RANGE_TOOL_DEF: dict[str, Any] = {
    "name": "loinc_reference_range",
    "description": (
        "Look up the published reference range for a LOINC code (e.g. "
        "'2160-0' for serum creatinine). Returns {low, high, units} from "
        "the local LOINC catalog. Use when judging whether a single value "
        "or aggregate is within published-normal range. Returns "
        "{status: 'unavailable'} if the catalog is absent or the code is "
        "unknown."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "loinc_code": {
                "type": "string",
                "description": "LOINC code in canonical NNNNN-N form (e.g. '2160-0').",
            },
        },
        "required": ["loinc_code"],
    },
}


ALL_TOOL_DEFS: list[dict[str, Any]] = [
    PUBMED_SEARCH_TOOL_DEF,
    MIMIC_DISTRIBUTION_TOOL_DEF,
    LOINC_REFERENCE_RANGE_TOOL_DEF,
]


TOOL_DISPATCH: dict[str, Any] = {
    "pubmed_search": pubmed_search,
    "mimic_distribution_lookup": mimic_distribution_lookup,
    "loinc_reference_range": loinc_reference_range,
}
"""Map of tool-name → callable, looked up by ``EvidenceAgent`` when the
model emits a ``tool_use`` block."""
