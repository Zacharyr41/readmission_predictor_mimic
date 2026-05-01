"""Health-evidence sub-system: a reusable ``EvidenceAgent`` plus the tools
it dispatches.

The agent encapsulates Anthropic's tool-use loop, observed-citations
tracking, anti-hallucination filtering, and graceful failure handling so
its callers (critic, sql_validator, clinical_consult) can focus on
prompt engineering and verdict parsing.

Public surface::

    from src.conversational.health_evidence import (
        EvidenceAgent, EvidenceResult, Citation, ToolCall,
        ALL_TOOL_DEFS, TOOL_DISPATCH,
        pubmed_search, mimic_distribution_lookup, loinc_reference_range,
        PUBMED_SEARCH_TOOL_DEF, MIMIC_DISTRIBUTION_TOOL_DEF,
        LOINC_REFERENCE_RANGE_TOOL_DEF,
    )
"""

from src.conversational.health_evidence.agent import EvidenceAgent
from src.conversational.health_evidence.models import (
    Citation,
    EvidenceResult,
    ToolCall,
)
from src.conversational.health_evidence.tool_defs import (
    ALL_TOOL_DEFS,
    LOINC_REFERENCE_RANGE_TOOL_DEF,
    MIMIC_DISTRIBUTION_TOOL_DEF,
    PUBMED_SEARCH_TOOL_DEF,
    TOOL_DISPATCH,
)
from src.conversational.health_evidence.tools import (
    loinc_reference_range,
    mimic_distribution_lookup,
    pubmed_search,
)

__all__ = [
    "ALL_TOOL_DEFS",
    "Citation",
    "EvidenceAgent",
    "EvidenceResult",
    "LOINC_REFERENCE_RANGE_TOOL_DEF",
    "MIMIC_DISTRIBUTION_TOOL_DEF",
    "PUBMED_SEARCH_TOOL_DEF",
    "TOOL_DISPATCH",
    "ToolCall",
    "loinc_reference_range",
    "mimic_distribution_lookup",
    "pubmed_search",
]
