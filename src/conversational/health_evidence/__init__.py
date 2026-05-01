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
    CODE_MAP_TOOL_DEF,
    ICD_AUTOCODE_TOOL_DEF,
    ICD_LOOKUP_TOOL_DEF,
    LOINC_REFERENCE_RANGE_TOOL_DEF,
    MIMIC_DISTRIBUTION_TOOL_DEF,
    OPENFDA_DRUG_LABEL_TOOL_DEF,
    PUBMED_SEARCH_TOOL_DEF,
    RXNORM_LOOKUP_TOOL_DEF,
    SNOMED_EXPAND_ECL_TOOL_DEF,
    SNOMED_SEARCH_TOOL_DEF,
    TOOL_DISPATCH,
    TRIALS_SEARCH_TOOL_DEF,
)
from src.conversational.health_evidence.tools import (
    code_map,
    icd_autocode,
    icd_lookup,
    loinc_reference_range,
    mimic_distribution_lookup,
    openfda_drug_label,
    pubmed_search,
    rxnorm_lookup,
    snomed_expand_ecl,
    snomed_search,
    trials_search,
)

__all__ = [
    "ALL_TOOL_DEFS",
    "CODE_MAP_TOOL_DEF",
    "Citation",
    "EvidenceAgent",
    "EvidenceResult",
    "ICD_AUTOCODE_TOOL_DEF",
    "ICD_LOOKUP_TOOL_DEF",
    "LOINC_REFERENCE_RANGE_TOOL_DEF",
    "MIMIC_DISTRIBUTION_TOOL_DEF",
    "OPENFDA_DRUG_LABEL_TOOL_DEF",
    "PUBMED_SEARCH_TOOL_DEF",
    "RXNORM_LOOKUP_TOOL_DEF",
    "SNOMED_EXPAND_ECL_TOOL_DEF",
    "SNOMED_SEARCH_TOOL_DEF",
    "TOOL_DISPATCH",
    "TRIALS_SEARCH_TOOL_DEF",
    "ToolCall",
    "code_map",
    "icd_autocode",
    "icd_lookup",
    "loinc_reference_range",
    "mimic_distribution_lookup",
    "openfda_drug_label",
    "pubmed_search",
    "rxnorm_lookup",
    "snomed_expand_ecl",
    "snomed_search",
    "trials_search",
]
