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


SNOMED_SEARCH_TOOL_DEF: dict[str, Any] = {
    "name": "snomed_search",
    "description": (
        "Search SNOMED CT for clinical concepts (diseases, findings, "
        "procedures, body structures) matching a free-text term. Returns "
        "concept IDs that the orchestrator can ground further lookups in. "
        "Use when you need to canonicalize a clinical phrase to a SNOMED "
        "concept ID before cross-mapping or searching other registries. "
        "Returns {status: 'unavailable'} if the Hermes MCP isn't installed."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "term": {
                "type": "string",
                "description": "Free-text clinical phrase to search for.",
            },
            "max_results": {
                "type": "integer",
                "description": "Records to return (default 10).",
                "default": 10,
                "maximum": 25,
            },
        },
        "required": ["term"],
    },
}


RXNORM_LOOKUP_TOOL_DEF: dict[str, Any] = {
    "name": "rxnorm_lookup",
    "description": (
        "Look up RxNorm RXCUIs (canonical drug identifiers) for a drug "
        "name via OMOPHub. Use this when you need to canonicalize a free-"
        "text drug mention (brand name, common name, abbreviation) to a "
        "structured identifier before further reasoning. Returns {status: "
        "'unavailable'} if OMOPHUB_MCP_URL is not configured."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "drug_name": {
                "type": "string",
                "description": (
                    "Drug name to look up. May be brand, generic, or "
                    "common abbreviation."
                ),
            },
            "max_results": {
                "type": "integer",
                "description": "Records to return (default 5).",
                "default": 5,
                "maximum": 10,
            },
        },
        "required": ["drug_name"],
    },
}


TRIALS_SEARCH_TOOL_DEF: dict[str, Any] = {
    "name": "trials_search",
    "description": (
        "Search ClinicalTrials.gov for studies matching a query (condition, "
        "intervention, outcome, etc.). Returns NCT IDs and brief summaries "
        "the orchestrator can use to ground evidence-based claims. "
        "Returns {status: 'unavailable'} if the ClinicalTrials MCP "
        "(bunx/npx clinicaltrialsgov-mcp-server) isn't available."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Search query — be specific: include condition, "
                    "intervention, and population."
                ),
            },
            "max_results": {
                "type": "integer",
                "description": "Records to return (default 10).",
                "default": 10,
                "maximum": 25,
            },
        },
        "required": ["query"],
    },
}


OPENFDA_DRUG_LABEL_TOOL_DEF: dict[str, Any] = {
    "name": "openfda_drug_label",
    "description": (
        "Look up the OpenFDA structured drug label for a brand or generic "
        "drug name. Returns indications, warnings, and other label fields "
        "useful for safety reasoning. Returns {status: 'unavailable'} if "
        "the OpenFDA MCP isn't installed."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "drug_name": {
                "type": "string",
                "description": "Brand or generic drug name.",
            },
        },
        "required": ["drug_name"],
    },
}


ICD_LOOKUP_TOOL_DEF: dict[str, Any] = {
    "name": "icd_lookup",
    "description": (
        "Look up ICD-10 (default) or ICD-11 codes matching a clinical "
        "phrase. Returns code + title + chapter. Use to canonicalize "
        "diagnoses to ICD codes. Returns {status: 'unavailable'} if "
        "ICD_MCP_URL is not set (the ICD MCP requires self-hosting per "
        "WHO licensing)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Clinical phrase to look up.",
            },
            "version": {
                "type": "string",
                "description": "ICD version: '10' (default) or '11'.",
                "default": "10",
            },
            "max_results": {
                "type": "integer",
                "description": "Records to return (default 10).",
                "default": 10,
                "maximum": 25,
            },
        },
        "required": ["query"],
    },
}


SNOMED_EXPAND_ECL_TOOL_DEF: dict[str, Any] = {
    "name": "snomed_expand_ecl",
    "description": (
        "Expand a SNOMED CT Expression Constraint Language (ECL) expression "
        "into the set of concepts that satisfy it. Use when you need a "
        "structured cohort definition rather than free-text search — e.g. "
        "<<73211009 |Diabetes mellitus| returns Diabetes mellitus and all "
        "its descendants. Returns {status: 'unavailable'} if Hermes isn't "
        "installed or the ECL parser rejected the expression."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": (
                    "ECL expression. Examples: '<<73211009' (descendants + "
                    "self), '<73211009 AND <<128605003' (intersection)."
                ),
            },
            "max_results": {
                "type": "integer",
                "description": "Concepts to return (default 200, max 1000).",
                "default": 200,
                "maximum": 1000,
            },
        },
        "required": ["expression"],
    },
}


CODE_MAP_TOOL_DEF: dict[str, Any] = {
    "name": "code_map",
    "description": (
        "Map a code from one clinical vocabulary to another via OMOPHub. "
        "Common pivots: ICD10CM → SNOMED for diagnosis canonicalization; "
        "ICD10CM → RxNorm via treats-relationship; LOINC → SNOMED for "
        "specimen-aware lab mapping. Use when grounding spans vocabularies. "
        "Returns {status: 'unavailable'} if OMOPHUB_MCP_URL is not set."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "source_vocabulary": {
                "type": "string",
                "description": (
                    "Source OMOP vocabulary id (e.g. 'ICD10CM', 'SNOMED', "
                    "'LOINC', 'RxNorm')."
                ),
            },
            "source_code": {
                "type": "string",
                "description": "Code to translate (e.g. 'E11.9', '2160-0').",
            },
            "target_vocabulary": {
                "type": "string",
                "description": (
                    "Target OMOP vocabulary id you want the code mapped to."
                ),
            },
            "max_results": {
                "type": "integer",
                "description": "Mappings to return (default 25, max 50).",
                "default": 25,
                "maximum": 50,
            },
        },
        "required": ["source_vocabulary", "source_code", "target_vocabulary"],
    },
}


ICD_AUTOCODE_TOOL_DEF: dict[str, Any] = {
    "name": "icd_autocode",
    "description": (
        "Suggest ICD-10 (default) or ICD-11 codes for free-text clinical "
        "narrative. Returns ranked candidates with confidence scores when "
        "the upstream provides them. Use to translate prose (admission-note "
        "phrases, problem-list bullets) into structured codes. Returns "
        "{status: 'unavailable'} when ICD_MCP_URL is not set."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "Clinical text to autocode.",
            },
            "version": {
                "type": "string",
                "description": "ICD version: '10' (default) or '11'.",
                "default": "10",
            },
            "max_results": {
                "type": "integer",
                "description": "Candidates to return (default 10, max 25).",
                "default": 10,
                "maximum": 25,
            },
        },
        "required": ["text"],
    },
}


ALL_TOOL_DEFS: list[dict[str, Any]] = [
    PUBMED_SEARCH_TOOL_DEF,
    MIMIC_DISTRIBUTION_TOOL_DEF,
    LOINC_REFERENCE_RANGE_TOOL_DEF,
    SNOMED_SEARCH_TOOL_DEF,
    SNOMED_EXPAND_ECL_TOOL_DEF,
    RXNORM_LOOKUP_TOOL_DEF,
    CODE_MAP_TOOL_DEF,
    TRIALS_SEARCH_TOOL_DEF,
    OPENFDA_DRUG_LABEL_TOOL_DEF,
    ICD_LOOKUP_TOOL_DEF,
    ICD_AUTOCODE_TOOL_DEF,
]


TOOL_DISPATCH: dict[str, Any] = {
    "pubmed_search": pubmed_search,
    "mimic_distribution_lookup": mimic_distribution_lookup,
    "loinc_reference_range": loinc_reference_range,
    "snomed_search": snomed_search,
    "snomed_expand_ecl": snomed_expand_ecl,
    "rxnorm_lookup": rxnorm_lookup,
    "code_map": code_map,
    "trials_search": trials_search,
    "openfda_drug_label": openfda_drug_label,
    "icd_lookup": icd_lookup,
    "icd_autocode": icd_autocode,
}
"""Map of tool-name → callable, looked up by ``EvidenceAgent`` when the
model emits a ``tool_use`` block."""
