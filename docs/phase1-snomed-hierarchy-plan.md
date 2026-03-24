# Phase 1: SNOMED Concept Hierarchy Resolver

## Problem

The conversational pipeline can't handle category-level queries like "show all antibiotics" or "what vasopressors were used" because MIMIC stores specific drug names (Vancomycin, Norepinephrine) while users ask about categories. The same issue exists for lab categories ("liver function tests"), diagnosis groups ("cardiovascular diseases"), and organism classes ("gram-negative bacteria").

Currently the LLM prompt hacks around this by telling Claude to expand categories into specific names, but this is brittle — the LLM doesn't know which of the 182 mapped drugs are antibiotics, and will miss many.

## What Exists Today

The project already has:
- **6 SNOMED mapping files** in `data/mappings/` — flat concept-code-to-SNOMED-code mappings for drugs (182), lab items (1,403), chart items (44), organisms (468), ICD codes (5,031), comorbidities (42)
- **`SnomedMapper`** class (`src/graph_construction/terminology/snomed_mapper.py`) — lazy-loading lookup by identifier type
- **`MappingChain`** (`src/graph_construction/terminology/mapping_chain.py`) — waterfall pattern: try static JSON, fall back to UMLS API
- **RF2 parser** (`scripts/parse_loinc_snomed_rf2.py`) — existing code for parsing SNOMED RF2 release files
- **SNOMED triples in the RDF graph** — every event gets `mimic:hasSnomedCode`, `mimic:hasSnomedTerm`, `mimic:hasSnomedConcept` triples via `_add_snomed_triples()` in `event_writers.py`

**What's missing**: The IS-A hierarchy. We have codes but no parent-child relationships. Can't answer "is Vancomycin an antibiotic?" without knowing that SNOMED code 372735009 (Vancomycin) IS-A 372532007 (Antibiotic).

## Architecture

```
User: "What antibiotics were prescribed?"
                    │
                    ▼
         ┌──────────────────┐
         │   Decomposer     │  concept_name="antibiotics", concept_type="drug"
         └────────┬─────────┘
                  │
                  ▼
     ┌─────────────────────────┐
     │  Concept Resolver (NEW) │  "antibiotics" → SNOMED 36020009
     │                         │  → query IS-A children
     │                         │  → [vancomycin, ceftriaxone, meropenem, ...]
     │                         │  → map back to MIMIC drug names
     └────────────┬────────────┘
                  │
                  ▼
         ┌──────────────────┐
         │    Extractor      │  WHERE drug IN ('Vancomycin', 'CeftriaXONE', ...)
         └──────────────────┘
```

## Implementation Plan

### Step 0: Obtain SNOMED CT Data

**Source**: SNOMED CT International Edition RF2 release (free for NLM UMLS licensees)

Download from: https://www.nlm.nih.gov/healthit/snomedct/us_edition.html
- Need: `SnomedCT_USEditionRF2_PRODUCTION_*/Snapshot/Terminology/`
  - `sct2_Relationship_Snapshot_US*.txt` — the IS-A hierarchy (this is the critical file)
  - `sct2_Description_Snapshot-en_US*.txt` — preferred terms (already parsed in `parse_loinc_snomed_rf2.py`)

Alternatively, use the UMLS REST API to traverse ancestors per-concept (slower but no download needed).

### Step 1: Parse RF2 IS-A Relationships

**New script**: `scripts/build_snomed_hierarchy.py`

Parse `sct2_Relationship_Snapshot` to extract IS-A relationships:
- Filter rows where `typeId = 116680003` (IS-A relationship) and `active = 1`
- Build parent map: `{child_sctid: [parent_sctids]}`
- Compute transitive closure (all ancestors) via BFS/DFS
- Assign root categories based on SNOMED top-level concepts:
  - `373873005` → Pharmaceutical / biologic product
  - `404684003` → Clinical finding
  - `71388002` → Procedure
  - `105590001` → Substance
  - `123037004` → Body structure

**Output**: `data/mappings/snomed_hierarchy.json`
```json
{
  "_metadata": {"source": "SNOMED CT US Edition RF2", "concepts": 350000},
  "372735009": {
    "term": "Vancomycin",
    "parents": ["372532007"],
    "ancestors": ["372532007", "373225007", "373873005"],
    "root_category": "pharmaceutical"
  }
}
```

Estimated size: ~50MB JSON for full hierarchy. Can be filtered to only concepts referenced in our existing mappings to reduce to ~5MB.

Follow the existing pattern from `parse_loinc_snomed_rf2.py` for RF2 file handling.

### Step 2: Hierarchy Provider

**New class**: `src/graph_construction/terminology/snomed_hierarchy.py`

```python
class SnomedHierarchy:
    """Lazy-loaded SNOMED IS-A hierarchy for concept resolution."""

    def __init__(self, hierarchy_path: Path):
        self._path = hierarchy_path
        self._data: dict | None = None

    def get_ancestors(self, sctid: str) -> list[str]:
        """Return all ancestor SNOMED codes (transitive IS-A)."""

    def get_children(self, sctid: str) -> list[str]:
        """Return all descendant SNOMED codes."""

    def is_a(self, child_sctid: str, parent_sctid: str) -> bool:
        """Check if child IS-A parent (directly or transitively)."""

    def get_category(self, sctid: str) -> str | None:
        """Return root category (pharmaceutical, finding, substance, etc.)."""
```

Follows the lazy-loading pattern from `SnomedMapper`.

### Step 3: Concept Resolver for the Conversational Pipeline

**New module**: `src/conversational/concept_resolver.py`

This sits between the decomposer and the extractor. When the decomposer outputs a category-level concept name (e.g. "antibiotics"), the resolver:

1. Looks up the category in a **category-to-SNOMED mapping** (curated, ~50 entries):
   ```json
   {
     "antibiotics": "372532007",
     "vasopressors": "372881000",
     "sedatives": "372614000",
     "liver function tests": "250639003",
     "electrolytes": "86355000",
     "gram-negative bacteria": "81325006"
   }
   ```

2. Queries `SnomedHierarchy.get_children(sctid)` to find all descendant concepts

3. Cross-references descendants against the existing SNOMED mappings (`drug_to_snomed.json`, `labitem_to_snomed.json`, etc.) to find which MIMIC identifiers belong to that category

4. Returns a list of specific concept names to use in extraction

```python
class ConceptResolver:
    def __init__(self, hierarchy: SnomedHierarchy, mapper: SnomedMapper):
        ...

    def resolve(self, concept: ClinicalConcept) -> list[str]:
        """Resolve a possibly-categorical concept to specific MIMIC names.

        Returns [concept.name] if already specific, or a list of specific
        names if the concept is a category with SNOMED children.
        """
```

### Step 4: Integrate into Pipeline

**Modify**: `src/conversational/extractor.py`

In `_extract_concept()`, before calling the type-specific handler:
1. Call `resolver.resolve(concept)` to get specific names
2. If multiple names returned, query for each and merge results
3. Pass resolved names to the handler's SQL `WHERE drug ILIKE ?` clause

**Modify**: `src/conversational/orchestrator.py`

Initialize `ConceptResolver` in the pipeline constructor (lazy — only loads hierarchy when first needed).

### Step 5: Update Decomposer Prompt

Remove the "IMPORTANT: Use specific drug names" hack. Instead, let the LLM freely use category names like "antibiotics", "vasopressors", "liver function tests" — the resolver handles expansion.

## Key Design Decisions

- **Offline hierarchy**: Parse RF2 once, cache as JSON. No runtime API calls for hierarchy traversal.
- **Lazy loading**: Hierarchy JSON only loaded when a category-level concept is detected.
- **Filtered hierarchy**: Only include SNOMED codes that have corresponding MIMIC identifiers, not the full 350K concept tree.
- **Curated category map**: The category-name-to-SNOMED-root mapping is small and hand-curated (~50 entries). This is the LLM↔SNOMED bridge.
- **Fallback**: If a concept name isn't in the category map and isn't resolvable, pass it through unchanged (current behavior).

## Files

| File | Status | Description |
|---|---|---|
| `scripts/build_snomed_hierarchy.py` | NEW | Parse RF2 IS-A relationships, build hierarchy JSON |
| `data/mappings/snomed_hierarchy.json` | NEW (generated) | Concept hierarchy with parents/ancestors |
| `data/mappings/category_to_snomed.json` | NEW (curated) | Category names → SNOMED root codes |
| `src/graph_construction/terminology/snomed_hierarchy.py` | NEW | Hierarchy provider class |
| `src/conversational/concept_resolver.py` | NEW | Category → specific names resolver |
| `src/conversational/extractor.py` | MODIFY | Use resolver before extraction |
| `src/conversational/orchestrator.py` | MODIFY | Initialize resolver |
| `src/conversational/prompts.py` | MODIFY | Remove specific-drug-name hack |

## Prerequisites

- SNOMED CT US Edition RF2 release (download from NLM, requires free UMLS license)
- Or: UMLS API key for per-concept ancestor queries (slower, no download needed)

## Verification

```
# After implementation, these should all work:
"What antibiotics were prescribed to sepsis patients?"
  → Resolves to: vancomycin, ceftriaxone, meropenem, piperacillin, ...

"Show liver function tests for patients over 65"
  → Resolves to: ALT, AST, alkaline phosphatase, bilirubin, albumin, ...

"What vasopressors were used in the first 24 hours?"
  → Resolves to: norepinephrine, vasopressin, phenylephrine, epinephrine, ...

"Are there any gram-negative organisms in the cultures?"
  → Resolves to: E. coli, Klebsiella, Pseudomonas, ...
```
