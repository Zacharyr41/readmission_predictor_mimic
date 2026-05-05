# Phase H — Externally-grounded critic v2

## Why

The pre-Phase-H critic (`src/conversational/critic.py`) reasoned from a
24-row reference table baked into `CRITIC_SYSTEM_PROMPT`. That worked for
common analytes but didn't generalize:

- Niche analytes outside the table (e.g. procalcitonin, troponin T) had
  no in-prompt range — the model fell back to recall.
- Cohort-specific distributions weren't covered. The lactate-7.99-mmol/L
  in-sepsis case (manually validated against an NIH editorial) was the
  motivating example: the value sits well above the general-population
  upper bound but is published-typical for severity-selected sepsis.
- The reference table contradicted user-feedback memory
  (`feedback_general_over_specific.md`): "general principles over
  lab-specific tables — lab-specific tables don't generalize."

Phase H replaces the recall-from-table pattern with **lookup-via-tool**.
The critic gains access to all 11 health-evidence tools, the
population-specific reference table is removed, and the universal
biological-impossibility floor stays in-prompt (because physics doesn't
change by cohort).

This phase combines what the planning doc called Bucket 1 (broaden the
critic) and Bucket 2 (give the 5 Phase-G MCPs production reach). The
unifying move: route all 11 tools through the default-on critic, so
every Phase-G MCP gains a production consumer immediately — no
`enable_contextualization` flag flip required.

## What landed

| Change | File | Notes |
|---|---|---|
| `_CRITIC_TOOLS` tuple | `src/conversational/critic.py` | One-line addition adds a tool |
| Filtered `_CRITIC_TOOL_DEFS` | `src/conversational/critic.py` | Subset of `ALL_TOOL_DEFS` |
| Late-binding-safe dispatch | `src/conversational/critic.py:_critic_tool_dispatch` | Comprehension with `_n=name` default-arg trick |
| 11-tool reach | `src/conversational/critic.py` | Was 1 (pubmed_search), now all 11 |
| Prompt shrink | `src/conversational/prompts.py` | 24-row table → 10-row impossibility floor |
| Tool enumeration in prompt | `src/conversational/prompts.py` | "Available tools" section rewritten |
| Multi-source citation shape | `src/conversational/models.py:CriticVerdict` | Accepts `{source, id}` AND legacy `{type, pmid}` |
| `CitationSource` extended | `src/conversational/health_evidence/models.py` | Added: snomed, rxnorm, icd, openfda, clinicaltrials |
| `_record_observations` for 8 sources | `src/conversational/health_evidence/agent.py` | Citation tracking for all citing tools |
| `tool_calls` field on verdict | `src/conversational/models.py:CriticVerdict` | Per-tool telemetry, populated by `critique()` |
| Orchestrator counters | `src/conversational/orchestrator.py:__init__` | `_critic_invocations`, `_critic_tool_calls`, `_critic_tool_unavailable` |
| Counter drain in `_critique` | `src/conversational/orchestrator.py:_critique` | Increments per critique that returned a verdict |
| Smoke script | `scripts/critic_evidence_smoke.py` | 8 probes, env-var-aware, exits 0 on any backend ok |

## How it differs from v1

### Tool reach

| Tool | Pre-Phase H | Phase H |
|---|---|---|
| `pubmed_search` | ✓ | ✓ |
| `loinc_reference_range` | – | ✓ |
| `mimic_distribution_lookup` | – | ✓ |
| `snomed_search` | – | ✓ |
| `snomed_expand_ecl` | – | ✓ |
| `rxnorm_lookup` | – | ✓ |
| `code_map` | – | ✓ |
| `trials_search` | – | ✓ |
| `openfda_drug_label` | – | ✓ |
| `icd_lookup` | – | ✓ |
| `icd_autocode` | – | ✓ |

### Prompt deltas

- **Removed:** the 24-row reference table with population-typical and
  ICU-plausible upper-bound columns. The model is told to call
  `loinc_reference_range` (general-population) or
  `mimic_distribution_lookup` (cohort-typical) instead of recalling.
- **Kept:** the failure-mode taxonomy (6 modes), the cohort-selection
  principle (this is a reasoning rule, not a table), the self-healing
  LOINC suggestion path (orthogonal to external grounding).
- **Added:** a 10-row universal-impossibility floor (physics: lactate >
  20 mmol/L sustained, sodium < 100 or > 180 mEq/L, mortality > 1.0,
  age > 130, etc.). These don't depend on cohort and don't need a tool
  call.
- **Added:** explicit per-tool guidance in the "Available tools"
  section ("call loinc_reference_range when... call rxnorm_lookup
  when..." etc.), so the model picks the right tool by purpose.
- **Updated:** the `cited_sources` schema documents both shapes —
  multi-source `{source, id, ...}` (preferred) and legacy
  `{type: pubmed, pmid: ...}` (back-compat).

## Setup for optional MCPs

Three of the 11 tools work fully offline:

- `pubmed_search` — public NCBI E-utilities (no auth required;
  `NCBI_API_KEY` raises rate limits if set).
- `loinc_reference_range` — reads
  `data/ontology_cache/loinc_reference_ranges.json`.
- `mimic_distribution_lookup` — reads `data/processed/lab_distributions.json`.

The remaining eight require external MCPs. Configure via env vars; each
backend independently degrades to `status=unavailable` if missing, and
the critic continues with whatever's left.

| Tool | Env vars | Backend |
|---|---|---|
| `pubmed_search` | `PUBMED_BACKEND`, `NCBI_API_KEY`, `PUBMED_MCP_URL` | NCBI / Anthropic-hosted MCP / self-hosted |
| `snomed_search`, `snomed_expand_ecl` | `HERMES_MCP_COMMAND`, `HERMES_MCP_DB` | Hermes (stdio) |
| `rxnorm_lookup`, `code_map` | `OMOPHUB_MCP_URL`, `OMOPHUB_API_KEY` | OMOPHub (HTTP) |
| `trials_search` | `CLINICALTRIALS_MCP_COMMAND` (defaults to `bunx`/`npx clinicaltrialsgov-mcp-server@latest`) | cyanheads stdio MCP |
| `openfda_drug_label` | `OPENFDA_MCP_COMMAND` (defaults to `npx openfda-mcp-server`), `OPENFDA_API_KEY` | npx stdio MCP |
| `icd_lookup`, `icd_autocode` | `ICD_MCP_URL` | Self-hosted ICD MCP (WHO licensing) |

## Graceful degradation

When a backend is unavailable, its tool returns:

```json
{"status": "unavailable", "error": "<reason>"}
```

The agent loop never raises — it logs the failure, marks the per-tool
`tool_calls` entry with `status="unavailable"`, and the model continues
to the next iteration. The `_critic_tool_unavailable` counter
increments per occurrence.

If every tool the critic tries is unavailable, the verdict still
renders — falling back to the universal-impossibility floor and the
cohort-selection principle. There's no path where a missing backend
causes a turn to fail.

## How to run the smoke script

```bash
.venv/bin/python scripts/critic_evidence_smoke.py
```

The script probes all 8 citation-producing tools. For each backend:

- If the env var / binary is missing, prints `skipped (reason)` and
  moves on.
- Otherwise, calls `critique()` against a synthetic
  `(CompetencyQuestion, AnswerResult)` pair designed to make that tool
  the natural choice.
- Reports: invocation status, latency, citation in verdict.

Exit code 0 when at least one configured backend was actually exercised
and returned `ok`. Exit 1 otherwise.

This is the manual verification gate before flipping any default. It
does NOT replace the broader real-MCP CI gate (Bucket 5 of the
remaining-work plan); that's a separate phase.

## Telemetry surface

All counters cumulative across `ask()` calls:

```python
pipeline._critic_invocations           # int — critique() returned a verdict
pipeline._critic_tool_calls            # dict[str, int] — invocations per tool
pipeline._critic_tool_unavailable      # dict[str, int] — unavailable-status subset
```

Mirrors the pre-existing `_pre_validator_*` counter pattern. The
`CriticVerdict.tool_calls` field carries the same data per-verdict
(safe to surface in the chat UI debug pane; not shown in user-visible
verdict text).

## Known caveats

- **Latency ceiling.** `_MAX_TOOL_ITERATIONS=3` × per-MCP latency
  (~1–3s) = up to ~9s overhead per critique. Mitigated by Sonnet 4.6
  prompt caching on the (large, stable) system block.
- **3-tool cap per critique.** The model will not call more than 3
  tools per verdict. The cap is enforced via
  `tool_choice={"type":"none"}` on the cap iteration.
- **Disambiguate / clarify / contextualize paths still narrow.** Phase
  H broadens only the critic. The decomposer-driven `disambiguate` and
  `clarify` helpers (`src/conversational/clinical_consult.py`) still
  use the 3-tool subset (pubmed + mimic + loinc). Broadening those is
  Bucket 3 of the remaining-work plan.
- **`code_map` skips citation tracking** by design — it produces
  cross-vocabulary mappings rather than primary records. The model
  can still reason about the result in narrative; it just won't show
  up in `cited_sources`.
- **Snapshot test.** `tests/test_conversational/fixtures/critic_prompt_snapshot.txt`
  was regenerated. Future prompt edits will fail the snapshot test
  with a clear "rm snapshot, re-run, commit" path.

## Verification (when this lands)

1. `.venv/bin/python -m pytest tests/test_conversational/ -q` — expect
   the prior 863 passes plus ~25 new tests (884+ total). Zero new
   skips, zero new xfails.
2. Diff `tests/test_conversational/fixtures/critic_prompt_snapshot.txt`
   — confirm prompt shape changes are intentional (table removed,
   tools enumerated, cohort-selection preserved).
3. Run the smoke script with at least one Phase-G MCP configured —
   expect a non-zero number of `YES`/`ok` rows.
4. Live chat-UI: ask a question with a niche analyte ("typical
   procalcitonin in septic shock?") and verify the verdict's
   `cited_sources` references `loinc` or `pubmed` rather than a
   recalled range.
5. Live chat-UI: the lactate-7.99-in-sepsis scenario from
   `memory/project_critic_external_grounding.md` — verify the verdict
   returns `severity=info` with `cited_sources` referencing
   `mimic_distribution` (if MIMIC distributions JSON has the data) or
   `pubmed` (fallback).
