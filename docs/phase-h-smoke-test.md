# Phase H Smoke Test — Externally-grounded critic, by hand

Hand-run verification that the Phase H critic actually calls its tools,
cites them correctly, and falls back gracefully when a backend is
missing. ~10 minutes if everything is already configured.

This is the **manual** companion to `scripts/critic_evidence_smoke.py`.
The script is good for "is the wiring intact"; this guide is for "does
it actually behave the way I want when a clinician asks a real
question."

## What you need

- Anthropic API key in `.env` (`ANTHROPIC_API_KEY=sk-ant-...`).
- The two offline catalogs the critic reads from:
  - `data/ontology_cache/loinc_reference_ranges.json`
  - `data/processed/lab_distributions.json`
- (Optional) Any subset of the 5 Phase-G MCPs configured per env vars
  in `docs/phase-h-externally-grounded-critic.md`. The smoke runs
  fine with zero of them — the LOINC + MIMIC + PubMed offline path
  is the floor.

## 0 — Pre-flight

```bash
# Confirm the catalogs exist (Phase H needs at least one).
ls -lh data/ontology_cache/loinc_reference_ranges.json
ls -lh data/processed/lab_distributions.json

# Confirm the critic carries all 11 tools.
.venv/bin/python -c "from src.conversational.critic import _CRITIC_TOOLS; print(len(_CRITIC_TOOLS), _CRITIC_TOOLS)"
# Expect: 11 ('pubmed_search', 'loinc_reference_range', ...)

# Confirm the snapshot fixture is committed (regeneration must be intentional).
git status tests/test_conversational/fixtures/critic_prompt_snapshot.txt
# Expect: clean (no diff).
```

If any of those fail, stop and fix before continuing.

## 1 — Run the automated probe

```bash
set -a && source .env && set +a
.venv/bin/python scripts/critic_evidence_smoke.py
```

What you'll see (varies by which MCPs you've configured):

```
Phase H critic-evidence smoke — 8 probes

BACKEND                          STATUS         LATENCY    INVOKED  CITATION
-----------------------------------------------------------------------------------------------
PubMed (NCBI E-utilities)        ok             4.12       YES      pubmed/27621888
LOINC catalog (offline)          ok             3.98       YES      loinc/33747-0
MIMIC distribution (offline)     ok             4.01       YES      mimic_distribution/50912
SNOMED via Hermes                skipped        -          -        (Hermes binary not on PATH...)
RxNorm via OMOPHub               skipped        -          -        (OMOPHUB_MCP_URL not set)
ICD via self-hosted MCP          skipped        -          -        (ICD_MCP_URL not set)
OpenFDA via npx MCP              ok             5.20       YES      openfda/Levophed
ClinicalTrials.gov via npx MCP   skipped        -          -        (need bunx or npx)

Exit 0: at least one configured backend was invoked and returned ok.
```

**Pass criteria:** at least one row says `ok` with `INVOKED=YES`. The
offline backends (PubMed, LOINC, MIMIC) should always succeed; if they
don't, that's a regression, not a setup issue.

**Don't expect every backend to fire on every probe.** The model is
free to skip a tool if the scenario doesn't compel it. The probes are
crafted to make each tool the natural choice, but model behavior is
non-deterministic. A `no` in the INVOKED column with an unconfigured
backend means the scenario didn't trip the model — not necessarily a
bug. Re-run once or twice; if a configured backend never fires across
3 runs, investigate.

## 2 — Inspect what the critic actually did

Run the canonical lactate-in-sepsis case directly through `critique()`
to see the verdict structure. Copy-paste:

```bash
.venv/bin/python <<'PY'
import os, json, anthropic
from src.conversational.critic import critique
from src.conversational.models import (
    AnswerResult, ClinicalConcept, CompetencyQuestion,
)

api_key = os.environ.get("ANTHROPIC_API_KEY") or open(".env").read().split("ANTHROPIC_API_KEY=")[1].split("\n")[0].strip()
client = anthropic.Anthropic(api_key=api_key)

cq = CompetencyQuestion(
    original_question="What is the mean lactate in our sepsis cohort?",
    clinical_concepts=[ClinicalConcept(name="lactate", concept_type="biomarker")],
    return_type="text_and_table", scope="cohort",
    interpretation_summary="Cohort: ICU admissions with sepsis. Aggregate: mean serum lactate (mmol/L).",
)
answer = AnswerResult(
    text_summary="Mean lactate 7.99 mmol/L (n=512) in sepsis cohort.",
    data_table=[{"mean_lactate_mmol_per_L": 7.99, "n": 512}],
)

verdict = critique(client, cq, answer)
print(json.dumps(verdict.model_dump() if verdict else None, indent=2, default=str))
PY
```

**What success looks like** for this scenario:

- `severity: "info"` (the value is shifted-but-plausible for sepsis,
  per the cohort-selection principle).
- `cited_sources` non-null with at least one entry, ideally referencing
  `mimic_distribution` or `pubmed`. A `loinc` citation alone is a yellow
  flag — the LOINC normal range says 0.5–2.2 mmol/L, so the model
  citing only LOINC suggests it didn't account for cohort selection.
- `tool_calls` non-null and non-empty, with at least one `status: "ok"`
  entry.

**What failure looks like:**

- `severity: "warn"` or `"block"` for this question would be a
  regression — pre-Phase-H the critic could have flagged it as
  "polluted with LDH" without checking the cohort distribution. Phase
  H should call `mimic_distribution_lookup` first.
- `cited_sources: null` AND the verdict reasoning mentions a specific
  range — that means the model recalled instead of looking up. Check
  the `tool_calls` field; if it's empty, the prompt isn't routing the
  model to tools strongly enough.
- `verdict is None` means `critique()` failed entirely. Check the log;
  the agent's exception is logged at WARNING level.

## 3 — Niche-analyte scenario (forces a LOINC lookup)

```bash
.venv/bin/python <<'PY'
import os, json, anthropic
from src.conversational.critic import critique
from src.conversational.models import (
    AnswerResult, ClinicalConcept, CompetencyQuestion,
)

api_key = os.environ.get("ANTHROPIC_API_KEY") or open(".env").read().split("ANTHROPIC_API_KEY=")[1].split("\n")[0].strip()
client = anthropic.Anthropic(api_key=api_key)

cq = CompetencyQuestion(
    original_question="What is the typical procalcitonin in our septic shock cohort?",
    clinical_concepts=[ClinicalConcept(name="procalcitonin", concept_type="biomarker")],
    return_type="text_and_table", scope="cohort",
    interpretation_summary="Cohort: ICU septic shock. Aggregate: mean procalcitonin (ng/mL).",
)
answer = AnswerResult(
    text_summary="Mean procalcitonin 12.4 ng/mL (n=84) in septic shock cohort.",
    data_table=[{"mean_procalcitonin_ng_per_mL": 12.4, "n": 84}],
)
verdict = critique(client, cq, answer)
print(json.dumps(verdict.model_dump() if verdict else None, indent=2, default=str))
PY
```

Procalcitonin is **deliberately not in the (now-shrunk) reference
table** — the pre-Phase-H critic would either skip the analyte or
guess. Phase H should call `loinc_reference_range` (LOINC 33747-0) or
`mimic_distribution_lookup`, or both, and cite the result.

**Success:** `tool_calls` includes `loinc_reference_range` OR
`mimic_distribution_lookup` (status=ok); `cited_sources` references
that source.

**Failure:** verdict references a hard-coded "0.0–0.5 ng/mL" with no
tool call — the model recalled. The prompt's "don't recall, look up"
instruction isn't biting hard enough.

## 4 — Cohort-canonicalization scenario (only if SNOMED/RxNorm/ICD configured)

```bash
.venv/bin/python <<'PY'
import os, json, anthropic
from src.conversational.critic import critique
from src.conversational.models import (
    AnswerResult, ClinicalConcept, CompetencyQuestion,
)

api_key = os.environ.get("ANTHROPIC_API_KEY") or open(".env").read().split("ANTHROPIC_API_KEY=")[1].split("\n")[0].strip()
client = anthropic.Anthropic(api_key=api_key)

cq = CompetencyQuestion(
    original_question="How many heart-failure patients received Levophed?",
    clinical_concepts=[
        ClinicalConcept(name="heart failure", concept_type="diagnosis"),
        ClinicalConcept(name="Levophed", concept_type="drug"),
    ],
    return_type="text_and_table", scope="cohort",
    interpretation_summary="Cohort: heart failure (ICD I50.9). Drug: Levophed (norepinephrine).",
)
answer = AnswerResult(
    text_summary="312 of 1,041 heart-failure patients received Levophed (29.97%).",
    data_table=[{"n_received": 312, "n_total": 1041, "pct": 29.97}],
)
verdict = critique(client, cq, answer)
print(json.dumps(verdict.model_dump() if verdict else None, indent=2, default=str))
PY
```

**With ICD MCP configured:** `tool_calls` should show `icd_lookup`
calling for "I50.9" or similar.

**With OMOPHub configured:** `rxnorm_lookup` should resolve
"Levophed" → norepinephrine RXCUI 7980.

**Without either configured:** the critic falls back to PubMed or
returns severity=info without external citations. The verdict should
still render — graceful degradation.

## 5 — Inspect telemetry

After running ≥1 of the above, the cumulative counters show the
critic's tool reach. From a fresh Python REPL or the chat UI's debug
pane:

```python
from src.conversational.orchestrator import ConversationalPipeline
from pathlib import Path
import os

pipeline = ConversationalPipeline(
    db_path=Path("data/processed/mimiciv.duckdb"),
    ontology_dir=Path("ontology/definition"),
    api_key=os.environ["ANTHROPIC_API_KEY"],
    data_source="local",
    enable_critic=True,
)
result = pipeline.ask("What is the mean lactate in our sepsis cohort?")

print("invocations:", pipeline._critic_invocations)
print("tool_calls:", pipeline._critic_tool_calls)
print("unavailable:", pipeline._critic_tool_unavailable)
```

**Expected shape:**
- `invocations: 1` (one critique() returned a verdict)
- `tool_calls: {'mimic_distribution_lookup': 1, 'pubmed_search': 1}` or similar
- `unavailable: {}` if all configured backends worked, or a few entries
  for the ones that aren't configured (e.g. `{'rxnorm_lookup': 1}` if
  the model tried RxNorm and OMOPHub isn't set)

**Red flag:** `invocations: 1` with `tool_calls: {}` means the critic
ran but didn't call any tools. The prompt-shrink may have been
too aggressive, or the scenario didn't compel a lookup. Iterate on the
question to verify.

## 6 — Live chat-UI run (full pipeline)

```bash
set -a && source .env && set +a && .venv/bin/streamlit run src/conversational/app.py
```

In the chat UI, ask:

1. **"What is the mean lactate in our sepsis cohort?"** — primary
   externally-grounded test. Verdict should be `severity=info` with a
   citation visible in the verdict pane.
2. **"What is the typical procalcitonin in septic shock?"** — niche
   analyte; verifies the LOINC tool path.
3. **"Mean creatinine in our AKI cohort?"** — verifies cohort-shifted
   creatinine doesn't get falsely flagged.

For each, expand the verdict's debug pane (or inspect
`result.critic_verdict.tool_calls` programmatically). Look for:

- A non-empty `tool_calls` array.
- A `cited_sources` array with `source: "loinc"`, `"pubmed"`, or
  `"mimic_distribution"` entries (not all three on every turn — the
  model picks).

## Failure-mode triage

| Symptom | Likely cause | Fix |
|---|---|---|
| Smoke script exits 1 with 0 backends invoked | Critic prompt isn't routing model to tools | Re-check `CRITIC_SYSTEM_PROMPT` for "Available tools" section enumeration |
| `cited_sources: null` always | Validator dropping multi-source entries | `_validate_cited_sources_shape` in `models.py` should accept `{source, id}` — check git log for regression |
| Citations show but no `tool_calls` | `critique()` not populating from `EvidenceResult.tool_calls` | Check `critic.py:critique` — verdict construction must pass `tool_calls=[tc.model_dump() ...]` |
| `_critic_tool_unavailable` stays `{}` even with no MCPs configured | Model never tried optional tools — possibly fine | Run a scenario that should trip a missing MCP and re-check |
| Snapshot test fails | Prompt edited without snapshot update | `rm tests/test_conversational/fixtures/critic_prompt_snapshot.txt && pytest tests/test_conversational/test_critic.py::TestCriticPromptSnapshot` |
| Verdict is `None` always | Anthropic auth, model name, or schema-validation bug | Check `critic.py` log output — it logs the raw response on schema failures |
| Telemetry counters increment by 0 across multiple `ask()` calls | `enable_critic=False` or critic returning None | `pipeline._enable_critic` must be `True`; check `critique()` returns non-None |

## When to stop

You're done when:
- The smoke script (Step 1) exits 0.
- At least one of the manual scenarios (Steps 2–4) produces a verdict
  with non-empty `tool_calls` and a `cited_sources` entry that matches
  the called tool's source registry.
- The telemetry counters (Step 5) show `_critic_invocations >= 1` and
  `_critic_tool_calls` non-empty.
- The chat UI (Step 6) renders a verdict for the lactate-in-sepsis
  question without crashing.

That's enough signal to ship Phase H. The broader real-MCP CI gate
(Bucket 5 of the original remaining-work plan) is a separate phase.
