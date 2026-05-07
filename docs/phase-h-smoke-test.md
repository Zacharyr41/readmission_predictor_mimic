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

## ⚠️ Tier D: `DATA_SOURCE` must match your session backend

The on-the-fly cohort compute fallback reads `DATA_SOURCE` from
`.env` (not from the streamlit sidebar). For the critic's compute
results to match what the chat UI is actually querying:

```bash
# In .env — match this to whichever backend your sidebar normally uses
DATA_SOURCE=bigquery   # or "local" for DuckDB
```

Restart streamlit after changing this so the env var gets re-sourced.
The catalog cache is source-agnostic and works regardless; only the
on-the-fly path is affected.

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

## 2 — Canonical case: single-patient lactate 7.99 in sepsis (→ info)

The canonical case from `memory/project_critic_external_grounding.md`:
**a single patient** in sepsis with a lactate measurement of 7.99 mmol/L.
This is a right-tail individual value — published lactate-mortality
strata (J Thorac Dis 2016;8(7):1388 etc.) put 4–8 mmol/L as typical
for severe septic shock. Expected verdict: `severity="info"`.

Note the `scope="single_patient"` and `n=1` framing — this is what
makes it a per-row distribution question, not an aggregate.

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
    original_question="A sepsis patient's lactate reading was 7.99 mmol/L. Is that plausible?",
    clinical_concepts=[ClinicalConcept(name="lactate", concept_type="biomarker")],
    return_type="text_and_table", scope="single_patient",
    interpretation_summary="Single sepsis admission, individual lactate measurement.",
)
answer = AnswerResult(
    text_summary="The patient's lactate was 7.99 mmol/L (single measurement, n=1) — they have sepsis.",
    data_table=[{"lactate_mmol_per_L": 7.99, "patient_count": 1}],
)
verdict = critique(client, cq, answer)
print(json.dumps(verdict.model_dump() if verdict else None, indent=2, default=str))
PY
```

**What success looks like** for this scenario:

- `severity: "info"` (the value is shifted-but-plausible for a sepsis
  patient — right tail of the cohort distribution).
- `tool_calls` non-null and non-empty, ideally including a
  `mimic_distribution_lookup` call with `cohort="sepsis"` so the model
  is consulting the empirical distribution (not just recalling
  training-data ranges).
- If the model also called `pubmed_search`, that's a bonus — sepsis
  lactate-mortality literature is the right population-level evidence.

For reference (from the Tier-D production catalog):
- All-MIMIC lactate p95: **4.7 mmol/L**
- Sepsis-cohort lactate p95: **6.9 mmol/L**
- Septic-shock-cohort lactate p95: **8.5 mmol/L**
- Hepatic-failure-cohort lactate p95: **10.2 mmol/L**

So a single-patient value of 7.99 mmol/L sits below septic_shock p95
and well below hepatic_failure p95 — clearly within the right-tail of
severity-selected distributions.

**What failure looks like:**

- `severity: "warn"` or `"block"` for this question would be a
  regression — the critic ignored the cohort-selection principle.
- `cited_sources: null` AND the verdict reasoning mentions a specific
  range — that means the model recalled instead of looking up. Check
  the `tool_calls` field; if it's empty, the prompt isn't routing the
  model to tools strongly enough.
- `verdict is None` means `critique()` failed entirely. Check the log;
  the agent's exception is logged at WARNING level.

## 2b — Counter-case: cohort *mean* of 7.99 across 512 sepsis patients (→ warn)

Same value, different question: instead of a single patient, this is
a *cohort mean* across 512 sepsis admissions. That distinction matters
hugely. Sepsis-cohort lactate **mean** in MIMIC is ~2.4 mmol/L; even
septic_shock cohort mean is ~2.8 mmol/L. A cohort mean of 7.99 means
the *entire distribution* has shifted up — which suggests pollution
(e.g., LDH-pooled values), units mismatch, or unusually narrow
selection that the user should justify. Expected: `severity="warn"`.

Note the `scope="cohort"` and `n=512` framing.

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

**What success looks like:**

- `severity: "warn"` with a concern that **distinguishes the mean
  from per-row p95** — the model should explicitly note that 7.99 is
  ~3× the cohort mean (2.4) and that a cohort mean shifted that far
  isn't right-tail variance, it's distributional shift.
- `tool_calls` includes `mimic_distribution_lookup(cohort="sepsis")`
  AND ideally `cohort="septic_shock"` (the model should sibling-check
  before flagging — see the prompt's worked example).

**What failure looks like:**

- `severity: "info"` — the model conflated single-row right-tail
  with cohort-mean shift. The cohort-mean signal is the load-bearing
  reason this should warn.
- The concern says "above p95" without distinguishing mean vs p95 —
  that's an apples-to-oranges comparison the prompt now warns against.

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

**Success (post-Tier-D):** `tool_calls` includes
`mimic_itemid_search(query="procalcitonin")` returning `n=0`
(procalcitonin is genuinely absent from MIMIC-IV), then a pivot
to `pubmed_search` for population evidence. The critic should
recognise that a "MIMIC-derived n=84 procalcitonin cohort" is
contradictory and produce `severity="warn"` flagging the
data-provenance mismatch — this is a sharper verdict than the
pre-Tier-D version, which incorrectly accepted the fictional MIMIC
sample as plausible based only on PubMed ranges.

For analytes that ARE in MIMIC but the model doesn't already know
the itemid, success is: `mimic_itemid_search` returns the itemid,
then `mimic_distribution_lookup(itemid=<resolved>, cohort=...)`
is called for the cohort distribution. The chain is
`search → distribution_lookup`.

**Failure:** verdict references a hard-coded "0.0–0.5 ng/mL" with no
tool call — the model recalled. OR: the critic guesses an itemid
without calling `mimic_itemid_search` first (regression to the
pre-Tier-D behaviour).

## 3b — Tier D raw-prefix scenario (cohort not in registry)

This exercises the on-the-fly compute fallback. We pick a cohort
that's deliberately NOT in `clinical_cohorts.json` — delirium
(ICD-10 F05) — and frame the question with a **borderline-suspicious
value** that compels the critic to verify against the actual cohort
distribution rather than dismiss from training recall.

> **Why a borderline value?** An uncontroversial number (e.g. mean
> creatinine 1.6 mg/dL — well within population-typical ICU range)
> won't trip the prompt's "use tools sparingly" discipline; the
> critic will rightly skip the lookup. To exercise the on-the-fly
> compute INFRASTRUCTURE, the value needs to be high enough that the
> model can't confidently judge it without checking. **3.8 mg/dL** is
> ~2.5× the all-MIMIC creatinine mean (~1.5) and sits near the AKI
> cohort p95 (~3.9) — a delirium cohort might or might not have that
> shifted distribution depending on renal-impairment overlap. Genuinely
> needs lookup.

The critic has two paths: (a) chain `icd_lookup(query="delirium")`
to resolve, then call `mimic_distribution_lookup(icd10_prefixes=["F05"])`;
(b) recall the ICD code from training and pass prefixes directly.
Either path is valid — the user never sees ICD codes.

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
    original_question="Mean creatinine in our delirium patients",
    clinical_concepts=[ClinicalConcept(name="creatinine", concept_type="biomarker")],
    return_type="text_and_table", scope="cohort",
    interpretation_summary="Cohort: ICU admissions with delirium (ICD F05). Aggregate: mean serum creatinine.",
)
answer = AnswerResult(
    text_summary="Mean creatinine 3.8 mg/dL (n=320) in delirium patients.",
    data_table=[{"mean_creatinine_mg_dl": 3.8, "n": 320}],
)
verdict = critique(client, cq, answer)
print(json.dumps(verdict.model_dump() if verdict else None, indent=2, default=str))
PY
```

**Success:** `tool_calls` includes `mimic_distribution_lookup` with
either `cohort="delirium"` (which falls through to compute since the
cohort isn't in the registry) or `icd10_prefixes=["F05"]`. The result
record has `source: "computed"` (not `"catalog"`). Latency ~50–200ms
locally, ~1–3s on BigQuery.

The verdict itself can be `info` OR `warn` — both are defensible
depending on what the live computed delirium-cohort distribution
shows. What matters for this smoke is that **the lookup actually
fired**, demonstrating the on-the-fly compute path works end-to-end
with a real backend.

**Failure:** `tool_calls` is empty (model dismissed from training
recall — value wasn't suspicious enough to compel lookup; bump to a
higher value like 5.0 mg/dL). Or: `source: "computed"` but `units`
is wrong — that suggests the fixture path is broken; check
`MIMIC_COMPUTE_DUCKDB_PATH`.

**Note on uncontroversial values:** if you run this scenario with
a value like 1.6 mg/dL instead, expect `severity=info` /
`tool_calls=null`. That's correct behavior — the critic's
`Tool-use discipline` section says only invoke tools when judgment
genuinely turns on a quantitative question. 1.6 mg/dL in any ICU
cohort doesn't, so tool use would be wasteful. The on-the-fly
compute path is unit-tested separately; this live scenario tests
that the integration works when the model decides it's needed.

## 4 — Cohort-canonicalization scenario (works once `OMOPHUB_API_KEY` is set)

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

**With OMOPHub configured (`OMOPHUB_API_KEY` set):** the model has
both `rxnorm_lookup` and `icd_lookup` available, both routed through
the hosted OMOPHub MCP. Whether they actually fire depends on the
prompt's tool-use discipline — for a clearly-uncontroversial
percentage like "30% Levophed in HF", the model may correctly skip
tools and verdict from training recall (`severity=info`). To force
tool use, change the percentage to something borderline (e.g. 90%
or 5%) where the model can't dismiss from training.

If the model DOES invoke tools:
- `rxnorm_lookup("Levophed")` returns SPL/RxNorm records (the OMOPHub
  search ranks SPL high for brand names; both are drug-related and
  useful for disambiguation).
- `icd_lookup("heart failure")` returns ICD-10-CM codes like I50.814
  (right heart failure due to left heart failure).

**Without OMOPHub configured:** the critic falls back to training
recall or PubMed. The verdict still renders — graceful degradation.

**With a self-hosted ICD MCP (`ICD_MCP_URL` set):** ICD-11 lookups
also work; the legacy dialect uses tool names `lookup` and
`autocode` rather than OMOPHub's `search_concepts` /
`semantic_search`.

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
