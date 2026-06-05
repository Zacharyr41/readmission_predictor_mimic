"""Non-interactive end-to-end smoke for the described-profile cohort path.

Drives the *exact* ``ConversationalPipeline.ask()`` path the chat UI uses
(decompose → build cohort definition → run_cohort against BigQuery), but
headless and timed, so we can confirm the demo question returns a non-empty
ranked cohort in seconds — not the ~21-minute wedge the O(n^2) param build +
per-id placeholder explosion used to produce on the ~199k EW EMER./DIRECT EMER.
pool.

The pre-execution MCP validator runs by default so this mirrors the live
dashboard exactly (it is what caught — falsely, until the array-param fix — the
``IN UNNEST(?)`` cohort feature-fetch). Pass ``--no-validator`` to isolate the
scoring path from the MCP round-trip when you only care about the scaling fix;
the validator fails *open*, so disabling it never changes the result, only
latency.

Usage:
    .venv/bin/python scripts/_e2e_cohort_smoke.py
    .venv/bin/python scripts/_e2e_cohort_smoke.py "some other question"
    .venv/bin/python scripts/_e2e_cohort_smoke.py --no-validator
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import Settings  # noqa: E402
from src.conversational.orchestrator import ConversationalPipeline  # noqa: E402

_DEMO_QUESTION = (
    "Find emergency patients similar to a 68-year-old whose creatinine ran "
    "high and whose platelets dropped, with a relatively short hospital stay."
)


def main() -> int:
    args = [a for a in sys.argv[1:]]
    use_validator = "--no-validator" not in args
    args = [a for a in args if a != "--no-validator"]
    question = args[0] if args else _DEMO_QUESTION
    settings = Settings()
    if settings.data_source != "bigquery":
        print(f"!! data_source={settings.data_source!r}, expected 'bigquery'.")
    if not settings.bigquery_project:
        print("!! BIGQUERY_PROJECT not set; aborting.")
        return 2
    if not settings.anthropic_api_key:
        print("!! ANTHROPIC_API_KEY not set; aborting.")
        return 2

    ontology_dir = Path(__file__).resolve().parent.parent / "ontology" / "definition"
    print(f"backend     : bigquery ({settings.bigquery_project})")
    print(f"validator   : {'ON (prod-faithful)' if use_validator else 'OFF'}")
    print(f"question    : {question}")
    print("-" * 78)

    pipeline = ConversationalPipeline(
        db_path=settings.duckdb_path,
        ontology_dir=ontology_dir,
        api_key=settings.anthropic_api_key,
        data_source="bigquery",
        bigquery_project=settings.bigquery_project,
        enable_pre_validator=use_validator,  # see module docstring
    )

    t0 = time.monotonic()

    def on_stage(label: str) -> None:
        print(f"  [{time.monotonic() - t0:6.1f}s] stage: {label}", flush=True)

    answer = pipeline.ask(question, progress_callback=on_stage)
    elapsed = time.monotonic() - t0

    print("-" * 78)
    print(f"completed in {elapsed:.1f}s")
    print("-" * 78)
    print(answer.text_summary)
    rows = getattr(answer, "data_table", None) or []
    print("-" * 78)
    print(f"ranked members returned: {len(rows)}")
    for r in rows[:5]:
        print(f"  {r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
