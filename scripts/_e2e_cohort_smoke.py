"""Non-interactive end-to-end smoke for the described-profile cohort path.

Drives the *exact* ``ConversationalPipeline.ask()`` path the chat UI uses
(decompose → build cohort definition → run_cohort against BigQuery), but
headless and timed, so we can confirm the demo question returns a non-empty
ranked cohort in seconds — not the ~21-minute wedge the O(n^2) param build +
per-id placeholder explosion used to produce on the ~199k EW EMER./DIRECT EMER.
pool.

The pre-execution MCP validator is left OFF here: it is an orthogonal piece of
infra that fails *open* (a timeout just yields ``verdict=None`` and the query
still runs), and in this environment it has been timing out — including it would
add latency noise without changing correctness, muddying the signal about the
scaling fix this smoke exists to verify.

Usage:
    .venv/bin/python scripts/_e2e_cohort_smoke.py
    .venv/bin/python scripts/_e2e_cohort_smoke.py "some other question"
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
    question = sys.argv[1] if len(sys.argv) > 1 else _DEMO_QUESTION
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
    print(f"question    : {question}")
    print("-" * 78)

    pipeline = ConversationalPipeline(
        db_path=settings.duckdb_path,
        ontology_dir=ontology_dir,
        api_key=settings.anthropic_api_key,
        data_source="bigquery",
        bigquery_project=settings.bigquery_project,
        enable_pre_validator=False,  # see module docstring
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
