"""Reproduce the lactate query end-to-end through the orchestrator with
explicit per-step timing logs. Helps pin where the Streamlit hang lives."""

import logging
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

from src.conversational.orchestrator import ConversationalPipeline


def main() -> None:
    api_key = (
        open(".env").read().split("ANTHROPIC_API_KEY=")[1].split("\n")[0].strip()
    )
    t0 = time.time()
    print(f"[{time.time() - t0:6.2f}s] constructing pipeline...", flush=True)
    pipeline = ConversationalPipeline(
        db_path=Path("data/processed/mimiciv.duckdb"),
        ontology_dir=Path("ontology/definition"),
        api_key=api_key,
        data_source="bigquery",
        bigquery_project="mimic-485500",
        enable_critic=True,
    )
    print(f"[{time.time() - t0:6.2f}s] pipeline built", flush=True)

    print(f"[{time.time() - t0:6.2f}s] calling ask('average lactate ICU')...", flush=True)
    result = pipeline.ask("What is the average lactate for ICU patients?")
    print(f"[{time.time() - t0:6.2f}s] ask returned", flush=True)
    print()
    print("=" * 60)
    print(f"text_summary: {result.text_summary}")
    print(f"critic_verdict: {result.critic_verdict}")
    print()
    print("Total elapsed:", round(time.time() - t0, 2), "sec")


if __name__ == "__main__":
    main()
