#!/usr/bin/env python3
"""Generate loinc_to_snomed.json via UMLS REST API crosswalk.

Queries NLM UMLS REST API to crosswalk each LOINC code found in
labitem_to_snomed.json to its SNOMED-CT equivalent.

Usage:
    python scripts/generate_loinc_snomed_map.py --umls-api-key YOUR_KEY

Prerequisites:
    - Free UMLS API key from https://uts.nlm.nih.gov/uts/profile
    - data/mappings/labitem_to_snomed.json must exist (for LOINC codes)
"""

import argparse
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

UMLS_BASE = "https://uts-ws.nlm.nih.gov/rest"
SCTID_RE = re.compile(r"^\d{5,18}$")


def _crosswalk_loinc(session: requests.Session, loinc_code: str, api_key: str) -> dict | None:
    """Query UMLS crosswalk for a LOINC code, return {snomed_code, snomed_term} or None."""
    url = f"{UMLS_BASE}/crosswalk/current/source/LNC/{loinc_code}"
    params = {"targetSource": "SNOMEDCT_US", "apiKey": api_key}
    for attempt in range(3):
        try:
            resp = session.get(url, params=params, timeout=30)
            if resp.status_code == 429:
                time.sleep(2 ** (attempt + 1))
                continue
            if resp.status_code in (404, 400):
                return None
            resp.raise_for_status()
            results = resp.json().get("result", [])
            if not results:
                return None
            non_obsolete = [r for r in results if not r.get("obsolete", False)]
            best = non_obsolete[0] if non_obsolete else results[0]
            sctid = best.get("ui", "")
            if SCTID_RE.match(sctid):
                return {"snomed_code": sctid, "snomed_term": best.get("name", "")}
            return None
        except requests.exceptions.RequestException:
            time.sleep(2 ** attempt)
    return None


def main():
    parser = argparse.ArgumentParser(description="Generate LOINC->SNOMED mapping via UMLS")
    parser.add_argument("--umls-api-key", required=True, help="UMLS API key")
    parser.add_argument("--labitem-file", type=Path,
                        default=Path("data/mappings/labitem_to_snomed.json"),
                        help="labitem_to_snomed.json to read LOINC codes from")
    parser.add_argument("--output", type=Path,
                        default=Path("data/mappings/loinc_to_snomed.json"),
                        help="Output path")
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()

    # Collect unique LOINC codes
    with open(args.labitem_file) as f:
        labitem_data = json.load(f)
    labitem_data.pop("_metadata", None)

    loinc_codes = sorted({
        v["loinc"] for v in labitem_data.values()
        if v.get("loinc") and re.match(r"\d+-\d+", v["loinc"])
    })
    logger.info("Found %d unique LOINC codes to look up", len(loinc_codes))

    session = requests.Session()
    results = {}
    total = len(loinc_codes)
    done = 0
    start = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(_crosswalk_loinc, session, code, args.umls_api_key): code
            for code in loinc_codes
        }
        for future in as_completed(futures):
            code = futures[future]
            done += 1
            try:
                result = future.result()
                if result:
                    results[code] = result
            except Exception as e:
                logger.debug("Error for %s: %s", code, e)
            if done % 100 == 0 or done == total:
                elapsed = time.time() - start
                rate = done / elapsed if elapsed > 0 else 0
                logger.info("  %d/%d (%d mapped, %.1f/sec)", done, total, len(results), rate)

    output = {
        "_metadata": {
            "source": "NLM UMLS REST API crosswalk (LNC -> SNOMEDCT_US)",
            "generated": time.strftime("%Y-%m-%d"),
            "total_entries": len(results),
        },
        **dict(sorted(results.items())),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    logger.info("Wrote %d LOINC->SNOMED mappings to %s", len(results), args.output)


if __name__ == "__main__":
    main()
