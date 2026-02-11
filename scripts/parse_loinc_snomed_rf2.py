#!/usr/bin/env python3
"""Parse the LOINC–SNOMED CT RF2 package to produce loinc_to_snomed.json.

The RF2 package is available from https://loincsnomed.org/downloads
(one-time free registration).  After downloading and extracting the zip
you will have a directory tree like:

    LOINC_2.78-SNOMEDCT_2024-09-01/
      Full/
        Refset/
          Map/
            der2_iisssccRefset_LOINCExpressionAssociationFull_LOINC-SNOMEDCT_...
      Snapshot/
        Content/
          sct2_Description_Snapshot-en_LOINC-SNOMEDCT_...
        Refset/
          Map/
            der2_iisssccRefset_LOINCExpressionAssociationSnapshot_LOINC-SNOMEDCT_...

This script:
  1. Finds the Expression-Association *Snapshot* refset file (maps LOINC → SNOMED).
  2. Parses tab-delimited data, filters active=1 rows.
  3. Finds the Description *Snapshot* file to resolve SNOMED concept IDs to terms.
  4. Writes data/mappings/loinc_to_snomed.json.

Usage:
    python scripts/parse_loinc_snomed_rf2.py /path/to/extracted/RF2_dir
    python scripts/parse_loinc_snomed_rf2.py /path/to/extracted/RF2_dir --output data/mappings/loinc_to_snomed.json
"""

import argparse
import csv
import json
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# FSN typeId in RF2
FSN_TYPE_ID = "900000000000003001"
# Preferred synonym typeId
SYNONYM_TYPE_ID = "900000000000013009"


def find_file(root: Path, pattern: str) -> Path | None:
    """Glob for a single file matching *pattern* under *root*."""
    matches = sorted(root.rglob(pattern))
    if not matches:
        return None
    return matches[0]


def load_descriptions(desc_path: Path) -> dict[str, str]:
    """Load SNOMED descriptions, returning {conceptId: preferred_term}.

    Prefers the FSN (Fully Specified Name) stripped of its semantic tag,
    falls back to first active synonym.
    """
    logger.info("Loading descriptions from %s ...", desc_path.name)
    concepts: dict[str, dict] = {}  # conceptId -> {"fsn": ..., "syn": ...}

    with open(desc_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row.get("active") != "1":
                continue
            cid = row["conceptId"]
            term = row["term"]
            tid = row["typeId"]
            if cid not in concepts:
                concepts[cid] = {}
            if tid == FSN_TYPE_ID and "fsn" not in concepts[cid]:
                # Strip semantic tag, e.g. "Creatinine (substance)" -> "Creatinine"
                clean = term.rsplit("(", 1)[0].strip() if "(" in term else term
                concepts[cid]["fsn"] = clean
            elif tid == SYNONYM_TYPE_ID and "syn" not in concepts[cid]:
                concepts[cid]["syn"] = term

    result = {}
    for cid, terms in concepts.items():
        result[cid] = terms.get("fsn") or terms.get("syn", "")
    logger.info("  Loaded terms for %d concepts", len(result))
    return result


def load_expression_association(refset_path: Path) -> dict[str, str]:
    """Parse the Expression-Association refset, returning {LOINC_code: SNOMED_SCTID}.

    The refset file has columns including:
      referencedComponentId  — the LOINC SCTID (not the LOINC code itself)
      mapTarget              — the SNOMED expression / concept
    But different RF2 releases use different column layouts.

    We look for the Identifier file to map LOINC SCTIDs -> LOINC codes,
    or fall back to the Simple-Map refset that directly maps LOINC code -> SCTID.
    """
    logger.info("Loading expression association from %s ...", refset_path.name)
    loinc_to_sctid: dict[str, str] = {}

    with open(refset_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        fieldnames = reader.fieldnames or []
        logger.info("  Columns: %s", fieldnames)

        for row in reader:
            if row.get("active") != "1":
                continue
            # The mapTarget or referencedComponentId columns vary by release
            loinc_code = row.get("referencedComponentId", "")
            sctid = row.get("mapTarget", "")
            if not sctid:
                # Try other possible column names
                sctid = row.get("targetComponentId", "")
            if loinc_code and sctid and sctid.isdigit():
                loinc_to_sctid[loinc_code] = sctid

    logger.info("  Parsed %d active LOINC->SNOMED mappings", len(loinc_to_sctid))
    return loinc_to_sctid


def load_simple_map(refset_path: Path) -> dict[str, str]:
    """Parse a SimpleMap refset (mapTarget is the LOINC code, referencedComponentId is the SCTID)."""
    logger.info("Loading simple map from %s ...", refset_path.name)
    loinc_to_sctid: dict[str, str] = {}

    with open(refset_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        fieldnames = reader.fieldnames or []
        logger.info("  Columns: %s", fieldnames)

        for row in reader:
            if row.get("active") != "1":
                continue
            sctid = row.get("referencedComponentId", "")
            loinc_code = row.get("mapTarget", "")
            if loinc_code and "-" in loinc_code and sctid and sctid.isdigit():
                loinc_to_sctid[loinc_code] = sctid

    logger.info("  Parsed %d active LOINC->SNOMED mappings", len(loinc_to_sctid))
    return loinc_to_sctid


def load_identifier_file(ident_path: Path) -> dict[str, str]:
    """Parse sct2_Identifier file: maps alternateIdentifier (LOINC code) -> referencedComponentId (SCTID)."""
    logger.info("Loading identifier file from %s ...", ident_path.name)
    loinc_to_sctid: dict[str, str] = {}

    with open(ident_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row.get("active") != "1":
                continue
            loinc_code = row.get("alternateIdentifier", "")
            sctid = row.get("referencedComponentId", "")
            if loinc_code and "-" in loinc_code and sctid and sctid.isdigit():
                loinc_to_sctid[loinc_code] = sctid

    logger.info("  Parsed %d LOINC code -> SCTID mappings", len(loinc_to_sctid))
    return loinc_to_sctid


def main():
    parser = argparse.ArgumentParser(description="Parse LOINC-SNOMED RF2 to JSON")
    parser.add_argument("rf2_dir", type=Path, help="Path to extracted RF2 directory")
    parser.add_argument("--output", type=Path, default=Path("data/mappings/loinc_to_snomed.json"),
                        help="Output JSON path")
    args = parser.parse_args()

    rf2_dir = args.rf2_dir
    if not rf2_dir.exists():
        logger.error("RF2 directory not found: %s", rf2_dir)
        return

    start = time.time()

    # Strategy 1: Use sct2_Identifier file (LOINC code -> SCTID) + Descriptions
    ident_file = find_file(rf2_dir, "sct2_Identifier_*Snapshot*")
    desc_file = find_file(rf2_dir, "sct2_Description_*Snapshot*")

    # Strategy 2: Use SimpleMap refset (direct LOINC code -> SCTID)
    simple_map_file = find_file(rf2_dir, "der2_sRefset_SimpleMap*Snapshot*")

    # Strategy 3: Use ExpressionAssociation refset
    expr_file = find_file(rf2_dir, "der2_*ExpressionAssociation*Snapshot*")

    logger.info("Found files:")
    logger.info("  Identifier:   %s", ident_file)
    logger.info("  Description:  %s", desc_file)
    logger.info("  SimpleMap:    %s", simple_map_file)
    logger.info("  Expression:   %s", expr_file)

    # Load SNOMED descriptions for term resolution
    terms: dict[str, str] = {}
    if desc_file:
        terms = load_descriptions(desc_file)

    # Build LOINC -> SCTID mapping using best available strategy
    loinc_to_sctid: dict[str, str] = {}

    if ident_file:
        loinc_to_sctid = load_identifier_file(ident_file)
        logger.info("Strategy 1 (Identifier file): %d mappings", len(loinc_to_sctid))

    if not loinc_to_sctid and simple_map_file:
        loinc_to_sctid = load_simple_map(simple_map_file)
        logger.info("Strategy 2 (SimpleMap): %d mappings", len(loinc_to_sctid))

    if not loinc_to_sctid and expr_file:
        loinc_to_sctid = load_expression_association(expr_file)
        logger.info("Strategy 3 (ExpressionAssociation): %d mappings", len(loinc_to_sctid))

    if not loinc_to_sctid:
        logger.error("No LOINC->SNOMED mappings could be extracted. Check RF2 directory structure.")
        return

    # Build final JSON: {loinc_code: {snomed_code, snomed_term}}
    mapping: dict[str, dict] = {}
    for loinc_code, sctid in sorted(loinc_to_sctid.items()):
        term = terms.get(sctid, "")
        mapping[loinc_code] = {
            "snomed_code": sctid,
            "snomed_term": term,
        }

    output = {
        "_metadata": {
            "source": "LOINC-SNOMED CT Cooperative Package (RF2)",
            "generated": time.strftime("%Y-%m-%d"),
            "total_entries": len(mapping),
            "with_terms": sum(1 for v in mapping.values() if v["snomed_term"]),
        },
        **mapping,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    elapsed = time.time() - start
    logger.info("Wrote %d LOINC->SNOMED mappings to %s (%.1fs)", len(mapping), args.output, elapsed)


if __name__ == "__main__":
    main()
