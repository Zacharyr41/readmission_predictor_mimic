#!/usr/bin/env python3
"""Generate comprehensive SNOMED-CT mapping files from authoritative sources.

Data sources:
- ICD-10-CM -> SNOMED: NLM UMLS REST API crosswalk (requires API key)
- Lab items -> SNOMED: srdc/mimic-iv-to-fhir labitems-to-loinc.csv + UMLS LOINC->SNOMED crosswalk
- Organisms -> SNOMED: srdc/mimic-iv-to-fhir org-itemids-to-snomed.csv (direct)
- Drugs -> SNOMED: NLM RxNorm REST API + UMLS crosswalk
- Chart items -> SNOMED: Manual curation (standard ICU vital signs)
- Comorbidities -> SNOMED: Manual curation (Charlson/Elixhauser categories)

Usage:
    python scripts/generate_snomed_mappings.py --umls-api-key YOUR_KEY
    python scripts/generate_snomed_mappings.py --umls-api-key YOUR_KEY --comprehensive
"""

import argparse
import csv
import io
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

UMLS_BASE = "https://uts-ws.nlm.nih.gov/rest"
RXNORM_BASE = "https://rxnav.nlm.nih.gov/REST"
SRDC_BASE = (
    "https://raw.githubusercontent.com/srdc/mimic-iv-to-fhir/main"
    "/R4/terminology-systems/MIMICTerminologyService"
)

# Common MIMIC-IV drug names (top ~200 by frequency)
COMMON_DRUGS = [
    "Acetaminophen", "Albumin 25%", "Albumin 5%", "Albuterol 0.083% Neb Soln",
    "Amlodipine", "Amiodarone", "Aspirin", "Atorvastatin", "Atropine Sulfate",
    "Azithromycin", "Bisacodyl", "Calcium Gluconate", "Cefazolin", "Cefepime",
    "Ceftriaxone", "Chlorhexidine Gluconate", "Ciprofloxacin", "Clopidogrel",
    "Dexamethasone", "Dexmedetomidine", "Digoxin", "Diltiazem",
    "Diphenhydramine", "Docusate Sodium", "Dobutamine", "Dopamine",
    "Enoxaparin", "Epinephrine", "Erythromycin", "Esmolol",
    "Famotidine", "Fentanyl Citrate", "Fluconazole", "Furosemide",
    "Gabapentin", "Gentamicin", "Glucagon", "Haloperidol",
    "Heparin", "Hydralazine", "Hydrochlorothiazide", "Hydromorphone",
    "Hydroxychloroquine", "Ibuprofen", "Insulin", "Ipratropium Bromide",
    "Isoproterenol", "Ketorolac", "Labetalol", "Lactulose",
    "Levofloxacin", "Levothyroxine", "Lidocaine", "Linezolid",
    "Lisinopril", "Lorazepam", "Losartan", "Magnesium Sulfate",
    "Mannitol", "Meropenem", "Metformin", "Methylprednisolone",
    "Metoclopramide", "Metoprolol Tartrate", "Metronidazole", "Micafungin",
    "Midazolam", "Milrinone", "Morphine Sulfate", "Nafcillin",
    "Naloxone", "Nicardipine", "Nitroglycerin", "Nitroprusside",
    "Norepinephrine", "Nystatin", "Ofloxacin", "Omeprazole",
    "Ondansetron", "Oseltamivir", "Oxacillin", "Oxycodone",
    "Pantoprazole", "Phenylephrine", "Phenytoin", "Piperacillin-Tazobactam",
    "Potassium Chloride", "Potassium Phosphate", "Prednisone", "Propofol",
    "Propranolol", "Quetiapine", "Ranitidine", "Rifampin",
    "Rocuronium", "Sennosides", "Simvastatin", "Sodium Bicarbonate",
    "Sodium Chloride 0.9%", "Sodium Phosphate", "Spironolactone", "Succinylcholine",
    "Tacrolimus", "Theophylline", "Ticagrelor", "Tobramycin",
    "Tramadol", "Trimethoprim-Sulfamethoxazole", "Valproic Acid", "Vancomycin",
    "Vasopressin", "Vecuronium", "Verapamil", "Warfarin",
    "Alteplase", "Ampicillin", "Ampicillin-Sulbactam", "Argatroban",
    "Bivalirudin", "Bumetanide", "Carvedilol", "Ceftazidime",
    "Cisatracurium", "Clindamycin", "Colistin", "Daptomycin",
    "Dexamethasone", "Diazepam", "Enalapril", "Epoprostenol",
    "Etomidate", "Fluticasone", "Fondaparinux", "Hydrocortisone",
    "Imipenem-Cilastatin", "Ketamine", "Levetiracetam", "Levofloxacin",
    "Methadone", "Nimodipine", "Nitrofurantoin", "Octreotide",
    "Olanzapine", "Oxcarbazepine", "Penicillin G", "Phenobarbital",
    "Polymyxin B", "Pregabalin", "Remifentanil", "Rivaroxaban",
    "Sevoflurane", "Tigecycline", "Torsemide", "Voriconazole",
    "Acetazolamide", "Aminophylline", "Amoxicillin", "Aztreonam",
    "Betamethasone", "Brivaracetam", "Caspofungin", "Cefotaxime",
    "Cephalexin", "Dantrolene", "Ertapenem", "Flecainide",
    "Flumazenil", "Foscarnet", "Ganciclovir", "Inamrinone",
    "Isavuconazonium", "Lacosamide", "Methylene Blue", "Mycophenolate",
    "Neostigmine", "Pentamidine", "Phenoxybenzamine", "Phentolamine",
    "Phytonadione", "Protamine", "Pyridostigmine", "Sodium Polystyrene Sulfonate",
    "Sugammadex", "Thrombin", "Tirofiban", "Zonisamide",
]


def _umls_crosswalk(session: requests.Session, source: str, code: str, api_key: str):
    """Query UMLS crosswalk, return first non-obsolete SNOMED result or first obsolete."""
    url = f"{UMLS_BASE}/crosswalk/current/source/{source}/{code}"
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
            # Prefer non-obsolete
            non_obsolete = [r for r in results if not r.get("obsolete", False)]
            best = non_obsolete[0] if non_obsolete else results[0]
            return {"snomed_code": best["ui"], "snomed_term": best["name"]}
        except requests.exceptions.RequestException:
            time.sleep(2 ** attempt)
    return None


def _rxnorm_lookup(session: requests.Session, drug_name: str):
    """Look up RxCUI for a drug name via RxNorm API."""
    url = f"{RXNORM_BASE}/rxcui.json"
    for search_type in [0, 2]:  # exact then approximate
        try:
            resp = session.get(url, params={"name": drug_name, "search": search_type}, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                ids = data.get("idGroup", {}).get("rxnormId", [])
                if ids:
                    return ids[0]
        except requests.exceptions.RequestException:
            pass
    return None


def _batch_crosswalk(api_key: str, source: str, codes: list[str], desc: str, workers: int = 15):
    """Run crosswalk queries concurrently, return {code: {snomed_code, snomed_term}}."""
    results = {}
    session = requests.Session()
    total = len(codes)
    done = 0
    start = time.time()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_umls_crosswalk, session, source, code, api_key): code
            for code in codes
        }
        for future in as_completed(futures):
            code = futures[future]
            done += 1
            try:
                result = future.result()
                if result:
                    results[code] = result
            except Exception as e:
                logger.debug(f"Error for {code}: {e}")
            if done % 200 == 0 or done == total:
                elapsed = time.time() - start
                rate = done / elapsed if elapsed > 0 else 0
                logger.info(f"  {desc}: {done}/{total} ({len(results)} mapped, {rate:.1f}/sec)")

    return results


# ==================== Generators ====================


def generate_icd_to_snomed(api_key: str, output_path: Path, comprehensive: bool = False):
    """Generate ICD-10-CM to SNOMED mapping via UMLS crosswalk."""
    logger.info("Generating ICD-10-CM -> SNOMED mapping...")

    # Phase 1: All 3-character category codes
    codes_3char = []
    for letter in "ABCDEFGHIJKLMNOPQRSTVWXYZ":  # Skip U
        for num in range(100):
            codes_3char.append(f"{letter}{num:02d}")

    results = _batch_crosswalk(api_key, "ICD10CM", codes_3char, "ICD 3-char")
    logger.info(f"  Phase 1: {len(results)} 3-char codes mapped")

    if comprehensive:
        # Phase 2: 4-char subcodes (.0-.9) for ALL chapters
        codes_4char = []
        for letter in "ABCDEFGHIJKLMNOPQRSTVWXYZ":
            for num in range(100):
                base = f"{letter}{num:02d}"
                for digit in range(10):
                    code = f"{base}.{digit}"
                    codes_4char.append(code)

        results_4 = _batch_crosswalk(api_key, "ICD10CM", codes_4char, "ICD 4-char")
        logger.info(f"  Phase 2: {len(results_4)} 4-char codes mapped")

        # Phase 3: 5-char subcodes for key chapters (I, E, G, J, K, N, D, R)
        codes_5char = []
        for letter in "IEGJKND":
            for num in range(100):
                base = f"{letter}{num:02d}"
                for d1 in range(10):
                    for d2 in range(10):
                        codes_5char.append(f"{base}.{d1}{d2}")

        results_5 = _batch_crosswalk(api_key, "ICD10CM", codes_5char, "ICD 5-char")
        logger.info(f"  Phase 3: {len(results_5)} 5-char codes mapped")

        results.update(results_4)
        results.update(results_5)
    else:
        # Phase 2 (basic): 4-char subcodes for key chapters only
        codes_4char = []
        for letter in "IEGIJKNDRJABCST":
            for num in range(100):
                base = f"{letter}{num:02d}"
                for digit in range(10):
                    codes_4char.append(f"{base}.{digit}")

        results_4 = _batch_crosswalk(api_key, "ICD10CM", codes_4char, "ICD 4-char (key)")
        logger.info(f"  Phase 2: {len(results_4)} 4-char codes mapped")
        results.update(results_4)

    # Convert dotted format to MIMIC undotted format
    mapping = {}
    for code, data in results.items():
        mimic_code = code.replace(".", "")
        mapping[mimic_code] = data

    output = {
        "_metadata": {
            "source": "NLM UMLS REST API crosswalk (ICD10CM -> SNOMEDCT_US)",
            "generated": time.strftime("%Y-%m-%d"),
            "total_codes": len(mapping),
            "comprehensive": comprehensive,
        },
        **dict(sorted(mapping.items())),
    }

    output_path.write_text(json.dumps(output, indent=2))
    logger.info(f"  Wrote {len(mapping)} ICD->SNOMED mappings to {output_path}")


def _umls_term_to_snomed(session: requests.Session, term: str, api_key: str):
    """Search UMLS for a clinical term and return the best SNOMED concept."""
    for search_term in [f"{term} measurement", term]:
        try:
            resp = session.get(f"{UMLS_BASE}/search/current", params={
                "string": search_term, "sab": "SNOMEDCT_US",
                "searchType": "words", "apiKey": api_key,
            }, timeout=15)
            if resp.status_code != 200:
                continue
            results = resp.json().get("result", {}).get("results", [])
            if not results or results[0].get("ui") == "NONE":
                continue
            cui = results[0]["ui"]
            # Get SNOMEDCT_US atoms for this CUI
            resp2 = session.get(f"{UMLS_BASE}/content/current/CUI/{cui}/atoms", params={
                "sab": "SNOMEDCT_US", "apiKey": api_key, "pageSize": 5,
            }, timeout=15)
            if resp2.status_code != 200:
                continue
            atoms = resp2.json().get("result", [])
            for atom in atoms:
                if atom.get("rootSource") == "SNOMEDCT_US":
                    # Extract SCTID from sourceConcept URI
                    sc_uri = atom.get("sourceConcept", "")
                    if "/" in sc_uri:
                        sctid = sc_uri.rstrip("/").rsplit("/", 1)[-1]
                    else:
                        sctid = sc_uri
                    return {"snomed_code": sctid, "snomed_term": atom.get("name", "")}
            # No SNOMEDCT_US atom found — do NOT store the CUI as a SNOMED code
            return None
        except Exception:
            pass
    return None


def _load_loinc_snomed_map(mappings_dir: Path) -> dict:
    """Load loinc_to_snomed.json if available, return {loinc_code: {snomed_code, snomed_term}}."""
    path = mappings_dir / "loinc_to_snomed.json"
    if not path.exists():
        logger.warning("  loinc_to_snomed.json not found at %s — LOINC-first lookup disabled", path)
        return {}
    with open(path) as f:
        data = json.load(f)
    data.pop("_metadata", None)
    logger.info("  Loaded %d LOINC->SNOMED entries from %s", len(data), path.name)
    return data


def _is_valid_sctid(code: str) -> bool:
    """Return True if *code* looks like a valid SNOMED SCTID (5-18 digits)."""
    import re
    return bool(re.fullmatch(r"\d{5,18}", str(code)))


def generate_labitem_to_snomed(api_key: str, output_path: Path):
    """Generate lab item -> SNOMED mapping.

    Strategy (LOINC-first):
      1. Download item→LOINC from srdc GitHub.
      2. Look up each LOINC code in loinc_to_snomed.json (from RF2 package).
      3. Fall back to UMLS term search for any remaining unmapped items.
    """
    logger.info("Generating lab item -> SNOMED mapping...")

    # Download labitems-to-loinc.csv from srdc
    resp = requests.get(f"{SRDC_BASE}/labitems-to-loinc.csv", timeout=30)
    resp.raise_for_status()
    reader = csv.DictReader(io.StringIO(resp.text))

    # Parse itemid -> LOINC mapping
    item_loinc = {}
    for row in reader:
        itemid = row["source_code"]
        loinc = row["target_code"]
        label = row["source_display"]
        if loinc:
            item_loinc[itemid] = {"loinc": loinc, "label": label}

    logger.info(f"  Downloaded {len(item_loinc)} lab item -> LOINC mappings from srdc")

    # --- Phase 1: LOINC-first lookup via loinc_to_snomed.json ---
    loinc_snomed = _load_loinc_snomed_map(output_path.parent)

    mapping = {}
    unmapped_labels = {}  # label -> [itemid, ...] for UMLS fallback

    for itemid, info in item_loinc.items():
        loinc = info["loinc"]
        label = info["label"]
        entry = {"loinc": loinc, "label": label}

        loinc_entry = loinc_snomed.get(loinc)
        if loinc_entry and _is_valid_sctid(loinc_entry.get("snomed_code", "")):
            entry["snomed_code"] = loinc_entry["snomed_code"]
            entry["snomed_term"] = loinc_entry.get("snomed_term", "")
        else:
            # Queue for UMLS fallback
            if label not in unmapped_labels:
                unmapped_labels[label] = []
            unmapped_labels[label].append(itemid)

        mapping[itemid] = entry

    loinc_mapped = sum(1 for v in mapping.values() if "snomed_code" in v)
    logger.info(f"  Phase 1 (LOINC-first): {loinc_mapped}/{len(mapping)} mapped via LOINC->SNOMED")

    # --- Phase 2: UMLS term search fallback for remaining items ---
    if unmapped_labels and api_key:
        logger.info(f"  Phase 2: UMLS term search for {len(unmapped_labels)} unique unmapped labels...")
        session = requests.Session()
        label_to_snomed = {}
        for i, label in enumerate(unmapped_labels):
            result = _umls_term_to_snomed(session, label, api_key)
            if result and _is_valid_sctid(result.get("snomed_code", "")):
                label_to_snomed[label] = result
            if (i + 1) % 100 == 0:
                logger.info(f"    UMLS lookup: {i+1}/{len(unmapped_labels)} ({len(label_to_snomed)} mapped)")

        logger.info(f"  Phase 2: {len(label_to_snomed)}/{len(unmapped_labels)} labels mapped via UMLS")

        # Apply UMLS results
        for label, snomed in label_to_snomed.items():
            for itemid in unmapped_labels[label]:
                mapping[itemid]["snomed_code"] = snomed["snomed_code"]
                mapping[itemid]["snomed_term"] = snomed["snomed_term"]

    mapped_count = sum(1 for v in mapping.values() if "snomed_code" in v)
    output = {
        "_metadata": {
            "source": "srdc/mimic-iv-to-fhir labitems-to-loinc.csv + LOINC-SNOMED RF2 + UMLS fallback",
            "generated": time.strftime("%Y-%m-%d"),
            "total_items": len(mapping),
            "snomed_mapped": mapped_count,
        },
        **dict(sorted(mapping.items(), key=lambda x: int(x[0]))),
    }

    output_path.write_text(json.dumps(output, indent=2))
    logger.info(f"  Wrote {len(mapping)} lab items ({mapped_count} with SNOMED) to {output_path}")


def generate_organism_to_snomed(output_path: Path):
    """Generate organism -> SNOMED mapping from srdc GitHub (already has SNOMED)."""
    logger.info("Generating organism -> SNOMED mapping...")

    resp = requests.get(f"{SRDC_BASE}/org-itemids-to-snomed.csv", timeout=30)
    resp.raise_for_status()
    reader = csv.DictReader(io.StringIO(resp.text))

    mapping = {}
    for row in reader:
        org_name = row["source_display"].strip()
        snomed_code = row["target_code"].strip()
        snomed_term = row["target_display"].strip()
        if org_name and snomed_code:
            # Key by uppercase organism name (MIMIC convention)
            mapping[org_name.upper()] = {
                "snomed_code": snomed_code,
                "snomed_term": snomed_term,
            }

    output = {
        "_metadata": {
            "source": "srdc/mimic-iv-to-fhir org-itemids-to-snomed.csv",
            "generated": time.strftime("%Y-%m-%d"),
            "total_organisms": len(mapping),
        },
        **dict(sorted(mapping.items())),
    }

    output_path.write_text(json.dumps(output, indent=2))
    logger.info(f"  Wrote {len(mapping)} organism -> SNOMED mappings to {output_path}")


def generate_drug_to_snomed(api_key: str, output_path: Path):
    """Generate drug -> SNOMED mapping via RxNorm + UMLS crosswalk."""
    logger.info("Generating drug -> SNOMED mapping...")

    session = requests.Session()
    # Step 1: Drug name -> RxCUI via RxNorm
    drug_rxcui = {}
    for i, drug in enumerate(COMMON_DRUGS):
        rxcui = _rxnorm_lookup(session, drug)
        if rxcui:
            drug_rxcui[drug] = rxcui
        if (i + 1) % 50 == 0:
            logger.info(f"  RxNorm lookup: {i+1}/{len(COMMON_DRUGS)} ({len(drug_rxcui)} found)")

    logger.info(f"  Found RxCUI for {len(drug_rxcui)}/{len(COMMON_DRUGS)} drugs")

    # Step 2: RxCUI -> SNOMED via UMLS crosswalk
    unique_rxcuis = list(set(drug_rxcui.values()))
    rxcui_to_snomed = _batch_crosswalk(api_key, "RXNORM", unique_rxcuis, "RxNorm->SNOMED")

    # Combine
    mapping = {}
    for drug, rxcui in drug_rxcui.items():
        key = drug.lower()
        entry = {"rxcui": rxcui, "drug_name": drug}
        snomed = rxcui_to_snomed.get(rxcui)
        if snomed:
            entry["snomed_code"] = snomed["snomed_code"]
            entry["snomed_term"] = snomed["snomed_term"]
        mapping[key] = entry

    mapped_count = sum(1 for v in mapping.values() if "snomed_code" in v)
    output = {
        "_metadata": {
            "source": "NLM RxNorm REST API + UMLS crosswalk (RXNORM -> SNOMEDCT_US)",
            "generated": time.strftime("%Y-%m-%d"),
            "total_drugs": len(mapping),
            "snomed_mapped": mapped_count,
        },
        **dict(sorted(mapping.items())),
    }

    output_path.write_text(json.dumps(output, indent=2))
    logger.info(f"  Wrote {len(mapping)} drug mappings ({mapped_count} with SNOMED) to {output_path}")


def generate_chartitem_to_snomed(output_path: Path):
    """Generate chart item (vital sign) -> SNOMED mapping. Manually curated."""
    logger.info("Generating chart item -> SNOMED mapping...")

    mapping = {
        # Heart Rate
        "220045": {"snomed_code": "364075005", "snomed_term": "Heart rate", "loinc": "8867-4"},
        "220046": {"snomed_code": "364075005", "snomed_term": "Heart rate alarm - high", "loinc": "8867-4"},
        "220047": {"snomed_code": "364075005", "snomed_term": "Heart rate alarm - low", "loinc": "8867-4"},
        # Arterial Blood Pressure
        "220050": {"snomed_code": "271649006", "snomed_term": "Systolic blood pressure", "loinc": "8480-6"},
        "220051": {"snomed_code": "271650006", "snomed_term": "Diastolic blood pressure", "loinc": "8462-4"},
        "220052": {"snomed_code": "6797001", "snomed_term": "Mean arterial pressure", "loinc": "8478-0"},
        # Non-invasive Blood Pressure
        "220179": {"snomed_code": "271649006", "snomed_term": "Non-invasive systolic blood pressure", "loinc": "8480-6"},
        "220180": {"snomed_code": "271650006", "snomed_term": "Non-invasive diastolic blood pressure", "loinc": "8462-4"},
        "220181": {"snomed_code": "6797001", "snomed_term": "Non-invasive mean blood pressure", "loinc": "8478-0"},
        # Respiratory
        "220210": {"snomed_code": "86290005", "snomed_term": "Respiratory rate", "loinc": "9279-1"},
        "224422": {"snomed_code": "86290005", "snomed_term": "Respiratory rate (total)", "loinc": "9279-1"},
        "224689": {"snomed_code": "86290005", "snomed_term": "Respiratory rate (spontaneous)", "loinc": "9279-1"},
        "227243": {"snomed_code": "86290005", "snomed_term": "Respiratory rate (set)", "loinc": "9279-1"},
        # Oxygen Saturation
        "220277": {"snomed_code": "431314004", "snomed_term": "Peripheral oxygen saturation", "loinc": "59408-5"},
        "220227": {"snomed_code": "442476006", "snomed_term": "Arterial oxygen saturation", "loinc": "2708-6"},
        # Central Venous Pressure
        "220074": {"snomed_code": "71420008", "snomed_term": "Central venous pressure", "loinc": "60985-9"},
        # Temperature
        "223761": {"snomed_code": "276885007", "snomed_term": "Core body temperature", "loinc": "8310-5"},
        "223762": {"snomed_code": "276885007", "snomed_term": "Core body temperature", "loinc": "8310-5"},
        "226329": {"snomed_code": "276885007", "snomed_term": "Body temperature", "loinc": "8310-5"},
        "226512": {"snomed_code": "276885007", "snomed_term": "Core body temperature", "loinc": "8310-5"},
        "223835": {"snomed_code": "276885007", "snomed_term": "Skin temperature", "loinc": ""},
        "224027": {"snomed_code": "276885007", "snomed_term": "Blood temperature", "loinc": ""},
        # Weight/Height
        "224639": {"snomed_code": "27113001", "snomed_term": "Body weight", "loinc": "29463-7"},
        "226512": {"snomed_code": "276885007", "snomed_term": "Core body temperature", "loinc": "8310-5"},
        "226730": {"snomed_code": "50373000", "snomed_term": "Body height", "loinc": "8302-2"},
        "226531": {"snomed_code": "27113001", "snomed_term": "Body weight", "loinc": "29463-7"},
        "226846": {"snomed_code": "27113001", "snomed_term": "Daily weight", "loinc": "29463-7"},
        # Glasgow Coma Scale
        "220739": {"snomed_code": "248263006", "snomed_term": "GCS - Eye opening", "loinc": "9267-6"},
        "223900": {"snomed_code": "248263006", "snomed_term": "GCS - Verbal response", "loinc": "9270-0"},
        "223901": {"snomed_code": "248263006", "snomed_term": "GCS - Motor response", "loinc": "9268-4"},
        # RASS
        "228096": {"snomed_code": "457441000124102", "snomed_term": "Richmond Agitation-Sedation Scale score", "loinc": "54635-8"},
        # Cardiac Output / Hemodynamics
        "220088": {"snomed_code": "82799009", "snomed_term": "Cardiac output", "loinc": ""},
        "220227": {"snomed_code": "442476006", "snomed_term": "Arterial oxygen saturation", "loinc": "2708-6"},
        # Ventilator settings
        "220339": {"snomed_code": "250854009", "snomed_term": "FiO2", "loinc": "3150-0"},
        "224700": {"snomed_code": "250854009", "snomed_term": "FiO2", "loinc": "3150-0"},
        "226873": {"snomed_code": "250854009", "snomed_term": "Inspired oxygen concentration", "loinc": "3150-0"},
        "220339": {"snomed_code": "250854009", "snomed_term": "FiO2 setting", "loinc": "3150-0"},
        # Tidal Volume
        "224685": {"snomed_code": "250816004", "snomed_term": "Tidal volume", "loinc": ""},
        "224686": {"snomed_code": "250816004", "snomed_term": "Tidal volume (observed)", "loinc": ""},
        "224421": {"snomed_code": "250816004", "snomed_term": "Tidal volume (set)", "loinc": ""},
        # PEEP
        "220339": {"snomed_code": "250854009", "snomed_term": "FiO2", "loinc": ""},
        "224700": {"snomed_code": "250854009", "snomed_term": "FiO2", "loinc": ""},
        # Urine output
        "226559": {"snomed_code": "404231006", "snomed_term": "Urine output", "loinc": ""},
        "226560": {"snomed_code": "404231006", "snomed_term": "Urine output (Foley)", "loinc": ""},
        "226561": {"snomed_code": "404231006", "snomed_term": "Urine output (condom catheter)", "loinc": ""},
        "226584": {"snomed_code": "404231006", "snomed_term": "Urine output (void)", "loinc": ""},
        # ICP
        "220765": {"snomed_code": "250844005", "snomed_term": "Intracranial pressure", "loinc": "60956-0"},
        # Pain scale
        "223791": {"snomed_code": "225908003", "snomed_term": "Pain score", "loinc": ""},
        "228300": {"snomed_code": "225908003", "snomed_term": "Pain level", "loinc": ""},
    }

    output = {
        "_metadata": {
            "source": "Manual curation of MIMIC-IV d_items vital signs to SNOMED-CT",
            "generated": time.strftime("%Y-%m-%d"),
            "total_items": len(mapping),
        },
        **dict(sorted(mapping.items(), key=lambda x: int(x[0]))),
    }

    output_path.write_text(json.dumps(output, indent=2))
    logger.info(f"  Wrote {len(mapping)} chart item -> SNOMED mappings to {output_path}")


def generate_comorbidity_to_snomed(output_path: Path):
    """Generate comorbidity -> SNOMED mapping. Charlson/Elixhauser categories."""
    logger.info("Generating comorbidity -> SNOMED mapping...")

    mapping = {
        # Charlson Comorbidity Index categories
        "myocardial_infarction": {"snomed_code": "22298006", "snomed_term": "Myocardial infarction"},
        "congestive_heart_failure": {"snomed_code": "42343007", "snomed_term": "Congestive heart failure"},
        "peripheral_vascular_disease": {"snomed_code": "400047006", "snomed_term": "Peripheral vascular disease"},
        "cerebrovascular_disease": {"snomed_code": "62914000", "snomed_term": "Cerebrovascular disease"},
        "dementia": {"snomed_code": "52448006", "snomed_term": "Dementia"},
        "chronic_pulmonary_disease": {"snomed_code": "13645005", "snomed_term": "Chronic obstructive lung disease"},
        "connective_tissue_disease": {"snomed_code": "105969002", "snomed_term": "Connective tissue disease"},
        "peptic_ulcer_disease": {"snomed_code": "13200003", "snomed_term": "Peptic ulcer"},
        "mild_liver_disease": {"snomed_code": "235856003", "snomed_term": "Chronic liver disease"},
        "diabetes_without_complications": {"snomed_code": "73211009", "snomed_term": "Diabetes mellitus"},
        "diabetes_with_complications": {"snomed_code": "73211009", "snomed_term": "Diabetes mellitus with complication"},
        "paraplegia": {"snomed_code": "60389000", "snomed_term": "Paraplegia"},
        "renal_disease": {"snomed_code": "709044004", "snomed_term": "Chronic kidney disease"},
        "cancer": {"snomed_code": "363346000", "snomed_term": "Malignant neoplastic disease"},
        "moderate_severe_liver_disease": {"snomed_code": "19943007", "snomed_term": "Cirrhosis of liver"},
        "metastatic_cancer": {"snomed_code": "128462008", "snomed_term": "Metastatic malignant neoplasm"},
        "aids": {"snomed_code": "62479008", "snomed_term": "Acquired immune deficiency syndrome"},
        # Elixhauser additional categories
        "hypertension_uncomplicated": {"snomed_code": "38341003", "snomed_term": "Essential hypertension"},
        "hypertension_complicated": {"snomed_code": "38341003", "snomed_term": "Hypertensive disorder"},
        "cardiac_arrhythmia": {"snomed_code": "698247007", "snomed_term": "Cardiac arrhythmia"},
        "valvular_disease": {"snomed_code": "368009", "snomed_term": "Heart valve disorder"},
        "pulmonary_circulation_disease": {"snomed_code": "233940005", "snomed_term": "Pulmonary circulation disorder"},
        "obesity": {"snomed_code": "414916001", "snomed_term": "Obesity"},
        "weight_loss": {"snomed_code": "89362005", "snomed_term": "Weight loss"},
        "fluid_electrolyte_disorder": {"snomed_code": "237840007", "snomed_term": "Disorder of fluid and electrolyte balance"},
        "blood_loss_anemia": {"snomed_code": "267469009", "snomed_term": "Acute blood loss anemia"},
        "deficiency_anemia": {"snomed_code": "271737000", "snomed_term": "Anemia"},
        "alcohol_abuse": {"snomed_code": "7200002", "snomed_term": "Alcoholism"},
        "drug_abuse": {"snomed_code": "66214007", "snomed_term": "Substance abuse"},
        "psychosis": {"snomed_code": "69322001", "snomed_term": "Psychotic disorder"},
        "depression": {"snomed_code": "35489007", "snomed_term": "Depressive disorder"},
        "coagulopathy": {"snomed_code": "234466008", "snomed_term": "Coagulation disorder"},
        "hypothyroidism": {"snomed_code": "40930008", "snomed_term": "Hypothyroidism"},
        "liver_disease": {"snomed_code": "235856003", "snomed_term": "Chronic liver disease"},
        # Common simple names used in code
        "diabetes": {"snomed_code": "73211009", "snomed_term": "Diabetes mellitus"},
        "hypertension": {"snomed_code": "38341003", "snomed_term": "Essential hypertension"},
        "atrial_fibrillation": {"snomed_code": "49436004", "snomed_term": "Atrial fibrillation"},
        "copd": {"snomed_code": "13645005", "snomed_term": "Chronic obstructive lung disease"},
        "ckd": {"snomed_code": "709044004", "snomed_term": "Chronic kidney disease"},
        "chf": {"snomed_code": "42343007", "snomed_term": "Congestive heart failure"},
        "stroke": {"snomed_code": "230690007", "snomed_term": "Cerebrovascular accident"},
        "epilepsy": {"snomed_code": "84757009", "snomed_term": "Epilepsy"},
    }

    output = {
        "_metadata": {
            "source": "Manual curation: Charlson/Elixhauser comorbidity categories to SNOMED-CT",
            "generated": time.strftime("%Y-%m-%d"),
            "total_comorbidities": len(mapping),
        },
        **dict(sorted(mapping.items())),
    }

    output_path.write_text(json.dumps(output, indent=2))
    logger.info(f"  Wrote {len(mapping)} comorbidity -> SNOMED mappings to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate SNOMED-CT mapping files")
    parser.add_argument("--umls-api-key", required=True, help="UMLS API key (free from uts.nlm.nih.gov)")
    parser.add_argument("--output-dir", default="data/mappings", help="Output directory for JSON files")
    parser.add_argument("--comprehensive", action="store_true", help="Generate 5-char ICD subcodes (slower, more complete)")
    parser.add_argument("--only", choices=["icd", "lab", "chart", "drug", "organism", "comorbidity"],
                       help="Generate only a specific mapping")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generators = {
        "organism": lambda: generate_organism_to_snomed(output_dir / "organism_to_snomed.json"),
        "chart": lambda: generate_chartitem_to_snomed(output_dir / "chartitem_to_snomed.json"),
        "comorbidity": lambda: generate_comorbidity_to_snomed(output_dir / "comorbidity_to_snomed.json"),
        "lab": lambda: generate_labitem_to_snomed(args.umls_api_key, output_dir / "labitem_to_snomed.json"),
        "drug": lambda: generate_drug_to_snomed(args.umls_api_key, output_dir / "drug_to_snomed.json"),
        "icd": lambda: generate_icd_to_snomed(args.umls_api_key, output_dir / "icd10cm_to_snomed.json", args.comprehensive),
    }

    if args.only:
        generators[args.only]()
    else:
        start = time.time()
        for name, gen in generators.items():
            logger.info(f"\n{'='*60}")
            gen()
        elapsed = time.time() - start
        logger.info(f"\nAll mappings generated in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
