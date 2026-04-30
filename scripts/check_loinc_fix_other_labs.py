"""Predict before/after means for several common labs so the chat
demo can be checked against ground truth.

For each lab, prints:
  LIKE-pooled mean   — what the pre-fix path would have returned
  LOINC-grounded mean — what the post-fix path returns
  Clinical reference — typical expected range for orientation

The deltas illustrate which labs are most affected by LIKE pollution
(those with multiple specimen types in MIMIC) vs. those that are
mostly unaffected because there's only one specimen.
"""

from google.cloud import bigquery

LABS = [
    # name, LOINC, LIKE pattern, clinical reference (typical adult range)
    ("Lactate (serum)", "2524-7", "%lactate%", "0.5-2.0 mmol/L"),
    ("Sodium (serum)", "2951-2", "%sodium%", "135-145 mEq/L"),
    ("Glucose (serum)", "2345-7", "%glucose%", "70-180 mg/dL"),
    ("Bilirubin (total)", "1975-2", "%bilirubin%", "0.1-1.2 mg/dL"),
    ("Hemoglobin", "718-7", "%hemoglobin%", "12-17 g/dL"),
]


def itemids_for_loinc(client: bigquery.Client, loinc: str) -> list[int]:
    import json
    from pathlib import Path
    m = json.loads(
        Path("data/mappings/labitem_to_snomed.json").read_text()
    )
    return sorted(
        int(k) for k, v in m.items()
        if isinstance(v, dict) and v.get("loinc") == loinc
    )


def run(client: bigquery.Client, name: str, loinc: str, like: str, ref: str) -> None:
    itemids = itemids_for_loinc(client, loinc)
    if not itemids:
        print(f"\n{name:<20s} (LOINC {loinc})")
        print("  WARNING: no MIMIC labitems map to this LOINC — skipping")
        return

    in_list = ",".join(str(i) for i in itemids)  # ints from mapping file, safe to inline

    pre_sql = (
        "SELECT AVG(l.valuenum) AS m, COUNT(*) AS n "
        "FROM `physionet-data.mimiciv_3_1_hosp.labevents` l "
        "JOIN `physionet-data.mimiciv_3_1_hosp.d_labitems` d "
        "  ON l.itemid = d.itemid "
        f"WHERE LOWER(d.label) LIKE LOWER('{like}') AND l.valuenum IS NOT NULL"
    )
    post_sql = (
        "SELECT AVG(l.valuenum) AS m, COUNT(*) AS n "
        "FROM `physionet-data.mimiciv_3_1_hosp.labevents` l "
        f"WHERE l.itemid IN ({in_list}) AND l.valuenum IS NOT NULL"
    )

    pre = next(iter(client.query(pre_sql).result()))
    post = next(iter(client.query(post_sql).result()))

    delta = abs(pre.m - post.m) / post.m * 100 if post.m else 0
    flag = " ← polluted" if delta > 20 else ""
    print(f"\n{name:<20s} (LOINC {loinc}, itemids={itemids})")
    print(f"  Reference range:    {ref}")
    print(f"  LIKE-pooled mean:   {pre.m:>10.3f}   (n={pre.n:>9,}){flag}")
    print(f"  LOINC-grounded:     {post.m:>10.3f}   (n={post.n:>9,})")
    print(f"  Delta from LIKE:    {delta:>10.1f}%")


if __name__ == "__main__":
    client = bigquery.Client(project="mimic-485500")
    for args in LABS:
        run(client, *args)
