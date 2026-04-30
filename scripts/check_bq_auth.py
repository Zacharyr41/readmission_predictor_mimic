from google.cloud import bigquery

client = bigquery.Client(project="mimic-485500")
sql = "SELECT COUNT(*) AS n FROM `physionet-data.mimiciv_3_1_hosp.admissions`"
row = list(client.query(sql).result())[0]
print(f"OK — {row.n:,} admissions reachable in physionet-data")
