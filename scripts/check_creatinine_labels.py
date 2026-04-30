from google.cloud import bigquery

client = bigquery.Client(project="mimic-485500")

sql = """
SELECT
  d.label,
  COUNT(*) AS n_rows,
  ROUND(AVG(l.valuenum), 2) AS mean_value,
  ROUND(MIN(l.valuenum), 2) AS min_value,
  ROUND(MAX(l.valuenum), 2) AS max_value
FROM `physionet-data.mimiciv_3_1_hosp.labevents` l
JOIN `physionet-data.mimiciv_3_1_hosp.d_labitems` d ON l.itemid = d.itemid
WHERE LOWER(d.label) LIKE '%creatinine%'
  AND l.valuenum IS NOT NULL
GROUP BY d.label
ORDER BY n_rows DESC
"""

print(f"{'label':<40s} {'n_rows':>12s} {'mean':>8s} {'min':>8s} {'max':>8s}")
for row in client.query(sql).result():
    print(f"{row.label:<40s} {row.n_rows:>12,} {row.mean_value:>8} {row.min_value:>8} {row.max_value:>8}")
