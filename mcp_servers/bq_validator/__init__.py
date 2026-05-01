"""bq-validator: deterministic pre-execution SQL validator MCP server.

The server runs as a stdio MCP that the orchestrator's McpClient connects
to. It validates a SQL query in five stages:

1. SQLGlot parse  — rejects syntactic garbage and multi-statement injection.
2. Read-only policy — rejects anything that's not SELECT / WITH / UNION.
3. Dataset scope — rejects table references outside the configured allowlist
   (defaults to physionet-data.mimiciv_3_1_*).
4. BigQuery dry_run — semantic + permission check; returns
   total_bytes_processed.
5. Cost gate — rejects if the dry-run cost or byte total exceeds limits.

Designed to run inside the GCP-BAA project boundary so dry-run calls
don't leave the trust boundary. Tools never see row data — only SQL
text and parameters the caller supplies.
"""
