"""Tests for the bq-validator MCP server.

These exercise ``_run_stages`` directly — no MCP wire format needed for
these tests since FastMCP just dispatches the tool function. The MCP
end-to-end path is exercised by ``test_sql_validator_dry_run.py`` via
the McpClient.

We mock ``bigquery.Client.query`` so tests run without GCP credentials
or network.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from mcp_servers.bq_validator.server import _run_stages, _to_bq_params, validate_sql


# ---------------------------------------------------------------------------
# Stage 1: parse
# ---------------------------------------------------------------------------


class TestParseStage:
    def test_clean_select_passes_to_dry_run(self):
        with patch("mcp_servers.bq_validator.server._bq") as bq:
            job = MagicMock()
            job.total_bytes_processed = 100
            job.referenced_tables = []
            bq.return_value.query.return_value = job
            verdict = _run_stages(
                "SELECT 1 FROM `physionet-data.mimic_iv_demo.admissions`",
                [], 0.5, 10**10,
            )
            assert verdict["ok"]
            assert verdict["stage"] == "ok"

    def test_garbage_sql_blocks_at_parse(self):
        verdict = _run_stages(
            "this is not valid SQL at all !!", [], 0.5, 10**10,
        )
        assert verdict["ok"] is False
        assert verdict["stage"] == "parse"


# ---------------------------------------------------------------------------
# Stage 2: read-only policy
# ---------------------------------------------------------------------------


class TestPolicyStage:
    @pytest.mark.parametrize("sql", [
        "DROP TABLE x",
        "DELETE FROM x WHERE y = 1",
        "INSERT INTO x VALUES (1)",
        "UPDATE x SET y = 1",
        "CREATE TABLE x (y INT64)",
        "TRUNCATE TABLE x",
    ])
    def test_writes_blocked_at_policy(self, sql):
        verdict = _run_stages(sql, [], 0.5, 10**10)
        # Could fail at parse or policy depending on shape; either is fine
        # because both are pre-execution rejections.
        assert verdict["ok"] is False
        assert verdict["stage"] in {"parse", "policy"}


# ---------------------------------------------------------------------------
# Stage 3: dataset scope
# ---------------------------------------------------------------------------


class TestScopeStage:
    def test_disallowed_dataset_blocked(self):
        verdict = _run_stages(
            "SELECT 1 FROM `some-other-project.evil.table`",
            [], 0.5, 10**10,
        )
        assert verdict["ok"] is False
        assert verdict["stage"] == "scope"

    def test_allowed_mimic_dataset_passes_scope(self):
        with patch("mcp_servers.bq_validator.server._bq") as bq:
            job = MagicMock()
            job.total_bytes_processed = 100
            job.referenced_tables = []
            bq.return_value.query.return_value = job
            verdict = _run_stages(
                "SELECT * FROM `physionet-data.mimiciv_3_1_hosp.admissions`",
                [], 0.5, 10**10,
            )
            assert verdict["ok"] is True

    def test_cte_reference_does_not_trigger_scope(self):
        with patch("mcp_servers.bq_validator.server._bq") as bq:
            job = MagicMock()
            job.total_bytes_processed = 0
            job.referenced_tables = []
            bq.return_value.query.return_value = job
            verdict = _run_stages(
                "WITH cohort AS (SELECT 1 AS x) "
                "SELECT * FROM cohort",
                [], 0.5, 10**10,
            )
            assert verdict["ok"] is True


# ---------------------------------------------------------------------------
# Stage 4: dry_run failures
# ---------------------------------------------------------------------------


class TestDryRunStage:
    def test_dry_run_exception_blocks(self):
        with patch("mcp_servers.bq_validator.server._bq") as bq:
            bq.return_value.query.side_effect = RuntimeError("dry-run boom")
            verdict = _run_stages(
                "SELECT 1 FROM `physionet-data.mimic_iv_demo.admissions`",
                [], 0.5, 10**10,
            )
            assert verdict["ok"] is False
            assert verdict["stage"] == "dry_run"
            assert "boom" in verdict["error"]


# ---------------------------------------------------------------------------
# Stage 5: cost gate
# ---------------------------------------------------------------------------


class TestCostGate:
    def test_under_budget_passes(self):
        with patch("mcp_servers.bq_validator.server._bq") as bq:
            job = MagicMock()
            job.total_bytes_processed = 1024**3  # 1 GiB
            job.referenced_tables = []
            bq.return_value.query.return_value = job
            verdict = _run_stages(
                "SELECT * FROM `physionet-data.mimiciv_3_1_hosp.admissions`",
                [],
                max_usd=0.50,
                max_bytes=10 * 1024**3,
            )
            assert verdict["ok"] is True
            assert verdict["bytes_processed"] == 1024**3

    def test_over_byte_cap_blocks(self):
        with patch("mcp_servers.bq_validator.server._bq") as bq:
            job = MagicMock()
            job.total_bytes_processed = 50 * 1024**3  # 50 GiB
            job.referenced_tables = []
            bq.return_value.query.return_value = job
            verdict = _run_stages(
                "SELECT * FROM `physionet-data.mimiciv_3_1_hosp.admissions`",
                [],
                max_usd=10.0,
                max_bytes=10 * 1024**3,
            )
            assert verdict["ok"] is False
            assert verdict["stage"] == "bytes"
            assert verdict["bytes_processed"] == 50 * 1024**3

    def test_over_usd_cap_blocks(self):
        with patch("mcp_servers.bq_validator.server._bq") as bq:
            job = MagicMock()
            # 5 GiB ≈ ~$0.03, so set max_usd=0.001 to trip it.
            job.total_bytes_processed = 5 * 1024**3
            job.referenced_tables = []
            bq.return_value.query.return_value = job
            verdict = _run_stages(
                "SELECT * FROM `physionet-data.mimiciv_3_1_hosp.admissions`",
                [],
                max_usd=0.001,
                max_bytes=10 * 1024**4,  # huge byte cap
            )
            assert verdict["ok"] is False
            assert verdict["stage"] == "cost"
            assert verdict["estimated_usd"] > 0.001


# ---------------------------------------------------------------------------
# validate_sql tool entry point
# ---------------------------------------------------------------------------


class TestValidateSqlTool:
    def test_returns_envelope_with_verdict(self):
        with patch("mcp_servers.bq_validator.server._bq") as bq:
            job = MagicMock()
            job.total_bytes_processed = 100
            job.referenced_tables = []
            bq.return_value.query.return_value = job
            payload = validate_sql(
                "SELECT 1 FROM `physionet-data.mimic_iv_demo.admissions`",
            )
            data = json.loads(payload)
            assert data["status"] == "ok"
            assert len(data["results"]) == 1
            verdict = data["results"][0]
            assert verdict["ok"] is True


# ---------------------------------------------------------------------------
# Param conversion
# ---------------------------------------------------------------------------


class TestParamConversion:
    def test_typed_params(self):
        out = _to_bq_params([1, 2.5, "hello", True])
        types = [p.type_ for p in out]
        # bool first because bool is subclass of int — check ordering
        assert types[0] == "INT64"
        assert types[1] == "FLOAT64"
        assert types[2] == "STRING"
        assert types[3] == "BOOL"
