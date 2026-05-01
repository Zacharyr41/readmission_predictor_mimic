"""Tests for the deterministic dry-run SQL validator wrapper.

Mocks ``McpClient.call_tool`` to feed the wrapper canned envelopes
(success / failure / each verdict tier). The MCP server itself is
tested separately in ``mcp_servers/bq_validator/test_server.py``.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.conversational.models import SqlValidationVerdict
from src.conversational.sql_fastpath import SqlFastpathQuery
from src.conversational.sql_validator_dry_run import (
    validate_sql_deterministic,
)


def _query() -> SqlFastpathQuery:
    return SqlFastpathQuery(
        sql="SELECT 1 FROM x", params=[1, 2, 3], columns=["a"],
    )


def _mock_client(envelope: dict) -> MagicMock:
    client = MagicMock()
    client.call_tool.return_value = envelope
    return client


# ---------------------------------------------------------------------------
# Pass path
# ---------------------------------------------------------------------------


class TestPassPath:
    def test_ok_verdict_returns_pass(self):
        client = _mock_client({
            "status": "ok",
            "results": [{
                "ok": True, "stage": "ok",
                "bytes_processed": 1024 * 1024 * 100,  # 100 MiB
                "estimated_usd": 0.0006,
                "referenced_tables": ["physionet-data.mimic_iv_demo.admissions"],
            }],
        })
        verdict = validate_sql_deterministic(
            _query(), mcp_client=client, resolved_itemids=[1, 2, 3],
        )
        assert verdict is not None
        assert verdict.verdict == "pass"
        assert verdict.bytes_processed == 1024 * 1024 * 100
        assert verdict.estimated_usd == pytest.approx(0.0006)
        assert verdict.reference_used == "dry_run:ok"

    def test_call_passes_sql_and_params_to_mcp(self):
        client = _mock_client({
            "status": "ok",
            "results": [{"ok": True, "stage": "ok",
                          "bytes_processed": 0, "estimated_usd": 0}],
        })
        validate_sql_deterministic(
            SqlFastpathQuery(sql="SELECT 42", params=["a", "b"], columns=[]),
            mcp_client=client,
            resolved_itemids=[1],
            max_usd=1.0,
            max_bytes=2 * 1024**3,
        )
        client.call_tool.assert_called_once()
        args = client.call_tool.call_args
        assert args.args[0] == "validate_sql"
        payload = args.args[1]
        assert payload["sql"] == "SELECT 42"
        assert payload["params"] == ["a", "b"]
        assert payload["max_usd"] == 1.0
        assert payload["max_bytes"] == 2 * 1024**3


# ---------------------------------------------------------------------------
# Block / warn paths
# ---------------------------------------------------------------------------


class TestBlockPath:
    def test_clean_loinc_failure_returns_block(self):
        """LOINC grounding succeeded (resolved_itemids is non-None) and
        no fallback warning — server failure → block."""
        client = _mock_client({
            "status": "ok",
            "results": [{
                "ok": False, "stage": "bytes",
                "bytes_processed": 50 * 1024**3,
                "estimated_usd": 0.30,
                "error": "Estimated 50 GiB exceeds limit",
            }],
        })
        verdict = validate_sql_deterministic(
            _query(),
            mcp_client=client,
            fallback_warning=None,
            resolved_itemids=[50912, 51081],
        )
        assert verdict is not None
        assert verdict.verdict == "block"
        assert "exceeds limit" in (verdict.concern or "")
        assert verdict.reference_used == "dry_run:bytes"
        assert verdict.bytes_processed == 50 * 1024**3
        assert verdict.suggested_fix is not None

    def test_parse_failure_with_clean_loinc_returns_block(self):
        client = _mock_client({
            "status": "ok",
            "results": [{
                "ok": False, "stage": "parse",
                "error": "syntax error at line 1",
            }],
        })
        verdict = validate_sql_deterministic(
            _query(), mcp_client=client, resolved_itemids=[1, 2, 3],
        )
        assert verdict is not None
        assert verdict.verdict == "block"
        assert verdict.suggested_fix is not None


class TestWarnPath:
    def test_fallback_warning_downgrades_block_to_warn(self):
        """Even if the validator says block, a LIKE-fallback path warns
        instead so the critic's self-healing retry gets a shot."""
        client = _mock_client({
            "status": "ok",
            "results": [{
                "ok": False, "stage": "scope",
                "error": "Table outside allowlist",
            }],
        })
        verdict = validate_sql_deterministic(
            _query(),
            mcp_client=client,
            fallback_warning="LOINC grounding failed; falling back to LIKE",
            resolved_itemids=None,
        )
        assert verdict is not None
        assert verdict.verdict == "warn"

    def test_no_resolved_itemids_downgrades_to_warn(self):
        """Same downgrade rule when itemids are None (no biomarker
        grounding attempted)."""
        client = _mock_client({
            "status": "ok",
            "results": [{
                "ok": False, "stage": "cost",
                "error": "too expensive",
            }],
        })
        verdict = validate_sql_deterministic(
            _query(),
            mcp_client=client,
            fallback_warning=None,
            resolved_itemids=None,
        )
        assert verdict is not None
        assert verdict.verdict == "warn"


# ---------------------------------------------------------------------------
# Failure modes — must always return None
# ---------------------------------------------------------------------------


class TestFailureModes:
    def test_mcp_unavailable_returns_none(self):
        client = _mock_client({
            "status": "unavailable", "error": "subprocess crashed",
        })
        verdict = validate_sql_deterministic(
            _query(), mcp_client=client, resolved_itemids=[1],
        )
        assert verdict is None

    def test_malformed_envelope_returns_none(self):
        client = _mock_client({"status": "ok", "results": []})
        verdict = validate_sql_deterministic(
            _query(), mcp_client=client, resolved_itemids=[1],
        )
        assert verdict is None

    def test_non_dict_result_returns_none(self):
        client = _mock_client({"status": "ok", "results": ["not a dict"]})
        verdict = validate_sql_deterministic(
            _query(), mcp_client=client, resolved_itemids=[1],
        )
        assert verdict is None

    def test_call_raising_returns_none(self):
        """Defense in depth: even if the McpClient violates its
        never-raise contract, the wrapper still returns None."""
        client = MagicMock()
        client.call_tool.side_effect = RuntimeError("boom")
        verdict = validate_sql_deterministic(
            _query(), mcp_client=client, resolved_itemids=[1],
        )
        assert verdict is None


# ---------------------------------------------------------------------------
# Telemetry passthrough
# ---------------------------------------------------------------------------


class TestTelemetryPassthrough:
    def test_pass_carries_cost_fields(self):
        client = _mock_client({
            "status": "ok",
            "results": [{"ok": True, "stage": "ok",
                          "bytes_processed": 12345, "estimated_usd": 0.001}],
        })
        verdict = validate_sql_deterministic(
            _query(), mcp_client=client, resolved_itemids=[1],
        )
        assert verdict.bytes_processed == 12345
        assert verdict.estimated_usd == pytest.approx(0.001)

    def test_block_also_carries_cost_fields(self):
        client = _mock_client({
            "status": "ok",
            "results": [{"ok": False, "stage": "cost",
                          "bytes_processed": 99 * 1024**3,
                          "estimated_usd": 0.60,
                          "error": "exceeds limit"}],
        })
        verdict = validate_sql_deterministic(
            _query(), mcp_client=client, resolved_itemids=[1],
        )
        assert verdict.verdict == "block"
        assert verdict.bytes_processed == 99 * 1024**3
        assert verdict.estimated_usd == pytest.approx(0.60)
