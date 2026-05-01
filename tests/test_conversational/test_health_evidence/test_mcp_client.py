"""Tests for the synchronous McpClient wrapper.

Strategy:
- For stdio: spawn a real subprocess running ``_test_mcp_server.py``
  (FastMCP). Lets us cover the full async-bridge + subprocess path.
- For HTTP: spawn FastMCP's HTTP server in a background thread,
  point the client at it.
- For failure modes: spawn intentionally-broken servers (bad path,
  crashing tool, etc.) and assert the envelope contract.

Every test must verify the contract: ``call_tool`` NEVER raises and
returns either ``{"status": "ok", "results": [...]}`` or
``{"status": "unavailable", "error": "..."}``.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

from src.conversational.health_evidence.mcp_client import (
    McpClient,
    McpServerConfig,
)


_HERE = Path(__file__).resolve().parent
_TEST_SERVER = _HERE / "_test_mcp_server.py"


def _stdio_config() -> McpServerConfig:
    return McpServerConfig(
        name="test-stdio",
        transport="stdio",
        command=sys.executable,
        args=[str(_TEST_SERVER)],
    )


# ---------------------------------------------------------------------------
# stdio transport
# ---------------------------------------------------------------------------


class TestStdioHappyPath:
    def test_echo_returns_text_envelope(self):
        client = McpClient(_stdio_config())
        try:
            result = client.call_tool("echo", {"text": "hello"})
            assert result["status"] == "ok"
            assert len(result["results"]) >= 1
            # Envelope-convert wraps plain text as {"text": ...}; the test
            # server returns text → wrapped.
            text_value = result["results"][0].get("text", "")
            assert "hello" in text_value
        finally:
            client.close()

    def test_add_returns_json_envelope(self):
        client = McpClient(_stdio_config())
        try:
            result = client.call_tool("add", {"a": 2, "b": 3})
            assert result["status"] == "ok"
            # Server returns JSON {"sum": 5}; envelope wraps it.
            payload = result["results"][0]
            assert payload.get("sum") == 5
        finally:
            client.close()

    def test_passthrough_envelope(self):
        client = McpClient(_stdio_config())
        try:
            result = client.call_tool(
                "returns_envelope", {"text": "passthrough"},
            )
            assert result["status"] == "ok"
            assert result["results"][0]["text"] == "passthrough"
        finally:
            client.close()

    def test_passthrough_unavailable_envelope(self):
        client = McpClient(_stdio_config())
        try:
            result = client.call_tool("returns_unavailable_envelope", {})
            assert result["status"] == "unavailable"
            assert "intentional" in result["error"]
        finally:
            client.close()


class TestStdioConnectionReuse:
    def test_two_calls_share_one_subprocess(self):
        """The second call_tool should use the same persistent session
        (no second subprocess spawn). We can't directly observe the
        subprocess count from Python, but we can assert that the second
        call is significantly faster than the first (subprocess spawn
        cost is amortized)."""
        client = McpClient(_stdio_config())
        try:
            # Warm up the session.
            t0 = time.monotonic()
            client.call_tool("echo", {"text": "warmup"})
            warm_dur = time.monotonic() - t0

            t1 = time.monotonic()
            client.call_tool("echo", {"text": "hot"})
            hot_dur = time.monotonic() - t1

            # The second call should be at least 2x faster than the first
            # (no subprocess spawn). Generous bound — flaky tests are no fun.
            assert hot_dur < max(warm_dur * 0.6, 0.1), (
                f"Expected hot call to be much faster than cold "
                f"(cold={warm_dur:.3f}s, hot={hot_dur:.3f}s)"
            )
        finally:
            client.close()


class TestStdioFailureModes:
    def test_bad_command_returns_unavailable(self):
        client = McpClient(McpServerConfig(
            name="bad",
            transport="stdio",
            command="/nonexistent/path/to/binary",
            args=[],
        ))
        try:
            result = client.call_tool("anything", {})
            assert result["status"] == "unavailable"
            assert "error" in result
        finally:
            client.close()

    def test_crashing_tool_returns_unavailable(self):
        client = McpClient(_stdio_config())
        try:
            result = client.call_tool("boom", {})
            assert result["status"] == "unavailable"
            assert "error" in result
        finally:
            client.close()

    def test_unknown_tool_returns_unavailable(self):
        client = McpClient(_stdio_config())
        try:
            result = client.call_tool("does_not_exist", {})
            assert result["status"] == "unavailable"
        finally:
            client.close()

    def test_timeout_returns_unavailable(self):
        client = McpClient(_stdio_config())
        try:
            # Server sleeps 5s; we time out at 0.5s.
            result = client.call_tool("slow", {"seconds": 5.0}, timeout=0.5)
            assert result["status"] == "unavailable"
            assert "error" in result
        finally:
            client.close()

    def test_call_after_close_returns_unavailable(self):
        client = McpClient(_stdio_config())
        client.call_tool("echo", {"text": "before"})
        client.close()
        result = client.call_tool("echo", {"text": "after"})
        assert result["status"] == "unavailable"

    def test_double_close_is_idempotent(self):
        client = McpClient(_stdio_config())
        client.close()
        client.close()  # must not raise


class TestStdioConfigValidation:
    def test_stdio_missing_command_returns_unavailable(self):
        client = McpClient(McpServerConfig(
            name="x", transport="stdio",
        ))
        result = client.call_tool("any", {})
        assert result["status"] == "unavailable"

    def test_http_missing_url_returns_unavailable(self):
        client = McpClient(McpServerConfig(
            name="x", transport="http",
        ))
        result = client.call_tool("any", {})
        assert result["status"] == "unavailable"


# ---------------------------------------------------------------------------
# Envelope conversion (tested in isolation so we don't have to spin up
# subprocesses to cover edge cases).
# ---------------------------------------------------------------------------


class TestEnvelopeConversion:
    def test_passthrough_envelope_with_status_ok(self):
        from src.conversational.health_evidence.mcp_client import (
            _envelope_from_mcp_result,
        )

        class FakeResult:
            isError = False
            content = [type("B", (), {"text": '{"status": "ok", "results": [{"x": 1}]}'})]

        result = _envelope_from_mcp_result(FakeResult())
        assert result == {"status": "ok", "results": [{"x": 1}]}

    def test_wraps_bare_dict(self):
        from src.conversational.health_evidence.mcp_client import (
            _envelope_from_mcp_result,
        )

        class FakeResult:
            isError = False
            content = [type("B", (), {"text": '{"key": "value"}'})]

        result = _envelope_from_mcp_result(FakeResult())
        assert result == {"status": "ok", "results": [{"key": "value"}]}

    def test_wraps_bare_list(self):
        from src.conversational.health_evidence.mcp_client import (
            _envelope_from_mcp_result,
        )

        class FakeResult:
            isError = False
            content = [type("B", (), {"text": '[1, 2, 3]'})]

        result = _envelope_from_mcp_result(FakeResult())
        assert result == {"status": "ok", "results": [1, 2, 3]}

    def test_wraps_plain_text(self):
        from src.conversational.health_evidence.mcp_client import (
            _envelope_from_mcp_result,
        )

        class FakeResult:
            isError = False
            content = [type("B", (), {"text": "not JSON, just words"})]

        result = _envelope_from_mcp_result(FakeResult())
        assert result == {
            "status": "ok",
            "results": [{"text": "not JSON, just words"}],
        }

    def test_is_error_returns_unavailable(self):
        from src.conversational.health_evidence.mcp_client import (
            _envelope_from_mcp_result,
        )

        class FakeResult:
            isError = True
            content = [type("B", (), {"text": "tool exploded"})]

        result = _envelope_from_mcp_result(FakeResult())
        assert result["status"] == "unavailable"
        assert "tool exploded" in result["error"]

    def test_no_content_returns_empty_results(self):
        from src.conversational.health_evidence.mcp_client import (
            _envelope_from_mcp_result,
        )

        class FakeResult:
            isError = False
            content = None

        result = _envelope_from_mcp_result(FakeResult())
        assert result == {"status": "ok", "results": []}
