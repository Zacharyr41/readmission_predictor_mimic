"""Tiny FastMCP server used as a stdio subprocess by McpClient tests.

Invoked by the test suite as::

    python tests/test_conversational/test_health_evidence/_test_mcp_server.py

Tools exposed:
- ``echo(text)`` — returns the input text.
- ``add(a, b)`` — returns a + b as JSON.
- ``returns_envelope(text)`` — returns a dict already in our envelope shape.
- ``returns_unavailable_envelope()`` — returns the unavailable envelope.
- ``boom()`` — raises an exception (server reports tool error).
- ``slow(seconds)`` — sleeps then returns; used for timeout tests.
"""

from __future__ import annotations

import json
import time

from fastmcp import FastMCP

mcp = FastMCP("test-server")


@mcp.tool
def echo(text: str) -> str:
    return text


@mcp.tool
def add(a: int, b: int) -> str:
    return json.dumps({"sum": a + b})


@mcp.tool
def returns_envelope(text: str) -> str:
    return json.dumps({"status": "ok", "results": [{"text": text}]})


@mcp.tool
def returns_unavailable_envelope() -> str:
    return json.dumps({"status": "unavailable", "error": "intentional"})


@mcp.tool
def boom() -> str:
    raise RuntimeError("intentional crash")


@mcp.tool
def slow(seconds: float) -> str:
    time.sleep(seconds)
    return "done"


if __name__ == "__main__":
    mcp.run(transport="stdio")
