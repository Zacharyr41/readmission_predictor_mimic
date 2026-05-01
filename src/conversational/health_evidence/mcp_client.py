"""Synchronous MCP client wrapper.

The MCP Python SDK is async-first. Our orchestrator is synchronous and
calls tool functions inline from the EvidenceAgent's tool-use loop. This
module bridges the gap: a single ``McpClient`` instance owns a background
asyncio loop running in a daemon thread, plus one persistent
``ClientSession`` per remote/local MCP server. ``call_tool`` is fully
synchronous — it submits a coroutine to the background loop via
``run_coroutine_threadsafe`` and waits for the result.

Why a persistent session instead of one-shot connections per call:
stdio MCPs spawn a subprocess (~50–200 ms cost on each connect). HTTP
MCPs negotiate TLS + the JSON-RPC initialize handshake. Both costs are
prohibitive on a hot path that may issue 5+ tool calls per turn.

Failure semantics: ``call_tool`` NEVER raises. On any failure (subprocess
crash, network error, malformed JSON-RPC, server-side tool error,
timeout) it returns the project's standard envelope::

    {"status": "unavailable", "error": "<message>"}

Successful calls return::

    {"status": "ok", "results": [...]}

The result-shape mapping handles three common MCP server conventions:
1) tool returns a JSON object with our envelope shape (pass-through),
2) tool returns a JSON object/array with no envelope (wrapped in
   ``results``), 3) tool returns plain text (wrapped as
   ``[{"text": "..."}]``).
"""

from __future__ import annotations

import asyncio
import atexit
import json
import logging
import threading
import weakref
from typing import Any, Literal

from pydantic import BaseModel

logger = logging.getLogger(__name__)


_DEFAULT_TIMEOUT = 8.0
_INIT_TIMEOUT = 12.0  # generous; subprocess + handshake + tool listing


class McpServerConfig(BaseModel):
    """Configuration for one MCP server connection."""

    name: str
    transport: Literal["stdio", "http"]
    # stdio fields
    command: str | None = None
    args: list[str] = []
    env: dict[str, str] | None = None
    # http fields
    url: str | None = None


# Track all live clients so we can cleanup at process exit (avoids zombie
# subprocesses if the caller forgot to call .close()).
_LIVE_CLIENTS: weakref.WeakSet["McpClient"] = weakref.WeakSet()


def _cleanup_at_exit() -> None:
    for c in list(_LIVE_CLIENTS):
        try:
            c.close()
        except Exception:  # noqa: BLE001
            pass


atexit.register(_cleanup_at_exit)


class McpClient:
    """Synchronous facade around an asynchronous MCP ClientSession.

    Lazy: the background loop and the session are created on the first
    ``call_tool`` invocation. Reused across all subsequent calls. Use
    ``close()`` to tear down (also called via ``atexit``).
    """

    def __init__(self, config: McpServerConfig) -> None:
        self._config = config
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._session: Any = None
        self._stream_ctx: Any = None
        self._session_ctx: Any = None
        self._closed = False
        _LIVE_CLIENTS.add(self)

    # -- public sync API -------------------------------------------------

    def call_tool(
        self,
        name: str,
        arguments: dict | None = None,
        *,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> dict:
        """Call a tool by name. Returns the project's envelope dict.

        NEVER raises. On any failure returns
        ``{"status": "unavailable", "error": "..."}``.
        """
        if self._closed:
            return {"status": "unavailable", "error": "client closed"}
        try:
            self._ensure_loop()
            assert self._loop is not None
            future = asyncio.run_coroutine_threadsafe(
                self._async_call(name, arguments or {}, timeout),
                self._loop,
            )
            # +2s grace over the per-call timeout so we don't race
            # the future.result() against the inner asyncio.wait_for.
            result = future.result(timeout=timeout + 2.0)
            return _envelope_from_mcp_result(result)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "McpClient[%s] call_tool(%s) failed: %s (%s)",
                self._config.name, name, exc, type(exc).__name__,
            )
            return {"status": "unavailable", "error": str(exc)}

    def close(self) -> None:
        """Tear down the session and the background loop. Idempotent."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            loop = self._loop
            self._loop = None
            thread = self._thread
            self._thread = None

        if loop is None:
            return
        try:
            future = asyncio.run_coroutine_threadsafe(
                self._async_close(), loop,
            )
            try:
                future.result(timeout=3.0)
            except Exception:  # noqa: BLE001
                pass
        finally:
            try:
                loop.call_soon_threadsafe(loop.stop)
            except Exception:  # noqa: BLE001
                pass
            if thread is not None:
                thread.join(timeout=2.0)

    # -- async implementation -------------------------------------------

    def _ensure_loop(self) -> None:
        with self._lock:
            if self._loop is not None:
                return
            if self._closed:
                raise RuntimeError("client closed")
            loop = asyncio.new_event_loop()
            thread = threading.Thread(
                target=loop.run_forever,
                daemon=True,
                name=f"mcp-client-{self._config.name}",
            )
            thread.start()
            self._loop = loop
            self._thread = thread

    async def _async_call(
        self, name: str, arguments: dict, timeout: float,
    ) -> Any:
        await self._async_init_session()
        return await asyncio.wait_for(
            self._session.call_tool(name=name, arguments=arguments),
            timeout=timeout,
        )

    async def _async_init_session(self) -> None:
        if self._session is not None:
            return
        from mcp import ClientSession

        if self._config.transport == "stdio":
            from mcp.client.stdio import StdioServerParameters, stdio_client

            if not self._config.command:
                raise ValueError("stdio config missing command")
            params = StdioServerParameters(
                command=self._config.command,
                args=list(self._config.args or []),
                env=self._config.env,
            )
            self._stream_ctx = stdio_client(params)
            streams = await asyncio.wait_for(
                self._stream_ctx.__aenter__(),
                timeout=_INIT_TIMEOUT,
            )
            read, write = streams[0], streams[1]
        elif self._config.transport == "http":
            from mcp.client.streamable_http import streamablehttp_client

            if not self._config.url:
                raise ValueError("http config missing url")
            self._stream_ctx = streamablehttp_client(self._config.url)
            streams = await asyncio.wait_for(
                self._stream_ctx.__aenter__(),
                timeout=_INIT_TIMEOUT,
            )
            read, write = streams[0], streams[1]
        else:
            raise ValueError(f"unknown transport: {self._config.transport!r}")

        self._session_ctx = ClientSession(read, write)
        self._session = await self._session_ctx.__aenter__()
        await asyncio.wait_for(
            self._session.initialize(), timeout=_INIT_TIMEOUT,
        )

    async def _async_close(self) -> None:
        if self._session_ctx is not None:
            try:
                await self._session_ctx.__aexit__(None, None, None)
            except Exception:  # noqa: BLE001
                pass
        if self._stream_ctx is not None:
            try:
                await self._stream_ctx.__aexit__(None, None, None)
            except Exception:  # noqa: BLE001
                pass
        self._session = None
        self._session_ctx = None
        self._stream_ctx = None


# ---------------------------------------------------------------------------
# Result envelope conversion
# ---------------------------------------------------------------------------


def _envelope_from_mcp_result(result: Any) -> dict:
    """Convert an MCP CallToolResult into the project's envelope shape."""
    if getattr(result, "isError", False):
        text = _extract_text(getattr(result, "content", None))
        return {
            "status": "unavailable",
            "error": text or "tool returned isError",
        }

    text = _extract_text(getattr(result, "content", None))
    if not text:
        return {"status": "ok", "results": []}

    # Try parsing as JSON.
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return {"status": "ok", "results": [{"text": text}]}

    # Server already returned an envelope: pass through (after sanity check).
    if isinstance(parsed, dict) and "status" in parsed:
        if parsed["status"] in {"ok", "unavailable"}:
            return parsed
        # Status field present but invalid — wrap.
        return {"status": "ok", "results": [parsed]}
    if isinstance(parsed, dict):
        return {"status": "ok", "results": [parsed]}
    if isinstance(parsed, list):
        return {"status": "ok", "results": parsed}
    return {"status": "ok", "results": [{"value": parsed}]}


def _extract_text(content: Any) -> str:
    """Concatenate text from MCP content blocks. Returns '' if none."""
    if content is None:
        return ""
    parts: list[str] = []
    for block in content:
        # MCP TextContent has .text; ImageContent / others don't.
        text = getattr(block, "text", None)
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts)
