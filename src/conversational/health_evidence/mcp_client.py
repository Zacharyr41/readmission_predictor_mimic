"""Synchronous MCP client wrapper.

The MCP Python SDK is async-first. Our orchestrator is synchronous and
calls tool functions inline from the EvidenceAgent's tool-use loop. This
module bridges the gap: a single ``McpClient`` instance owns a background
asyncio loop running in a daemon thread, plus one persistent
``ClientSession`` per remote/local MCP server. ``call_tool`` is fully
synchronous — it hands a request to the background loop and waits for the
result on a thread-safe future.

Why a single long-lived "session owner" task:
the SDK's transports (``stdio_client``, ``streamablehttp_client``) and
``ClientSession`` are built on anyio task groups / cancel scopes, which
MUST be entered and exited in the *same* asyncio task. A persistent
session that's opened on one ``run_coroutine_threadsafe`` task and torn
down on another violates that rule and raises
``RuntimeError: Attempted to exit cancel scope in a different task than
it was entered in`` at shutdown. To avoid that, the whole session
lifecycle lives inside one coroutine (``_session_owner``): it opens the
streams + session under a single ``async with``, serves tool calls off a
queue, and unwinds the ``async with`` in that same task on shutdown.

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
import concurrent.futures
import json
import logging
import threading
import weakref
from contextlib import AsyncExitStack
from typing import Any, Literal

from pydantic import BaseModel

logger = logging.getLogger(__name__)


_DEFAULT_TIMEOUT = 8.0
_INIT_TIMEOUT = 12.0  # generous; subprocess + handshake + tool listing

# Unique sentinel pushed onto the request queue to tell the owner task to
# break its serve loop and unwind the session (in its own task).
_SHUTDOWN = object()


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
    # Optional HTTP headers (e.g. ``{"Authorization": "Bearer ..."}``).
    # Used by hosted MCPs that authenticate the client request rather
    # than reading credentials from a server-side env var. Leave None
    # for unauthenticated transports. Never logged.
    headers: dict[str, str] | None = None
    # When True, the first call that times out marks the client dead so
    # subsequent calls fast-fail (see McpClient circuit breaker). Use for
    # OPTIONAL MCPs that degrade gracefully (e.g. the bq-validator pre-check),
    # never for ones whose failure should be retried.
    circuit_break_on_timeout: bool = False


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

    Lazy: the background loop and the session-owner task are created on
    the first ``call_tool`` invocation and reused across all subsequent
    calls. Use ``close()`` to tear down (also called via ``atexit``).
    """

    def __init__(self, config: McpServerConfig) -> None:
        self._config = config
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        # asyncio.Queue, created on the loop by the owner task.
        self._requests: asyncio.Queue | None = None
        # Resolves True on successful init, or with the init exception.
        self._ready: concurrent.futures.Future | None = None
        # Resolves once the owner task has fully unwound the session.
        self._done: concurrent.futures.Future | None = None
        self._owner_started = False
        self._session_ready = False
        self._closed = False
        # Circuit breaker: when ``config.circuit_break_on_timeout`` is set, the
        # first call that times out marks the client dead so every subsequent
        # call fast-fails (returns ``unavailable`` immediately) instead of
        # paying the timeout again. Used for the optional bq-validator, whose
        # stdio handshake can stall in detached/backgrounded contexts.
        self._dead = False
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
        if self._dead:
            return {"status": "unavailable", "error": "circuit-broken (prior timeout)"}
        try:
            self._ensure_session()
            loop = self._loop
            requests = self._requests
            if loop is None or requests is None:
                raise RuntimeError("mcp session not available")
            fut: concurrent.futures.Future = concurrent.futures.Future()
            loop.call_soon_threadsafe(
                requests.put_nowait,
                (name, arguments or {}, timeout, fut),
            )
            # +2s grace over the per-call timeout so we don't race
            # the future.result() against the inner asyncio.wait_for.
            result = fut.result(timeout=timeout + 2.0)
            return _envelope_from_mcp_result(result)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "McpClient[%s] call_tool(%s) failed: %s (%s)",
                self._config.name, name, exc, type(exc).__name__,
            )
            if isinstance(exc, TimeoutError) and getattr(
                self._config, "circuit_break_on_timeout", False
            ):
                self._dead = True
                logger.warning(
                    "McpClient[%s] circuit-broken after timeout; further calls "
                    "skip this MCP for the session.", self._config.name,
                )
            return {"status": "unavailable", "error": str(exc)}

    def close(self) -> None:
        """Tear down the session and the background loop. Idempotent."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            loop = self._loop
            thread = self._thread
            done = self._done
            owner_started = self._owner_started
            self._loop = None
            self._thread = None
            self._requests = None
            self._session_ready = False

        if loop is None:
            if thread is not None:
                thread.join(timeout=2.0)
            return

        # Signal the owner task to break its serve loop and unwind the
        # session's ``async with`` IN ITS OWN TASK — this is what avoids
        # the anyio cross-task cancel-scope RuntimeError. Then wait for
        # that clean teardown before stopping the loop.
        try:
            loop.call_soon_threadsafe(self._signal_shutdown)
            if owner_started and done is not None:
                try:
                    done.result(timeout=4.0)
                except Exception:  # noqa: BLE001
                    pass
        finally:
            try:
                loop.call_soon_threadsafe(loop.stop)
            except Exception:  # noqa: BLE001
                pass
            if thread is not None:
                thread.join(timeout=2.0)

    # -- session lifecycle ----------------------------------------------

    def _ensure_session(self) -> None:
        """Start the loop + owner task and block until the session is
        ready. Raises on init failure (``call_tool`` catches it)."""
        if self._session_ready:
            return
        with self._lock:
            if self._session_ready:
                return
            if self._closed:
                raise RuntimeError("client closed")
            if not self._owner_started:
                self._start_locked()
            ready = self._ready
        assert ready is not None
        try:
            ready.result(timeout=_INIT_TIMEOUT + 4.0)
        except Exception:
            # Init failed or timed out. Tear the dead loop down so the
            # next call_tool retries cleanly, then re-raise so this call
            # returns an "unavailable" envelope.
            self._reset_after_failed_init()
            raise
        self._session_ready = True

    def _start_locked(self) -> None:
        """Spawn the background loop and schedule the owner task. Caller
        must hold ``self._lock``."""
        loop = asyncio.new_event_loop()
        thread = threading.Thread(
            target=loop.run_forever,
            daemon=True,
            name=f"mcp-client-{self._config.name}",
        )
        thread.start()
        self._loop = loop
        self._thread = thread
        self._ready = concurrent.futures.Future()
        self._done = concurrent.futures.Future()
        self._owner_started = True
        asyncio.run_coroutine_threadsafe(self._session_owner(), loop)

    def _reset_after_failed_init(self) -> None:
        """Stop the dead loop/thread and clear owner state so a later
        ``call_tool`` can start a fresh session."""
        with self._lock:
            if self._closed:
                return
            loop = self._loop
            thread = self._thread
            self._loop = None
            self._thread = None
            self._requests = None
            self._ready = None
            self._done = None
            self._owner_started = False
            self._session_ready = False
        if loop is not None:
            try:
                loop.call_soon_threadsafe(loop.stop)
            except Exception:  # noqa: BLE001
                pass
        if thread is not None:
            thread.join(timeout=2.0)

    def _signal_shutdown(self) -> None:
        """Push the shutdown sentinel onto the request queue. Runs on the
        loop thread (scheduled via ``call_soon_threadsafe``)."""
        requests = self._requests
        if requests is not None:
            try:
                requests.put_nowait(_SHUTDOWN)
            except Exception:  # noqa: BLE001
                pass

    # -- async implementation (all runs on the one owner task) ----------

    async def _session_owner(self) -> None:
        """Own the entire session lifecycle within a single task.

        Opens the streams + ``ClientSession`` under one ``async with``,
        signals readiness, serves tool calls off the queue, then unwinds
        the ``async with`` in THIS task on the shutdown sentinel. Keeping
        enter and exit on the same task is what avoids anyio's
        cross-task cancel-scope RuntimeError.
        """
        from mcp import ClientSession

        requests: asyncio.Queue = asyncio.Queue()
        self._requests = requests
        ready = self._ready
        done = self._done
        try:
            async with AsyncExitStack() as stack:
                read, write = await self._open_streams(stack)
                session = await stack.enter_async_context(
                    ClientSession(read, write),
                )
                await asyncio.wait_for(
                    session.initialize(), timeout=_INIT_TIMEOUT,
                )
                if ready is not None and not ready.done():
                    ready.set_result(True)
                await self._serve(session, requests)
        except BaseException as exc:  # noqa: BLE001
            # Init failure (or a crash before/at the serve loop). If
            # nobody is waiting on readiness anymore, just swallow it.
            if ready is not None and not ready.done():
                ready.set_exception(exc)
        finally:
            self._fail_pending(requests)
            self._requests = None
            if done is not None and not done.done():
                done.set_result(True)

    async def _open_streams(self, stack: AsyncExitStack) -> tuple[Any, Any]:
        """Open the transport and return ``(read, write)`` streams. The
        context manager is entered on ``stack`` so it's exited later in
        this same task when the stack unwinds."""
        cfg = self._config
        if cfg.transport == "stdio":
            from mcp.client.stdio import StdioServerParameters, stdio_client

            if not cfg.command:
                raise ValueError("stdio config missing command")
            params = StdioServerParameters(
                command=cfg.command,
                args=list(cfg.args or []),
                env=cfg.env,
            )
            streams = await stack.enter_async_context(stdio_client(params))
        elif cfg.transport == "http":
            from mcp.client.streamable_http import streamablehttp_client

            if not cfg.url:
                raise ValueError("http config missing url")
            # Pass ``headers`` only when configured (None / absent
            # otherwise). Empty dict is NOT equivalent — some SDK
            # versions attach malformed Authorization headers from
            # empty mappings. The headers content is sensitive (Bearer
            # tokens etc.) and is intentionally NEVER logged below.
            if cfg.headers:
                cm = streamablehttp_client(cfg.url, headers=cfg.headers)
            else:
                cm = streamablehttp_client(cfg.url)
            streams = await stack.enter_async_context(cm)
        else:
            raise ValueError(f"unknown transport: {cfg.transport!r}")
        return streams[0], streams[1]

    async def _serve(
        self, session: Any, requests: asyncio.Queue,
    ) -> None:
        """Pull requests off the queue and execute them on this task,
        one at a time, until the shutdown sentinel arrives."""
        while True:
            item = await requests.get()
            if item is _SHUTDOWN:
                return
            name, arguments, timeout, fut = item
            if fut.done():  # caller already gave up
                continue
            try:
                result = await asyncio.wait_for(
                    session.call_tool(name=name, arguments=arguments),
                    timeout=timeout,
                )
            except BaseException as exc:  # noqa: BLE001
                if not fut.done():
                    fut.set_exception(exc)
            else:
                if not fut.done():
                    fut.set_result(result)

    @staticmethod
    def _fail_pending(requests: asyncio.Queue | None) -> None:
        """Fail any still-queued requests so their callers stop waiting.
        Runs on the owner task during teardown."""
        if requests is None:
            return
        while True:
            try:
                item = requests.get_nowait()
            except asyncio.QueueEmpty:
                break
            if item is _SHUTDOWN:
                continue
            fut = item[3]
            if not fut.done():
                fut.set_exception(RuntimeError("mcp client closed"))


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
