"""Backward-compatibility shim for the critic's tool surface.

The canonical implementation now lives in
:mod:`src.conversational.health_evidence.tools` and
:mod:`src.conversational.health_evidence.tool_defs`. This module re-exports
the symbols that existing tests / external callers reference, so the
refactor doesn't break any import paths.

Specifically: ``tests/test_conversational/test_critic_tools.py``
monkeypatches ``src.conversational.critic_tools.requests`` (via the
re-exported ``requests`` module) — keeping that path live requires
re-importing ``requests`` here so it appears in this module's namespace.

Do NOT delete this file in v1.
"""

from __future__ import annotations

# Re-export the request module so existing monkeypatches on
# ``src.conversational.critic_tools.requests.get`` continue to take
# effect. The canonical pubmed_search reads requests via its own module
# import, so the patches on this name only affect callers that explicitly
# look up ``critic_tools.requests`` (legacy tests that pre-date the
# refactor); the new test_tools.py uses the canonical health_evidence
# path. Both work.
import requests  # noqa: F401 — re-exported for legacy monkeypatch path

from src.conversational.health_evidence.tool_defs import (  # noqa: F401
    PUBMED_SEARCH_TOOL_DEF,
    TOOL_DISPATCH,
)
from src.conversational.health_evidence.tools import (  # noqa: F401
    _MAX_TOOL_RESULT_BYTES,
    _enforce_size_budget,
    pubmed_search,
)

__all__ = [
    "PUBMED_SEARCH_TOOL_DEF",
    "TOOL_DISPATCH",
    "pubmed_search",
    "_MAX_TOOL_RESULT_BYTES",
    "_enforce_size_budget",
]
