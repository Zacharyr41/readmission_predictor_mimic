"""Enums and typing aliases for pygower."""

from __future__ import annotations

from enum import Enum


class Kind(str, Enum):
    """Variable type that selects the partial-similarity kernel."""

    QUANTITATIVE = "quantitative"
    NOMINAL = "nominal"
    ORDINAL = "ordinal"
    BINARY = "binary"


class Direction(str, Enum):
    """Monotonic direction for a quantitative column.

    ``SYMMETRIC`` is ordinary distance-to-a-point. The two one-sided values
    make *more-extreme-in-the-stated-direction* score as at-least-as-similar:
    a candidate that exceeds the reference is never penalized.
    """

    SYMMETRIC = "symmetric"
    HIGHER_MORE_SIMILAR = "higher_more_similar"
    LOWER_MORE_SIMILAR = "lower_more_similar"


class Missing(str, Enum):
    """Per-column policy for NaN values.

    ``EXCLUDE`` drops the column from a pair's weighted average (Gower's
    classic delta=0). ``ZERO_SIMILARITY`` instead counts the column with
    similarity 0 (a detractor) — needed to reproduce the existing
    contextual-similarity semantics.
    """

    EXCLUDE = "exclude"
    ZERO_SIMILARITY = "zero_similarity"
