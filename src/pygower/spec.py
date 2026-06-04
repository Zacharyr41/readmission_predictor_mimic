"""Column configuration and type inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import pandas as pd

from ._types import Direction, Kind, Missing


@dataclass
class ColumnSpec:
    """Per-column configuration for a Gower computation.

    Parameters
    ----------
    kind:
        Variable type (:class:`Kind` or its string value).
    weight:
        Non-negative per-column weight in the aggregation.
    range_:
        ``(lo, hi)`` for a quantitative column. If ``None`` the range is
        learned at fit time. Pass an explicit (frozen) range to keep
        distances comparable across calls.
    categories:
        Explicit low->high ordering for an ordinal column.
    asymmetric:
        For binary columns, whether both-absent pairs are excluded from the
        average (asymmetric binary) rather than counted as a match.
    present_value:
        The value that counts as "present" for a binary column.
    direction:
        Monotonic direction for a quantitative column (see :class:`Direction`).
        Non-symmetric directions select the one-sided kernel.
    missing:
        NaN policy for this column (see :class:`Missing`).
    kernel:
        Optional explicit kernel callable that overrides type/direction-based
        selection. Used to plug a custom partial-similarity rule.
    """

    kind: Kind | str
    weight: float = 1.0
    range_: tuple[float, float] | None = None
    categories: Sequence | None = None
    asymmetric: bool = False
    present_value: object = True
    direction: Direction | str = Direction.SYMMETRIC
    missing: Missing | str = Missing.EXCLUDE
    kernel: Callable | None = None

    def __post_init__(self) -> None:
        self.kind = Kind(self.kind)
        self.direction = Direction(self.direction)
        self.missing = Missing(self.missing)
        if self.weight < 0:
            raise ValueError("weight must be non-negative")


def infer_spec(df: pd.DataFrame) -> dict[str, ColumnSpec]:
    """Heuristic per-column type inference. An explicit spec always overrides."""
    spec: dict[str, ColumnSpec] = {}
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_bool_dtype(s):
            spec[col] = ColumnSpec(Kind.BINARY)
        elif isinstance(s.dtype, pd.CategoricalDtype) and s.cat.ordered:
            spec[col] = ColumnSpec(Kind.ORDINAL, categories=list(s.cat.categories))
        elif pd.api.types.is_numeric_dtype(s):
            spec[col] = ColumnSpec(Kind.QUANTITATIVE)
        else:
            spec[col] = ColumnSpec(Kind.NOMINAL)
    return spec
