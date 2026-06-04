"""pygower — Gower's distance for mixed-type tabular data.

A correct, vectorized, fit/transform-separated implementation of Gower's
distance with first-class support for:

  * the four classic kinds (quantitative, nominal, ordinal, binary),
  * a one-sided / monotonic kernel for directional traits ("worse", "rising"),
  * per-column NaN policy (exclude vs zero-similarity),
  * per-column similarity exposure for signed-contribution explanations,
  * frozen ranges so distances stay comparable across queries.

See ``docs/gower_distance_design.md`` for the design.
"""

from __future__ import annotations

from ._types import Direction, Kind, Missing
from .api import gower_distances, gower_matrix, gower_topn
from .core import ColumnContribution
from .sklearn import GowerDistance
from .spec import ColumnSpec, infer_spec

__all__ = [
    "ColumnContribution",
    "ColumnSpec",
    "Direction",
    "GowerDistance",
    "Kind",
    "Missing",
    "gower_distances",
    "gower_matrix",
    "gower_topn",
    "infer_spec",
]
