# Design Document: `pygower` — A Full-Featured Gower's Distance Module for Python

**Status:** Draft
**Audience:** Implementers (human or coding agent)
**Goal:** A correct, fast, well-tested, ergonomic Python library for computing Gower's distance/similarity on mixed-type tabular data.

---

## 1. Background & Motivation

Gower's distance measures dissimilarity between observations described by a mix of variable types (continuous, categorical, ordinal, binary). For each variable it computes a partial similarity using a type-appropriate rule, then aggregates via a (weighted) mean into a single value in `[0, 1]`.

For variables $k$, observations $i, j$:

$$
d_{ij} = \frac{\sum_k w_k\,\delta_{ijk}\,(1 - s_{ijk})}{\sum_k w_k\,\delta_{ijk}}
$$

- $s_{ijk}$ — partial *similarity* for variable $k$ (in `[0,1]`).
- $\delta_{ijk}$ — validity indicator (0 when either value is missing, or for asymmetric-binary negative matches; else 1).
- $w_k$ — per-variable weight.

Partial similarities:

| Type | Partial similarity $s_{ijk}$ | Validity $\delta$ |
|---|---|---|
| Quantitative | $1 - \lvert x_i - x_j \rvert / R_k$ | 0 if either missing |
| Nominal | $\mathbb{1}[x_i = x_j]$ | 0 if either missing |
| Ordinal | rank-transform, then quantitative rule | 0 if either missing |
| Binary (symmetric) | $\mathbb{1}[x_i = x_j]$ | 0 if either missing |
| Binary (asymmetric) | $\mathbb{1}[x_i = x_j = \text{present}]$ | 0 if both absent or either missing |

Existing tools (`gower` on PyPI, R's `cluster::daisy`) are useful but have gaps this module addresses: limited type control, no first-class ordinal handling, weak missing-data semantics, no sparse/blocked computation for large $n$, and limited extensibility.

### Design goals
1. **Correctness first** — match `cluster::daisy` outputs within floating tolerance on reference datasets.
2. **Explicit typing** — never silently guess a column's type when the user wants control; infer sensibly by default.
3. **Performance** — vectorized NumPy core; optional chunked/parallel computation for large matrices.
4. **Extensibility** — pluggable per-type similarity functions.
5. **Good ergonomics** — pandas-friendly, scikit-learn-compatible.

### Non-goals
- Building clustering algorithms (we produce distance matrices that feed PAM, hierarchical, DBSCAN, etc.).
- GPU support in v1 (leave a clean seam for it).

---

## 2. Public API

```python
import numpy as np
import pandas as pd
from pygower import (
    gower_matrix,        # pairwise distance matrix
    gower_distances,     # X vs Y (or X vs X)
    gower_topn,          # top-n nearest neighbours, memory-efficient
    GowerDistance,       # sklearn-compatible transformer
    ColumnSpec,          # per-column configuration
)

# --- Quickest path: let the module infer types ---
D = gower_matrix(df)                       # (n, n) symmetric, zero diagonal

# --- X vs Y ---
D_xy = gower_distances(df_query, df_reference)   # (n_query, n_ref)

# --- Nearest neighbours without materializing the full matrix ---
idx, dist = gower_topn(df_query, df_reference, n=5)

# --- Full control over columns ---
spec = {
    "age":      ColumnSpec(kind="quantitative"),
    "income":   ColumnSpec(kind="quantitative", weight=2.0, range_=(0, 250_000)),
    "city":     ColumnSpec(kind="nominal"),
    "rating":   ColumnSpec(kind="ordinal", categories=["low", "med", "high"]),
    "is_member":ColumnSpec(kind="binary", asymmetric=True, present_value=True),
}
D = gower_matrix(df, spec=spec)

# --- sklearn pipeline ---
gd = GowerDistance(spec=spec).fit(df_train)
D_test = gd.transform(df_test)            # distances of test rows vs train rows
```

### Key signatures

```python
def gower_matrix(
    X: pd.DataFrame | np.ndarray,
    *,
    spec: Mapping[str, "ColumnSpec"] | None = None,
    weights: Sequence[float] | Mapping[str, float] | None = None,
    return_similarity: bool = False,
    n_jobs: int = 1,
    chunk_size: int | None = None,
    dtype: np.dtype = np.float64,
) -> np.ndarray: ...

def gower_distances(
    X, Y=None, *, spec=None, weights=None, n_jobs=1, chunk_size=None,
) -> np.ndarray: ...

def gower_topn(
    X, Y=None, *, n: int = 5, spec=None, weights=None, n_jobs=1,
) -> tuple[np.ndarray, np.ndarray]: ...   # (indices, distances)
```

---

## 3. Architecture

```
pygower/
├── __init__.py            # public exports
├── spec.py                # ColumnSpec, type inference
├── ranges.py              # range / scaling estimation, fit state
├── kernels.py             # per-type partial-similarity kernels (vectorized)
├── core.py                # aggregation engine (dense + chunked)
├── api.py                 # gower_matrix / gower_distances / gower_topn
├── sklearn.py             # GowerDistance transformer
├── validation.py          # input checks, NaN handling, dtype coercion
└── _types.py              # enums, typing aliases
```

**Data flow:** `validate input → resolve ColumnSpec per column → fit ranges/categories → encode columns into numeric arrays + type tags → kernel pass per column → weighted aggregation → distance matrix`.

The crucial separation is **fit** (learn ranges, category orderings, present-values) vs **transform** (apply them). This makes X-vs-Y and train/test correct: ranges must come from a fixed reference, not be recomputed per call.

---

## 4. Component design with code

### 4.1 `_types.py`

```python
from enum import Enum

class Kind(str, Enum):
    QUANTITATIVE = "quantitative"
    NOMINAL = "nominal"
    ORDINAL = "ordinal"
    BINARY = "binary"
```

### 4.2 `spec.py` — column configuration & inference

```python
from dataclasses import dataclass, field
from typing import Sequence
import numpy as np
import pandas as pd
from ._types import Kind

@dataclass
class ColumnSpec:
    kind: Kind | str
    weight: float = 1.0
    # quantitative
    range_: tuple[float, float] | None = None   # if None, learned at fit
    # ordinal
    categories: Sequence | None = None           # explicit order, low→high
    # binary
    asymmetric: bool = False
    present_value: object = True

    def __post_init__(self):
        self.kind = Kind(self.kind)
        if self.weight < 0:
            raise ValueError("weight must be non-negative")


def infer_spec(df: pd.DataFrame) -> dict[str, ColumnSpec]:
    """Heuristic type inference. Explicit user spec always overrides."""
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
```

### 4.3 `ranges.py` — fit state

```python
import numpy as np
import pandas as pd
from ._types import Kind

class FittedColumn:
    """Encodes one column to a numeric array + carries fit-time state."""
    def __init__(self, name, spec):
        self.name = name
        self.spec = spec
        self.range_: float | None = None
        self.categories_: dict | None = None  # value -> rank for ordinal

    def fit(self, s: pd.Series) -> "FittedColumn":
        k = self.spec.kind
        if k == Kind.QUANTITATIVE:
            if self.spec.range_ is not None:
                lo, hi = self.spec.range_
            else:
                lo, hi = np.nanmin(s.to_numpy(float)), np.nanmax(s.to_numpy(float))
            self.range_ = float(hi - lo) or 1.0   # guard zero range
        elif k == Kind.ORDINAL:
            cats = self.spec.categories or sorted(s.dropna().unique())
            self.categories_ = {c: i for i, c in enumerate(cats)}
            self.range_ = float(len(cats) - 1) or 1.0
        return self

    def encode(self, s: pd.Series) -> np.ndarray:
        """Return float array; NaN marks missing. Categoricals -> codes."""
        k = self.spec.kind
        if k in (Kind.QUANTITATIVE,):
            return s.to_numpy(dtype=float)
        if k == Kind.ORDINAL:
            return s.map(self.categories_).to_numpy(dtype=float)
        if k == Kind.BINARY:
            present = (s == self.spec.present_value)
            arr = present.to_numpy(dtype=float)
            arr[s.isna().to_numpy()] = np.nan
            return arr
        # nominal: factorize to integer codes, NaN preserved
        codes, _ = pd.factorize(s, use_na_sentinel=True)
        out = codes.astype(float)
        out[codes == -1] = np.nan
        return out
```

### 4.4 `kernels.py` — vectorized partial similarities

Each kernel takes one column of `X` (shape `(n,)`) and one of `Y` (shape `(m,)`) and returns two `(n, m)` arrays: partial **similarity** `s` and validity mask `delta`. Broadcasting does the pairwise expansion.

```python
import numpy as np
from ._types import Kind

def _missing(a, b):
    """delta=0 where either side is NaN."""
    return ~(np.isnan(a)[:, None] | np.isnan(b)[None, :])

def quantitative_kernel(a, b, *, range_, **_):
    diff = np.abs(a[:, None] - b[None, :]) / range_
    s = 1.0 - diff
    delta = _missing(a, b)
    return s, delta

def nominal_kernel(a, b, **_):
    s = (a[:, None] == b[None, :]).astype(float)
    delta = _missing(a, b)
    return s, delta

ordinal_kernel = quantitative_kernel   # operates on rank-encoded values

def binary_kernel(a, b, *, asymmetric, **_):
    eq = (a[:, None] == b[None, :])
    s = eq.astype(float)
    delta = _missing(a, b)
    if asymmetric:
        # exclude negative matches (both == 0/absent) from the average
        both_absent = (a[:, None] == 0) & (b[None, :] == 0)
        delta = delta & ~both_absent
    return s, delta

KERNELS = {
    Kind.QUANTITATIVE: quantitative_kernel,
    Kind.NOMINAL: nominal_kernel,
    Kind.ORDINAL: ordinal_kernel,
    Kind.BINARY: binary_kernel,
}
```

### 4.5 `core.py` — aggregation engine

```python
import numpy as np
from ._types import Kind
from .kernels import KERNELS

def _accumulate(cols_X, cols_Y, fitted, weights):
    """
    cols_X[c], cols_Y[c]: encoded 1-D arrays for column c.
    Returns distance matrix (n, m).
    """
    n = cols_X[0].shape[0]
    m = cols_Y[0].shape[0]
    num = np.zeros((n, m))          # sum of w * delta * (1 - s)
    den = np.zeros((n, m))          # sum of w * delta
    for c, fc in enumerate(fitted):
        kernel = KERNELS[fc.spec.kind]
        s, delta = kernel(
            cols_X[c], cols_Y[c],
            range_=fc.range_,
            asymmetric=fc.spec.asymmetric,
        )
        w = fc.spec.weight
        d = delta.astype(float)
        num += w * d * (1.0 - s)
        den += w * d
    with np.errstate(invalid="ignore", divide="ignore"):
        D = np.where(den > 0, num / den, np.nan)  # all-missing pair -> NaN
    return D


def gower_full(cols_X, cols_Y, fitted, weights, chunk_size=None):
    """Dense or row-chunked aggregation to bound memory at O(chunk * m)."""
    n = cols_X[0].shape[0]
    m = cols_Y[0].shape[0]
    if chunk_size is None:
        return _accumulate(cols_X, cols_Y, fitted, weights)
    D = np.empty((n, m))
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        sub_X = [c[start:stop] for c in cols_X]
        D[start:stop] = _accumulate(sub_X, cols_Y, fitted, weights)
    return D
```

### 4.6 `api.py` — top-level functions

```python
import numpy as np
import pandas as pd
from .spec import infer_spec, ColumnSpec
from .ranges import FittedColumn
from .core import gower_full
from .validation import to_frame, check_same_columns

def _fit_columns(df, spec):
    spec = spec or infer_spec(df)
    fitted = [FittedColumn(col, spec[col]).fit(df[col]) for col in df.columns]
    return fitted, spec

def _encode(df, fitted):
    return [fc.encode(df[fc.name]) for fc in fitted]

def gower_distances(X, Y=None, *, spec=None, weights=None,
                    n_jobs=1, chunk_size=None):
    X = to_frame(X)
    fitted, spec = _fit_columns(X, spec)          # ranges learned from X
    if Y is None:
        Y = X
    else:
        Y = to_frame(Y)
        check_same_columns(X, Y)
    cols_X = _encode(X, fitted)
    cols_Y = _encode(Y, fitted)
    return gower_full(cols_X, cols_Y, fitted, weights, chunk_size=chunk_size)

def gower_matrix(X, **kw):
    D = gower_distances(X, None, **kw)
    np.fill_diagonal(D, 0.0)       # enforce exact zero self-distance
    return D

def gower_topn(X, Y=None, *, n=5, **kw):
    D = gower_distances(X, Y, **kw)
    idx = np.argsort(D, axis=1)[:, :n]
    dist = np.take_along_axis(D, idx, axis=1)
    return idx, dist
```

> For `gower_topn` at very large scale, replace the materialize-then-sort with a row-chunked pass that keeps only the running top-n per query row (a heap or partial argsort per chunk), so peak memory is `O(chunk * m)` rather than `O(n * m)`.

### 4.7 `sklearn.py` — transformer

```python
from sklearn.base import BaseEstimator, TransformerMixin
from .api import _fit_columns, _encode
from .core import gower_full

class GowerDistance(BaseEstimator, TransformerMixin):
    """Learns ranges/categories on fit; transform = distances vs the fit data."""
    def __init__(self, spec=None, weights=None, chunk_size=None):
        self.spec = spec
        self.weights = weights
        self.chunk_size = chunk_size

    def fit(self, X, y=None):
        from .validation import to_frame
        self._X = to_frame(X)
        self.fitted_, self.spec_ = _fit_columns(self._X, self.spec)
        self._cols_ref = _encode(self._X, self.fitted_)
        return self

    def transform(self, X):
        from .validation import to_frame, check_same_columns
        Xf = to_frame(X)
        check_same_columns(self._X, Xf)
        cols_X = _encode(Xf, self.fitted_)
        return gower_full(cols_X, self._cols_ref, self.fitted_,
                          self.weights, chunk_size=self.chunk_size)
```

### 4.8 `validation.py`

```python
import numpy as np
import pandas as pd

def to_frame(X):
    if isinstance(X, pd.DataFrame):
        return X
    if isinstance(X, np.ndarray):
        return pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    raise TypeError(f"Unsupported input type: {type(X)!r}")

def check_same_columns(X, Y):
    if list(X.columns) != list(Y.columns):
        raise ValueError("X and Y must have identical columns in the same order")
```

---

## 5. Correctness considerations

- **Zero-range numeric columns:** a constant column has range 0 → division blows up. Guard by treating range 0 as 1.0; all pairs then get partial distance 0, which is correct (no information to separate them).
- **Symmetric matrix & exact zero diagonal:** floating accumulation can leave the diagonal at `~1e-17`; force it to 0 in `gower_matrix`.
- **All-missing pairs:** if `den == 0` for a pair (every shared variable missing), the distance is undefined. Return NaN rather than 0 and document it; downstream clusterers must decide how to handle.
- **Range from reference only:** in X-vs-Y / train-test, ranges and category orderings must be fit on the reference and reused, never recomputed on the query — otherwise distances are not comparable across calls.
- **Asymmetric binary:** the "present" value must be explicit; defaulting to truthiness is a common bug source.

---

## 6. Performance plan

| Lever | Approach |
|---|---|
| Vectorization | All kernels broadcast to `(n, m)`; no Python-level pair loops. |
| Memory | `chunk_size` bounds peak allocation to `O(chunk * m)` per kernel. |
| Parallelism | `joblib.Parallel` over row chunks; `n_jobs` threads/processes. |
| Dtype | `float32` option roughly halves memory for large matrices. |
| Top-n | Streaming partial-sort to avoid full matrix when only neighbours needed. |
| Optional acceleration | Numba `@njit` on the inner accumulation as a later optimization; keep the pure-NumPy path as reference/fallback. |

Complexity is `O(n · m · p)` time for `p` columns and `O(n · m)` memory (or `O(chunk · m)` chunked).

---

## 7. Testing strategy

1. **Golden tests vs R `cluster::daisy`** on canonical mixed datasets (e.g. `flower`); assert agreement within `1e-10`.
2. **Property tests** (Hypothesis):
   - `D[i,i] == 0`, symmetry `D == D.T`, range `0 ≤ D ≤ 1`.
   - Permuting rows permutes the matrix consistently.
   - Adding a constant column leaves distances unchanged.
3. **Missing-data tests:** NaN exclusion matches a hand-computed reference; all-missing pair → NaN.
4. **Type-specific unit tests** for each kernel, including asymmetric binary negative-match exclusion and ordinal rank encoding.
5. **Fit/transform invariance:** `GowerDistance().fit_transform(X)` equals `gower_matrix(X)`; train/test ranges come only from train.
6. **Performance regression:** benchmark `n=2000` stays under a threshold; chunked path equals dense path numerically.

---

## 8. Roadmap

- **v0.1** — dense `gower_matrix`/`gower_distances`, inference, all four kernels, golden tests.
- **v0.2** — `GowerDistance` transformer, weights, chunking, `gower_topn`.
- **v0.3** — joblib parallelism, float32, streaming top-n.
- **v0.4** — Numba inner loop, optional sparse-categorical handling, docs site.

---

## 9. Open questions

- Should weights be normalized to sum to 1, or left raw (affects only interpretability, not ranking)?
- Default ordinal behavior when an unseen category appears at transform time: error, or treat as missing? (Lean toward explicit error with an opt-in `handle_unknown="missing"`.)
- Expose similarity (`1 - d`) as a first-class output everywhere, or only via `return_similarity`?
