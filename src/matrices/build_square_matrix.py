"""Phase 02 -- build the square unified-weight ``Z`` matrix.

Phase 02 is the **only** owner of the pre/post -> unified-weight computation::

    s = (|pre_z| + |post_z|) / 2
    g = tanh(alpha * log(post_z / (pre_z + eps)))
    w = s * g

It consumes the Phase 01 artifact ``parsed_graph.json`` (``pairs`` carry
``pre_z`` / ``post_z`` verbatim) and produces the dense square matrix ``Z``
(rows = source neurons, columns = target neurons, missing edges = ``0``,
diagonal = ``0``).  Phase 02 performs no spectral analysis, no clustering and no
pair filtering -- those belong to later phases.

Pipeline order (fixed, and recorded in ``config``)::

    fuse (unified weight)  ->  assemble Z  ->  normalize  ->  symmetrize

Artifacts written per input (``<stem>`` = the source ``.gv`` stem, so Phase 02
lands next to its input)::

    <outdir>/<stem>/z_matrix.json         # canonical, self-sufficient JSON
    <outdir>/<stem>/z_matrix.npz          # derived array cache (--no-sidecar opts out)
    <outdir>/<stem>/z_matrix.png          # heatmap                     (--plot)
    <outdir>/<stem>/z_matrix.data.json    # per-pair weights + stats   (--save-data)

For a **non-default** matrix config the variant segment ``.<config_hash8>`` is
inserted (``z_matrix.<config_hash8>.json``) so a variant run can never clobber
the canonical artifact; ``--config-hash`` forces the segment even for a default
config.  A Phase 01 *filtered* input adds ``.filtered`` instead.  No CSV files
are produced or consumed anywhere in this project -- JSON (plus the ``.npz``
array cache) is the interchange format.

CLI
---
::

    python -m src.matrices.build_square_matrix \\
        -i data/processed/FB4Yaffect_FB45_999prePost_001_all/parsed_graph.json \\
        -o data/processed \\
        [--glob 'parsed_graph.json'] [--include-filtered] \\
        [--eps 0.1] [--alpha 1.0] [--zero-policy {zero-in-zero-out,formula}] \\
        [--symmetric] [--symmetrize {mean,sum,max-abs,min-abs}] \\
        [--normalize {none,rows,cols,unit,spectral,zscore-nonzero}] \\
        [--config-hash] [--no-sidecar] \\
        [--plot] [--no-popup] [--cmap viridis] [--interactive] \\
        [--save-data] [--stats] [--hist-bins 10] \\
        [--strict] [--dry-run] [--log-level INFO]

Terminal behaviour
------------------
``--stats`` prints a statistics block and **always** prints a boxed Phase 02
completion summary; both are terminal-only.  ``--plot`` and ``--interactive``
are the only options that open a GUI window -- both are skipped automatically
on a non-interactive backend.  ``--plot`` saves ``z_matrix.png`` and, on an
interactive backend, shows it with ``plt.show()`` (``--no-popup`` forces the
window off).  ``--interactive`` opens the same heatmap in an
:mod:`mplcursors` window whose hover tooltips report the source neuron, the
target neuron, the unified weight and the ``(i, j)`` indices of the hovered
cell; it writes no artifact of its own.  When both flags are given the plain
``--plot`` popup is suppressed, so exactly one window opens.

Loading (Phases 03+)
--------------------
* :func:`load_z_matrix` -- integrity-checked object (verifies the ``.npz`` cache
  against the canonical JSON's SHA-256, ignoring a stale cache with a warning).
* :func:`load_sidecar_arrays` -- fast array access, ``allow_pickle=False``.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import logging
import math
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from src.parsing.parse_graphviz import (
    DEFAULT_ARTIFACT_NAME as PARSED_ARTIFACT_NAME,
    FILTERED_ARTIFACT_NAME as PARSED_FILTERED_ARTIFACT_NAME,
    ValidationReport,
    validate_payload_schema,
)
from src.utils.io import (
    dumps_json,
    ensure_dir,
    read_json,
    sha256_file,
    utc_timestamp,
    write_json_atomic,
)

__all__ = [
    "CANONICAL_STEM",
    "CODE_ARTIFACT_WRITE",
    "CODE_CONFIG",
    "CONFIG_KEYS",
    "CODE_DIAGONAL",
    "CODE_DUPLICATE_COORDINATE",
    "CODE_EMPTY_PAIRS",
    "CODE_ENDPOINT_NOT_NEURON",
    "CODE_INPUT_SCHEMA",
    "CODE_MATRIX_SHAPE",
    "CODE_NEURON_ID",
    "CODE_NONFINITE",
    "CODE_PAYLOAD_SCHEMA",
    "CODE_PLOT",
    "CODE_SELF_LOOP",
    "CODE_SIDECAR_CACHE",
    "CODE_SMALL_GRAPH",
    "CODE_SYMMETRY",
    "CODE_Z_INVALID",
    "DEFAULT_ALPHA",
    "DEFAULT_CMAP",
    "DEFAULT_DATA_NAME",
    "DEFAULT_EPS",
    "DEFAULT_HIST_BINS",
    "DEFAULT_NORMALIZATION",
    "DEFAULT_PLOT_NAME",
    "DEFAULT_SIDECAR_NAME",
    "DEFAULT_SYMMETRIZE",
    "DEFAULT_ZERO_POLICY",
    "INTERACTIVE_ANNOTATION_KWARGS",
    "MATRIX_CONFIG_FIELDS",
    "METADATA_KEYS",
    "NON_INTERACTIVE_BACKENDS",
    "NORMALIZATIONS",
    "PROVENANCE_KEYS",
    "SAVE_DATA_KEYS",
    "STATS_KEYS",
    "SYMMETRIZE_METHODS",
    "TOP_LEVEL_KEYS",
    "UNIFICATION_RULE",
    "WEIGHTED_PAIR_KEYS",
    "WEIGHT_STATS_KEYS",
    "ZERO_POLICIES",
    "ZMatrix",
    "ZMatrixConfig",
    "ZMatrixValidationError",
    "WeightedPair",
    "artifact_paths",
    "attach_cell_cursor",
    "build_argument_parser",
    "build_interactive_figure",
    "build_sidecar_arrays",
    "build_z_matrix",
    "build_z_matrix_from_file",
    "can_popup",
    "compute_diagnostics",
    "format_cell_tooltip",
    "load_sidecar_arrays",
    "load_z_matrix",
    "main",
    "normalize_matrix",
    "plot_matrix",
    "render_statistics",
    "render_summary_box",
    "resolve_inputs",
    "save_data_payload",
    "show_interactive_matrix",
    "sidecar_path",
    "symmetrize",
    "unified_weight",
    "unified_weights",
    "validate_z_payload_schema",
    "write_artifact_set",
    "write_npz_atomic",
    "write_sidecar",
]

LOGGER = logging.getLogger("src.matrices.build_square_matrix")

# ---------------------------------------------------------------------------
# Artifact names and format constants
# ---------------------------------------------------------------------------
CANONICAL_STEM = "z_matrix"
DEFAULT_ARTIFACT_NAME = f"{CANONICAL_STEM}.json"
DEFAULT_SIDECAR_NAME = f"{CANONICAL_STEM}.npz"
DEFAULT_PLOT_NAME = f"{CANONICAL_STEM}.png"
DEFAULT_DATA_NAME = f"{CANONICAL_STEM}.data.json"
DEFAULT_INPUT_PATTERN = PARSED_ARTIFACT_NAME
SUPPORTED_SUFFIX = ".json"

#: Unification-rule defaults (see ``execution-plans/zscore_context.md``).
DEFAULT_EPS = 0.1
DEFAULT_ALPHA = 1.0
ZERO_POLICIES = ("zero-in-zero-out", "formula")
DEFAULT_ZERO_POLICY = "zero-in-zero-out"
SYMMETRIZE_METHODS = ("mean", "sum", "max-abs", "min-abs")
DEFAULT_SYMMETRIZE = "mean"
NORMALIZATIONS = ("none", "rows", "cols", "unit", "spectral", "zscore-nonzero")
DEFAULT_NORMALIZATION = "none"
DEFAULT_DTYPE = "float64"
DEFAULT_CMAP = "viridis"
DEFAULT_HIST_BINS = 10

#: Backends that can only render to a file -- matplotlib's own non-interactive set,
#: used as the classification fallback when the backend registry is unavailable.
NON_INTERACTIVE_BACKENDS = frozenset({"agg", "cairo", "pdf", "pgf", "ps", "svg", "template"})

#: Tooltip (annotation) styling for the ``--interactive`` mplcursors hover cursor.
#: Mirrors mplcursors' own default look (round yellow box, short arrow) with a
#: slightly more opaque background so long neuron ids stay readable over the
#: heatmap's warm/cold cells.
INTERACTIVE_ANNOTATION_KWARGS: dict[str, Any] = {
    "bbox": {"boxstyle": "round,pad=0.5", "fc": "lightyellow", "ec": "black", "alpha": 0.92},
    "arrowprops": {"arrowstyle": "->", "shrinkB": 0},
}

#: Human-readable form of the rule, embedded in every artifact for provenance.
UNIFICATION_RULE = "w = s * tanh(alpha * log(post_z / (pre_z + eps))), s = (|pre_z| + |post_z|) / 2"

#: Every field that *defines the matrix* (and is therefore part of the config
#: hash).  Presentation/IO flags (``--plot``, ``--save-data``, ``--stats``,
#: ``--no-sidecar``, ``--config-hash``, ``--dry-run``) are deliberately excluded:
#: they do not change a single matrix entry, and hashing them would stop a
#: default-config run from producing the canonical ``z_matrix.json``.
MATRIX_CONFIG_FIELDS = (
    "unification",
    "eps",
    "alpha",
    "zero_policy",
    "symmetric",
    "symmetrize",
    "normalize",
    "dtype",
    "filters",
)

#: ``is_default()`` compares only these CLI-controlled fields (``filters`` is
#: hashed but must not force a hash segment: a filtered input already gets the
#: ``.filtered`` segment from the artifact name).
DEFAULT_DECISION_FIELDS = ("eps", "alpha", "zero_policy", "symmetric", "symmetrize", "normalize", "dtype")

#: Projection of ``pairs`` written by :meth:`WeightedPair.to_dict`.
WEIGHTED_PAIR_KEYS = frozenset({"source", "target", "pre_z", "post_z", "weight"})
PROVENANCE_KEYS = frozenset(
    {
        "source_artifact",
        "source_artifact_sha256",
        "source_file",
        "source_file_sha256",
        "parsed_created_utc",
        "phase",
    }
)
CONFIG_KEYS = frozenset(
    {"unification", "eps", "alpha", "zero_policy", "symmetric", "symmetrize", "normalize", "dtype", "filters", "config_hash"}
)
TOP_LEVEL_KEYS = frozenset(
    {"provenance", "config", "neuron_order", "pairs", "matrix", "matrix_symmetric", "metadata"}
)
STATS_KEYS = frozenset({"min", "max", "mean", "std"})
WEIGHT_STATS_KEYS = frozenset({"min", "max", "mean", "mean_abs", "std", "sum"})
METADATA_KEYS = frozenset(
    {
        "n_neurons",
        "n_edges",
        "n_stored_nonzero",
        "n_matrix_zeros",
        "n_weight_positive",
        "n_weight_negative",
        "n_weight_zero",
        "n_self_loops",
        "sparsity",
        "density",
        "reciprocity",
        "matrix_stats",
        "weight_stats",
        "row_norm_stats",
        "col_norm_stats",
        "frobenius_norm",
        "largest_singular_value",
        "spectral_radius",
        "symmetrized_spectral_radius",
        "normalization_scale",
        "config_hash",
        "numpy_version",
        "generator",
        "created_utc",
    }
)
SAVE_DATA_KEYS = frozenset(
    {
        "provenance",
        "config",
        "symmetric",
        "normalize",
        "pairs",
        "z_stats",
        "histogram",
        "sparsity",
        "n_positive",
        "n_negative",
        "n_zero",
        "row_norms",
        "col_norms",
        "row_norm_stats",
        "col_norm_stats",
        "spectral_radius",
        "density",
        "reciprocity",
        "generator",
        "created_utc",
    }
)

SEVERITY_ERROR = "error"
SEVERITY_WARNING = "warning"

# Validation issue codes (one per rule).
CODE_INPUT_SCHEMA = "input_schema"
CODE_NEURON_ID = "neuron_id"
CODE_ENDPOINT_NOT_NEURON = "endpoint_not_neuron"
CODE_DUPLICATE_COORDINATE = "duplicate_coordinate"
CODE_SELF_LOOP = "self_loop"
CODE_Z_INVALID = "z_invalid"
CODE_CONFIG = "config"
CODE_MATRIX_SHAPE = "matrix_shape"
CODE_DIAGONAL = "diagonal"
CODE_SYMMETRY = "symmetry"
CODE_NONFINITE = "nonfinite"
CODE_PAYLOAD_SCHEMA = "payload_schema"
CODE_EMPTY_PAIRS = "empty_pairs"
CODE_SMALL_GRAPH = "small_graph"
CODE_SIDECAR_CACHE = "sidecar_cache"
CODE_ARTIFACT_WRITE = "artifact_write"
CODE_PLOT = "plot"

GENERATOR = "src.matrices.build_square_matrix"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ZMatrixConfig:
    """The complete, hashable description of *how* ``Z`` was built."""

    eps: float = DEFAULT_EPS
    alpha: float = DEFAULT_ALPHA
    zero_policy: str = DEFAULT_ZERO_POLICY
    symmetric: bool = False
    symmetrize: str = DEFAULT_SYMMETRIZE
    normalize: str = DEFAULT_NORMALIZATION
    dtype: str = DEFAULT_DTYPE
    #: Phase 01 ``metadata.filters`` of the input artifact (provenance only).
    filters: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON payload written to ``config``."""
        payload = self.hash_fields()
        payload["config_hash"] = self.config_hash()
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ZMatrixConfig":
        """Rebuild a config from an artifact payload (ignores derived keys)."""
        return cls(
            eps=float(payload.get("eps", DEFAULT_EPS)),
            alpha=float(payload.get("alpha", DEFAULT_ALPHA)),
            zero_policy=str(payload.get("zero_policy", DEFAULT_ZERO_POLICY)),
            symmetric=bool(payload.get("symmetric", False)),
            symmetrize=str(payload.get("symmetrize", DEFAULT_SYMMETRIZE)),
            normalize=str(payload.get("normalize", DEFAULT_NORMALIZATION)),
            dtype=str(payload.get("dtype", DEFAULT_DTYPE)),
            filters=list(payload.get("filters") or []),
        )

    def hash_fields(self) -> dict[str, Any]:
        """The matrix-defining subset hashed into ``config_hash``.

        Built directly (never via :meth:`to_dict`) -- ``to_dict`` adds
        ``config_hash``, so routing through it here would recurse forever.
        """
        return {
            "unification": UNIFICATION_RULE,
            "eps": float(self.eps),
            "alpha": float(self.alpha),
            "zero_policy": self.zero_policy,
            "symmetric": bool(self.symmetric),
            "symmetrize": self.symmetrize,
            "normalize": self.normalize,
            "dtype": self.dtype,
            "filters": [dict(item) for item in self.filters],
        }

    def config_hash(self) -> str:
        """Deterministic 8-hex-character digest of :meth:`hash_fields`."""
        digest = hashlib.sha256(dumps_json(self.hash_fields()).encode("utf-8")).hexdigest()
        return digest[:8]

    def is_default(self) -> bool:
        """True when every CLI-controlled matrix parameter is at its default."""
        return all(
            getattr(self, name) == getattr(_DEFAULT_CONFIG, name) for name in DEFAULT_DECISION_FIELDS
        )


#: Reference instance used by :meth:`ZMatrixConfig.is_default` comparisons.
_DEFAULT_CONFIG = ZMatrixConfig()


@dataclass(frozen=True)
class WeightedPair:
    """One collapsed functional edge and its unified weight."""

    source: str
    target: str
    pre_z: float
    post_z: float
    weight: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "pre_z": float(self.pre_z),
            "post_z": float(self.post_z),
            "weight": float(self.weight),
        }


@dataclass
class ZMatrix:
    """In-memory Phase 02 artifact.

    ``matrix`` is the **effective** matrix: ``matrix_symmetric`` when
    ``config.symmetric`` is set, otherwise the normalized directed ``Z``.
    ``matrix_symmetric`` is always the symmetrized matrix (identical to
    ``matrix`` in symmetric mode), so Phase 03's eigen path always has a
    symmetric matrix available.
    """

    source_artifact: Path
    source_artifact_sha256: str
    source_file: str
    source_file_sha256: str
    parsed_created_utc: str
    config: ZMatrixConfig
    neuron_order: list[str]
    pairs: list[WeightedPair]
    matrix: np.ndarray
    matrix_symmetric: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    #: Where this object was read from (JSON or NPZ), ``None`` for a freshly built
    #: one.  Deliberately *not* serialized: ``provenance`` always describes the
    #: Phase 01 input artifact, so ``to_dict()`` round-trips byte-for-byte.
    loaded_from: Path | None = None

    @property
    def n_neurons(self) -> int:
        return len(self.neuron_order)

    @property
    def n_edges(self) -> int:
        return len(self.pairs)

    @property
    def index(self) -> dict[str, int]:
        """Neuron-id -> row/column index lookup."""
        return {name: position for position, name in enumerate(self.neuron_order)}

    def to_dict(self) -> dict[str, Any]:
        """Return the exact JSON payload written to ``z_matrix.json``."""
        return {
            "provenance": {
                "source_artifact": str(self.source_artifact),
                "source_artifact_sha256": self.source_artifact_sha256,
                "source_file": self.source_file,
                "source_file_sha256": self.source_file_sha256,
                "parsed_created_utc": self.parsed_created_utc,
                "phase": "02",
            },
            "config": self.config.to_dict(),
            "neuron_order": list(self.neuron_order),
            "pairs": [pair.to_dict() for pair in self.pairs],
            "matrix": self.matrix.tolist(),
            "matrix_symmetric": self.matrix_symmetric.tolist(),
            "metadata": dict(self.metadata),
        }


class ZMatrixValidationError(ValueError):
    """Raised when a Phase 02 input/output violates a validation rule.

    The full :class:`src.parsing.parse_graphviz.ValidationReport` is available as
    :attr:`report`, mirroring ``GraphvizValidationError`` from Phase 01.
    """

    def __init__(self, message: str, report: ValidationReport | None = None) -> None:
        super().__init__(message)
        self.report = report


# ---------------------------------------------------------------------------
# Unification rule
# ---------------------------------------------------------------------------
def unified_weights(
    pre_z: Any,
    post_z: Any,
    *,
    eps: float = DEFAULT_EPS,
    alpha: float = DEFAULT_ALPHA,
    zero_policy: str = DEFAULT_ZERO_POLICY,
) -> np.ndarray:
    """Vectorized unified weight ``w = s * tanh(alpha * log(post/(pre+eps)))``.

    ``s = (|pre_z| + |post_z|) / 2``.  ``zero_policy = "zero-in-zero-out"``
    (default) maps any pair with ``pre_z == 0`` or ``post_z == 0`` to exactly
    ``0.0`` -- a zero z-score means "no measurable relationship", whereas the raw
    formula would drive it to the extreme attenuation limit ``-s``.  Negative
    zero is canonicalized to ``0.0`` so JSON and hashed bytes stay stable.
    """
    pre = np.asarray(pre_z, dtype=np.float64)
    post = np.asarray(post_z, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        strength = (np.abs(pre) + np.abs(post)) / 2.0
        gain = np.tanh(float(alpha) * np.log(post / (pre + float(eps))))
        weights = strength * gain
    if zero_policy == "zero-in-zero-out":
        weights = np.where((pre == 0.0) | (post == 0.0), 0.0, weights)
    # `-0.0` serializes as "-0.0" and would break byte-level determinism.
    return np.where(weights == 0.0, 0.0, weights)


def unified_weight(
    pre_z: float,
    post_z: float,
    *,
    eps: float = DEFAULT_EPS,
    alpha: float = DEFAULT_ALPHA,
    zero_policy: str = DEFAULT_ZERO_POLICY,
) -> float:
    """Scalar unified weight; delegates to :func:`unified_weights` for parity."""
    values = unified_weights(
        np.array([pre_z], dtype=np.float64),
        np.array([post_z], dtype=np.float64),
        eps=eps,
        alpha=alpha,
        zero_policy=zero_policy,
    )
    return float(values[0])


# ---------------------------------------------------------------------------
# Normalization and symmetrization
# ---------------------------------------------------------------------------
def normalize_matrix(
    matrix: Any, *, method: str = DEFAULT_NORMALIZATION
) -> tuple[np.ndarray, dict[str, Any]]:
    """Normalize *matrix* and return ``(normalized, info)``.

    ``none``        -- identity (the default)
    ``rows``        -- divide each row by its L1 norm ``sum_j |Z_ij|``
    ``cols``        -- divide each column by its L1 norm
    ``unit``        -- divide every entry by ``max |Z|`` (bounded to ``[-1, 1]``)
    ``spectral``    -- divide by the largest singular value
    ``zscore-nonzero`` -- standardize the *stored* weights in place; zeros stay
                          zero (a global standardisation would densify the matrix)

    All-zero inputs are returned unchanged (no division by zero).
    """
    values = np.asarray(matrix, dtype=np.float64)
    if method == "none":
        return values.copy(), {"method": "none", "kind": "identity", "scale": 1.0}

    if method in {"rows", "cols"}:
        axis = 1 if method == "rows" else 0
        norms = np.abs(values).sum(axis=axis, keepdims=True)
        scale = np.zeros_like(norms)
        np.divide(1.0, norms, out=scale, where=norms > 0)
        return values * scale, {"method": method, "kind": "l1-vector", "scale": None}

    if method == "unit":
        peak = float(np.abs(values).max()) if values.size else 0.0
        if peak == 0.0:
            return values.copy(), {"method": method, "kind": "scalar", "scale": None}
        factor = 1.0 / peak
        return values * factor, {"method": method, "kind": "scalar", "scale": factor}

    if method == "spectral":
        if values.size == 0 or not np.any(values):
            return values.copy(), {"method": method, "kind": "scalar", "scale": None}
        largest = float(np.linalg.svd(values, compute_uv=False)[0])
        if largest == 0.0:
            return values.copy(), {"method": method, "kind": "scalar", "scale": None}
        factor = 1.0 / largest
        return values * factor, {"method": method, "kind": "scalar", "scale": factor}

    if method == "zscore-nonzero":
        stored = values != 0
        if not stored.any():
            return values.copy(), {"method": method, "kind": "nonzero-standardize", "scale": None}
        sample = values[stored]
        mean = float(sample.mean())
        deviation = float(sample.std())
        result = values.copy()
        if deviation > 0.0:
            result[stored] = (sample - mean) / deviation
            return result, {
                "method": method,
                "kind": "nonzero-standardize",
                "scale": deviation,
                "mean": mean,
            }
        result[stored] = 0.0
        return result, {"method": method, "kind": "nonzero-standardize", "scale": None, "mean": mean}

    raise ZMatrixValidationError(f"unknown normalization method: {method!r}")


def symmetrize(matrix: Any, *, method: str = DEFAULT_SYMMETRIZE) -> np.ndarray:
    """Return a symmetric matrix built from *matrix*.

    ``mean``    -- ``(Z + Z.T) / 2`` (default; exactly symmetric in IEEE-754)
    ``sum``     -- ``Z + Z.T`` (= 2 x ``mean``)
    ``max-abs`` -- entry with the larger magnitude, sign preserved
    ``min-abs`` -- entry with the smaller magnitude, sign preserved

    Symmetric matrices have real eigenvalues and orthogonal eigenvectors, which
    is what Phase 03's eigen-decomposition path requires.
    """
    values = np.asarray(matrix, dtype=np.float64)
    transpose = values.T
    if method == "mean":
        return (values + transpose) / 2.0
    if method == "sum":
        return values + transpose
    if method == "max-abs":
        return np.where(np.abs(values) >= np.abs(transpose), values, transpose)
    if method == "min-abs":
        return np.where(np.abs(values) <= np.abs(transpose), values, transpose)
    raise ZMatrixValidationError(f"unknown symmetrization method: {method!r}")


# ---------------------------------------------------------------------------
# Diagnostics (single source of truth for metadata, --stats and --save-data)
# ---------------------------------------------------------------------------
def _summary(values: np.ndarray) -> dict[str, float | None]:
    """``min/max/mean/std`` (ddof=0) over *values*, or ``None`` when empty."""
    if values.size == 0:
        return {"min": None, "max": None, "mean": None, "std": None}
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
    }


def _histogram(values: np.ndarray, bins: int) -> dict[str, Any]:
    """Histogram of the *stored* (non-zero) entries of *values*."""
    zeros = int(np.count_nonzero(values == 0)) if values.size else 0
    stored = values[values != 0]
    if stored.size == 0 or bins < 1:
        return {"bins": [], "counts": [], "n_bins": 0, "range": None, "zero_count": zeros, "degenerate": True}

    first = float(stored.min())
    last = float(stored.max())
    # A numerically degenerate range (e.g. every stored weight is -1.0 after
    # `--normalize rows`, or -0.5 after symmetrizing it) cannot be split into
    # `bins` finite-sized bins: numpy raises "Too many bins for data range"
    # because the float spacing is wider than the span.  Report a single bin
    # holding every stored value instead.  The span test is relative so it
    # behaves the same for tiny and huge weight scales.
    scale = max(abs(first), abs(last), 1.0)
    if last <= first or (last - first) < 1e-12 * scale * int(bins):
        edge = float(stored[0])
        return {
            "bins": [edge, edge],
            "counts": [int(stored.size)],
            "n_bins": 1,
            "range": [edge, edge],
            "zero_count": zeros,
            "degenerate": True,
        }
    counts, edges = np.histogram(stored, bins=int(bins))
    return {
        "bins": [float(edge) for edge in edges],
        "counts": [int(count) for count in counts],
        "n_bins": int(bins),
        "range": [float(edges[0]), float(edges[-1])],
        "zero_count": zeros,
        "degenerate": False,
    }


def compute_diagnostics(
    matrix: Any,
    matrix_symmetric: Any,
    weights: Any,
    *,
    symmetric: bool,
    density: float | None = None,
    reciprocity: int | None = None,
    hist_bins: int = DEFAULT_HIST_BINS,
) -> dict[str, Any]:
    """Compute every statistic Phase 02 reports (metadata / ``--stats`` / ``--save-data``).

    Conventions (documented in the Phase 02 plan):

    * ``matrix_stats`` is over the **stored (non-zero) entries** of the effective
      matrix; the absent-edge zeros are summarized by ``sparsity`` instead.
    * ``weight_stats`` is over the raw unified weights of *all* pairs (zeros included).
    * ``row_norms`` / ``col_norms`` are L1 norms (``sum_j |Z_ij|``).
    * ``spectral_radius`` is ``max |eigenvalue|`` of the **effective** matrix and is
      ``None`` unless symmetric mode was used; ``symmetrized_spectral_radius`` is the
      same quantity for ``matrix_symmetric`` and is always available (informational).
    """
    values = np.asarray(matrix, dtype=np.float64)
    symmetric_values = np.asarray(matrix_symmetric, dtype=np.float64)
    unified = np.asarray(weights, dtype=np.float64)

    stored = values[values != 0]
    n_entries = int(values.size)
    n_stored = int(np.count_nonzero(values))
    sparsity = float((n_entries - n_stored) / n_entries) if n_entries else 0.0

    row_norms = np.abs(values).sum(axis=1) if values.size else np.zeros(0)
    col_norms = np.abs(values).sum(axis=0) if values.size else np.zeros(0)

    spectral_radius: float | None = None
    if symmetric and symmetric_values.size:
        spectral_radius = float(np.abs(np.linalg.eigvalsh(symmetric_values)).max())

    symmetrized_radius = (
        float(np.abs(np.linalg.eigvalsh(symmetric_values)).max()) if symmetric_values.size else None
    )
    largest_singular = (
        float(np.linalg.svd(values, compute_uv=False)[0]) if values.size and np.any(values) else 0.0
    )

    return {
        "shape": [int(values.shape[0]), int(values.shape[1])] if values.ndim == 2 else [int(values.size)],
        "n_entries": n_entries,
        "n_stored_nonzero": n_stored,
        "n_matrix_zeros": n_entries - n_stored,
        "sparsity": sparsity,
        "n_weight_positive": int((stored > 0).sum()),
        "n_weight_negative": int((stored < 0).sum()),
        "n_weight_zero": int((unified == 0).sum()),
        "matrix_stats": _summary(stored),
        "weight_stats": {
            **_summary(unified),
            "mean_abs": float(np.abs(unified).mean()) if unified.size else None,
            "sum": float(unified.sum()) if unified.size else 0.0,
        },
        "row_norms": [float(value) for value in row_norms],
        "col_norms": [float(value) for value in col_norms],
        "row_norm_stats": _summary(row_norms),
        "col_norm_stats": _summary(col_norms),
        "frobenius_norm": float(np.linalg.norm(values)) if values.size else 0.0,
        "largest_singular_value": largest_singular,
        "spectral_radius": spectral_radius,
        "symmetrized_spectral_radius": symmetrized_radius,
        "histogram": _histogram(values, hist_bins),
        "density": None if density is None else float(density),
        "reciprocity": None if reciprocity is None else int(reciprocity),
    }


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------
def _validate_config(config: ZMatrixConfig, report: ValidationReport) -> None:
    """Validate the unification / normalization / symmetrization parameters."""
    if not (isinstance(config.eps, (int, float)) and math.isfinite(config.eps) and config.eps > 0):
        report.error(CODE_CONFIG, f"eps must be finite and > 0 (needed for the ratio), got {config.eps!r}")
    if not (isinstance(config.alpha, (int, float)) and math.isfinite(config.alpha) and config.alpha > 0):
        report.error(CODE_CONFIG, f"alpha must be finite and > 0, got {config.alpha!r}")
    if config.zero_policy not in ZERO_POLICIES:
        report.error(CODE_CONFIG, f"zero_policy must be one of {ZERO_POLICIES}, got {config.zero_policy!r}")
    if config.symmetrize not in SYMMETRIZE_METHODS:
        report.error(CODE_CONFIG, f"symmetrize must be one of {SYMMETRIZE_METHODS}, got {config.symmetrize!r}")
    if config.normalize not in NORMALIZATIONS:
        report.error(CODE_CONFIG, f"normalize must be one of {NORMALIZATIONS}, got {config.normalize!r}")
    if config.dtype != DEFAULT_DTYPE:
        report.error(CODE_CONFIG, f"only dtype {DEFAULT_DTYPE!r} is supported, got {config.dtype!r}")


def _collect_pairs(
    payload: dict[str, Any],
    index: dict[str, int],
    report: ValidationReport,
    config: ZMatrixConfig,
) -> list[WeightedPair]:
    """Validate ``pairs`` and compute one unified weight per pair."""
    raw_pairs = payload.get("pairs") or []
    pre: list[float] = []
    post: list[float] = []
    sources: list[str] = []
    targets: list[str] = []
    seen: dict[tuple[int, int], int] = {}

    for position, raw in enumerate(raw_pairs):
        if not isinstance(raw, dict):
            report.error(CODE_INPUT_SCHEMA, f"pairs[{position}] is not an object: {raw!r}")
            continue
        source = raw.get("source")
        target = raw.get("target")
        if not isinstance(source, str) or not isinstance(target, str):
            report.error(CODE_ENDPOINT_NOT_NEURON, f"pairs[{position}] has non-string endpoints {source!r} -> {target!r}")
            continue
        missing = [name for name in (source, target) if name not in index]
        if missing:
            report.error(
                CODE_ENDPOINT_NOT_NEURON,
                f"pairs[{position}] endpoint(s) {missing!r} are not in the neuron list",
            )
            continue
        if source == target:
            report.error(CODE_SELF_LOOP, f"pairs[{position}] encodes the self-loop {source!r}")
            continue
        coordinate = (index[source], index[target])
        if coordinate in seen:
            report.error(
                CODE_DUPLICATE_COORDINATE,
                f"pairs[{position}] duplicates the coordinate {coordinate} already used by pairs[{seen[coordinate]}]",
            )
            continue
        seen[coordinate] = position
        try:
            pre_z = float(raw["pre_z"])
            post_z = float(raw["post_z"])
        except (KeyError, TypeError, ValueError):
            report.error(CODE_Z_INVALID, f"pairs[{position}] has non-numeric pre_z/post_z: {raw!r}")
            continue
        if not (math.isfinite(pre_z) and math.isfinite(post_z)) or pre_z < 0 or post_z < 0:
            report.error(CODE_Z_INVALID, f"pairs[{position}] has invalid z-scores pre_z={pre_z!r} post_z={post_z!r}")
            continue
        sources.append(source)
        targets.append(target)
        pre.append(pre_z)
        post.append(post_z)

    if report.has_errors or not pre:
        return []

    weights = unified_weights(pre, post, eps=config.eps, alpha=config.alpha, zero_policy=config.zero_policy)
    pairs = [
        WeightedPair(source, target, pre_z, post_z, float(weight))
        for source, target, pre_z, post_z, weight in zip(sources, targets, pre, post, weights)
    ]
    # Deterministic order: `pairs` drives `matrix` scatter order and the artifact layout.
    return sorted(pairs, key=lambda item: (item.source, item.target))


def _collect_neuron_order(payload: dict[str, Any], report: ValidationReport) -> list[str]:
    """Return the row/column order verbatim from the Phase 01 ``neurons`` list."""
    neurons = payload.get("neurons") or []
    if not isinstance(neurons, list) or not neurons:
        report.error(CODE_INPUT_SCHEMA, "the input artifact has no `neurons` list")
        return []
    order: list[str] = []
    for position, neuron in enumerate(neurons):
        if not isinstance(neuron, dict):
            report.error(CODE_INPUT_SCHEMA, f"neurons[{position}] is not an object: {neuron!r}")
            continue
        name = neuron.get("neuron_id")
        if not isinstance(name, str) or not name:
            report.error(CODE_NEURON_ID, f"neurons[{position}] has no usable neuron_id: {name!r}")
            continue
        if "--" in name:
            report.error(CODE_NEURON_ID, f"neuron_id {name!r} looks like a hub name (contains '--')")
            continue
        if name in order:
            report.error(CODE_NEURON_ID, f"neuron_id {name!r} appears twice in the neuron list")
            continue
        order.append(name)
    return order


def build_z_matrix(
    payload: dict[str, Any],
    *,
    source_artifact: str | Path | None = None,
    config: ZMatrixConfig | None = None,
    strict: bool = False,
    now: Any = None,
    hist_bins: int = DEFAULT_HIST_BINS,
) -> tuple[ZMatrix, ValidationReport]:
    """Build the Phase 02 artifact from a Phase 01 payload.

    Returns ``(ZMatrix, ValidationReport)`` and never raises for validation
    problems -- inspect ``report.has_errors`` / use :func:`build_z_matrix_from_file`
    for the raising variant.
    """
    active = config or ZMatrixConfig()
    report = ValidationReport()

    for problem in validate_payload_schema(payload):
        report.error(CODE_INPUT_SCHEMA, f"input artifact does not conform to the Phase 01 schema: {problem}")

    _validate_config(active, report)

    neuron_order = _collect_neuron_order(payload, report)
    index = {name: position for position, name in enumerate(neuron_order)}
    n_neurons = len(neuron_order)

    pairs = _collect_pairs(payload, index, report, active)

    if not pairs:
        report.warning(CODE_EMPTY_PAIRS, "no usable pairs: the Z-matrix is all zeros")
    if 0 < n_neurons < 2:
        report.warning(CODE_SMALL_GRAPH, f"only {n_neurons} neuron(s): the matrix is 1x1 and density is undefined")

    if report.has_errors:
        return (
            ZMatrix(
                source_artifact=Path(source_artifact) if source_artifact else Path("<in-memory>"),
                source_artifact_sha256="",
                source_file=str(payload.get("source_file") or ""),
                source_file_sha256=str(payload.get("file_sha256") or ""),
                parsed_created_utc=str((payload.get("metadata") or {}).get("created_utc") or ""),
                config=active,
                neuron_order=neuron_order,
                pairs=pairs,
                matrix=np.zeros((n_neurons, n_neurons), dtype=np.float64),
                matrix_symmetric=np.zeros((n_neurons, n_neurons), dtype=np.float64),
                metadata={},
                diagnostics={},
            ),
            report,
        )

    # 1. assemble the directed matrix (missing edges and the diagonal are zero)
    directed = np.zeros((n_neurons, n_neurons), dtype=np.float64)
    rows = np.array([index[pair.source] for pair in pairs], dtype=np.intp)
    cols = np.array([index[pair.target] for pair in pairs], dtype=np.intp)
    weights = np.array([pair.weight for pair in pairs], dtype=np.float64)
    directed[rows, cols] = weights

    # 2. normalize, then 3. symmetrize (fixed pipeline order)
    normalized, normalization_info = normalize_matrix(directed, method=active.normalize)
    symmetric_matrix = symmetrize(normalized, method=active.symmetrize)
    # 4. the effective matrix Phase 03 consumes
    effective = symmetric_matrix if active.symmetric else normalized
    effective = np.asarray(effective, dtype=np.float64)

    phase_one_metadata = payload.get("metadata") or {}
    density = phase_one_metadata.get("density")
    if density is None:
        density = len(pairs) / (n_neurons * (n_neurons - 1)) if n_neurons > 1 else 0.0
    reciprocity = phase_one_metadata.get("reciprocity")
    if reciprocity is None:
        keys = {(pair.source, pair.target) for pair in pairs}
        reciprocity = sum(1 for pair in pairs if (pair.target, pair.source) in keys)

    diagnostics = compute_diagnostics(
        effective,
        symmetric_matrix,
        weights,
        symmetric=active.symmetric,
        density=density,
        reciprocity=int(reciprocity),
        hist_bins=hist_bins,
    )

    matrix_stats = dict(diagnostics["matrix_stats"])
    weight_stats = dict(diagnostics["weight_stats"])

    metadata: dict[str, Any] = {
        "n_neurons": int(n_neurons),
        "n_edges": int(len(pairs)),
        "n_stored_nonzero": int(diagnostics["n_stored_nonzero"]),
        "n_matrix_zeros": int(diagnostics["n_matrix_zeros"]),
        "n_weight_positive": int(diagnostics["n_weight_positive"]),
        "n_weight_negative": int(diagnostics["n_weight_negative"]),
        "n_weight_zero": int(diagnostics["n_weight_zero"]),
        "n_self_loops": 0,
        "sparsity": float(diagnostics["sparsity"]),
        "density": float(density),
        "reciprocity": int(reciprocity),
        "matrix_stats": {key: (None if value is None else float(value)) for key, value in matrix_stats.items()},
        "weight_stats": {
            key: (None if value is None else float(value)) for key, value in weight_stats.items()
        },
        "row_norm_stats": {
            key: (None if value is None else float(value))
            for key, value in diagnostics["row_norm_stats"].items()
        },
        "col_norm_stats": {
            key: (None if value is None else float(value))
            for key, value in diagnostics["col_norm_stats"].items()
        },
        "frobenius_norm": float(diagnostics["frobenius_norm"]),
        "largest_singular_value": float(diagnostics["largest_singular_value"]),
        "spectral_radius": (
            None if diagnostics["spectral_radius"] is None else float(diagnostics["spectral_radius"])
        ),
        "symmetrized_spectral_radius": (
            None
            if diagnostics["symmetrized_spectral_radius"] is None
            else float(diagnostics["symmetrized_spectral_radius"])
        ),
        "normalization_scale": normalization_info.get("scale"),
        "config_hash": active.config_hash(),
        "numpy_version": np.__version__,
        "generator": GENERATOR,
        "created_utc": utc_timestamp(now),
    }

    artifact_path = Path(source_artifact) if source_artifact else Path("<in-memory>")
    z_matrix = ZMatrix(
        source_artifact=artifact_path,
        source_artifact_sha256=sha256_file(artifact_path) if artifact_path.is_file() else "",
        source_file=str(payload.get("source_file") or ""),
        source_file_sha256=str(payload.get("file_sha256") or ""),
        parsed_created_utc=str(phase_one_metadata.get("created_utc") or ""),
        config=active,
        neuron_order=neuron_order,
        pairs=pairs,
        matrix=effective,
        matrix_symmetric=symmetric_matrix,
        metadata=metadata,
        diagnostics=diagnostics,
    )

    # Structural self-checks on the produced payload.
    for problem in validate_z_payload_schema(z_matrix.to_dict()):
        report.error(CODE_PAYLOAD_SCHEMA, f"artifact does not conform to the Phase 02 schema: {problem}")
    if effective.shape != (n_neurons, n_neurons):
        report.error(CODE_MATRIX_SHAPE, f"matrix shape {effective.shape} != ({n_neurons}, {n_neurons})")
    if effective.size and np.any(np.diag(effective) != 0.0):
        report.error(CODE_DIAGONAL, "the matrix diagonal must be exactly 0 (no self-loops)")
    if effective.size and not np.array_equal(symmetric_matrix, symmetric_matrix.T):
        report.error(CODE_SYMMETRY, "matrix_symmetric is not exactly symmetric")
    if effective.size and not np.isfinite(effective).all():
        report.error(CODE_NONFINITE, "the matrix contains non-finite entries")
    if effective.size and not np.isfinite(symmetric_matrix).all():
        report.error(CODE_NONFINITE, "matrix_symmetric contains non-finite entries")
    if active.symmetric and effective.size and not np.array_equal(effective, symmetric_matrix):
        report.error(CODE_SYMMETRY, "symmetric mode requires matrix == matrix_symmetric")

    if strict and report.warnings:
        report.error(CODE_CONFIG, f"--strict escalates {len(report.warnings)} warning(s) to errors")

    return z_matrix, report


def build_z_matrix_from_file(
    source_path: str | Path,
    *,
    config: ZMatrixConfig | None = None,
    strict: bool = False,
    now: Any = None,
    hist_bins: int = DEFAULT_HIST_BINS,
) -> tuple[ZMatrix, ValidationReport]:
    """Load a Phase 01 artifact and build the Phase 02 matrix from it."""
    path = Path(source_path)
    payload = read_json(path)
    return build_z_matrix(
        payload,
        source_artifact=path,
        config=config,
        strict=strict,
        now=now,
        hist_bins=hist_bins,
    )


def validate_z_payload_schema(payload: dict[str, Any]) -> list[str]:
    """Hand-rolled schema check for the Phase 02 JSON artifact.

    ``jsonschema``/``pydantic`` are not installed in this environment, so this
    mirrors Phase 01's ``validate_payload_schema`` approach: return a list of
    human-readable problems (empty list == conforming).
    """
    problems: list[str] = []

    def _require_keys(actual: Iterable[str], expected: frozenset[str], where: str) -> None:
        missing = sorted(expected - set(actual))
        extra = sorted(set(actual) - expected)
        if missing:
            problems.append(f"{where}: missing key(s) {missing}")
        if extra:
            problems.append(f"{where}: unexpected key(s) {extra}")

    if not isinstance(payload, dict):
        return ["payload is not an object"]

    _require_keys(payload.keys(), TOP_LEVEL_KEYS, "top level")

    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        problems.append("provenance: not an object")
    else:
        _require_keys(provenance.keys(), PROVENANCE_KEYS, "provenance")

    config = payload.get("config")
    if not isinstance(config, dict):
        problems.append("config: not an object")
    else:
        _require_keys(config.keys(), CONFIG_KEYS, "config")
        if not isinstance(config.get("symmetric"), bool):
            problems.append("config.symmetric: not a boolean")
        if config.get("normalize") not in NORMALIZATIONS:
            problems.append(f"config.normalize: {config.get('normalize')!r} is not one of {NORMALIZATIONS}")
        if config.get("symmetrize") not in SYMMETRIZE_METHODS:
            problems.append(f"config.symmetrize: {config.get('symmetrize')!r} is not one of {SYMMETRIZE_METHODS}")
        if not isinstance(config.get("config_hash"), str) or len(config["config_hash"]) != 8:
            problems.append("config.config_hash: expected 8 hex characters")

    neuron_order = payload.get("neuron_order")
    if not isinstance(neuron_order, list) or not all(isinstance(name, str) for name in neuron_order):
        problems.append("neuron_order: expected a list of strings (a dict would lose the index order)")
        neuron_order = []

    pairs = payload.get("pairs")
    if not isinstance(pairs, list):
        problems.append("pairs: not a list")
    else:
        for position, pair in enumerate(pairs):
            if not isinstance(pair, dict):
                problems.append(f"pairs[{position}]: not an object")
                continue
            _require_keys(pair.keys(), WEIGHTED_PAIR_KEYS, f"pairs[{position}]")
            if pair.get("source") not in neuron_order or pair.get("target") not in neuron_order:
                problems.append(f"pairs[{position}]: endpoint is not in neuron_order")

    n_neurons = len(neuron_order)
    for key in ("matrix", "matrix_symmetric"):
        matrix = payload.get(key)
        if not isinstance(matrix, list) or len(matrix) != n_neurons:
            problems.append(f"{key}: expected a list of {n_neurons} rows")
            continue
        for position, row in enumerate(matrix):
            if not isinstance(row, list) or len(row) != n_neurons:
                problems.append(f"{key}[{position}]: expected a list of {n_neurons} floats")
                break
            if not all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in row):
                problems.append(f"{key}[{position}]: contains a non-numeric entry")
                break

    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        problems.append("metadata: not an object")
    else:
        _require_keys(metadata.keys(), METADATA_KEYS, "metadata")
        for key in ("matrix_stats", "row_norm_stats", "col_norm_stats"):
            if not isinstance(metadata.get(key), dict):
                problems.append(f"metadata.{key}: not an object")
            else:
                _require_keys(metadata[key].keys(), STATS_KEYS, f"metadata.{key}")
        if not isinstance(metadata.get("weight_stats"), dict):
            problems.append("metadata.weight_stats: not an object")
        else:
            _require_keys(metadata["weight_stats"].keys(), WEIGHT_STATS_KEYS, "metadata.weight_stats")

    return problems


# ---------------------------------------------------------------------------
# Artifact paths
# ---------------------------------------------------------------------------
def _is_filtered_input(source_artifact: str | Path | None) -> bool:
    """True when the input artifact is a Phase 01 *filtered* artifact."""
    if source_artifact is None:
        return False
    name = Path(source_artifact).name
    return name == PARSED_FILTERED_ARTIFACT_NAME or ".filtered." in name


def variant_stem(
    config: ZMatrixConfig,
    *,
    source_artifact: str | Path | None = None,
    force_config_hash: bool = False,
) -> str:
    """Return ``z_matrix[.filtered][.<config_hash8>]`` for *config*.

    The hash segment is added automatically for any non-default matrix config
    (``--config-hash`` forces it even for a default config) so a variant run can
    never overwrite the canonical artifact.
    """
    stem = CANONICAL_STEM
    if _is_filtered_input(source_artifact):
        stem += ".filtered"
    if force_config_hash or not config.is_default():
        stem += f".{config.config_hash()}"
    return stem


def artifact_paths(
    z_matrix: ZMatrix,
    outdir: str | Path,
    *,
    force_config_hash: bool = False,
) -> dict[str, Path]:
    """Return the four Phase 02 artifact paths for *z_matrix*.

    ``{"json", "npz", "png", "data"}`` under ``<outdir>/<gv-stem>/`` -- the same
    directory Phase 01 writes its artifact to.
    """
    stem = variant_stem(
        z_matrix.config,
        source_artifact=z_matrix.source_artifact,
        force_config_hash=force_config_hash,
    )
    directory = Path(outdir) / Path(z_matrix.source_file).stem if z_matrix.source_file else Path(outdir) / stem
    return {
        "json": directory / f"{stem}.json",
        "npz": directory / f"{stem}.npz",
        "png": directory / f"{stem}.png",
        "data": directory / f"{stem}.data.json",
    }


def sidecar_path(json_path: str | Path) -> Path:
    """``z_matrix[.variant].json`` -> ``z_matrix[.variant].npz``."""
    return Path(json_path).with_suffix(".npz")


# ---------------------------------------------------------------------------
# Sidecar (derived npz array cache)
# ---------------------------------------------------------------------------
def build_sidecar_arrays(z_matrix: ZMatrix, *, source_json_sha256: str) -> dict[str, np.ndarray]:
    """Return the ordered array bundle stored in the ``.npz`` cache.

    Ordering and dtypes are fixed (``<f8`` / ``<i8`` / ``<U*``) so the archive is
    byte-reproducible: numpy writes every zip entry with a constant
    ``date_time`` and the ``.npy`` header carries no timestamp.
    """
    index = z_matrix.index
    order_length = max((len(name) for name in z_matrix.neuron_order), default=1)
    return {
        "matrix": np.asarray(z_matrix.matrix, dtype="<f8"),
        "matrix_symmetric": np.asarray(z_matrix.matrix_symmetric, dtype="<f8"),
        "neuron_order": np.asarray(z_matrix.neuron_order, dtype=f"<U{order_length}"),
        "edge_rows": np.asarray([index[pair.source] for pair in z_matrix.pairs], dtype="<i8"),
        "edge_cols": np.asarray([index[pair.target] for pair in z_matrix.pairs], dtype="<i8"),
        "edge_weights": np.asarray([pair.weight for pair in z_matrix.pairs], dtype="<f8"),
        "source_json_sha256": np.asarray([source_json_sha256], dtype="<U64"),
        "numpy_version": np.asarray([np.__version__], dtype="<U64"),
    }


def write_npz_atomic(path: str | Path, arrays: dict[str, np.ndarray]) -> Path:
    """Write *arrays* to *path* (``.npz``) atomically and byte-reproducibly.

    The archive is built in memory first, then written to a temporary file in the
    destination directory, flushed and ``fsync``-ed before ``os.replace`` -- so a
    reader sees either the previous cache or the complete new one.
    """
    destination = Path(path)
    ensure_dir(destination.parent)

    buffer = io.BytesIO()
    np.savez(buffer, **arrays)  # deterministic: numpy pins the zip date_time
    payload = buffer.getvalue()

    handle = tempfile.NamedTemporaryFile(
        mode="wb",
        dir=str(destination.parent),
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def write_sidecar(z_matrix: ZMatrix, json_path: str | Path) -> Path:
    """Write the ``.npz`` cache for *z_matrix*, keyed by the JSON's SHA-256."""
    json_path = Path(json_path)
    digest = sha256_file(json_path)
    arrays = build_sidecar_arrays(z_matrix, source_json_sha256=digest)
    return write_npz_atomic(sidecar_path(json_path), arrays)


def load_sidecar_arrays(path: str | Path) -> dict[str, np.ndarray]:
    """Fast array access to a ``.npz`` cache (``allow_pickle=False``).

    Accepts either the ``.npz`` path or the ``.json`` path it belongs to.
    Raises :class:`FileNotFoundError` when the cache does not exist.
    """
    candidate = Path(path)
    if candidate.suffix != ".npz":
        candidate = sidecar_path(candidate)
    if not candidate.is_file():
        raise FileNotFoundError(f"no npz sidecar at {candidate}")
    with np.load(candidate, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _verify_sidecar(
    arrays: dict[str, np.ndarray],
    z_matrix: ZMatrix,
    json_path: Path,
) -> None:
    """Raise :class:`ZMatrixValidationError` when the cache disagrees with the JSON."""
    expected_sha = sha256_file(json_path)
    recorded = arrays.get("source_json_sha256")
    if recorded is None or str(recorded[0]) != expected_sha:
        raise ZMatrixValidationError(
            f"sidecar {sidecar_path(json_path).name} was built from a different JSON "
            f"(recorded {None if recorded is None else str(recorded[0])[:12]}, actual {expected_sha[:12]})"
        )
    if not np.array_equal(arrays["matrix"], z_matrix.matrix):
        raise ZMatrixValidationError("sidecar `matrix` differs from the JSON matrix")
    if not np.array_equal(arrays["matrix_symmetric"], z_matrix.matrix_symmetric):
        raise ZMatrixValidationError("sidecar `matrix_symmetric` differs from the JSON matrix")
    if list(arrays["neuron_order"]) != list(z_matrix.neuron_order):
        raise ZMatrixValidationError("sidecar `neuron_order` differs from the JSON order")


def _z_matrix_from_payload(
    payload: dict[str, Any],
    *,
    loaded_from: Path | None = None,
    hist_bins: int = DEFAULT_HIST_BINS,
) -> ZMatrix:
    """Rebuild a :class:`ZMatrix` (with diagnostics) from a JSON payload.

    Every ``provenance`` field is taken verbatim from the payload so that
    ``load_z_matrix(path).to_dict()`` reproduces the artifact byte-for-byte;
    *loaded_from* only records where the object was read from.
    """
    config = ZMatrixConfig.from_dict(payload.get("config") or {})
    provenance = payload.get("provenance") or {}
    neuron_order = [str(name) for name in payload.get("neuron_order") or []]
    pairs = [
        WeightedPair(
            source=pair["source"],
            target=pair["target"],
            pre_z=float(pair["pre_z"]),
            post_z=float(pair["post_z"]),
            weight=float(pair["weight"]),
        )
        for pair in payload.get("pairs") or []
    ]
    matrix = np.asarray(payload.get("matrix") or [], dtype=np.float64)
    matrix_symmetric = np.asarray(payload.get("matrix_symmetric") or [], dtype=np.float64)
    metadata = dict(payload.get("metadata") or {})

    diagnostics = compute_diagnostics(
        matrix,
        matrix_symmetric,
        np.array([pair.weight for pair in pairs], dtype=np.float64),
        symmetric=config.symmetric,
        density=metadata.get("density"),
        reciprocity=metadata.get("reciprocity"),
        hist_bins=hist_bins,
    )

    return ZMatrix(
        source_artifact=Path(str(provenance.get("source_artifact") or "<in-memory>")),
        source_artifact_sha256=str(provenance.get("source_artifact_sha256") or ""),
        source_file=str(provenance.get("source_file") or ""),
        source_file_sha256=str(provenance.get("source_file_sha256") or ""),
        parsed_created_utc=str(provenance.get("parsed_created_utc") or ""),
        config=config,
        neuron_order=neuron_order,
        pairs=pairs,
        matrix=matrix,
        matrix_symmetric=matrix_symmetric,
        metadata=metadata,
        diagnostics=diagnostics,
        loaded_from=loaded_from,
    )


def load_z_matrix(
    path: str | Path,
    *,
    prefer_sidecar: bool = True,
    hist_bins: int = DEFAULT_HIST_BINS,
) -> ZMatrix:
    """Load a Phase 02 artifact (the entry point for Phases 03-06).

    The canonical JSON is always the source of truth: metadata, config and pairs
    are read from it.  When the sibling ``.npz`` cache exists it is *verified*
    against the JSON's SHA-256 and arrays; a stale or tampered cache is ignored
    with a warning instead of raising.  ``path`` may also be the ``.npz`` file
    (Phase 03 often has one), in which case the sibling JSON is required.
    """
    candidate = Path(path)

    if candidate.suffix == ".npz":
        json_path = candidate.with_suffix(".json")
        if not json_path.is_file():
            raise FileNotFoundError(f"the npz sidecar {candidate} has no sibling JSON artifact at {json_path}")
        z_matrix = _z_matrix_from_payload(read_json(json_path), loaded_from=json_path, hist_bins=hist_bins)
        _verify_sidecar(load_sidecar_arrays(candidate), z_matrix, json_path)
        return z_matrix

    payload = read_json(candidate)
    z_matrix = _z_matrix_from_payload(payload, loaded_from=candidate, hist_bins=hist_bins)

    if prefer_sidecar:
        cache = sidecar_path(candidate)
        if cache.is_file():
            try:
                _verify_sidecar(load_sidecar_arrays(cache), z_matrix, candidate)
            except Exception as exc:  # stale/tampered/legacy cache: fall back to JSON
                LOGGER.warning("ignoring sidecar %s: %s", cache.name, exc)
    return z_matrix


# ---------------------------------------------------------------------------
# Diagnostics output: --save-data, --stats, the completion box, --plot
# ---------------------------------------------------------------------------
def save_data_payload(
    z_matrix: ZMatrix,
    *,
    hist_bins: int = DEFAULT_HIST_BINS,
    now: Any = None,
) -> dict[str, Any]:
    """Return the ``z_matrix.data.json`` payload (per-pair weights + statistics)."""
    diagnostics = compute_diagnostics(
        z_matrix.matrix,
        z_matrix.matrix_symmetric,
        np.array([pair.weight for pair in z_matrix.pairs], dtype=np.float64),
        symmetric=z_matrix.config.symmetric,
        density=z_matrix.metadata.get("density"),
        reciprocity=z_matrix.metadata.get("reciprocity"),
        hist_bins=hist_bins,
    )
    return {
        "provenance": {
            "source_artifact": str(z_matrix.source_artifact),
            "source_artifact_sha256": z_matrix.source_artifact_sha256,
            "source_file": z_matrix.source_file,
            "source_file_sha256": z_matrix.source_file_sha256,
            "parsed_created_utc": z_matrix.parsed_created_utc,
            "phase": "02",
        },
        "config": z_matrix.config.to_dict(),
        "symmetric": bool(z_matrix.config.symmetric),
        "normalize": z_matrix.config.normalize,
        "pairs": [pair.to_dict() for pair in z_matrix.pairs],
        "z_stats": dict(diagnostics["matrix_stats"]),
        "histogram": dict(diagnostics["histogram"]),
        "sparsity": float(diagnostics["sparsity"]),
        "n_positive": int(diagnostics["n_weight_positive"]),
        "n_negative": int(diagnostics["n_weight_negative"]),
        "n_zero": int(diagnostics["n_weight_zero"]),
        "row_norms": list(diagnostics["row_norms"]),
        "col_norms": list(diagnostics["col_norms"]),
        "row_norm_stats": dict(diagnostics["row_norm_stats"]),
        "col_norm_stats": dict(diagnostics["col_norm_stats"]),
        "spectral_radius": diagnostics["spectral_radius"],
        "density": diagnostics["density"],
        "reciprocity": diagnostics["reciprocity"],
        "generator": GENERATOR,
        "created_utc": utc_timestamp(now),
    }


def _format(value: Any, digits: int = 6) -> str:
    """Format an optional float for terminal output."""
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def render_statistics(z_matrix: ZMatrix, diagnostics: dict[str, Any] | None = None) -> str:
    """Terminal-only statistics block printed by ``--stats``."""
    diagnostics = diagnostics or z_matrix.diagnostics
    matrix_stats = diagnostics.get("matrix_stats") or {}
    row_stats = diagnostics.get("row_norm_stats") or {}
    col_stats = diagnostics.get("col_norm_stats") or {}
    symmetric = z_matrix.config.symmetric
    radius = diagnostics.get("spectral_radius")
    lines = [
        "Phase 02 -- Z-matrix statistics",
        f"  shape               : {z_matrix.n_neurons} x {z_matrix.n_neurons}",
        f"  stored entries      : {diagnostics.get('n_stored_nonzero')} of {diagnostics.get('n_entries')}",
        f"  min / max           : {_format(matrix_stats.get('min'))} / {_format(matrix_stats.get('max'))}",
        f"  mean / std          : {_format(matrix_stats.get('mean'))} / {_format(matrix_stats.get('std'))}",
        f"  sparsity            : {_format(diagnostics.get('sparsity'))} (fraction of zero entries)",
        (
            "  positive / negative : "
            f"{diagnostics.get('n_weight_positive')} / {diagnostics.get('n_weight_negative')}"
            f"  ({diagnostics.get('n_weight_zero')} exact zero weight(s))"
        ),
        (
            "  row norms (L1)      : "
            f"min {_format(row_stats.get('min'))} max {_format(row_stats.get('max'))} "
            f"mean {_format(row_stats.get('mean'))} std {_format(row_stats.get('std'))}"
        ),
        (
            "  col norms (L1)      : "
            f"min {_format(col_stats.get('min'))} max {_format(col_stats.get('max'))} "
            f"mean {_format(col_stats.get('mean'))} std {_format(col_stats.get('std'))}"
        ),
        (
            "  spectral radius     : "
            + (
                f"{_format(radius)} (|eigenvalue| of the symmetric matrix)"
                if symmetric
                else "n/a (run with --symmetric)"
            )
        ),
        f"  symmetrized radius  : {_format(diagnostics.get('symmetrized_spectral_radius'))} (|eigenvalue| of (Z + Z.T)/2)",
        f"  frobenius norm      : {_format(diagnostics.get('frobenius_norm'))}",
        f"  density / reciprocity: {_format(diagnostics.get('density'))} / {diagnostics.get('reciprocity')}",
    ]
    return "\n".join(lines)


def render_summary_box(
    z_matrix: ZMatrix,
    paths: dict[str, Path],
    *,
    written: bool = True,
    sidecar: bool = True,
    plot: bool = False,
    save_data: bool = False,
    stats: bool = False,
    interactive: bool = False,
    interactive_shown: bool = False,
) -> str:
    """The boxed terminal-only Phase 02 completion summary.

    The ``Interactive:`` line is emitted **only** when ``interactive`` is set, so
    a run without ``--interactive`` produces exactly the same box as before.
    """
    def _name(key: str) -> str:
        return paths[key].name

    def _skip(reason: str) -> str:
        return f"skipped ({reason})"

    if written and sidecar:
        npz_line = _name("npz")
    elif sidecar:
        npz_line = _skip("--dry-run")
    else:
        npz_line = _skip("--no-sidecar")

    if not written:
        interactive_line = _skip("--dry-run")
    elif interactive_shown:
        interactive_line = "window shown (--interactive)"
    else:
        interactive_line = "skipped (non-interactive backend)"

    lines = [
        "Phase 02 complete" if written else "Phase 02 dry-run (nothing written)",
        f"Z matrix shape: {z_matrix.n_neurons}x{z_matrix.n_neurons}",
        "Unified weights: OK",
        f"Symmetric mode: {'yes' if z_matrix.config.symmetric else 'no'}",
        f"Normalization: {z_matrix.config.normalize}",
        f"JSON written: {_name('json') if written else _skip('--dry-run')}",
        f"NPZ written:  {npz_line}",
        f"Plot:        {_name('png') if (written and plot) else ('not requested (--plot)' if not plot else _skip('--dry-run'))}",
        *([f"Interactive: {interactive_line}"] if interactive else []),
        (
            "Save-data:   "
            + (
                _name("data")
                if (written and save_data)
                else ("not requested (--save-data)" if not save_data else _skip("--dry-run"))
            )
        ),
        (
            "Stats: min/max/mean/std printed above (--stats)"
            if stats
            else "Stats: not requested (--stats)"
        ),
    ]
    width = max(max(len(line) for line in lines), 60)
    border = "+" + "-" * (width + 2) + "+"
    body = [f"| {line.ljust(width)} |" for line in lines]
    return "\n".join([border, *body, border])


def _backend_gui_framework(backend: str) -> str | None:
    """Return the GUI framework *backend* is bound to (``None`` when file-only).

    Thin wrapper around matplotlib's backend registry so that the headless
    classification has one testable seam; raises for an unrecognised backend
    name, which :func:`can_popup` treats as "ask the name set instead".
    """
    from matplotlib.backends import backend_registry
    return backend_registry.resolve_backend(backend)[1]


def can_popup() -> bool:
    """True when the matplotlib backend can actually open a window.

    Non-interactive backends (``Agg`` and friends) are detected so ``--plot`` and
    ``--interactive`` never block in a headless session; ``--no-popup`` forces the
    ``--plot`` window off.

    The answer comes from matplotlib's backend registry:
    ``backend_registry.resolve_backend(name)`` returns the GUI framework the
    backend is bound to, which is ``None`` for file-only backends.  A plain
    substring test on ``"agg"`` is **wrong** here -- ``qtagg``, ``tkagg``,
    ``gtk3agg`` and ``wxagg`` all contain it, which used to misclassify the
    desktop backend as headless and silently suppress every window.
    """
    try:
        import matplotlib
    except Exception:  # pragma: no cover - matplotlib is a pinned dependency
        return False
    backend = str(matplotlib.get_backend()).strip()
    try:
        return _backend_gui_framework(backend) is not None
    except Exception:  # pragma: no cover - matplotlib < 3.9 / unregistered name
        return backend.lower() not in NON_INTERACTIVE_BACKENDS


def plot_matrix(
    matrix: Any,
    out_path: str | Path,
    *,
    cmap: str = DEFAULT_CMAP,
    popup: bool = True,
    title: str | None = None,
    log: logging.Logger | None = None,
) -> Path:
    """Save a heatmap PNG of *matrix* and optionally show a GUI popup.

    This is the ``--plot`` popup only: ``--interactive`` opens a separate
    cursor-enabled window through :func:`show_interactive_matrix`.  matplotlib is
    imported lazily so that importing this module never selects a backend.
    """
    logger = log or LOGGER
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - matplotlib is a pinned dependency
        raise ZMatrixValidationError(f"matplotlib is required for --plot: {exc}") from exc

    destination = Path(out_path)
    ensure_dir(destination.parent)

    values = np.asarray(matrix, dtype=np.float64)
    figure, axes = plt.subplots(figsize=(7.5, 6.5), dpi=150)
    image = axes.imshow(values, cmap=cmap, interpolation="nearest", aspect="equal")
    figure.colorbar(image, ax=axes, label="unified weight w", shrink=0.85)
    axes.set_xlabel("target neuron index")
    axes.set_ylabel("source neuron index")
    axes.set_title(title or f"Phase 02 unified-weight Z matrix ({values.shape[0]}x{values.shape[1]})")
    figure.tight_layout()
    figure.savefig(destination, dpi=150, bbox_inches="tight")

    if popup and can_popup():
        logger.info("showing the heatmap popup for %s (close the window to continue)", destination.name)
        plt.show()  # GUI popup: blocks until the user closes the window
    else:
        logger.debug(
            "wrote %s (GUI popup skipped: %s)",
            destination,
            "--no-popup" if not popup else "non-interactive matplotlib backend",
        )
    plt.close(figure)
    return destination


# ---------------------------------------------------------------------------
# --interactive: the hover-tooltip window (matplotlib + mplcursors)
# ---------------------------------------------------------------------------
def _raw_weight_lookup(z_matrix: ZMatrix) -> dict[tuple[int, int], float]:
    """Map ``(source_index, target_index) -> pair.weight`` for the *stored* edges.

    The pair weight is the **pre-normalization** unified weight, so it differs
    from the matrix entry whenever ``--normalize`` or ``--symmetric`` is active;
    the tooltip reports it as a labelled extra line.  Pure: no I/O, no matplotlib.
    """
    index = z_matrix.index
    return {(index[pair.source], index[pair.target]): pair.weight for pair in z_matrix.pairs}


def format_cell_tooltip(
    z_matrix: ZMatrix,
    row: int,
    col: int,
    *,
    raw_weights: dict[tuple[int, int], float] | None = None,
) -> str:
    """Return the hover-tooltip text for cell ``(row, col)`` of the effective matrix.

    Always reports the source neuron, the target neuron, the displayed unified
    weight and the ``(i, j)`` indices -- the four facts ``--interactive`` is
    specified to show.  Two refinements keep the tooltip truthful:

    * a coordinate with no stored pair is marked ``stored edge: none (absent)``
      (presence is decided by the pair lookup, so a stored weight of exactly
      ``0.0`` from ``--zero-policy zero-in-zero-out`` is *not* called absent);
    * when ``--normalize`` or ``--symmetric`` makes the displayed value differ
      from the unified weight of the originating pair, that raw weight is added
      as a clearly labelled extra line.

    Deliberately matplotlib-free and total: an out-of-range index degrades to a
    ``<row n>`` / ``<col n>`` placeholder instead of raising, because this text is
    produced from inside a GUI event callback.
    """
    n_neurons = z_matrix.n_neurons
    values = z_matrix.matrix
    in_range = 0 <= row < n_neurons and 0 <= col < n_neurons and values.size > 0
    value = float(values[row, col]) if in_range else 0.0
    source = z_matrix.neuron_order[row] if 0 <= row < n_neurons else f"<row {row}>"
    target = z_matrix.neuron_order[col] if 0 <= col < n_neurons else f"<col {col}>"

    lookup = _raw_weight_lookup(z_matrix) if raw_weights is None else raw_weights
    forwards = lookup.get((row, col))
    # The reverse coordinate only carries meaning in symmetric mode, where the
    # displayed cell is built from both directions; in directed mode a missing
    # (row, col) pair is simply an absent edge.
    backwards = lookup.get((col, row)) if z_matrix.config.symmetric else None

    weight_line = f"unified weight: {_format(value)}"
    lines = [f"source: {source}", f"target: {target}", weight_line, f"(i, j) = ({row}, {col})"]

    if forwards is None and backwards is None:
        lines[2] = f"{weight_line} (absent edge)"
        lines.append("stored edge: none (absent)")
    elif z_matrix.config.normalize != "none" or z_matrix.config.symmetric:
        if forwards is not None:
            lines.append(f"raw unified weight: {_format(forwards)}")
        else:
            lines.append(f"raw unified weight (reverse direction): {_format(backwards)}")
    return "\n".join(lines)


def build_interactive_figure(
    z_matrix: ZMatrix,
    *,
    cmap: str = DEFAULT_CMAP,
    title: str | None = None,
) -> tuple[Any, Any, Any]:
    """Build the ``--interactive`` heatmap figure; return ``(figure, axes, image)``.

    Presentation mirrors the ``--plot`` PNG (same colormap, ``nearest``
    interpolation, colorbar labelled ``unified weight w``, index axes) but is
    sized for a screen window at 100 dpi rather than a 150 dpi output file.
    No cursor and no ``plt.show()`` here, so the figure can be built, inspected
    and embedded headlessly.

    ``plot_matrix`` is intentionally **not** refactored to share this code: the
    ``--plot`` path stays byte-for-byte unchanged.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - matplotlib is a pinned dependency
        raise ZMatrixValidationError(f"matplotlib is required for --interactive: {exc}") from exc

    values = np.asarray(z_matrix.matrix, dtype=np.float64)
    figure, axes = plt.subplots(figsize=(8.0, 7.5), dpi=100)
    image = axes.imshow(values, cmap=cmap, interpolation="nearest", aspect="equal")
    figure.colorbar(image, ax=axes, label="unified weight w", shrink=0.85)
    axes.set_xlabel("target neuron index (hover a cell for its neurons and weight)")
    axes.set_ylabel("source neuron index (hover a cell for its neurons and weight)")
    axes.set_title(
        title
        or f"Phase 02 unified-weight Z matrix ({values.shape[0]}x{values.shape[1]}) -- hover a cell"
    )
    figure.tight_layout()
    return figure, axes, image


def attach_cell_cursor(image: Any, z_matrix: ZMatrix, *, log: logging.Logger | None = None) -> Any:
    """Attach an mplcursors hover cursor to *image*; return the ``mplcursors.Cursor``.

    Every hovered cell annotates itself with :func:`format_cell_tooltip`
    (source neuron, target neuron, unified weight, ``(i, j)``).  ``hover=True``
    means no click is needed; ``highlight=False`` is required because
    mplcursors has no ``make_highlight`` handler for ``AxesImage`` and the
    default one would emit a ``UserWarning`` and copy the image on every hover.

    mplcursors is imported lazily (like matplotlib) so that importing this
    module never pulls it in; a missing dependency raises a
    :class:`ZMatrixValidationError` with the conda-forge install hint.
    """
    try:
        import mplcursors
    except Exception as exc:
        raise ZMatrixValidationError(
            "mplcursors is required for --interactive "
            "(install it with `conda install -c conda-forge mplcursors`): "
            f"{exc}"
        ) from exc

    cursor = mplcursors.cursor(
        image,
        hover=True,
        highlight=False,
        annotation_kwargs=INTERACTIVE_ANNOTATION_KWARGS,
    )
    raw_weights = _raw_weight_lookup(z_matrix)
    logger = log or LOGGER
    logger.debug(
        "attached an mplcursors hover cursor to the Z-matrix heatmap (%d cells readable)",
        z_matrix.n_neurons * z_matrix.n_neurons,
    )

    @cursor.connect("add")
    def _on_add(selection: Any) -> None:  # pragma: no cover - driven by the GUI event loop
        try:
            # For an AxesImage mplcursors reports index == (row, col); anything
            # else (a future artist swap) is ignored rather than crashing.
            row, col = (int(part) for part in selection.index)
        except Exception:
            return
        selection.annotation.set_text(
            format_cell_tooltip(z_matrix, row, col, raw_weights=raw_weights)
        )
        patch = selection.annotation.get_bbox_patch()
        if patch is not None:
            patch.set_alpha(INTERACTIVE_ANNOTATION_KWARGS["bbox"]["alpha"])

    return cursor


def show_interactive_matrix(
    z_matrix: ZMatrix,
    *,
    cmap: str = DEFAULT_CMAP,
    title: str | None = None,
    show: bool = True,
    log: logging.Logger | None = None,
) -> bool:
    """Open the ``--interactive`` window: the effective Z matrix with hover tooltips.

    Returns ``True`` when the GUI window was actually shown.  On a
    non-interactive backend (``Agg`` and friends, i.e. headless CI) the figure
    and the cursor are still constructed, ``plt.show()`` is skipped with a
    warning and ``False`` is returned -- so a batch run never blocks and never
    fails for lack of a display.  ``show=False`` is the same path without
    entering an event loop (used by tests and by embedding callers).

    Writes no artifact and reads nothing from disk: the tooltip data comes from
    the in-memory :class:`ZMatrix`, so it is correct with ``--no-sidecar`` too.
    """
    logger = log or LOGGER
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - matplotlib is a pinned dependency
        raise ZMatrixValidationError(f"matplotlib is required for --interactive: {exc}") from exc

    figure, _axes, image = build_interactive_figure(z_matrix, cmap=cmap, title=title)
    attach_cell_cursor(image, z_matrix, log=logger)

    shown = False
    if show and can_popup():
        try:  # window titles are backend-dependent; never fatal
            figure.canvas.manager.set_window_title("Phase 02 -- Z matrix (hover a cell)")
        except Exception:  # pragma: no cover - Agg/BboxImage-like managers
            pass
        logger.info(
            "showing the interactive Z-matrix window for %dx%d neurons "
            "(hover a cell for its source, target, unified weight and (i, j); "
            "close the window to continue)",
            z_matrix.n_neurons,
            z_matrix.n_neurons,
        )
        plt.show()  # GUI window: blocks until the user closes it
        shown = True
    else:
        logger.warning(
            "interactive Z-matrix window not shown (%s); the artifacts on disk are unaffected",
            "non-interactive matplotlib backend" if show else "show=False",
        )
    plt.close(figure)
    return shown


def write_artifact_set(
    z_matrix: ZMatrix,
    outdir: str | Path,
    *,
    sidecar: bool = True,
    plot: bool = False,
    save_data: bool = False,
    force_config_hash: bool = False,
    popup: bool = True,
    cmap: str = DEFAULT_CMAP,
    hist_bins: int = DEFAULT_HIST_BINS,
    report: ValidationReport | None = None,
    log: logging.Logger | None = None,
) -> dict[str, Path]:
    """Write the requested artifacts and return the paths that were written.

    The canonical JSON goes first (it is self-sufficient); the ``.npz`` cache,
    plot and save-data file follow.  Under ``--strict`` a failed derived write is
    an error, otherwise it is recorded as a warning/recoverable notice.
    """
    logger = log or LOGGER
    report = report if report is not None else ValidationReport()
    paths = artifact_paths(z_matrix, outdir, force_config_hash=force_config_hash)
    written: dict[str, Path] = {}

    written["json"] = write_json_atomic(paths["json"], z_matrix.to_dict())

    if sidecar:
        try:
            written["npz"] = write_sidecar(z_matrix, written["json"])
        except Exception as exc:
            report.error(CODE_ARTIFACT_WRITE, f"could not write the npz sidecar {paths['npz']}: {exc}")

    if plot:
        try:
            written["png"] = plot_matrix(z_matrix.matrix, paths["png"], cmap=cmap, popup=popup, log=logger)
        except Exception as exc:
            report.error(CODE_PLOT, f"could not write the diagnostic plot {paths['png']}: {exc}")

    if save_data:
        try:
            written["data"] = write_json_atomic(
                paths["data"], save_data_payload(z_matrix, hist_bins=hist_bins)
            )
        except Exception as exc:
            report.error(CODE_ARTIFACT_WRITE, f"could not write the save-data file {paths['data']}: {exc}")

    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def resolve_inputs(
    paths: Sequence[str | Path],
    pattern: str = DEFAULT_INPUT_PATTERN,
    *,
    include_filtered: bool = False,
) -> list[Path]:
    """Expand CLI inputs into a deterministic list of Phase 01 JSON artifacts.

    Directory inputs are searched **recursively** (``<outdir>/<stem>/parsed_graph.json``
    is one level below the processed root).  ``--include-filtered`` also picks up
    ``parsed_graph.filtered.json``; explicit files must carry a ``.json`` suffix.
    """
    patterns = [pattern]
    if include_filtered and PARSED_FILTERED_ARTIFACT_NAME not in patterns:
        patterns.append(PARSED_FILTERED_ARTIFACT_NAME)

    resolved: list[Path] = []
    for raw in paths:
        candidate = Path(raw)
        if candidate.is_dir():
            matches = {path for pat in patterns for path in candidate.rglob(pat) if path.is_file()}
            resolved.extend(sorted(matches))
        elif candidate.is_file():
            if candidate.suffix.lower() != SUPPORTED_SUFFIX:
                LOGGER.error("%s: unsupported suffix (expected %s)", candidate, SUPPORTED_SUFFIX)
                continue
            resolved.append(candidate)
        else:
            LOGGER.error("%s: no such file or directory", candidate)

    unique: list[Path] = []
    for item in resolved:
        if item not in unique:
            unique.append(item)
    return unique


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.matrices.build_square_matrix",
        description="Build data/processed/<stem>/z_matrix.json from a Phase 01 parsed_graph.json",
    )
    parser.add_argument(
        "--input", "-i", nargs="+", required=True, metavar="PATH",
        help="one or more parsed_graph.json files or directories (searched recursively)",
    )
    parser.add_argument(
        "--outdir", "-o", default="data/processed",
        help="output root; artifacts land in <outdir>/<stem>/ (default: data/processed)",
    )
    parser.add_argument(
        "--glob", dest="glob_pattern", default=DEFAULT_INPUT_PATTERN,
        help=f"glob used for directory inputs (default: {DEFAULT_INPUT_PATTERN})",
    )
    parser.add_argument(
        "--include-filtered", action="store_true",
        help="also process Phase 01 'parsed_graph.filtered.json' artifacts",
    )

    unification = parser.add_argument_group("unified-weight parameters")
    unification.add_argument("--eps", type=float, default=DEFAULT_EPS, help=f"ratio stabilizer (default: {DEFAULT_EPS})")
    unification.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help=f"gain sensitivity (default: {DEFAULT_ALPHA})")
    unification.add_argument(
        "--zero-policy", dest="zero_policy", choices=ZERO_POLICIES, default=DEFAULT_ZERO_POLICY,
        help="how to treat a zero pre/post z-score (default: zero-in-zero-out)",
    )

    structure = parser.add_argument_group("matrix structure")
    structure.add_argument(
        "--symmetric", action="store_true",
        help="make Z symmetric: Z_sym = (Z + Z.T) / 2 (real eigenvalues, orthogonal eigenvectors)",
    )
    structure.add_argument(
        "--symmetrize", choices=SYMMETRIZE_METHODS, default=DEFAULT_SYMMETRIZE,
        help=f"rule used by --symmetric and for the matrix_symmetric key (default: {DEFAULT_SYMMETRIZE})",
    )
    structure.add_argument(
        "--normalize", choices=NORMALIZATIONS, default=DEFAULT_NORMALIZATION,
        help="rows | cols | unit (max |w|) | spectral (largest singular value) | zscore-nonzero | none",
    )

    outputs = parser.add_argument_group("outputs")
    outputs.add_argument("--config-hash", dest="config_hash", action="store_true",
                         help="always include <config_hash8> in the artifact filenames")
    outputs.add_argument("--no-sidecar", dest="no_sidecar", action="store_true",
                         help="do not write z_matrix.npz (the JSON stays canonical and load_z_matrix still works)")
    outputs.add_argument("--plot", action="store_true", help="write z_matrix.png and show the heatmap popup")
    outputs.add_argument("--interactive", action="store_true",
                         help="open a window of the effective Z matrix with per-cell hover tooltips "
                              "(source neuron, target neuron, unified weight, (i, j)); needs mplcursors "
                              "and a GUI backend; writes no artifact")
    outputs.add_argument("--no-popup", dest="no_popup", action="store_true",
                         help="with --plot: save the PNG without opening the GUI window")
    outputs.add_argument("--cmap", default=DEFAULT_CMAP, help=f"heatmap colormap (default: {DEFAULT_CMAP})")
    outputs.add_argument("--save-data", dest="save_data", action="store_true",
                         help="write z_matrix.data.json (per-pair weights, histograms, norms, stats)")
    outputs.add_argument("--stats", action="store_true", help="print Z statistics to the terminal")
    outputs.add_argument("--hist-bins", dest="hist_bins", type=int, default=DEFAULT_HIST_BINS,
                         help=f"histogram bin count for --save-data (default: {DEFAULT_HIST_BINS})")

    parser.add_argument("--strict", action="store_true", help="fail on any validation issue (JSON and NPZ writing)")
    parser.add_argument("--dry-run", dest="dry_run", action="store_true",
                        help="validate, print statistics and the summary box without writing anything")
    parser.add_argument(
        "--log-level", dest="log_level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="logging verbosity (default: INFO)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; returns the process exit code.

    ``0`` on success, ``1`` when any input failed parsing/validation or an
    artifact could not be written, ``2`` when no input file matched.
    """
    args = build_argument_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(name)s: %(message)s")

    if args.hist_bins < 1:
        LOGGER.error("--hist-bins must be >= 1 (got %d)", args.hist_bins)
        return 1

    inputs = resolve_inputs(args.input, args.glob_pattern, include_filtered=args.include_filtered)
    if not inputs:
        LOGGER.error("no input files matched: %s", ", ".join(str(item) for item in args.input))
        return 2

    # --interactive is meant for inspecting one matrix; in a batch it degrades to
    # one blocking window per input, one after the other.
    if args.interactive and len(inputs) > 1:
        LOGGER.warning(
            "--interactive resolves to %d input(s): a window will be opened for each, "
            "one after the other (interactive mode is meant for a single artifact)",
            len(inputs),
        )

    failures = 0
    for source in inputs:
        try:
            payload = read_json(source)
        except Exception as exc:
            LOGGER.error("%s: could not read the input artifact: %s", source.name, exc)
            failures += 1
            continue

        config = ZMatrixConfig(
            eps=args.eps,
            alpha=args.alpha,
            zero_policy=args.zero_policy,
            symmetric=args.symmetric,
            symmetrize=args.symmetrize,
            normalize=args.normalize,
            # hashed but never default-forcing: a filtered input is named `.filtered`
            filters=list((payload.get("metadata") or {}).get("filters") or []),
        )

        try:
            z_matrix, report = build_z_matrix(
                payload,
                source_artifact=source,
                config=config,
                strict=False,
                hist_bins=args.hist_bins,
            )
        except Exception as exc:  # malformed/unreadable artifact
            LOGGER.error("%s: %s", source.name, exc)
            failures += 1
            continue

        for issue in report.warnings:
            LOGGER.warning("%s: [%s] %s", source.name, issue.code, issue.message)

        if report.has_errors:
            for issue in report.errors:
                LOGGER.error("%s: [%s] %s", source.name, issue.code, issue.message)
            LOGGER.error("%s: validation failed (%s)", source.name, report.summary())
            failures += 1
            continue

        paths = artifact_paths(z_matrix, args.outdir, force_config_hash=args.config_hash)
        metadata = z_matrix.metadata
        LOGGER.info(
            "%s: neurons=%d pairs=%d stored=%d sparsity=%.6f symmetric=%s normalize=%s config_hash=%s",
            source.name,
            metadata["n_neurons"],
            metadata["n_edges"],
            metadata["n_stored_nonzero"],
            metadata["sparsity"],
            config.symmetric,
            config.normalize,
            metadata["config_hash"],
        )

        if args.stats:
            print(render_statistics(z_matrix))

        if not args.dry_run:
            written = write_artifact_set(
                z_matrix,
                args.outdir,
                sidecar=not args.no_sidecar,
                plot=args.plot,
                save_data=args.save_data,
                force_config_hash=args.config_hash,
                # when --interactive is given, the richer cursor window replaces the
                # plain --plot popup so exactly one window opens
                popup=not args.no_popup and not args.interactive,
                cmap=args.cmap,
                hist_bins=args.hist_bins,
                report=report,
                log=LOGGER,
            )
            if args.plot and args.interactive:
                LOGGER.info(
                    "%s: plain --plot popup suppressed (--interactive opens the hover-tooltip window)",
                    source.name,
                )
            for issue in report.warnings:
                LOGGER.warning("%s: [%s] %s", source.name, issue.code, issue.message)
            if report.has_errors or (args.strict and report.warnings):
                for issue in report.errors:
                    LOGGER.error("%s: [%s] %s", source.name, issue.code, issue.message)
                if args.strict and report.warnings:
                    LOGGER.error("%s: --strict escalates %d warning(s) to errors", source.name, len(report.warnings))
                LOGGER.error("%s: artifact writing failed (%s)", source.name, report.summary())
                failures += 1
                continue
            for key, path in written.items():
                LOGGER.info("%s: wrote %s", source.name, path)

        # The GUI window is the last thing that happens for this input, so the
        # blocking plt.show() cannot hide the terminal output that follows.
        interactive_shown = False
        if args.interactive:
            if args.dry_run:
                LOGGER.warning("%s: --interactive is ignored under --dry-run (no window is opened)", source.name)
            else:
                try:
                    interactive_shown = show_interactive_matrix(z_matrix, cmap=args.cmap, log=LOGGER)
                except Exception as exc:
                    LOGGER.error("%s: could not open the interactive window: %s", source.name, exc)
                    failures += 1

        print(
            render_summary_box(
                z_matrix,
                paths,
                written=not args.dry_run,
                sidecar=not args.no_sidecar,
                plot=args.plot,
                save_data=args.save_data,
                stats=args.stats,
                interactive=args.interactive,
                interactive_shown=interactive_shown,
            )
        )

    if failures:
        LOGGER.error("%d of %d input file(s) failed", failures, len(inputs))
    return 1 if failures else 0


if __name__ == "__main__":  # pragma: no cover - exercised through subprocess tests
    sys.exit(main())
