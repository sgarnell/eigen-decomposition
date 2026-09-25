"""Phase 03 -- eigen / SVD decomposition of the Phase 02 ``Z`` matrix.

Phase 03 extracts the **latent functional modes** of the unified-weight matrix
built by Phase 02.  It owns exactly two decompositions and nothing else:

* **eigen** -- a true eigen-decomposition of a *symmetric* matrix
  (``scipy.linalg.eigh``): real eigenvalues, orthonormal eigenvectors.  This is
  the mathematically meaningful path for ``Z_sym = (Z + Z^T) / 2`` because a
  general directed ``Z`` has a complex spectrum.
* **svd** -- a singular value decomposition of the *directed* matrix
  (``scipy.linalg.svd``, ``gesdd``, ``full_matrices=False``): non-negative
  singular values plus the two orthonormal loading bases ``U`` (source /
  sender side) and ``V`` (target / receiver side) -- the "cross-type coupling
  axes" of the master plan.

``--method auto`` (the default) picks the eigen path when the selected source
matrix is numerically symmetric and the SVD path otherwise, so a default Phase 02
artifact decomposes with SVD and a ``--symmetric`` one with eigen, with no extra
flags.  ``--method eigen`` on an asymmetric source is a hard error.

Artifacts written per input (``<stem>`` = the source ``.gv`` stem, so Phase 03
lands next to the Phase 02 artifacts)::

    <outdir>/<stem>/eigen.json            # canonical: spectrum + modes + heuristics
    <outdir>/<stem>/eigen.npz             # derived array cache (--no-sidecar opts out)
    <outdir>/<stem>/eigen.png             # scree + cumulative variance + spectrum (--plot)
    <outdir>/<stem>/eigen.modes.png       # mode heatmaps (--plot, --plot-modes > 0)
    <outdir>/<stem>/eigen.data.json       # per-mode detail tables (--save-data)

The artifact stem **inherits the Phase 02 variant suffix** (``z_matrix.json`` ->
``eigen.json``; ``z_matrix.f8652585.json`` -> ``eigen.f8652585.json``) and adds
``.<config_hash8>`` for any non-default Phase 03 config, so a variant run can
never clobber the canonical ``eigen.json``.

Explained variance is defined **identically for both paths** as
``value**2 / sum(value**2)`` -- for SVD that is the classical ``sigma**2 /
sum(sigma**2)``; for a symmetric matrix it is ``lambda**2 / sum(lambda**2)``.
This is what makes the two paths comparable and gives the exact invariant
``sum(sigma**2) == sum(lambda**2) == ||A||_F**2``.

Everything is deterministic: the eigenvectors' arbitrary signs are fixed by
``--sign-convention max-abs-positive``, the heuristics are a pure function of the
spectrum, and ``created_utc`` honours ``SOURCE_DATE_EPOCH``, so two runs produce
byte-identical artifacts.  No CSV files are produced or consumed.

CLI
---
::

    python -m src.spectral.spectral_decomposition \\
        -i data/processed/<stem>/z_matrix.json -o data/processed \\
        [--glob 'z_matrix.json'] [--include-variants] \\
        [--method {auto,eigen,svd}] [--source {effective,symmetric}] \\
        [--k auto|0|N] [--rank-by {magnitude,value}] \\
        [--sign-convention {max-abs-positive,none}] [--driver {evr,evd,ev,evx}] \\
        [--variance-threshold 0.9] [--elbow] [--no-elbow] \\
        [--elbow-curve {cumulative,scree}] \\
        [--elbow-method {l-method,second-difference}] \\
        [--spectral-gap] [--no-spectral-gap] [--gap-window 25] \\
        [--gap-metric {ratio,gap}] [--residual-tol 1e-9] \\
        [--config-hash] [--no-sidecar] \\
        [--plot] [--no-popup] [--cmap viridis] [--plot-modes 4] \\
        [--save-data] [--stats] \\
        [--strict] [--dry-run] [--log-level INFO]

Loading (Phase 04+)
-------------------
* :func:`load_spectrum` -- integrity-checked object (verifies the ``.npz`` cache
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
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from src.matrices.build_square_matrix import (
    DEFAULT_ARTIFACT_NAME as Z_ARTIFACT_NAME,
    ZMatrix,
    can_popup,
    load_z_matrix,
    validate_z_payload_schema,
)
from src.parsing.parse_graphviz import ValidationReport
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
    "CODE_DEGENERATE",
    "CODE_DIMENSION",
    "CODE_INPUT_SCHEMA",
    "CODE_K_RANGE",
    "CODE_METHOD_INCOMPATIBLE",
    "CODE_NONFINITE",
    "CODE_NOT_SYMMETRIC",
    "CODE_PAYLOAD_SCHEMA",
    "CODE_PLOT",
    "CODE_RESIDUAL",
    "CODE_SIDECAR_CACHE",
    "CODE_SMALL_GRAPH",
    "CODE_TRIVIAL_MATRIX",
    "CONFIG_KEYS",
    "DEFAULT_CMAP",
    "DEFAULT_DEGENERATE_TOL",
    "DEFAULT_EIGH_DRIVER",
    "DEFAULT_ELBOW_CURVE",
    "DEFAULT_ELBOW_METHOD",
    "DEFAULT_GAP_METRIC",
    "DEFAULT_GAP_WINDOW",
    "DEFAULT_K",
    "DEFAULT_METHOD",
    "DEFAULT_PLOT_MODES",
    "DEFAULT_RANK_BY",
    "DEFAULT_RESIDUAL_TOL",
    "DEFAULT_SIGN_CONVENTION",
    "DEFAULT_SOURCE",
    "DEFAULT_VARIANCE_THRESHOLD",
    "DecompositionResult",
    "EIGH_DRIVERS",
    "ELBOW_CURVES",
    "ELBOW_METHODS",
    "GAP_METRICS",
    "GENERATOR",
    "HEURISTICS_KEYS",
    "HEURISTIC_KEYS",
    "INPUT_VARIANT_PREFIX",
    "MATRIX_CONFIG_FIELDS",
    "METADATA_KEYS",
    "METHODS",
    "MODE_KEYS",
    "PROVENANCE_KEYS",
    "RANK_BY",
    "SAVE_DATA_KEYS",
    "SIGN_CONVENTIONS",
    "SOURCES",
    "STATS_KEYS",
    "TOP_LEVEL_KEYS",
    "VALUE_KEYS",
    "HeuristicResult",
    "SpectralConfig",
    "SpectralDecomposition",
    "SpectralMode",
    "SpectralValidationError",
    "SpectralValue",
    "apply_sign_convention",
    "artifact_paths",
    "build_argument_parser",
    #: re-exported from Phase 02: the headless probe used by every plot
    "can_popup",
    "build_sidecar_arrays",
    "build_spectral_decomposition",
    "compute_diagnostics",
    "compute_heuristics",
    "decompose_matrix",
    "degenerate_groups",
    "input_variant",
    "l_method_knee",
    "load_sidecar_arrays",
    "load_spectrum",
    "main",
    "matrix_is_symmetric",
    "plot_mode_heatmaps",
    "plot_spectrum",
    "recommended_k",
    "render_statistics",
    "render_summary_box",
    "resolve_inputs",
    "resolve_method",
    "resolve_retained_k",
    "save_data_payload",
    "second_difference_knee",
    "select_source_matrix",
    "sidecar_path",
    "validate_spectral_payload_schema",
    "variant_stem",
    "write_artifact_set",
    "write_npz_atomic",
    "write_sidecar",
]

LOGGER = logging.getLogger("src.spectral.spectral_decomposition")

# ---------------------------------------------------------------------------
# Artifact names and format constants
# ---------------------------------------------------------------------------
CANONICAL_STEM = "eigen"
DEFAULT_ARTIFACT_NAME = f"{CANONICAL_STEM}.json"
DEFAULT_SIDECAR_NAME = f"{CANONICAL_STEM}.npz"
DEFAULT_PLOT_NAME = f"{CANONICAL_STEM}.png"
DEFAULT_MODES_PLOT_NAME = f"{CANONICAL_STEM}.modes.png"
DEFAULT_DATA_NAME = f"{CANONICAL_STEM}.data.json"

#: Phase 02 artifact stem whose variant suffix Phase 03 inherits.
INPUT_VARIANT_PREFIX = "z_matrix"
DEFAULT_INPUT_PATTERN = Z_ARTIFACT_NAME
VARIANT_INPUT_PATTERN = f"{INPUT_VARIANT_PREFIX}.*.json"
#: Artifact names that look like an input but are Phase 02 save-data files.
EXCLUDED_INPUT_SUFFIXES = (".data.json",)
SUPPORTED_SUFFIX = ".json"

#: Decomposition selection.
METHODS = ("auto", "eigen", "svd")
SOURCES = ("effective", "symmetric")
RANK_BY = ("magnitude", "value")
SIGN_CONVENTIONS = ("max-abs-positive", "none")
EIGH_DRIVERS = ("evr", "evd", "ev", "evx")
ELBOW_CURVES = ("cumulative", "scree")
ELBOW_METHODS = ("l-method", "second-difference")
GAP_METRICS = ("ratio", "gap")

#: Defaults.
DEFAULT_METHOD = "auto"
DEFAULT_SOURCE = "effective"
DEFAULT_RANK_BY = "magnitude"
DEFAULT_SIGN_CONVENTION = "max-abs-positive"
DEFAULT_K = "auto"
DEFAULT_VARIANCE_THRESHOLD = 0.9
DEFAULT_ELBOW_CURVE = "cumulative"
DEFAULT_ELBOW_METHOD = "l-method"
DEFAULT_GAP_WINDOW = 25
DEFAULT_GAP_METRIC = "ratio"
DEFAULT_EIGH_DRIVER = "evr"
DEFAULT_RESIDUAL_TOL = 1e-09
DEFAULT_DEGENERATE_TOL = 1e-10
DEFAULT_PLOT_MODES = 4
DEFAULT_CMAP = "viridis"

GENERATOR = "src.spectral.spectral_decomposition"

# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------
#: Matrix/spectrum-defining config fields hashed into ``config_hash``.
MATRIX_CONFIG_FIELDS = (
    "method",
    "source",
    "rank_by",
    "sign_convention",
    "k_requested",
    "variance_threshold",
    "elbow",
    "elbow_curve",
    "elbow_method",
    "spectral_gap",
    "gap_window",
    "gap_metric",
    "driver",
    "residual_tol",
)
#: CLI-controlled fields compared against the defaults by ``is_default()``.
DEFAULT_DECISION_FIELDS = MATRIX_CONFIG_FIELDS

PROVENANCE_KEYS = frozenset(
    {
        "source_artifact",
        "source_artifact_sha256",
        "source_file",
        "source_file_sha256",
        "parsed_created_utc",
        "z_matrix_config_hash",
        "z_matrix_config",
        "phase",
    }
)
CONFIG_KEYS = frozenset(
    {
        "method",
        "resolved_method",
        "source",
        "rank_by",
        "sign_convention",
        "k_requested",
        "k_resolved",
        "variance_threshold",
        "elbow",
        "elbow_curve",
        "elbow_method",
        "spectral_gap",
        "gap_window",
        "gap_metric",
        "driver",
        "residual_tol",
        "degenerate_tol",
        "symmetric_input",
        "config_hash",
    }
)
VALUE_KEYS = frozenset(
    {
        "rank",
        "original_index",
        "value",
        "abs_value",
        "energy",
        "explained_variance_ratio",
        "cumulative_variance_ratio",
    }
)
MODE_KEYS = VALUE_KEYS | {"left", "right", "left_norm", "right_norm", "sign_flipped", "residual"}
HEURISTICS_KEYS = frozenset(
    {
        "variance_threshold",
        "elbow",
        "spectral_gap",
        "participation_ratio",
        "recommended_k",
        "rule",
    }
)
#: Every heuristic entry carries at least ``k``; the rest depends on the heuristic.
HEURISTIC_KEYS = frozenset({"k", "threshold", "curve", "method", "window", "metric", "value"})
TOP_LEVEL_KEYS = frozenset(
    {"provenance", "config", "neuron_order", "values", "modes", "heuristics", "metadata"}
)
STATS_KEYS = frozenset({"min", "max", "mean", "std"})
METADATA_KEYS = frozenset(
    {
        "n_neurons",
        "n_modes_total",
        "n_modes_retained",
        "method",
        "source",
        "matrix_is_symmetric",
        "n_positive",
        "n_negative",
        "n_zero",
        "value_min",
        "value_max",
        "abs_value_max",
        "smallest_value",
        "sum_values",
        "sum_squares",
        "frobenius_norm",
        "trace",
        "numerical_rank",
        "condition_number",
        "explained_variance_top1",
        "explained_variance_topk",
        "cumulative_at_k",
        "participation_ratio",
        "reconstruction_error",
        "reconstruction_error_abs",
        "orthogonality_error",
        "max_mode_residual",
        "degenerate_groups",
        "n_sign_flipped",
        "heuristics",
        "config_hash",
        "scipy_version",
        "numpy_version",
        "generator",
        "created_utc",
    }
)
SAVE_DATA_KEYS = frozenset(
    {
        "provenance",
        "config",
        "method",
        "source",
        "values",
        "modes",
        "heuristics",
        "n_modes_total",
        "n_modes_retained",
        "reconstruction_error",
        "reconstruction_error_abs",
        "orthogonality_error",
        "max_mode_residual",
        "energy_share",
        "generator",
        "created_utc",
    }
)

# Validation issue codes (one per rule).
CODE_INPUT_SCHEMA = "input_schema"
CODE_NOT_SYMMETRIC = "not_symmetric"
CODE_METHOD_INCOMPATIBLE = "method_incompatible"
CODE_CONFIG = "config"
CODE_DIMENSION = "dimension"
CODE_SMALL_GRAPH = "small_graph"
CODE_NONFINITE = "nonfinite"
CODE_K_RANGE = "k_range"
CODE_RESIDUAL = "residual"
CODE_DEGENERATE = "degenerate"
CODE_TRIVIAL_MATRIX = "trivial_matrix"
CODE_PAYLOAD_SCHEMA = "payload_schema"
CODE_SIDECAR_CACHE = "sidecar_cache"
CODE_ARTIFACT_WRITE = "artifact_write"
CODE_PLOT = "plot"

#: Metadata sub-block key set for the per-mode loading summaries.
LOADING_STATS_KEYS = STATS_KEYS


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SpectralConfig:
    """The complete, hashable description of *how* the spectrum was computed."""

    method: str = DEFAULT_METHOD
    #: Derived: the decomposition actually run (``eigen`` or ``svd``).
    resolved_method: str = ""
    source: str = DEFAULT_SOURCE
    rank_by: str = DEFAULT_RANK_BY
    sign_convention: str = DEFAULT_SIGN_CONVENTION
    #: ``"auto"`` (resolved from the heuristics), ``0``/``"all"`` (keep every mode)
    #: or a positive integer.  Stored as a string so ``to_dict()`` stays stable.
    k_requested: str = DEFAULT_K
    #: Derived: the number of modes retained (never hashed, recorded verbatim).
    k_resolved: int = 0
    variance_threshold: float = DEFAULT_VARIANCE_THRESHOLD
    elbow: bool = True
    elbow_curve: str = DEFAULT_ELBOW_CURVE
    elbow_method: str = DEFAULT_ELBOW_METHOD
    spectral_gap: bool = True
    gap_window: int = DEFAULT_GAP_WINDOW
    gap_metric: str = DEFAULT_GAP_METRIC
    driver: str = DEFAULT_EIGH_DRIVER
    residual_tol: float = DEFAULT_RESIDUAL_TOL
    degenerate_tol: float = DEFAULT_DEGENERATE_TOL
    #: Derived from the input: ``True`` when the decomposed matrix was symmetric.
    symmetric_input: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON payload written to ``config`` (including the hash)."""
        payload = self.hash_fields()
        payload["resolved_method"] = self.resolved_method
        payload["k_resolved"] = int(self.k_resolved)
        payload["degenerate_tol"] = float(self.degenerate_tol)
        payload["symmetric_input"] = bool(self.symmetric_input)
        payload["config_hash"] = self.config_hash()
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SpectralConfig":
        """Rebuild a config from an artifact payload (ignores derived keys)."""
        return cls(
            method=str(payload.get("method", DEFAULT_METHOD)),
            resolved_method=str(payload.get("resolved_method", "")),
            source=str(payload.get("source", DEFAULT_SOURCE)),
            rank_by=str(payload.get("rank_by", DEFAULT_RANK_BY)),
            sign_convention=str(payload.get("sign_convention", DEFAULT_SIGN_CONVENTION)),
            k_requested=str(payload.get("k_requested", DEFAULT_K)),
            k_resolved=int(payload.get("k_resolved", 0)),
            variance_threshold=float(payload.get("variance_threshold", DEFAULT_VARIANCE_THRESHOLD)),
            elbow=bool(payload.get("elbow", True)),
            elbow_curve=str(payload.get("elbow_curve", DEFAULT_ELBOW_CURVE)),
            elbow_method=str(payload.get("elbow_method", DEFAULT_ELBOW_METHOD)),
            spectral_gap=bool(payload.get("spectral_gap", True)),
            gap_window=int(payload.get("gap_window", DEFAULT_GAP_WINDOW)),
            gap_metric=str(payload.get("gap_metric", DEFAULT_GAP_METRIC)),
            driver=str(payload.get("driver", DEFAULT_EIGH_DRIVER)),
            residual_tol=float(payload.get("residual_tol", DEFAULT_RESIDUAL_TOL)),
            degenerate_tol=float(payload.get("degenerate_tol", DEFAULT_DEGENERATE_TOL)),
            symmetric_input=bool(payload.get("symmetric_input", False)),
        )

    def hash_fields(self) -> dict[str, Any]:
        """The spectrum-defining subset hashed into ``config_hash``.

        Built directly (never via :meth:`to_dict`) -- ``to_dict`` adds derived
        keys and ``config_hash`` itself, so routing through it would recurse.
        Derived/observational fields (``k_resolved``, ``symmetric_input``,
        ``degenerate_tol``) are deliberately excluded: they do not change a
        single spectrum entry, and hashing them would stop a canonical run from
        producing the canonical ``eigen.json``.
        """
        return {
            "method": self.method,
            "source": self.source,
            "rank_by": self.rank_by,
            "sign_convention": self.sign_convention,
            "k_requested": str(self.k_requested),
            "variance_threshold": float(self.variance_threshold),
            "elbow": bool(self.elbow),
            "elbow_curve": self.elbow_curve,
            "elbow_method": self.elbow_method,
            "spectral_gap": bool(self.spectral_gap),
            "gap_window": int(self.gap_window),
            "gap_metric": self.gap_metric,
            "driver": self.driver,
            "residual_tol": float(self.residual_tol),
        }

    def config_hash(self) -> str:
        """Deterministic 8-hex-character digest of :meth:`hash_fields`."""
        digest = hashlib.sha256(dumps_json(self.hash_fields()).encode("utf-8")).hexdigest()
        return digest[:8]

    def is_default(self) -> bool:
        """True when every CLI-controlled spectral parameter is at its default."""
        return all(
            getattr(self, name) == getattr(_DEFAULT_CONFIG, name) for name in DEFAULT_DECISION_FIELDS
        )


#: Reference instance used by :meth:`SpectralConfig.is_default` comparisons.
_DEFAULT_CONFIG = SpectralConfig()


@dataclass(frozen=True)
class SpectralValue:
    """One ranked entry of the spectrum (present for **every** mode, 1..N)."""

    rank: int
    original_index: int
    value: float
    abs_value: float
    energy: float
    explained_variance_ratio: float
    cumulative_variance_ratio: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": int(self.rank),
            "original_index": int(self.original_index),
            "value": float(self.value),
            "abs_value": float(self.abs_value),
            "energy": float(self.energy),
            "explained_variance_ratio": float(self.explained_variance_ratio),
            "cumulative_variance_ratio": float(self.cumulative_variance_ratio),
        }


@dataclass(frozen=True)
class SpectralMode:
    """A retained mode: its ranked value plus the two loading vectors.

    ``left`` is the source/sender axis (``U`` for SVD, ``V`` for eigen) and
    ``right`` is the target/receiver axis (``V`` for both -- the eigen path uses
    the same vector twice, so ``left == right`` there and Phase 04 can always
    read ``loadings_left``/``loadings_right`` uniformly.
    """

    rank: int
    original_index: int
    value: float
    abs_value: float
    energy: float
    explained_variance_ratio: float
    cumulative_variance_ratio: float
    left: tuple[float, ...]
    right: tuple[float, ...]
    left_norm: float
    right_norm: float
    sign_flipped: bool
    residual: float

    @classmethod
    def from_value(cls, value: SpectralValue, **vectors: Any) -> "SpectralMode":
        """Build a mode from its ranked :class:`SpectralValue` plus vectors."""
        return cls(
            rank=value.rank,
            original_index=value.original_index,
            value=value.value,
            abs_value=value.abs_value,
            energy=value.energy,
            explained_variance_ratio=value.explained_variance_ratio,
            cumulative_variance_ratio=value.cumulative_variance_ratio,
            **vectors,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": int(self.rank),
            "original_index": int(self.original_index),
            "value": float(self.value),
            "abs_value": float(self.abs_value),
            "energy": float(self.energy),
            "explained_variance_ratio": float(self.explained_variance_ratio),
            "cumulative_variance_ratio": float(self.cumulative_variance_ratio),
            "left": [float(entry) for entry in self.left],
            "right": [float(entry) for entry in self.right],
            "left_norm": float(self.left_norm),
            "right_norm": float(self.right_norm),
            "sign_flipped": bool(self.sign_flipped),
            "residual": float(self.residual),
        }


@dataclass(frozen=True)
class HeuristicResult:
    """One dimensionality-selection estimate."""

    name: str
    k: int
    value: float | None = None
    threshold: float | None = None
    curve: str | None = None
    method: str | None = None
    window: int | None = None
    metric: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"k": int(self.k)}
        for key in ("threshold", "curve", "method", "window", "metric"):
            item = getattr(self, key)
            if item is not None:
                payload[key] = item
        if self.value is not None:
            payload["value"] = float(self.value)
        return payload


@dataclass(frozen=True)
class DecompositionResult:
    """The raw output of :func:`decompose_matrix` (arrays, ranked, deterministic)."""

    method: str
    symmetric_input: bool
    n_neurons: int
    #: Number of retained modes (``1 <= k <= n_neurons``).
    k: int
    #: Ranked spectrum -- **all** ``n_neurons`` values, descending per ``rank_by``.
    values: np.ndarray
    #: ``rank_indices[i]`` is the original solver index of the i-th ranked value.
    rank_indices: np.ndarray
    abs_values: np.ndarray
    energy: np.ndarray
    explained_variance_ratio: np.ndarray
    cumulative_variance_ratio: np.ndarray
    #: Retained loading vectors, ``(n_neurons, k)``.
    left: np.ndarray
    right: np.ndarray
    sign_flipped: tuple[bool, ...]
    #: Effective numerical rank (SVD only; ``None`` for the eigen path).
    numerical_rank: int | None = None

    @property
    def n_modes_total(self) -> int:
        return int(self.values.size)

    def spectrum_values(self) -> np.ndarray:
        """Return the ranked spectrum as a plain float array."""
        return np.asarray(self.values, dtype=np.float64)


@dataclass
class SpectralDecomposition:
    """In-memory Phase 03 artifact (see :meth:`to_dict` for the JSON schema).

    ``values`` carries every mode (all ``N``) while ``modes`` carries the
    retained loading vectors (``k``), so the ranked spectrum is never truncated
    and Phase 04 has a ready ``(N x k)`` loading matrix.
    """

    source_artifact: Path
    source_artifact_sha256: str
    source_file: str
    source_file_sha256: str
    parsed_created_utc: str
    z_matrix_config_hash: str
    z_matrix_config: dict[str, Any]
    config: SpectralConfig
    neuron_order: list[str]
    values: list[SpectralValue]
    modes: list[SpectralMode]
    heuristics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    #: Where this object was read from; never serialized so ``to_dict()``
    #: round-trips byte-for-byte.
    loaded_from: Path | None = None

    @property
    def n_neurons(self) -> int:
        return len(self.neuron_order)

    @property
    def n_modes(self) -> int:
        return len(self.modes)

    @property
    def k(self) -> int:
        return len(self.modes)

    @property
    def index(self) -> dict[str, int]:
        """Neuron-id -> row index lookup (the Phase 04 neuron index map)."""
        return {name: position for position, name in enumerate(self.neuron_order)}

    @property
    def spectrum(self) -> np.ndarray:
        """Ranked spectrum as an array (length ``N``)."""
        return np.asarray([value.value for value in self.values], dtype=np.float64)

    @property
    def abs_spectrum(self) -> np.ndarray:
        return np.asarray([value.abs_value for value in self.values], dtype=np.float64)

    @property
    def explained_variance_ratio(self) -> np.ndarray:
        return np.asarray([value.explained_variance_ratio for value in self.values], dtype=np.float64)

    @property
    def cumulative_variance_ratio(self) -> np.ndarray:
        return np.asarray(
            [value.cumulative_variance_ratio for value in self.values], dtype=np.float64
        )

    @property
    def loadings_left(self) -> np.ndarray:
        """``(N x k)`` source/sender loading matrix ``L`` (the Phase 04 input)."""
        if not self.modes:
            return np.zeros((self.n_neurons, 0), dtype=np.float64)
        return np.column_stack([mode.left for mode in self.modes])

    @property
    def loadings_right(self) -> np.ndarray:
        """``(N x k)`` target/receiver loading matrix (equals ``loadings_left`` for eigen)."""
        if not self.modes:
            return np.zeros((self.n_neurons, 0), dtype=np.float64)
        return np.column_stack([mode.right for mode in self.modes])

    def to_dict(self) -> dict[str, Any]:
        """Return the exact JSON payload written to ``eigen.json``."""
        return {
            "provenance": {
                "source_artifact": str(self.source_artifact),
                "source_artifact_sha256": self.source_artifact_sha256,
                "source_file": self.source_file,
                "source_file_sha256": self.source_file_sha256,
                "parsed_created_utc": self.parsed_created_utc,
                "z_matrix_config_hash": self.z_matrix_config_hash,
                "z_matrix_config": dict(self.z_matrix_config),
                "phase": "03",
            },
            "config": self.config.to_dict(),
            "neuron_order": list(self.neuron_order),
            "values": [value.to_dict() for value in self.values],
            "modes": [mode.to_dict() for mode in self.modes],
            "heuristics": dict(self.heuristics),
            "metadata": dict(self.metadata),
        }


class SpectralValidationError(ValueError):
    """Raised when a Phase 03 input/output violates a validation rule.

    The full :class:`src.parsing.parse_graphviz.ValidationReport` is available as
    :attr:`report`, mirroring ``GraphvizValidationError`` / ``ZMatrixValidationError``.
    """

    def __init__(self, message: str, report: ValidationReport | None = None) -> None:
        super().__init__(message)
        self.report = report


# ---------------------------------------------------------------------------
# Decomposition engine
# ---------------------------------------------------------------------------
def matrix_is_symmetric(matrix: Any, *, tol: float = 1e-12) -> bool:
    """True when *matrix* is square and numerically symmetric.

    An exact comparison is tried first (Phase 02 writes a bit-exactly symmetric
    ``matrix_symmetric``), then a scale-relative ``atol`` check.
    """
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        return False
    if np.array_equal(values, values.T):
        return True
    scale = float(np.abs(values).max()) if values.size else 0.0
    return bool(np.allclose(values, values.T, rtol=0.0, atol=tol * max(scale, 1.0)))


def resolve_method(matrix: Any, requested: str = DEFAULT_METHOD) -> str:
    """Resolve ``auto`` into ``eigen`` (symmetric) or ``svd`` (directed)."""
    if requested == "auto":
        return "eigen" if matrix_is_symmetric(matrix) else "svd"
    return requested


def _rank_indices(values: np.ndarray, rank_by: str) -> np.ndarray:
    """Deterministic rank order: ``magnitude`` (default) or signed ``value``.

    Ties are broken by the signed value (for ``magnitude``), then by the
    original solver index, so the ranking is a pure function of the spectrum.
    """
    count = int(values.size)
    positions = np.arange(count)
    if rank_by == "value":
        return np.lexsort((positions, -np.abs(values), -values))
    return np.lexsort((positions, -values, -np.abs(values)))


def apply_sign_convention(
    left: np.ndarray,
    right: np.ndarray,
    convention: str = DEFAULT_SIGN_CONVENTION,
) -> tuple[np.ndarray, np.ndarray, tuple[bool, ...]]:
    """Fix the arbitrary sign of each loading pair.

    ``max-abs-positive`` flips a mode so that the largest-|.| component of its
    **left** vector is positive (first index wins on ties), which is what makes
    the artifacts byte-reproducible.  ``none`` leaves the solver output untouched
    and is therefore only useful for diagnostics.

    Both vectors of a mode are flipped **together** -- an SVD triple obeys
    ``A v_i = sigma_i u_i``, so flipping only one of ``u_i``/``v_i`` would break
    the relation (and inflate every per-mode residual).  For the eigen path
    ``left == right``, so the coupled flip is exactly a flip of the eigenvector.
    """
    left = np.array(left, dtype=np.float64, copy=True)
    right = np.array(right, dtype=np.float64, copy=True)
    n_modes = int(left.shape[1])
    flipped: list[bool] = [False] * n_modes
    if convention == "none":
        return left, right, tuple(flipped)

    for mode in range(n_modes):
        pivot = int(np.argmax(np.abs(left[:, mode])))
        if left[pivot, mode] < 0.0:
            left[:, mode] = -left[:, mode]
            right[:, mode] = -right[:, mode]
            flipped[mode] = True
    return left, right, tuple(flipped)


def _numeric_rank(values: np.ndarray, n_neurons: int) -> int:
    """Effective rank: how many singular values exceed ``max * n * eps``."""
    if values.size == 0:
        return 0
    tolerance = float(values.max()) * n_neurons * float(np.finfo(np.float64).eps)
    return int(np.count_nonzero(values > tolerance))


def decompose_matrix(
    matrix: Any,
    *,
    method: str = DEFAULT_METHOD,
    k: int | None = None,
    rank_by: str = DEFAULT_RANK_BY,
    sign_convention: str = DEFAULT_SIGN_CONVENTION,
    driver: str = DEFAULT_EIGH_DRIVER,
) -> DecompositionResult:
    """Run the eigen or SVD decomposition and rank the resulting spectrum.

    ``k = None`` (or ``0``) retains every mode.  The returned spectrum always
    contains **all** ``n_neurons`` values, ranked per *rank_by*.  SciPy is the
    mandated engine (``eigh`` with a selectable LAPACK driver, ``svd`` with
    ``gesdd``); ``check_finite=True`` makes a non-finite input an immediate,
    explicit error instead of a silent ``NaN`` spectrum.
    """
    from scipy import linalg

    array = np.asarray(matrix, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise SpectralValidationError(f"matrix must be square, got shape {tuple(array.shape)}")
    if array.size and not bool(np.all(np.isfinite(array))):
        # checked before the symmetry test: `allclose` on a NaN emits a warning
        raise SpectralValidationError("matrix contains non-finite entries")
    n_neurons = int(array.shape[0])

    resolved = resolve_method(array, method)
    symmetric = matrix_is_symmetric(array)

    if resolved == "eigen":
        if not symmetric:
            raise SpectralValidationError(
                "the eigen path requires a symmetric matrix; use --method svd for the "
                "directed Z, --source symmetric, or rebuild the matrix with --symmetric"
            )
        values_raw, vectors = linalg.eigh(array, lower=True, driver=driver, check_finite=True)
        left_raw = np.array(vectors, dtype=np.float64, copy=True)
        right_raw = np.array(vectors, dtype=np.float64, copy=True)
        numerical_rank: int | None = None
    else:
        u, singular, vt = linalg.svd(
            array, full_matrices=False, compute_uv=True, lapack_driver="gesdd", check_finite=True
        )
        values_raw = np.asarray(singular, dtype=np.float64)
        left_raw = np.array(u, dtype=np.float64, copy=True)
        right_raw = np.array(vt, dtype=np.float64).T.copy()
        numerical_rank = _numeric_rank(values_raw, n_neurons)

    order = _rank_indices(values_raw, rank_by)
    values = np.asarray(values_raw[order], dtype=np.float64)
    abs_values = np.abs(values)
    energy = values**2
    total_energy = float(energy.sum())
    ratio = energy / total_energy if total_energy > 0.0 else np.zeros_like(energy)
    cumulative = np.cumsum(ratio)

    retained = n_neurons if not k else int(k)
    retained = max(1, min(retained, n_neurons))

    left = left_raw[:, order][:, :retained]
    right = right_raw[:, order][:, :retained]
    left, right, flipped = apply_sign_convention(left, right, sign_convention)

    return DecompositionResult(
        method=resolved,
        symmetric_input=symmetric,
        n_neurons=n_neurons,
        k=retained,
        values=values,
        rank_indices=np.asarray(order, dtype=np.int64),
        abs_values=abs_values,
        energy=energy,
        explained_variance_ratio=ratio,
        cumulative_variance_ratio=cumulative,
        left=left,
        right=right,
        sign_flipped=flipped,
        numerical_rank=numerical_rank,
    )


# ---------------------------------------------------------------------------
# Dimensionality-selection heuristics
# ---------------------------------------------------------------------------
def l_method_knee(curve: Any) -> int:
    """L-method knee: the point farthest from the chord joining the endpoints.

    Returns a **1-based** ``k``.  Self-contained (no ``kneed`` dependency) and
    deterministic; for the reference directed spectrum the cumulative-variance
    curve yields ``k = 20``.
    """
    values = np.asarray(curve, dtype=np.float64).ravel()
    count = int(values.size)
    if count < 3:
        return 1
    x = np.arange(count, dtype=np.float64)
    x1, y1 = x[0], values[0]
    x2, y2 = x[-1], values[-1]
    dx, dy = x2 - x1, y2 - y1
    denominator = math.hypot(dx, dy)
    if denominator == 0.0:
        return 1
    distances = np.abs(dy * (x - x1) - dx * (values - y1)) / denominator
    return int(np.argmax(distances)) + 1


def second_difference_knee(curve: Any) -> int:
    """Knee from the largest absolute second difference (1-based ``k``)."""
    values = np.asarray(curve, dtype=np.float64).ravel()
    if values.size < 3:
        return 1
    curvature = np.abs(np.diff(values, 2))
    return int(np.argmax(curvature)) + 2


def recommended_k(
    results: Sequence[HeuristicResult], *, fallback: int = 1, n_modes: int = 1
) -> int:
    """Median of the enabled heuristics' ``k`` values, clipped to ``[1, n_modes]``.

    A *recommendation*, not a claim: with the three default heuristics on the
    reference spectrum the estimates are 22 (variance), 20 (elbow) and 1
    (spectral gap), and the median -- 20 -- is returned.
    """
    candidates = sorted(int(result.k) for result in results)
    if not candidates:
        return max(1, min(int(fallback), int(n_modes)))
    median = int(math.floor(float(np.median(np.asarray(candidates, dtype=np.float64)))))
    return max(1, min(median, int(n_modes)))


def resolve_retained_k(requested: Any, *, recommended: int, n_modes: int) -> int:
    """Resolve ``--k`` (``auto`` / ``0`` / ``all`` / N) into a retained count."""
    token = str(requested).strip().lower()
    if token in ("auto", ""):
        chosen = recommended if recommended > 0 else n_modes
    elif token in ("0", "all", "none"):
        chosen = n_modes
    else:
        chosen = int(float(token))
    return max(1, min(int(chosen), int(n_modes)))


def compute_heuristics(
    values: Any,
    *,
    variance_threshold: float = DEFAULT_VARIANCE_THRESHOLD,
    elbow: bool = True,
    elbow_curve: str = DEFAULT_ELBOW_CURVE,
    elbow_method: str = DEFAULT_ELBOW_METHOD,
    spectral_gap: bool = True,
    gap_window: int = DEFAULT_GAP_WINDOW,
    gap_metric: str = DEFAULT_GAP_METRIC,
) -> tuple[list[HeuristicResult], dict[str, Any]]:
    """Estimate a sensible mode count from the ranked spectrum.

    Returns ``(results, payload)`` where *payload* is the JSON-serializable
    ``heuristics`` block.  All three heuristics are advisory: they never drop a
    mode by themselves, only ``--k auto`` follows ``recommended_k``.

    The spectral-gap search is restricted to the leading ``gap_window``
    components on purpose: an unrestricted ``argmax`` of the consecutive ratio
    lands on the numerical-null tail (it returns ``k = 109`` for the reference
    directed spectrum, where four singular values are ~1e-17 apart), which is a
    measurement artefact rather than a mode boundary.
    """
    ranked = np.asarray(values, dtype=np.float64).ravel()
    count = int(ranked.size)
    abs_values = np.abs(ranked)
    energy = ranked**2
    total_energy = float(energy.sum())
    ratio = energy / total_energy if total_energy > 0.0 else np.zeros_like(energy)
    cumulative = np.cumsum(ratio)

    results: list[HeuristicResult] = []
    payload: dict[str, Any] = {}

    if variance_threshold and 0.0 < float(variance_threshold) <= 1.0 and count:
        k = int(np.searchsorted(cumulative, float(variance_threshold)) + 1)
        k = max(1, min(k, count))
        results.append(
            HeuristicResult("variance_threshold", k, threshold=float(variance_threshold))
        )
        payload["variance_threshold"] = {"k": k, "threshold": float(variance_threshold)}
    else:
        payload["variance_threshold"] = None

    if elbow and count >= 3:
        curve = cumulative if elbow_curve == "cumulative" else ratio
        knee = l_method_knee(curve) if elbow_method == "l-method" else second_difference_knee(curve)
        knee = max(1, min(int(knee), count))
        results.append(HeuristicResult("elbow", knee, curve=elbow_curve, method=elbow_method))
        payload["elbow"] = {"k": knee, "curve": elbow_curve, "method": elbow_method}
    else:
        payload["elbow"] = None

    if spectral_gap and count >= 3:
        window = max(2, min(int(gap_window), count - 1))
        head = abs_values[: window + 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            if gap_metric == "gap":
                deltas = head[:-1] - head[1:]
            else:
                deltas = np.where(head[1:] > 0.0, head[:-1] / head[1:], np.inf)
        index = int(np.argmax(deltas))
        gap_value = float(deltas[index]) if math.isfinite(float(deltas[index])) else None
        entry: dict[str, Any] = {
            "k": int(index) + 1,
            "window": int(window),
            "metric": gap_metric,
        }
        if gap_value is not None:
            entry["value"] = gap_value
        results.append(
            HeuristicResult(
                "spectral_gap",
                int(index) + 1,
                value=gap_value,
                window=int(window),
                metric=gap_metric,
            )
        )
        payload["spectral_gap"] = entry
    else:
        payload["spectral_gap"] = None

    participation = (
        float(abs_values.sum() ** 2 / float((abs_values**2).sum())) if total_energy > 0.0 else 0.0
    )
    payload["participation_ratio"] = participation
    payload["recommended_k"] = recommended_k(results, fallback=count, n_modes=count)
    payload["rule"] = "median of the enabled heuristic k values, clipped to [1, N]"
    return results, payload


# ---------------------------------------------------------------------------
# Diagnostics (single source of truth for metadata, --stats and --save-data)
# ---------------------------------------------------------------------------
def _loading_stats(vectors: np.ndarray) -> dict[str, float | None]:
    """``min/max/mean/std`` (ddof=0) over the entries of a loading matrix."""
    flat = np.asarray(vectors, dtype=np.float64).ravel()
    if flat.size == 0:
        return {"min": None, "max": None, "mean": None, "std": None}
    return {
        "min": float(flat.min()),
        "max": float(flat.max()),
        "mean": float(flat.mean()),
        "std": float(flat.std()),
    }


def degenerate_groups(abs_values: Any, *, tolerance: float = DEFAULT_DEGENERATE_TOL) -> dict[str, Any]:
    """Group ``abs_values`` that are equal to within *tolerance* (relative).

    Within such a group the individual eigenvectors are not uniquely defined
    (any rotation of the subspace is valid), which no sign convention can fix --
    hence the explicit diagnostic.
    """
    values = np.asarray(abs_values, dtype=np.float64).ravel()
    if values.size == 0:
        return {"count": 0, "sizes": [], "tolerance": float(tolerance)}
    threshold = float(tolerance) * max(float(values.max()), 1.0)
    sizes: list[int] = []
    run = 1
    for index in range(1, values.size):
        if abs(float(values[index]) - float(values[index - 1])) < threshold:
            run += 1
        else:
            if run > 1:
                sizes.append(run)
            run = 1
    if run > 1:
        sizes.append(run)
    return {"count": len(sizes), "sizes": sizes, "tolerance": float(tolerance)}


def _mode_residuals(result: DecompositionResult, matrix: np.ndarray) -> np.ndarray:
    """Per-retained-mode residual ``||A v - value * (partner)||_inf``.

    For the eigen path the partner is the same vector; for SVD both the left and
    the right singular relations are checked, which is what catches a wrongly
    matched singular triple.
    """
    residuals = np.zeros(result.k, dtype=np.float64)
    for mode in range(result.k):
        value = float(result.values[mode])
        left = result.left[:, mode]
        right = result.right[:, mode]
        if result.method == "eigen":
            residual = float(np.abs(matrix @ right - value * right).max())
        else:
            residual = max(
                float(np.abs(matrix @ right - value * left).max()),
                float(np.abs(matrix.T @ left - value * right).max()),
            )
        residuals[mode] = residual
    return residuals


def compute_diagnostics(
    result: DecompositionResult,
    matrix: Any,
    *,
    source: str = DEFAULT_SOURCE,
    heuristics: dict[str, Any] | None = None,
    config_hash: str = "",
    created_utc: str = "",
    degenerate_tol: float = DEFAULT_DEGENERATE_TOL,
) -> dict[str, Any]:
    """Compute every statistic Phase 03 reports.

    Conventions (documented in the Phase 03 plan):

    * ``explained_variance_ratio`` is ``value**2 / sum(value**2)`` for **both**
      paths, so ``sum_squares == ||A||_F**2`` and the eigen and SVD spectra are
      directly comparable.
    * ``frobenius_norm`` is ``numpy.linalg.norm(A)`` (so it matches Phase 02's
      ``metadata.frobenius_norm``); ``sum_squares`` is the spectral identity
      check ``sum(value**2) == frobenius_norm**2``.
    * ``reconstruction_error`` is relative ``||A_rec(k) - A||_F / ||A||_F``; with
      ``k == N`` it is the *full* decomposition check, with ``k < N`` the
      (informational) truncation error.
    * ``condition_number`` and ``numerical_rank`` are SVD-only (``None`` for the
      eigen path).
    """
    import scipy

    array = np.asarray(matrix, dtype=np.float64)
    values = np.asarray(result.values, dtype=np.float64)
    abs_values = np.asarray(result.abs_values, dtype=np.float64)
    energy = np.asarray(result.energy, dtype=np.float64)
    ratio = np.asarray(result.explained_variance_ratio, dtype=np.float64)
    cumulative = np.asarray(result.cumulative_variance_ratio, dtype=np.float64)

    square_sum = float(energy.sum())
    frobenius = float(np.linalg.norm(array)) if array.size else 0.0

    retained_values = values[: result.k]
    reconstruction = result.left @ (retained_values[:, None] * result.right.T)
    difference = reconstruction - array
    error_abs = float(np.linalg.norm(difference))
    error_rel = error_abs / frobenius if frobenius > 0.0 else 0.0

    orthogonality = 0.0
    if result.k:
        identifier = np.eye(result.k, dtype=np.float64)
        for basis in (result.left, result.right):
            orthogonality = max(
                orthogonality, float(np.abs(basis.T @ basis - identifier).max())
            )

    residuals = _mode_residuals(result, array)
    max_residual = float(residuals.max()) if residuals.size else 0.0

    condition_number: float | None = None
    if result.method == "svd" and result.numerical_rank:
        ordered = np.sort(abs_values)[::-1]
        smallest = float(ordered[result.numerical_rank - 1])
        if smallest > 0.0:
            condition_number = float(abs_values.max() / smallest)

    heuristics = dict(heuristics or {})
    return {
        "n_neurons": int(result.n_neurons),
        "n_modes_total": int(result.n_modes_total),
        "n_modes_retained": int(result.k),
        "method": result.method,
        "source": source,
        "matrix_is_symmetric": bool(result.symmetric_input),
        "n_positive": int((values > 0.0).sum()),
        "n_negative": int((values < 0.0).sum()),
        "n_zero": int((values == 0.0).sum()),
        "value_min": float(values.min()) if values.size else 0.0,
        "value_max": float(values.max()) if values.size else 0.0,
        "abs_value_max": float(abs_values.max()) if abs_values.size else 0.0,
        "smallest_value": float(abs_values.min()) if abs_values.size else 0.0,
        "sum_values": float(values.sum()),
        "sum_squares": square_sum,
        "frobenius_norm": frobenius,
        "trace": float(np.trace(array)),
        "numerical_rank": result.numerical_rank,
        "condition_number": condition_number,
        "explained_variance_top1": float(ratio[0]) if ratio.size else 0.0,
        "explained_variance_topk": float(cumulative[result.k - 1]) if cumulative.size else 0.0,
        "cumulative_at_k": float(cumulative[result.k - 1]) if cumulative.size else 0.0,
        "participation_ratio": heuristics.get("participation_ratio"),
        "reconstruction_error": error_rel,
        "reconstruction_error_abs": error_abs,
        "orthogonality_error": orthogonality,
        "max_mode_residual": max_residual,
        "degenerate_groups": degenerate_groups(abs_values, tolerance=degenerate_tol),
        "n_sign_flipped": int(sum(1 for flag in result.sign_flipped if flag)),
        "heuristics": heuristics,
        "config_hash": config_hash,
        "scipy_version": str(scipy.__version__),
        "numpy_version": str(np.__version__),
        "generator": GENERATOR,
        "created_utc": created_utc,
    }


# ---------------------------------------------------------------------------
# Input handling and the top-level builder
# ---------------------------------------------------------------------------
def select_source_matrix(z_matrix: ZMatrix, source: str = DEFAULT_SOURCE) -> np.ndarray:
    """Return the matrix to decompose.

    ``effective`` uses Phase 02's effective matrix (``Z_sym`` when that run was
    ``--symmetric``, otherwise the directed ``Z``); ``symmetric`` always uses the
    bit-exactly symmetric ``matrix_symmetric``.
    """
    if source == "symmetric":
        return np.asarray(z_matrix.matrix_symmetric, dtype=np.float64)
    return np.asarray(z_matrix.matrix, dtype=np.float64)


def _validate_config(config: SpectralConfig, n_neurons: int, report: ValidationReport) -> None:
    """Validate the CLI-controlled spectral parameters."""
    if config.method not in METHODS:
        report.error(CODE_CONFIG, f"method {config.method!r} is not one of {METHODS}")
    if config.source not in SOURCES:
        report.error(CODE_CONFIG, f"source {config.source!r} is not one of {SOURCES}")
    if config.rank_by not in RANK_BY:
        report.error(CODE_CONFIG, f"rank_by {config.rank_by!r} is not one of {RANK_BY}")
    if config.sign_convention not in SIGN_CONVENTIONS:
        report.error(
            CODE_CONFIG, f"sign_convention {config.sign_convention!r} is not one of {SIGN_CONVENTIONS}"
        )
    if config.driver not in EIGH_DRIVERS:
        report.error(CODE_CONFIG, f"driver {config.driver!r} is not one of {EIGH_DRIVERS}")
    if config.elbow_curve not in ELBOW_CURVES:
        report.error(CODE_CONFIG, f"elbow_curve {config.elbow_curve!r} is not one of {ELBOW_CURVES}")
    if config.elbow_method not in ELBOW_METHODS:
        report.error(CODE_CONFIG, f"elbow_method {config.elbow_method!r} is not one of {ELBOW_METHODS}")
    if config.gap_metric not in GAP_METRICS:
        report.error(CODE_CONFIG, f"gap_metric {config.gap_metric!r} is not one of {GAP_METRICS}")
    if not math.isfinite(config.variance_threshold) or config.variance_threshold > 1.0:
        report.error(CODE_CONFIG, f"variance_threshold must be <= 1.0, got {config.variance_threshold}")
    if config.gap_window < 2:
        report.error(CODE_CONFIG, f"gap_window must be >= 2, got {config.gap_window}")
    if not math.isfinite(config.residual_tol) or config.residual_tol <= 0.0:
        report.error(CODE_CONFIG, f"residual_tol must be finite and > 0, got {config.residual_tol}")
    if not math.isfinite(config.degenerate_tol) or config.degenerate_tol <= 0.0:
        report.error(
            CODE_CONFIG, f"degenerate_tol must be finite and > 0, got {config.degenerate_tol}"
        )
    token = str(config.k_requested).strip().lower()
    if token not in ("auto", "", "all", "none", "0"):
        try:
            requested = int(float(token))
        except (TypeError, ValueError):
            report.error(CODE_K_RANGE, f"k {config.k_requested!r} is not 'auto', 0 or an integer")
        else:
            if requested < 0 or requested > n_neurons:
                report.error(CODE_K_RANGE, f"k must be in [0, {n_neurons}], got {requested}")


def _retained_result(result: DecompositionResult, k: int) -> DecompositionResult:
    """Slice a full decomposition down to the first *k* ranked modes."""
    k = max(1, min(int(k), result.n_neurons))
    if k == result.k:
        return result
    return replace(
        result,
        k=k,
        left=result.left[:, :k],
        right=result.right[:, :k],
        sign_flipped=result.sign_flipped[:k],
    )


def build_spectral_decomposition(
    z_matrix: ZMatrix,
    *,
    payload: dict[str, Any] | None = None,
    source_artifact: str | Path | None = None,
    config: SpectralConfig | None = None,
    report: ValidationReport | None = None,
    now: Any = None,
) -> tuple[SpectralDecomposition, ValidationReport]:
    """Decompose *z_matrix* and return the Phase 03 artifact plus a report.

    *payload* (the raw Phase 02 JSON) is optional; when given it is checked
    against the Phase 02 schema first, so a hand-edited artifact is reported
    rather than silently decomposed.  Warnings are escalated to errors by the
    CLI's ``--strict``, not here.
    """
    config = config or SpectralConfig()
    report = report if report is not None else ValidationReport()

    if payload is not None:
        for problem in validate_z_payload_schema(payload):
            report.error(CODE_INPUT_SCHEMA, problem)

    matrix = select_source_matrix(z_matrix, config.source)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        issue = report.error(
            CODE_DIMENSION, f"expected a square matrix, got shape {tuple(matrix.shape)}"
        )
        raise SpectralValidationError(issue.message, report)
    n_neurons = int(matrix.shape[0])
    if matrix.size and not bool(np.all(np.isfinite(matrix))):
        issue = report.error(CODE_NONFINITE, "the matrix contains non-finite entries")
        raise SpectralValidationError(issue.message, report)
    if n_neurons < 2:
        report.warning(CODE_SMALL_GRAPH, f"only {n_neurons} neuron(s): the spectrum is degenerate")

    _validate_config(config, n_neurons, report)
    if report.has_errors:
        raise SpectralValidationError(
            f"{len(report.errors)} validation error(s) (input schema / Phase 03 configuration); "
            "see the attached report",
            report,
        )

    if config.method == "eigen" and not matrix_is_symmetric(matrix):
        issue = report.error(
            CODE_METHOD_INCOMPATIBLE,
            "--method eigen needs a symmetric matrix; the selected source is directed -- "
            "use --method svd, or --source symmetric, or rebuild with --symmetric",
        )
        # raise rather than let decompose_matrix fail later: the report (and its
        # message) must reach the caller intact
        raise SpectralValidationError(issue.message, report)

    full = decompose_matrix(
        matrix,
        method=config.method,
        k=None,
        rank_by=config.rank_by,
        sign_convention=config.sign_convention,
        driver=config.driver,
    )

    _, heuristics = compute_heuristics(
        full.values,
        variance_threshold=config.variance_threshold,
        elbow=config.elbow,
        elbow_curve=config.elbow_curve,
        elbow_method=config.elbow_method,
        spectral_gap=config.spectral_gap,
        gap_window=config.gap_window,
        gap_metric=config.gap_metric,
    )
    retained = resolve_retained_k(
        config.k_requested, recommended=int(heuristics["recommended_k"]), n_modes=n_neurons
    )
    result = _retained_result(full, retained)
    resolved_config = replace(
        config,
        resolved_method=result.method,
        k_resolved=result.k,
        symmetric_input=result.symmetric_input,
    )

    metadata = compute_diagnostics(
        result,
        matrix,
        source=config.source,
        heuristics=heuristics,
        config_hash=resolved_config.config_hash(),
        created_utc=utc_timestamp(now),
        degenerate_tol=config.degenerate_tol,
    )

    _check_numerical_health(metadata, result, config, report)
    return (
        _assemble_decomposition(
            z_matrix, source_artifact, matrix, resolved_config, result, metadata, heuristics
        ),
        report,
    )


def _check_numerical_health(
    metadata: dict[str, Any],
    result: DecompositionResult,
    config: SpectralConfig,
    report: ValidationReport,
) -> None:
    """Turn the diagnostic numbers into validation findings.

    A full decomposition (``k == N``) must reconstruct the matrix to within
    ``--residual-tol``; a truncated one is only informational.

    Numerical-null singular values and degenerate eigenvalue clusters are
    reported as **informational log lines**, not warnings: a directed matrix
    with a numerical null space is expected here (the reference artifact has four
    singular values ~1e-17), and ``--strict`` must stay usable on the reference
    dataset.  Both quantities are always recorded in ``metadata.numerical_rank``
    and ``metadata.degenerate_groups``, and printed by ``--stats``.
    """
    if result.k == result.n_neurons and float(metadata["reconstruction_error"]) > config.residual_tol:
        report.error(
            CODE_RESIDUAL,
            f"full decomposition residual {metadata['reconstruction_error']:.3e} exceeds "
            f"--residual-tol {config.residual_tol:.1e}",
        )
    if (
        result.method == "svd"
        and result.numerical_rank is not None
        and result.numerical_rank < result.n_neurons
    ):
        null_count = result.n_neurons - result.numerical_rank
        LOGGER.info(
            "[%s] the directed matrix has %d numerical-null singular value(s) "
            "(effective rank %d of %d)",
            CODE_DIMENSION,
            null_count,
            result.numerical_rank,
            result.n_neurons,
        )
    groups = metadata.get("degenerate_groups") or {}
    if int(groups.get("count", 0)) > 0:
        LOGGER.info(
            "[%s] %d degenerate spectral cluster(s) of sizes %s -- the vectors inside a "
            "cluster are defined only up to a rotation",
            CODE_DEGENERATE,
            groups["count"],
            groups.get("sizes"),
        )
    if float(metadata["abs_value_max"]) == 0.0:
        report.warning(CODE_TRIVIAL_MATRIX, "the matrix is entirely zero: every mode is null")


def _assemble_decomposition(
    z_matrix: ZMatrix,
    source_artifact: str | Path | None,
    matrix: Any,
    config: SpectralConfig,
    result: DecompositionResult,
    metadata: dict[str, Any],
    heuristics: dict[str, Any],
) -> SpectralDecomposition:
    """Build the :class:`SpectralDecomposition` from the ranked arrays."""
    n_neurons = result.n_neurons
    values = [
        SpectralValue(
            rank=index + 1,
            original_index=int(result.rank_indices[index]),
            value=float(result.values[index]),
            abs_value=float(result.abs_values[index]),
            energy=float(result.energy[index]),
            explained_variance_ratio=float(result.explained_variance_ratio[index]),
            cumulative_variance_ratio=float(result.cumulative_variance_ratio[index]),
        )
        for index in range(n_neurons)
    ]

    residuals = _mode_residuals(result, np.asarray(matrix, dtype=np.float64))
    modes = [
        SpectralMode.from_value(
            values[index],
            left=tuple(float(entry) for entry in result.left[:, index]),
            right=tuple(float(entry) for entry in result.right[:, index]),
            left_norm=float(np.linalg.norm(result.left[:, index])),
            right_norm=float(np.linalg.norm(result.right[:, index])),
            sign_flipped=bool(result.sign_flipped[index]),
            residual=float(residuals[index]),
        )
        for index in range(result.k)
    ]

    provenance_path = (
        Path(source_artifact) if source_artifact is not None else z_matrix.source_artifact
    )
    return SpectralDecomposition(
        source_artifact=provenance_path,
        source_artifact_sha256=z_matrix.source_artifact_sha256,
        source_file=z_matrix.source_file,
        source_file_sha256=z_matrix.source_file_sha256,
        parsed_created_utc=z_matrix.parsed_created_utc,
        z_matrix_config_hash=z_matrix.config.config_hash(),
        z_matrix_config=z_matrix.config.to_dict(),
        config=config,
        neuron_order=list(z_matrix.neuron_order),
        values=values,
        modes=modes,
        heuristics=dict(heuristics),
        metadata=dict(metadata),
        diagnostics=dict(metadata),
    )


# ---------------------------------------------------------------------------
# Artifact paths
# ---------------------------------------------------------------------------
def input_variant(source_artifact: str | Path | None) -> str:
    """Variant suffix of a Phase 02 artifact (``z_matrix.f8652585`` -> ``.f8652585``).

    Phase 03 inherits it so that a spectrum computed from a variant matrix can
    never overwrite the canonical ``eigen.json``.
    """
    if source_artifact is None:
        return ""
    stem = Path(source_artifact).stem
    if stem == INPUT_VARIANT_PREFIX:
        return ""
    if stem.startswith(f"{INPUT_VARIANT_PREFIX}."):
        return stem[len(INPUT_VARIANT_PREFIX) :]
    return ""


def variant_stem(
    config: SpectralConfig,
    *,
    source_artifact: str | Path | None = None,
    force_config_hash: bool = False,
) -> str:
    """Return ``eigen[<input variant>][.<config_hash8>]`` for *config*.

    The hash segment is added automatically for any non-default spectral config
    (``--config-hash`` forces it even for a default config) so a variant run can
    never overwrite the canonical artifact.
    """
    stem = CANONICAL_STEM + input_variant(source_artifact)
    if force_config_hash or not config.is_default():
        stem += f".{config.config_hash()}"
    return stem


def artifact_paths(
    decomposition: SpectralDecomposition,
    outdir: str | Path,
    *,
    force_config_hash: bool = False,
) -> dict[str, Path]:
    """Return the five Phase 03 artifact paths for *decomposition*.

    ``{"json", "npz", "png", "modes", "data"}`` under ``<outdir>/<gv-stem>/`` --
    the same directory Phase 01 and Phase 02 write to.
    """
    stem = variant_stem(
        decomposition.config,
        source_artifact=decomposition.source_artifact,
        force_config_hash=force_config_hash,
    )
    directory = (
        Path(outdir) / Path(decomposition.source_file).stem
        if decomposition.source_file
        else Path(outdir) / stem
    )
    return {
        "json": directory / f"{stem}.json",
        "npz": directory / f"{stem}.npz",
        "png": directory / f"{stem}.png",
        "modes": directory / f"{stem}.modes.png",
        "data": directory / f"{stem}.data.json",
    }


def sidecar_path(json_path: str | Path) -> Path:
    """``eigen[.variant].json`` -> ``eigen[.variant].npz``."""
    return Path(json_path).with_suffix(".npz")


# ---------------------------------------------------------------------------
# Sidecar (derived npz array cache)
# ---------------------------------------------------------------------------
def build_sidecar_arrays(
    decomposition: SpectralDecomposition, *, source_json_sha256: str
) -> dict[str, np.ndarray]:
    """Return the ordered array bundle stored in the ``.npz`` cache.

    Ordering and dtypes are fixed (``<f8`` / ``<i8`` / ``<U*``) so the archive is
    byte-reproducible: numpy writes every zip entry with a constant ``date_time``
    and the ``.npy`` header carries no timestamp.
    """
    order_length = max((len(name) for name in decomposition.neuron_order), default=1)
    return {
        "values": np.asarray(decomposition.spectrum, dtype="<f8"),
        "loadings_left": np.asarray(decomposition.loadings_left, dtype="<f8"),
        "loadings_right": np.asarray(decomposition.loadings_right, dtype="<f8"),
        "explained_variance_ratio": np.asarray(
            decomposition.explained_variance_ratio, dtype="<f8"
        ),
        "rank_indices": np.asarray(
            [value.original_index for value in decomposition.values], dtype="<i8"
        ),
        "neuron_order": np.asarray(decomposition.neuron_order, dtype=f"<U{order_length}"),
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


def write_sidecar(decomposition: SpectralDecomposition, json_path: str | Path) -> Path:
    """Write the ``.npz`` cache for *decomposition*, keyed by the JSON's SHA-256."""
    json_path = Path(json_path)
    digest = sha256_file(json_path)
    arrays = build_sidecar_arrays(decomposition, source_json_sha256=digest)
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
    decomposition: SpectralDecomposition,
    json_path: Path,
) -> None:
    """Raise :class:`SpectralValidationError` when the cache disagrees with the JSON."""
    expected_sha = sha256_file(json_path)
    recorded = arrays.get("source_json_sha256")
    if recorded is None or str(recorded[0]) != expected_sha:
        raise SpectralValidationError(
            f"sidecar {sidecar_path(json_path).name} was built from a different JSON "
            f"(recorded {None if recorded is None else str(recorded[0])[:12]}, "
            f"actual {expected_sha[:12]})"
        )
    if not np.array_equal(arrays["values"], decomposition.spectrum):
        raise SpectralValidationError("sidecar `values` differs from the JSON spectrum")
    if not np.array_equal(arrays["loadings_left"], decomposition.loadings_left):
        raise SpectralValidationError("sidecar `loadings_left` differs from the JSON modes")
    if not np.array_equal(arrays["loadings_right"], decomposition.loadings_right):
        raise SpectralValidationError("sidecar `loadings_right` differs from the JSON modes")
    if list(arrays["neuron_order"]) != list(decomposition.neuron_order):
        raise SpectralValidationError("sidecar `neuron_order` differs from the JSON order")


def _spectrum_from_payload(
    payload: dict[str, Any], *, loaded_from: Path | None = None
) -> SpectralDecomposition:
    """Rebuild a :class:`SpectralDecomposition` from a JSON payload.

    Every ``provenance`` field is taken verbatim from the payload so that
    ``load_spectrum(path).to_dict()`` reproduces the artifact byte-for-byte;
    *loaded_from* only records where the object was read from.
    """
    config = SpectralConfig.from_dict(payload.get("config") or {})
    provenance = payload.get("provenance") or {}
    neuron_order = [str(name) for name in payload.get("neuron_order") or []]

    values = [
        SpectralValue(
            rank=int(entry["rank"]),
            original_index=int(entry["original_index"]),
            value=float(entry["value"]),
            abs_value=float(entry["abs_value"]),
            energy=float(entry["energy"]),
            explained_variance_ratio=float(entry["explained_variance_ratio"]),
            cumulative_variance_ratio=float(entry["cumulative_variance_ratio"]),
        )
        for entry in payload.get("values") or []
    ]
    modes = [
        SpectralMode(
            rank=int(entry["rank"]),
            original_index=int(entry["original_index"]),
            value=float(entry["value"]),
            abs_value=float(entry["abs_value"]),
            energy=float(entry["energy"]),
            explained_variance_ratio=float(entry["explained_variance_ratio"]),
            cumulative_variance_ratio=float(entry["cumulative_variance_ratio"]),
            left=tuple(float(item) for item in entry.get("left") or []),
            right=tuple(float(item) for item in entry.get("right") or []),
            left_norm=float(entry.get("left_norm", 0.0)),
            right_norm=float(entry.get("right_norm", 0.0)),
            sign_flipped=bool(entry.get("sign_flipped", False)),
            residual=float(entry.get("residual", 0.0)),
        )
        for entry in payload.get("modes") or []
    ]

    return SpectralDecomposition(
        source_artifact=Path(str(provenance.get("source_artifact") or "<in-memory>")),
        source_artifact_sha256=str(provenance.get("source_artifact_sha256") or ""),
        source_file=str(provenance.get("source_file") or ""),
        source_file_sha256=str(provenance.get("source_file_sha256") or ""),
        parsed_created_utc=str(provenance.get("parsed_created_utc") or ""),
        z_matrix_config_hash=str(provenance.get("z_matrix_config_hash") or ""),
        z_matrix_config=dict(provenance.get("z_matrix_config") or {}),
        config=config,
        neuron_order=neuron_order,
        values=values,
        modes=modes,
        heuristics=dict(payload.get("heuristics") or {}),
        metadata=dict(payload.get("metadata") or {}),
        diagnostics=dict(payload.get("metadata") or {}),
        loaded_from=loaded_from,
    )


def load_spectrum(
    path: str | Path,
    *,
    prefer_sidecar: bool = True,
) -> SpectralDecomposition:
    """Load a Phase 03 artifact (the entry point for Phases 04-06).

    The canonical JSON is always the source of truth: heuristics, metadata,
    config and modes are read from it.  When the sibling ``.npz`` cache exists it
    is *verified* against the JSON's SHA-256 and arrays; a stale or tampered cache
    is ignored with a warning instead of raising.  ``path`` may also be the
    ``.npz`` file, in which case the sibling JSON is required.
    """
    candidate = Path(path)

    if candidate.suffix == ".npz":
        json_path = candidate.with_suffix(".json")
        if not json_path.is_file():
            raise FileNotFoundError(
                f"the npz sidecar {candidate} has no sibling JSON artifact at {json_path}"
            )
        decomposition = _spectrum_from_payload(read_json(json_path), loaded_from=json_path)
        _verify_sidecar(load_sidecar_arrays(candidate), decomposition, json_path)
        return decomposition

    payload = read_json(candidate)
    decomposition = _spectrum_from_payload(payload, loaded_from=candidate)

    if prefer_sidecar:
        cache = sidecar_path(candidate)
        if cache.is_file():
            try:
                _verify_sidecar(load_sidecar_arrays(cache), decomposition, candidate)
            except Exception as exc:  # stale/tampered/legacy cache: fall back to JSON
                LOGGER.warning("ignoring sidecar %s: %s", cache.name, exc)
    return decomposition


# ---------------------------------------------------------------------------
# Payload validation
# ---------------------------------------------------------------------------
def validate_spectral_payload_schema(payload: dict[str, Any]) -> list[str]:
    """Hand-rolled schema check for the Phase 03 JSON artifact.

    ``jsonschema``/``pydantic`` are not installed in this environment, so this
    mirrors Phase 01/02: return a list of human-readable problems (empty list ==
    conforming).
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
        if config.get("resolved_method") not in ("eigen", "svd"):
            problems.append("config.resolved_method: expected 'eigen' or 'svd'")
        if config.get("rank_by") not in RANK_BY:
            problems.append(f"config.rank_by: {config.get('rank_by')!r} is not one of {RANK_BY}")
        if not isinstance(config.get("config_hash"), str) or len(str(config["config_hash"])) != 8:
            problems.append("config.config_hash: expected 8 hex characters")
        if not isinstance(config.get("symmetric_input"), bool):
            problems.append("config.symmetric_input: not a boolean")

    neuron_order = payload.get("neuron_order")
    if not isinstance(neuron_order, list) or not all(isinstance(name, str) for name in neuron_order):
        problems.append(
            "neuron_order: expected a list of strings (a dict would lose the index order)"
        )
        neuron_order = []
    n_neurons = len(neuron_order)

    values = payload.get("values")
    if not isinstance(values, list) or not values:
        problems.append("values: expected a non-empty list")
    else:
        if len(values) != n_neurons:
            problems.append(f"values: expected {n_neurons} ranked entries, got {len(values)}")
        for position, entry in enumerate(values):
            if not isinstance(entry, dict):
                problems.append(f"values[{position}]: not an object")
                continue
            _require_keys(entry.keys(), VALUE_KEYS, f"values[{position}]")
        ranks = [entry.get("rank") for entry in values if isinstance(entry, dict)]
        if ranks != list(range(1, len(values) + 1)):
            problems.append("values: ranks are not 1..N in order")

    modes = payload.get("modes")
    if not isinstance(modes, list) or not modes:
        problems.append("modes: expected a non-empty list")
    else:
        for position, entry in enumerate(modes):
            if not isinstance(entry, dict):
                problems.append(f"modes[{position}]: not an object")
                continue
            _require_keys(entry.keys(), MODE_KEYS, f"modes[{position}]")
            for side in ("left", "right"):
                vector = entry.get(side)
                if not isinstance(vector, list) or len(vector) != n_neurons:
                    problems.append(f"modes[{position}].{side}: expected {n_neurons} floats")
                    break

    heuristics = payload.get("heuristics")
    if not isinstance(heuristics, dict):
        problems.append("heuristics: not an object")
    else:
        _require_keys(heuristics.keys(), HEURISTICS_KEYS, "heuristics")
        for name in ("variance_threshold", "elbow", "spectral_gap"):
            entry = heuristics.get(name)
            if entry is not None and (not isinstance(entry, dict) or "k" not in entry):
                problems.append(f"heuristics.{name}: expected null or an object with 'k'")
        participation = heuristics.get("participation_ratio")
        if not isinstance(participation, (int, float)) or isinstance(participation, bool):
            problems.append("heuristics.participation_ratio: expected a number")
        if not isinstance(heuristics.get("recommended_k"), int):
            problems.append("heuristics.recommended_k: expected an integer")

    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        problems.append("metadata: not an object")
    else:
        _require_keys(metadata.keys(), METADATA_KEYS, "metadata")

    return problems


# ---------------------------------------------------------------------------
# Diagnostics output: --save-data, --stats, the completion box, --plot
# ---------------------------------------------------------------------------
def save_data_payload(decomposition: SpectralDecomposition, *, now: Any = None) -> dict[str, Any]:
    """The ``eigen.data.json`` payload (per-mode detail tables)."""
    spectrum_total = float(np.asarray([value.energy for value in decomposition.values]).sum())

    def _share(energy: float) -> float:
        return float(energy / spectrum_total) if spectrum_total else 0.0

    return {
        "provenance": decomposition.to_dict()["provenance"],
        "config": decomposition.config.to_dict(),
        "method": decomposition.metadata.get("method"),
        "source": decomposition.metadata.get("source"),
        "n_modes_total": decomposition.metadata.get("n_modes_total"),
        "n_modes_retained": decomposition.metadata.get("n_modes_retained"),
        "values": [value.to_dict() for value in decomposition.values],
        "modes": [
            {
                "rank": int(mode.rank),
                "original_index": int(mode.original_index),
                "value": float(mode.value),
                "abs_value": float(mode.abs_value),
                "energy": float(mode.energy),
                "energy_share": _share(mode.energy),
                "explained_variance_ratio": float(mode.explained_variance_ratio),
                "left_norm": float(mode.left_norm),
                "right_norm": float(mode.right_norm),
                "sign_flipped": bool(mode.sign_flipped),
                "residual": float(mode.residual),
                "left_argmax": int(np.argmax(np.abs(np.asarray(mode.left)))) if mode.left else 0,
                "right_argmax": int(np.argmax(np.abs(np.asarray(mode.right)))) if mode.right else 0,
                "left_stats": _loading_stats(np.asarray(mode.left)),
                "right_stats": _loading_stats(np.asarray(mode.right)),
            }
            for mode in decomposition.modes
        ],
        "heuristics": dict(decomposition.heuristics),
        "reconstruction_error": decomposition.metadata.get("reconstruction_error"),
        "reconstruction_error_abs": decomposition.metadata.get("reconstruction_error_abs"),
        "orthogonality_error": decomposition.metadata.get("orthogonality_error"),
        "max_mode_residual": decomposition.metadata.get("max_mode_residual"),
        "energy_share": [_share(mode.energy) for mode in decomposition.modes],
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


def _format_scientific(value: Any) -> str:
    """Format an optional float in scientific notation."""
    if value is None:
        return "n/a"
    return f"{float(value):.3e}"


def render_statistics(decomposition: SpectralDecomposition) -> str:
    """Terminal-only statistics block printed by ``--stats``."""
    metadata = decomposition.metadata
    heuristics = decomposition.heuristics
    top = decomposition.values[: min(5, len(decomposition.values))]

    def _k(name: str) -> str:
        entry = heuristics.get(name)
        return "off" if entry is None else str(entry.get("k"))

    lines = [
        "Phase 03 -- spectral statistics",
        f"  method / source     : {metadata.get('method')} / {metadata.get('source')}"
        f"  (symmetric input: {metadata.get('matrix_is_symmetric')})",
        f"  shape               : {decomposition.n_neurons} x {decomposition.n_neurons}",
        f"  values              : {metadata.get('n_modes_total')} total, "
        f"{metadata.get('n_modes_retained')} mode(s) retained",
        "  top 5               : " + ", ".join(_format(entry.value) for entry in top),
        f"  |value| max         : {_format(metadata.get('abs_value_max'))}",
        f"  explained var top1  : {_format(metadata.get('explained_variance_top1'))}",
        f"  cumulative @ k      : {_format(metadata.get('cumulative_at_k'))}",
        f"  heuristics          : threshold k={_k('variance_threshold')}"
        f" | elbow k={_k('elbow')} | gap k={_k('spectral_gap')}",
        f"  recommended k       : {heuristics.get('recommended_k')}",
        f"  participation ratio : {_format(heuristics.get('participation_ratio'))}",
        f"  frobenius norm      : {_format(metadata.get('frobenius_norm'))}",
        f"  sum(value^2)        : {_format(metadata.get('sum_squares'))}",
        f"  reconstruction error: {_format_scientific(metadata.get('reconstruction_error'))}"
        "  (relative, retained modes)",
        f"  orthogonality error : {_format_scientific(metadata.get('orthogonality_error'))}",
        f"  max mode residual   : {_format_scientific(metadata.get('max_mode_residual'))}",
        f"  numerical rank      : {metadata.get('numerical_rank')}"
        f"   condition number: {_format_scientific(metadata.get('condition_number'))}",
        f"  degenerate clusters : {(metadata.get('degenerate_groups') or {}).get('count')}",
        f"  sign flips          : {metadata.get('n_sign_flipped')}"
        f"   trace: {_format(metadata.get('trace'))}",
        f"  scipy / numpy       : {metadata.get('scipy_version')} / {metadata.get('numpy_version')}",
    ]
    return "\n".join(lines)


def render_summary_box(
    decomposition: SpectralDecomposition,
    paths: dict[str, Path],
    *,
    written: bool = True,
    sidecar: bool = True,
    plot: bool = False,
    plot_modes: int = 0,
    save_data: bool = False,
    stats: bool = False,
) -> str:
    """The boxed terminal-only Phase 03 completion summary."""
    metadata = decomposition.metadata
    heuristics = decomposition.heuristics

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

    if not plot:
        plot_line = "not requested (--plot)"
    elif written:
        plot_line = _name("png")
        if plot_modes > 0:
            plot_line += f" + {_name('modes')}"
    else:
        plot_line = _skip("--dry-run")

    def _k(name: str) -> str:
        entry = heuristics.get(name)
        return "off" if entry is None else str(entry.get("k"))

    lines = [
        "Phase 03 complete" if written else "Phase 03 dry-run (nothing written)",
        f"Method: {str(metadata.get('method')).upper()} (source: {metadata.get('source')})",
        f"Spectrum: {metadata.get('n_modes_total')} values, top |v| = "
        f"{_format(metadata.get('abs_value_max'))}",
        f"Modes retained: {metadata.get('n_modes_retained')} "
        f"(recommended_k={heuristics.get('recommended_k')})",
        f"Heuristics: threshold k={_k('variance_threshold')} | elbow k={_k('elbow')}"
        f" | gap k={_k('spectral_gap')}",
        f"Cum. variance @k: {_format(metadata.get('cumulative_at_k'))}",
        f"Reconstruction error: {_format_scientific(metadata.get('reconstruction_error'))} "
        "(relative)",
        f"JSON written: {_name('json') if written else _skip('--dry-run')}",
        f"NPZ written:  {npz_line}",
        f"Plot:        {plot_line}",
        (
            "Save-data:   "
            + (
                _name("data")
                if (written and save_data)
                else ("not requested (--save-data)" if not save_data else _skip("--dry-run"))
            )
        ),
        "Stats: printed above (--stats)" if stats else "Stats: not requested (--stats)",
    ]
    width = max(max(len(line) for line in lines), 60)
    border = "+" + "-" * (width + 2) + "+"
    body = [f"| {line.ljust(width)} |" for line in lines]
    return "\n".join([border, *body, border])


# ---------------------------------------------------------------------------
# Diagnostic plots (matplotlib imported lazily; only --plot pops up)
# ---------------------------------------------------------------------------
def plot_spectrum(
    decomposition: SpectralDecomposition,
    out_path: str | Path,
    *,
    cmap: str = DEFAULT_CMAP,
    popup: bool = True,
    title: str | None = None,
    log: logging.Logger | None = None,
) -> Path:
    """Save the scree / cumulative-variance / spectrum figure and optionally pop it up.

    This is the ``--plot`` figure and the **only** one that opens a window;
    :func:`plot_modes` is always saved silently.
    """
    logger = log or LOGGER
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - matplotlib is a pinned dependency
        raise SpectralValidationError(f"matplotlib is required for --plot: {exc}") from exc

    destination = Path(out_path)
    ensure_dir(destination.parent)

    metadata = decomposition.metadata
    heuristics = decomposition.heuristics
    method = str(metadata.get("method"))
    values = decomposition.spectrum
    ratio = decomposition.explained_variance_ratio
    cumulative = decomposition.cumulative_variance_ratio
    ranks = np.arange(1, values.size + 1)

    figure, axes = plt.subplots(1, 3, figsize=(16.0, 5.0), dpi=150)

    # (a) scree -- log-y because the reference spectrum spans ~17 decades
    shown = min(40, ratio.size)
    with np.errstate(divide="ignore"):
        axes[0].semilogy(ranks[:shown], np.where(ratio[:shown] > 0, ratio[:shown], np.nan),
                         marker="o", markersize=3, linewidth=1.0, color="#1f77b4")
    axes[0].set_xlabel("mode rank")
    axes[0].set_ylabel("explained variance ratio (log)")
    axes[0].set_title("Scree plot")
    axes[0].grid(alpha=0.3)

    # (b) cumulative variance with the threshold and the heuristic estimates
    axes[1].plot(ranks, cumulative, linewidth=1.5, color="#2ca02c")
    threshold = metadata["heuristics"].get("variance_threshold") or {}
    if threshold:
        axes[1].axhline(threshold["threshold"], linestyle="--", color="grey", linewidth=1.0)
        axes[1].axvline(threshold["k"], linestyle=":", color="grey", linewidth=1.0)
    for name, colour in (("elbow", "#ff7f0e"), ("spectral_gap", "#d62728")):
        entry = heuristics.get(name)
        if entry:
            axes[1].axvline(entry["k"], linestyle="--", color=colour, linewidth=1.0, label=name)
    axes[1].axvline(
        metadata["n_modes_retained"], color="black", linewidth=1.2, label="retained k"
    )
    axes[1].set_xlabel("mode rank")
    axes[1].set_ylabel("cumulative explained variance")
    axes[1].set_title("Cumulative variance and heuristic k")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    # (c) the spectrum itself: signed eigenvalues or singular values, with the
    # markers shaded by |value| through the requested colormap
    head = np.abs(values[:shown])
    scale = float(head.max())
    normalizer = plt.Normalize(vmin=0.0, vmax=scale if scale > 0.0 else 1.0)
    colours = plt.get_cmap(cmap)(normalizer(head))
    if method == "eigen":
        signed = np.sort(values)[::-1][:shown]
        axes[2].vlines(ranks[:shown], 0.0, signed, colors=colours, linewidth=1.2)
        axes[2].scatter(ranks[:shown], signed, c=colours, s=16, zorder=3)
        axes[2].axhline(0.0, color="black", linewidth=0.7)
        axes[2].set_ylabel("eigenvalue")
        axes[2].set_title("Eigenvalue spectrum (signed)")
    else:
        axes[2].set_yscale("log")
        axes[2].scatter(
            ranks[:shown], np.where(values[:shown] > 0, values[:shown], np.nan),
            c=colours, s=16, zorder=3,
        )
        axes[2].set_ylabel("singular value (log)")
        axes[2].set_title("Singular value spectrum")
        ranks_value = int(metadata.get("numerical_rank") or 0)
        if ranks_value:
            axes[2].axvline(ranks_value, linestyle=":", color="grey", linewidth=1.0)
    axes[2].set_xlabel("mode rank")
    axes[2].grid(alpha=0.3)

    figure.suptitle(
        title
        or (
            f"Phase 03 {method.upper()} spectrum "
            f"({decomposition.n_neurons} neurons, {metadata['n_modes_retained']} modes retained)"
        )
    )
    figure.tight_layout()
    figure.savefig(destination, dpi=150, bbox_inches="tight")

    if popup and can_popup():
        logger.info("showing the spectrum popup for %s (close the window to continue)", destination.name)
        plt.show()  # GUI popup: blocks until the user closes the window
    else:
        logger.debug(
            "wrote %s (GUI popup skipped: %s)",
            destination,
            "--no-popup" if not popup else "non-interactive matplotlib backend",
        )
    plt.close(figure)
    return destination


def plot_mode_heatmaps(
    decomposition: SpectralDecomposition,
    out_path: str | Path,
    *,
    count: int = DEFAULT_PLOT_MODES,
    cmap: str = DEFAULT_CMAP,
    title: str | None = None,
) -> Path:
    """Save the mode-heatmap figure: the loading matrix plus single-mode components.

    The left panel is the full ``(N x k)`` loading matrix (rows = neuron order,
    columns = mode rank); the remaining panels are the leading single-mode
    contributions ``value_i * u_i v_i^T`` (eigen: ``lambda_i v_i v_i^T``) -- the
    "cross-type coupling axes" that show where a mode acts on sources vs targets.
    Never opens a window.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - matplotlib is a pinned dependency
        raise SpectralValidationError(f"matplotlib is required for --plot: {exc}") from exc

    destination = Path(out_path)
    ensure_dir(destination.parent)

    left = decomposition.loadings_left
    right = decomposition.loadings_right
    values = decomposition.spectrum
    components = max(1, min(int(count), decomposition.k))
    columns = 1 + components
    figure, axes = plt.subplots(1, columns, figsize=(4.0 * columns, 5.0), dpi=150)
    if columns == 1:  # pragma: no cover - defensive; count >= 1 keeps >= 2 panels
        axes = [axes]

    image = axes[0].imshow(left, cmap=cmap, interpolation="nearest", aspect="auto")
    figure.colorbar(image, ax=axes[0], label="loading", shrink=0.85)
    axes[0].set_xlabel("mode rank")
    axes[0].set_ylabel("neuron index (source axis)")
    axes[0].set_title(f"Loading matrix ({decomposition.n_neurons} x {decomposition.k})")

    for panel in range(components):
        component = values[panel] * np.outer(left[:, panel], right[:, panel])
        image = axes[panel + 1].imshow(
            component, cmap=cmap, interpolation="nearest", aspect="equal"
        )
        figure.colorbar(image, ax=axes[panel + 1], label="contribution", shrink=0.85)
        axes[panel + 1].set_xlabel("target neuron index")
        axes[panel + 1].set_ylabel("source neuron index")
        axes[panel + 1].set_title(
            f"Mode {panel + 1}: value={values[panel]:.4f}\n"
            f"evr={decomposition.explained_variance_ratio[panel]:.4f}"
        )

    figure.suptitle(title or "Phase 03 mode heatmaps")
    figure.tight_layout()
    figure.savefig(destination, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return destination


def write_artifact_set(
    decomposition: SpectralDecomposition,
    outdir: str | Path,
    *,
    sidecar: bool = True,
    plot: bool = False,
    plot_modes: int = 0,
    save_data: bool = False,
    force_config_hash: bool = False,
    popup: bool = True,
    cmap: str = DEFAULT_CMAP,
    report: ValidationReport | None = None,
    log: logging.Logger | None = None,
) -> dict[str, Path]:
    """Write the requested artifacts and return the paths that were written.

    The canonical JSON goes first (it is self-sufficient); the ``.npz`` cache,
    the plots and the save-data file follow.  A failed derived write is recorded
    as an error in *report* (and is always a hard failure under ``--strict`` in
    the CLI), never a partial file: every writer builds its bytes first and then
    swaps them into place.
    """
    logger = log or LOGGER
    report = report if report is not None else ValidationReport()
    paths = artifact_paths(decomposition, outdir, force_config_hash=force_config_hash)
    written: dict[str, Path] = {}

    written["json"] = write_json_atomic(paths["json"], decomposition.to_dict())

    if sidecar:
        try:
            written["npz"] = write_sidecar(decomposition, written["json"])
        except Exception as exc:
            report.error(CODE_ARTIFACT_WRITE, f"could not write the npz sidecar {paths['npz']}: {exc}")

    if plot:
        try:
            written["png"] = plot_spectrum(
                decomposition, paths["png"], cmap=cmap, popup=popup, log=logger
            )
        except Exception as exc:
            report.error(CODE_PLOT, f"could not write the spectrum plot {paths['png']}: {exc}")
        if plot_modes > 0:
            try:
                written["modes"] = plot_mode_heatmaps(
                    decomposition, paths["modes"], count=plot_modes, cmap=cmap
                )
            except Exception as exc:
                report.error(CODE_PLOT, f"could not write the mode heatmaps {paths['modes']}: {exc}")

    if save_data:
        try:
            written["data"] = write_json_atomic(paths["data"], save_data_payload(decomposition))
        except Exception as exc:
            report.error(CODE_ARTIFACT_WRITE, f"could not write the save-data file {paths['data']}: {exc}")

    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _is_excluded_name(name: str) -> bool:
    """True for Phase 02 save-data artifacts (``z_matrix.data.json``)."""
    return name.endswith(EXCLUDED_INPUT_SUFFIXES)


def _looks_like_phase02(name: str) -> bool:
    """True for a Phase 02 artifact name (``z_matrix[.variant].json``)."""
    return name.startswith(INPUT_VARIANT_PREFIX) and not _is_excluded_name(name)


def resolve_inputs(
    paths: Sequence[str | Path],
    pattern: str = DEFAULT_INPUT_PATTERN,
    *,
    include_variants: bool = False,
) -> list[Path]:
    """Expand CLI inputs into a deterministic list of Phase 02 JSON artifacts.

    Directory inputs are searched **recursively** (``<outdir>/<stem>/z_matrix.json``
    is one level below the processed root) and only pick up Phase 02 artifacts, so
    a Phase 03 output (``eigen.json``) or a save-data file is never re-consumed.
    ``--include-variants`` also resolves ``z_matrix.<variant>.json``.
    """
    patterns = [pattern]
    if include_variants and VARIANT_INPUT_PATTERN not in patterns:
        patterns.append(VARIANT_INPUT_PATTERN)

    resolved: list[Path] = []
    for raw in paths:
        candidate = Path(raw)
        if candidate.is_dir():
            matches = {
                path
                for pat in patterns
                for path in candidate.rglob(pat)
                if path.is_file() and _looks_like_phase02(path.name)
            }
            resolved.extend(sorted(matches))
        elif candidate.is_file():
            if candidate.suffix.lower() != SUPPORTED_SUFFIX:
                LOGGER.error("%s: unsupported suffix (expected %s)", candidate, SUPPORTED_SUFFIX)
                continue
            if _is_excluded_name(candidate.name):
                LOGGER.error("%s: this is a save-data artifact, not a Z-matrix", candidate)
                continue
            resolved.append(candidate)
        else:
            LOGGER.error("%s: no such file or directory", candidate)

    return sorted(dict.fromkeys(resolved))


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.spectral.spectral_decomposition",
        description="Decompose a Phase 02 z_matrix.json into eigen/SVD modes (eigen.json)",
    )
    parser.add_argument(
        "--input", "-i", nargs="+", required=True, metavar="PATH",
        help="one or more z_matrix.json files or directories (searched recursively)",
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
        "--include-variants", dest="include_variants", action="store_true",
        help="also process Phase 02 variant artifacts (z_matrix.<config_hash8>.json)",
    )

    decomposition = parser.add_argument_group("decomposition")
    decomposition.add_argument(
        "--method", choices=METHODS, default=DEFAULT_METHOD,
        help="auto | eigen (symmetric Z) | svd (directed Z) -- auto picks per symmetry",
    )
    decomposition.add_argument(
        "--source", choices=SOURCES, default=DEFAULT_SOURCE,
        help="effective (Phase 02's matrix) | symmetric (matrix_symmetric, always symmetric)",
    )
    decomposition.add_argument(
        "--k", dest="k", default=DEFAULT_K,
        help="modes to retain: auto (heuristic median) | 0/all (every mode) | N",
    )
    decomposition.add_argument(
        "--rank-by", dest="rank_by", choices=RANK_BY, default=DEFAULT_RANK_BY,
        help="rank modes by |value| (magnitude) or signed value",
    )
    decomposition.add_argument(
        "--sign-convention", dest="sign_convention", choices=SIGN_CONVENTIONS,
        default=DEFAULT_SIGN_CONVENTION,
        help="max-abs-positive keeps the artifacts byte-reproducible; none leaves solver signs",
    )
    decomposition.add_argument(
        "--driver", choices=EIGH_DRIVERS, default=DEFAULT_EIGH_DRIVER,
        help=f"LAPACK driver for scipy.linalg.eigh (default: {DEFAULT_EIGH_DRIVER})",
    )
    decomposition.add_argument(
        "--residual-tol", dest="residual_tol", type=float, default=DEFAULT_RESIDUAL_TOL,
        help=f"relative reconstruction tolerance for a full decomposition (default: {DEFAULT_RESIDUAL_TOL})",
    )

    heuristics = parser.add_argument_group("dimensionality heuristics")
    heuristics.add_argument(
        "--variance-threshold", dest="variance_threshold", type=float,
        default=DEFAULT_VARIANCE_THRESHOLD,
        help=f"cumulative explained-variance target (default: {DEFAULT_VARIANCE_THRESHOLD})",
    )
    heuristics.add_argument(
        "--elbow", action=argparse.BooleanOptionalAction, default=True,
        help="estimate k from the elbow of the variance curve (--no-elbow disables)",
    )
    heuristics.add_argument(
        "--elbow-curve", dest="elbow_curve", choices=ELBOW_CURVES, default=DEFAULT_ELBOW_CURVE,
        help=f"curve the elbow is detected on (default: {DEFAULT_ELBOW_CURVE})",
    )
    heuristics.add_argument(
        "--elbow-method", dest="elbow_method", choices=ELBOW_METHODS,
        default=DEFAULT_ELBOW_METHOD,
        help=f"knee detector (default: {DEFAULT_ELBOW_METHOD})",
    )
    heuristics.add_argument(
        "--spectral-gap", dest="spectral_gap", action=argparse.BooleanOptionalAction, default=True,
        help="estimate k from the largest leading spectral gap (--no-spectral-gap disables)",
    )
    heuristics.add_argument(
        "--gap-window", dest="gap_window", type=int, default=DEFAULT_GAP_WINDOW,
        help=f"leading components searched for the spectral gap (default: {DEFAULT_GAP_WINDOW})",
    )
    heuristics.add_argument(
        "--gap-metric", dest="gap_metric", choices=GAP_METRICS, default=DEFAULT_GAP_METRIC,
        help=f"spectral-gap criterion (default: {DEFAULT_GAP_METRIC})",
    )

    outputs = parser.add_argument_group("outputs")
    outputs.add_argument("--config-hash", dest="config_hash", action="store_true",
                         help="always include <config_hash8> in the artifact filenames")
    outputs.add_argument("--no-sidecar", dest="no_sidecar", action="store_true",
                         help="do not write eigen.npz (the JSON stays canonical; load_spectrum still works)")
    outputs.add_argument("--plot", action="store_true",
                         help="write eigen.png (scree/cumulative/spectrum) and show its popup")
    outputs.add_argument("--no-popup", dest="no_popup", action="store_true",
                         help="with --plot: save the PNG without opening the GUI window")
    outputs.add_argument("--cmap", default=DEFAULT_CMAP, help=f"plot colormap (default: {DEFAULT_CMAP})")
    outputs.add_argument("--plot-modes", dest="plot_modes", type=int, default=DEFAULT_PLOT_MODES,
                         help="single-mode heatmap panels in eigen.modes.png, 0 disables "
                              f"(default: {DEFAULT_PLOT_MODES})")
    outputs.add_argument("--save-data", dest="save_data", action="store_true",
                         help="write eigen.data.json (per-mode loading stats, residuals, energy shares)")
    outputs.add_argument("--stats", action="store_true",
                         help="print the spectral statistics to the terminal")

    parser.add_argument("--strict", action="store_true",
                        help="fail on any validation issue (JSON, NPZ and plot writing)")
    parser.add_argument("--dry-run", dest="dry_run", action="store_true",
                        help="validate, print statistics and the summary box without writing anything")
    parser.add_argument(
        "--log-level", dest="log_level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="logging verbosity (default: INFO)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; returns the process exit code.

    ``0`` on success, ``1`` when any input failed validation/decomposition or an
    artifact could not be written, ``2`` when no input file matched.
    """
    args = build_argument_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level), format="%(levelname)s %(name)s: %(message)s"
    )

    if int(args.plot_modes) < 0:
        LOGGER.error("--plot-modes must be >= 0 (got %d)", args.plot_modes)
        return 1

    inputs = resolve_inputs(
        args.input, args.glob_pattern, include_variants=args.include_variants
    )
    if not inputs:
        LOGGER.error("no input files matched: %s", ", ".join(str(item) for item in args.input))
        return 2

    config = SpectralConfig(
        method=args.method,
        source=args.source,
        rank_by=args.rank_by,
        sign_convention=args.sign_convention,
        k_requested=str(args.k),
        variance_threshold=args.variance_threshold,
        elbow=args.elbow,
        elbow_curve=args.elbow_curve,
        elbow_method=args.elbow_method,
        spectral_gap=args.spectral_gap,
        gap_window=args.gap_window,
        gap_metric=args.gap_metric,
        driver=args.driver,
        residual_tol=args.residual_tol,
    )

    failures = 0
    for source in inputs:
        try:
            payload = read_json(source)
        except Exception as exc:
            LOGGER.error("%s: could not read the input artifact: %s", source.name, exc)
            failures += 1
            continue

        try:
            z_matrix = load_z_matrix(source)
        except Exception as exc:
            LOGGER.error("%s: could not load the Z matrix: %s", source.name, exc)
            failures += 1
            continue

        try:
            decomposition, report = build_spectral_decomposition(
                z_matrix, payload=payload, source_artifact=source, config=config
            )
        except Exception as exc:  # malformed/unreadable artifact or invalid config
            LOGGER.error("%s: %s", source.name, exc)
            failures += 1
            continue

        if report.has_errors:
            for issue in report.errors:
                LOGGER.error("%s: [%s] %s", source.name, issue.code, issue.message)
            LOGGER.error("%s: validation failed (%s)", source.name, report.summary())
            failures += 1
            continue

        metadata = decomposition.metadata
        LOGGER.info(
            "%s: %s neurons=%d values=%d retained=%d sparsity-free top|v|=%.6f cum@k=%.6f "
            "config_hash=%s",
            source.name,
            str(metadata["method"]).upper(),
            metadata["n_neurons"],
            metadata["n_modes_total"],
            metadata["n_modes_retained"],
            metadata["abs_value_max"],
            metadata["cumulative_at_k"],
            metadata["config_hash"],
        )

        if args.stats:
            print(render_statistics(decomposition))

        paths = artifact_paths(decomposition, args.outdir, force_config_hash=args.config_hash)
        if not args.dry_run:
            write_artifact_set(
                decomposition,
                args.outdir,
                sidecar=not args.no_sidecar,
                plot=args.plot,
                plot_modes=args.plot_modes if args.plot else 0,
                save_data=args.save_data,
                force_config_hash=args.config_hash,
                popup=not args.no_popup,
                cmap=args.cmap,
                report=report,
                log=LOGGER,
            )

        for issue in report.warnings:
            LOGGER.warning("%s: [%s] %s", source.name, issue.code, issue.message)

        if not args.dry_run and (report.has_errors or (args.strict and report.warnings)):
            for issue in report.errors:
                LOGGER.error("%s: [%s] %s", source.name, issue.code, issue.message)
            if args.strict and report.warnings:
                LOGGER.error(
                    "%s: --strict escalates %d warning(s) to errors",
                    source.name,
                    len(report.warnings),
                )
            LOGGER.error("%s: artifact writing failed (%s)", source.name, report.summary())
            failures += 1
            continue

        if not args.dry_run:
            for key, path in paths.items():
                if path.is_file():
                    LOGGER.info("%s: wrote %s [%s]", source.name, path, key)

        print(
            render_summary_box(
                decomposition,
                paths,
                written=not args.dry_run,
                sidecar=not args.no_sidecar,
                plot=args.plot,
                plot_modes=args.plot_modes if args.plot else 0,
                save_data=args.save_data,
                stats=args.stats,
            )
        )

    if failures:
        LOGGER.error("%d of %d input file(s) failed", failures, len(inputs))
    return 1 if failures else 0


if __name__ == "__main__":  # pragma: no cover - exercised through subprocess tests
    sys.exit(main())
