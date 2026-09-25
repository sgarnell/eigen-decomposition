"""Phase 03 -- eigen / SVD decomposition of the Phase 02 ``Z`` matrix.

Public surface (imported lazily so that
``python -m src.spectral.spectral_decomposition`` does not execute the module
twice, mirroring :mod:`src.parsing` and :mod:`src.matrices`):

* :func:`decompose_matrix` -- the pure engine (SciPy ``eigh``/``svd``, ranked).
* :func:`resolve_method` / :func:`matrix_is_symmetric` -- the auto path selection.
* :func:`compute_heuristics` / :func:`l_method_knee` / :func:`recommended_k` /
  :func:`resolve_retained_k` -- the dimensionality-selection grid.
* :func:`build_spectral_decomposition` -- artifact assembly from a ``ZMatrix``.
* :func:`load_spectrum` / :func:`load_sidecar_arrays` -- the entry points for
  Phases 04+ (canonical JSON, plus the verified ``.npz`` array cache).
* :func:`plot_spectrum` / :func:`plot_mode_heatmaps` -- the ``--plot`` figures.
* :func:`write_artifact_set` -- persist the requested artifact set.
* :func:`main` -- the command line interface.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - import-time only for type checkers
    from src.spectral.spectral_decomposition import (
        CANONICAL_STEM,
        CONFIG_KEYS,
        DEFAULT_CMAP,
        DEFAULT_DEGENERATE_TOL,
        DEFAULT_EIGH_DRIVER,
        DEFAULT_ELBOW_CURVE,
        DEFAULT_ELBOW_METHOD,
        DEFAULT_GAP_METRIC,
        DEFAULT_GAP_WINDOW,
        DEFAULT_K,
        DEFAULT_METHOD,
        DEFAULT_PLOT_MODES,
        DEFAULT_RANK_BY,
        DEFAULT_RESIDUAL_TOL,
        DEFAULT_SIGN_CONVENTION,
        DEFAULT_SOURCE,
        DEFAULT_VARIANCE_THRESHOLD,
        EIGH_DRIVERS,
        ELBOW_CURVES,
        ELBOW_METHODS,
        GAP_METRICS,
        GENERATOR,
        HEURISTICS_KEYS,
        HEURISTIC_KEYS,
        INPUT_VARIANT_PREFIX,
        MATRIX_CONFIG_FIELDS,
        METADATA_KEYS,
        METHODS,
        MODE_KEYS,
        PROVENANCE_KEYS,
        RANK_BY,
        SAVE_DATA_KEYS,
        SIGN_CONVENTIONS,
        SOURCES,
        TOP_LEVEL_KEYS,
        VALUE_KEYS,
        DecompositionResult,
        HeuristicResult,
        SpectralConfig,
        SpectralDecomposition,
        SpectralMode,
        SpectralValidationError,
        SpectralValue,
        apply_sign_convention,
        artifact_paths,
        build_argument_parser,
        build_sidecar_arrays,
        can_popup,
        build_spectral_decomposition,
        compute_diagnostics,
        compute_heuristics,
        decompose_matrix,
        degenerate_groups,
        input_variant,
        l_method_knee,
        load_sidecar_arrays,
        load_spectrum,
        main,
        matrix_is_symmetric,
        plot_mode_heatmaps,
        plot_spectrum,
        recommended_k,
        render_statistics,
        render_summary_box,
        resolve_inputs,
        resolve_method,
        resolve_retained_k,
        save_data_payload,
        second_difference_knee,
        select_source_matrix,
        sidecar_path,
        validate_spectral_payload_schema,
        variant_stem,
        write_artifact_set,
        write_npz_atomic,
        write_sidecar,
    )

_MODULE_NAME = "src.spectral.spectral_decomposition"

__all__ = [
    "CANONICAL_STEM",
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
    "build_sidecar_arrays",
    "can_popup",
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


def __getattr__(name: str) -> Any:
    """Resolve the public names from :mod:`src.spectral.spectral_decomposition` on demand."""
    if name in __all__:
        module = importlib.import_module(_MODULE_NAME)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
