"""Phase 02 -- square Z-matrix construction.

Public surface (imported lazily so that
``python -m src.matrices.build_square_matrix`` does not execute the module twice,
mirroring :mod:`src.parsing`):

* :func:`build_z_matrix` / :func:`build_z_matrix_from_file` -- assemble the
  unified-weight matrix from a Phase 01 artifact.
* :func:`unified_weight` / :func:`unified_weights` -- the pre/post unification rule.
* :func:`load_z_matrix` / :func:`load_sidecar_arrays` -- the entry points for
  Phases 03+, reading the canonical JSON and the ``.npz`` array cache.
* :func:`write_artifact_set` -- persist the requested artifact set.
* :func:`format_cell_tooltip` / :func:`build_interactive_figure` /
  :func:`attach_cell_cursor` / :func:`show_interactive_matrix` -- the
  ``--interactive`` hover-tooltip window (matplotlib + mplcursors).
* :func:`main` -- the command line interface.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - import-time only for type checkers
    from src.matrices.build_square_matrix import (
        CONFIG_KEYS,
        DEFAULT_ARTIFACT_NAME,
        DEFAULT_CMAP,
        DEFAULT_DATA_NAME,
        DEFAULT_EPS,
        DEFAULT_NORMALIZATION,
        DEFAULT_PLOT_NAME,
        DEFAULT_SIDECAR_NAME,
        INTERACTIVE_ANNOTATION_KWARGS,
        METADATA_KEYS,
        NON_INTERACTIVE_BACKENDS,
        NORMALIZATIONS,
        PROVENANCE_KEYS,
        SAVE_DATA_KEYS,
        STATS_KEYS,
        SYMMETRIZE_METHODS,
        TOP_LEVEL_KEYS,
        UNIFICATION_RULE,
        WEIGHTED_PAIR_KEYS,
        WEIGHT_STATS_KEYS,
        ZMatrix,
        ZMatrixConfig,
        ZMatrixValidationError,
        WeightedPair,
        artifact_paths,
        attach_cell_cursor,
        build_argument_parser,
        build_interactive_figure,
        build_sidecar_arrays,
        build_z_matrix,
        build_z_matrix_from_file,
        compute_diagnostics,
        format_cell_tooltip,
        load_sidecar_arrays,
        load_z_matrix,
        main,
        normalize_matrix,
        plot_matrix,
        render_statistics,
        render_summary_box,
        resolve_inputs,
        save_data_payload,
        show_interactive_matrix,
        sidecar_path,
        symmetrize,
        unified_weight,
        unified_weights,
        validate_z_payload_schema,
        write_artifact_set,
        write_npz_atomic,
        write_sidecar,
    )

_MODULE_NAME = "src.matrices.build_square_matrix"

__all__ = [
    "CONFIG_KEYS",
    "DEFAULT_ARTIFACT_NAME",
    "DEFAULT_CMAP",
    "DEFAULT_DATA_NAME",
    "DEFAULT_EPS",
    "DEFAULT_NORMALIZATION",
    "DEFAULT_PLOT_NAME",
    "DEFAULT_SIDECAR_NAME",
    "INTERACTIVE_ANNOTATION_KWARGS",
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
    "WeightedPair",
    "ZMatrix",
    "ZMatrixConfig",
    "ZMatrixValidationError",
    "artifact_paths",
    "attach_cell_cursor",
    "build_argument_parser",
    "build_interactive_figure",
    "build_sidecar_arrays",
    "build_z_matrix",
    "build_z_matrix_from_file",
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


def __getattr__(name: str) -> Any:
    """Resolve the public names from :mod:`src.matrices.build_square_matrix` on demand."""
    if name in __all__:
        module = importlib.import_module(_MODULE_NAME)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
