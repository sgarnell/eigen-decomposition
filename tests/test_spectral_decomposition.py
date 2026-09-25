"""Unit tests for the Phase 03 eigen / SVD decomposition.

The synthetic fixtures in ``tests/fixtures`` drive the fast path; the integration
tests against the real 727 KB dataset are marked ``slow`` and can be deselected
with ``-m "not slow"``.  The matplotlib backend is pinned to ``Agg`` for the whole
module so ``--plot`` never opens (or blocks on) a GUI window.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: --plot must save the PNG without a popup

import numpy as np  # noqa: E402  (after the backend pin, deliberately)
import pytest  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402  (after matplotlib.use("Agg"))

from src.matrices.build_square_matrix import (  # noqa: E402
    ZMatrixConfig,
    build_z_matrix_from_file,
    write_artifact_set,
)
from src.spectral.spectral_decomposition import write_npz_atomic  # noqa: E402
from src.parsing.parse_graphviz import build_parsed_graph, write_artifact  # noqa: E402
from src.spectral.spectral_decomposition import (  # noqa: E402
    CODE_CONFIG,
    CODE_DEGENERATE,
    CODE_INPUT_SCHEMA,
    CODE_METHOD_INCOMPATIBLE,
    CODE_PLOT,
    CODE_TRIVIAL_MATRIX,
    DEFAULT_GAP_WINDOW,
    DEFAULT_METHOD,
    DEFAULT_PLOT_MODES,
    MATRIX_CONFIG_FIELDS,
    METHODS,
    SAVE_DATA_KEYS,
    SpectralConfig,
    SpectralValidationError,
    apply_sign_convention,
    artifact_paths,
    build_spectral_decomposition,
    build_sidecar_arrays,
    compute_heuristics,
    decompose_matrix,
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
    second_difference_knee,
    select_source_matrix,
    sidecar_path,
    validate_spectral_payload_schema,
    variant_stem,
    write_artifact_set as write_spectral_artifacts,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
TINY = FIXTURES / "tiny_valid.gv"
REFERENCE_GV = PROJECT_ROOT / "data" / "raw_dot" / "FB4Yaffect_FB45_999prePost_001_all.gv"
REFERENCE_DIR = PROJECT_ROOT / "data" / "processed" / REFERENCE_GV.stem
REFERENCE_ARTIFACT = REFERENCE_DIR / "z_matrix.json"
VARIANT_ARTIFACT = REFERENCE_DIR / "z_matrix.f8652585.json"

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
A_ASYM = np.array(
    [
        [0.0, 0.5, -0.4, 0.1],
        [0.2, 0.0, 0.3, -0.6],
        [0.1, -0.2, 0.0, 0.4],
        [-0.3, 0.7, -0.5, 0.0],
    ]
)
A_SYM = (A_ASYM + A_ASYM.T) / 2.0


@pytest.fixture(scope="module")
def tiny_z_matrix(tmp_path_factory):
    """A Phase 02 object for ``tests/fixtures/tiny_valid.gv`` (real file path)."""
    outdir = tmp_path_factory.mktemp("phase03_tiny")
    parsed, _ = build_parsed_graph(TINY)
    write_artifact(parsed, outdir)
    artifact = outdir / TINY.stem / "parsed_graph.json"
    z_matrix, report = build_z_matrix_from_file(artifact, config=ZMatrixConfig())
    assert not report.has_errors
    return z_matrix


@pytest.fixture(scope="module")
def tiny_payload(tiny_z_matrix):
    return tiny_z_matrix.to_dict()


@pytest.fixture
def tiny_workspace(tmp_path):
    """A tmp tree holding ``parsed_graph.json`` + ``z_matrix.json`` for the tiny graph."""
    parsed, _ = build_parsed_graph(TINY)
    write_artifact(parsed, tmp_path)
    artifact = tmp_path / TINY.stem / "parsed_graph.json"
    z_matrix, _ = build_z_matrix_from_file(artifact, config=ZMatrixConfig())
    write_artifact_set(z_matrix, tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
def test_public_constants_are_consistent() -> None:
    assert DEFAULT_METHOD in METHODS
    assert DEFAULT_GAP_WINDOW >= 2
    assert DEFAULT_PLOT_MODES >= 1
    assert CODE_CONFIG == "config"
    assert CODE_PLOT == "plot"
    assert CODE_DEGENERATE == "degenerate"
    assert CODE_TRIVIAL_MATRIX == "trivial_matrix"
    assert set(MATRIX_CONFIG_FIELDS) <= set(SpectralConfig().hash_fields())


def test_matrix_is_symmetric_exact_and_tolerant() -> None:
    assert matrix_is_symmetric(A_SYM)
    assert not matrix_is_symmetric(A_ASYM)
    assert not matrix_is_symmetric(np.zeros((2, 3)))
    # a matrix that is symmetric only up to floating-point noise still counts
    noisy = A_SYM.copy()
    noisy[0, 1] += 1e-15
    assert matrix_is_symmetric(noisy)


def test_resolve_method_auto_picks_per_symmetry() -> None:
    assert resolve_method(A_SYM) == "eigen"
    assert resolve_method(A_ASYM) == "svd"
    assert resolve_method(A_ASYM, "auto") == "svd"
    assert resolve_method(A_ASYM, "eigen") == "eigen"
    assert resolve_method(A_SYM, "svd") == "svd"


def test_decompose_svd_matches_numpy() -> None:
    result = decompose_matrix(A_ASYM, method="svd", k=None)
    expected = np.linalg.svd(A_ASYM, compute_uv=False)
    assert result.method == "svd"
    assert result.n_neurons == 4
    assert np.allclose(result.values, expected, rtol=1e-12, atol=1e-14)


def test_decompose_eigen_matches_numpy() -> None:
    result = decompose_matrix(A_SYM, method="eigen", k=None)
    expected = np.sort(np.abs(np.linalg.eigvalsh(A_SYM)))[::-1]
    assert result.method == "eigen"
    assert np.allclose(result.abs_values, expected, rtol=1e-12, atol=1e-14)


def test_eigen_on_asymmetric_matrix_raises() -> None:
    with pytest.raises(SpectralValidationError, match="symmetric"):
        decompose_matrix(A_ASYM, method="eigen")


def test_non_square_matrix_raises() -> None:
    with pytest.raises(SpectralValidationError, match="square"):
        decompose_matrix(np.zeros((2, 3)), method="svd")


def test_eigen_and_svd_agree_on_a_symmetric_matrix() -> None:
    eigen = decompose_matrix(A_SYM, method="eigen", k=None)
    singular = decompose_matrix(A_SYM, method="svd", k=None)
    assert np.allclose(eigen.abs_values, singular.values, rtol=1e-12, atol=1e-14)


def test_svd_reconstruction_and_orthonormality() -> None:
    result = decompose_matrix(A_ASYM, method="svd", k=None)
    identifier = np.eye(result.n_neurons)
    assert np.allclose(result.left.T @ result.left, identifier, atol=1e-12)
    assert np.allclose(result.right.T @ result.right, identifier, atol=1e-12)
    rebuilt = result.left @ np.diag(result.values) @ result.right.T
    assert np.linalg.norm(rebuilt - A_ASYM) / np.linalg.norm(A_ASYM) < 1e-12


def test_eigen_reconstruction_and_orthonormality() -> None:
    result = decompose_matrix(A_SYM, method="eigen", k=None)
    identifier = np.eye(result.n_neurons)
    assert np.allclose(result.left.T @ result.left, identifier, atol=1e-12)
    rebuilt = result.left @ np.diag(result.values) @ result.right.T
    assert np.linalg.norm(rebuilt - A_SYM) / np.linalg.norm(A_SYM) < 1e-12
    # the eigen path uses the same vector on both axes
    assert np.array_equal(result.left, result.right)


def test_eigen_sum_of_values_equals_trace() -> None:
    result = decompose_matrix(A_SYM, method="eigen", k=None)
    assert float(np.sum(result.values)) == pytest.approx(float(np.trace(A_SYM)), abs=1e-12)


def test_svd_sum_of_squares_equals_frobenius_squared() -> None:
    result = decompose_matrix(A_ASYM, method="svd", k=None)
    assert float(np.sum(result.energy)) == pytest.approx(
        float(np.linalg.norm(A_ASYM) ** 2), rel=1e-12
    )


def test_sign_convention_makes_the_pivot_positive() -> None:
    result = decompose_matrix(A_ASYM, method="svd", k=4)
    for mode in range(result.k):
        column = result.left[:, mode]
        assert column[int(np.argmax(np.abs(column)))] > 0.0


def test_sign_convention_keeps_the_singular_relation() -> None:
    """A coupled flip is required: A v = sigma u must survive the convention."""
    result = decompose_matrix(A_ASYM, method="svd", k=4)
    for mode in range(result.k):
        left = result.left[:, mode]
        right = result.right[:, mode]
        assert np.allclose(A_ASYM @ right, result.values[mode] * left, atol=1e-12)
        assert np.allclose(A_ASYM.T @ left, result.values[mode] * right, atol=1e-12)


def test_sign_convention_none_flags_nothing() -> None:
    left = np.array([[1.0, -2.0], [0.5, 0.25]])
    right = np.array([[-1.0, 2.0], [0.5, -0.25]])
    untouched_left, untouched_right, flipped = apply_sign_convention(left, right, "none")
    assert flipped == (False, False)
    assert np.array_equal(untouched_left, left)
    assert np.array_equal(untouched_right, right)


def test_sign_convention_is_deterministic() -> None:
    first = decompose_matrix(A_ASYM, method="svd", k=3)
    second = decompose_matrix(A_ASYM, method="svd", k=3)
    assert np.array_equal(first.left, second.left)
    assert np.array_equal(first.right, second.right)


def test_sign_convention_handles_a_negative_leading_component() -> None:
    left = np.array([[-3.0, 1.0], [1.0, -2.0]])
    right = np.array([[-1.0, 0.5], [0.25, 1.0]])
    new_left, new_right, flipped = apply_sign_convention(left, right, "max-abs-positive")
    assert flipped == (True, True)
    assert new_left[0, 0] > 0.0 and new_left[1, 1] > 0.0
    # the pair is flipped together, never independently
    assert np.array_equal(new_right[:, 0], -right[:, 0])
    assert np.array_equal(new_right[:, 1], -right[:, 1])


def test_ranking_magnitude_vs_signed_value() -> None:
    diagonal = np.diag([1.0, -3.0, 2.0])
    magnitude = decompose_matrix(diagonal, method="eigen", k=None, rank_by="magnitude")
    signed = decompose_matrix(diagonal, method="eigen", k=None, rank_by="value")
    assert [round(float(value), 9) for value in magnitude.values] == [-3.0, 2.0, 1.0]
    assert [round(float(value), 9) for value in signed.values] == [2.0, 1.0, -3.0]
    # the original solver index travels with each ranked entry (eigh returns
    # ascending [-3, 1, 2], so |value| ranking maps to source indices 0, 2, 1)
    assert [int(index) for index in magnitude.rank_indices] == [0, 2, 1]
    assert [int(index) for index in signed.rank_indices] == [2, 1, 0]


def test_rank_indices_are_a_permutation() -> None:
    result = decompose_matrix(A_ASYM, method="svd", k=None)
    assert sorted(int(index) for index in result.rank_indices) == [0, 1, 2, 3]


def test_k_slicing_and_clamping() -> None:
    assert decompose_matrix(A_ASYM, method="svd", k=2).left.shape == (4, 2)
    assert decompose_matrix(A_ASYM, method="svd", k=0).left.shape == (4, 4)
    assert decompose_matrix(A_ASYM, method="svd", k=None).left.shape == (4, 4)
    assert decompose_matrix(A_ASYM, method="svd", k=99).left.shape == (4, 4)


def test_explained_variance_sums_to_one_and_is_monotone() -> None:
    result = decompose_matrix(A_ASYM, method="svd", k=None)
    assert float(np.sum(result.explained_variance_ratio)) == pytest.approx(1.0, abs=1e-12)
    assert np.all(np.diff(result.cumulative_variance_ratio) >= -1e-15)
    assert result.cumulative_variance_ratio[-1] == pytest.approx(1.0, abs=1e-12)
    assert np.allclose(
        result.explained_variance_ratio, result.energy / float(result.energy.sum()), atol=1e-15
    )


def test_zero_matrix_is_all_zero() -> None:
    result = decompose_matrix(np.zeros((3, 3)), method="svd", k=None)
    assert np.allclose(result.values, 0.0)
    assert np.allclose(result.explained_variance_ratio, 0.0)
    assert np.allclose(result.cumulative_variance_ratio, 0.0)
    assert result.numerical_rank == 0


def test_non_finite_matrix_is_rejected() -> None:
    bad = np.array([[0.0, np.nan], [1.0, 0.0]])
    with pytest.raises(SpectralValidationError, match="non-finite"):
        decompose_matrix(bad, method="svd")


# ---------------------------------------------------------------------------
# Dimensionality heuristics
# ---------------------------------------------------------------------------
SPECTRUM = np.array([10.0, 5.0, 2.0, 1.0, 0.5])
ELBOW_CURVE = np.array([0.0, 0.9, 0.95, 0.97, 0.98, 1.0])
GAPPED = np.array([10.0, 9.0, 8.0, 1.0, 0.5])


def test_variance_threshold_finds_the_crossing_rank() -> None:
    results, payload = compute_heuristics(SPECTRUM, variance_threshold=0.9)
    entry = payload["variance_threshold"]
    assert entry == {"k": 2, "threshold": 0.9}
    assert [result.k for result in results if result.name == "variance_threshold"] == [2]
    # cumulative: 100/130.25 = 0.7677 < 0.9 <= 125/130.25 = 0.9597


def test_variance_threshold_can_be_disabled() -> None:
    results, payload = compute_heuristics(SPECTRUM, variance_threshold=0.0)
    assert payload["variance_threshold"] is None
    assert all(result.name != "variance_threshold" for result in results)


def test_l_method_knee_on_a_known_elbow() -> None:
    assert l_method_knee(ELBOW_CURVE) == 2
    assert l_method_knee([0.0, 1.0]) == 1
    assert l_method_knee([1.0]) == 1


def test_second_difference_knee_on_a_known_elbow() -> None:
    assert second_difference_knee(ELBOW_CURVE) == 2
    assert second_difference_knee([1.0, 0.5]) == 1


def test_elbow_method_is_recorded_and_bounded() -> None:
    """``compute_heuristics`` derives its own curve, so only bounds are asserted here."""
    _, payload = compute_heuristics(SPECTRUM, variance_threshold=0.0)
    assert payload["elbow"]["method"] == "l-method"
    assert 1 <= payload["elbow"]["k"] <= SPECTRUM.size
    _, alternative = compute_heuristics(
        SPECTRUM, variance_threshold=0.0, elbow_method="second-difference"
    )
    assert alternative["elbow"]["method"] == "second-difference"
    assert 1 <= alternative["elbow"]["k"] <= SPECTRUM.size


def test_elbow_curve_can_be_the_scree() -> None:
    _, payload = compute_heuristics(SPECTRUM, elbow_curve="scree")
    assert payload["elbow"]["curve"] == "scree"
    assert 1 <= payload["elbow"]["k"] <= SPECTRUM.size


def test_spectral_gap_finds_a_clean_gap() -> None:
    _, payload = compute_heuristics(GAPPED, variance_threshold=0.0, elbow=False)
    assert payload["spectral_gap"]["k"] == 3
    assert payload["spectral_gap"]["metric"] == "ratio"
    assert payload["spectral_gap"]["value"] == pytest.approx(8.0, rel=1e-12)


def test_spectral_gap_window_limits_the_search() -> None:
    _, payload = compute_heuristics(
        GAPPED, variance_threshold=0.0, elbow=False, gap_window=2
    )
    assert payload["spectral_gap"]["window"] == 2
    assert payload["spectral_gap"]["k"] == 2


def test_spectral_gap_gap_metric() -> None:
    _, payload = compute_heuristics(
        GAPPED, variance_threshold=0.0, elbow=False, gap_metric="gap"
    )
    assert payload["spectral_gap"]["k"] == 3
    assert payload["spectral_gap"]["value"] == pytest.approx(7.0, rel=1e-12)


def test_spectral_gap_can_be_disabled() -> None:
    results, payload = compute_heuristics(GAPPED, spectral_gap=False)
    assert payload["spectral_gap"] is None
    assert all(result.name != "spectral_gap" for result in results)


def test_participation_ratio_is_the_inverse_herfindahl_index() -> None:
    _, payload = compute_heuristics(SPECTRUM, variance_threshold=0.0, elbow=False, spectral_gap=False)
    expected = float(SPECTRUM.sum() ** 2 / (SPECTRUM**2).sum())
    assert payload["participation_ratio"] == pytest.approx(expected, rel=1e-12)
    # (10+5+2+1+0.5)^2 / (100+25+4+1+0.25) = 342.25 / 130.25
    assert payload["participation_ratio"] == pytest.approx(2.6276391555, rel=1e-9)


def test_recommended_k_is_the_median_of_the_enabled_heuristics() -> None:
    results, payload = compute_heuristics(GAPPED, variance_threshold=0.9)
    candidates = sorted(result.k for result in results)
    assert payload["recommended_k"] == candidates[len(candidates) // 2]
    assert payload["rule"].startswith("median")


def test_recommended_k_without_heuristics_falls_back() -> None:
    _, payload = compute_heuristics(
        SPECTRUM, variance_threshold=0.0, elbow=False, spectral_gap=False
    )
    assert payload["variance_threshold"] is None
    assert payload["elbow"] is None
    assert payload["spectral_gap"] is None
    assert payload["recommended_k"] == SPECTRUM.size
    assert recommended_k([], fallback=7, n_modes=10) == 7
    assert recommended_k([], fallback=99, n_modes=10) == 10


def test_recommended_k_is_clipped_to_the_mode_count() -> None:
    from src.spectral.spectral_decomposition import HeuristicResult

    results = [HeuristicResult("elbow", 99)]
    assert recommended_k(results, n_modes=5) == 5
    assert recommended_k([HeuristicResult("elbow", 0)], n_modes=5) == 1


@pytest.mark.parametrize(
    ("requested", "recommended", "n_modes", "expected"),
    [
        ("auto", 7, 20, 7),
        ("AUTO", 7, 20, 7),
        ("", 7, 20, 7),
        ("0", 7, 20, 20),
        ("all", 7, 20, 20),
        ("5", 7, 20, 5),
        ("99", 7, 20, 20),
        ("auto", 0, 20, 20),
    ],
)
def test_resolve_retained_k(requested, recommended, n_modes, expected) -> None:
    assert resolve_retained_k(requested, recommended=recommended, n_modes=n_modes) == expected


def test_heuristics_payload_keys_are_stable() -> None:
    _, payload = compute_heuristics(SPECTRUM)
    assert set(payload) == {
        "variance_threshold",
        "elbow",
        "spectral_gap",
        "participation_ratio",
        "recommended_k",
        "rule",
    }
    for name in ("variance_threshold", "elbow", "spectral_gap"):
        assert payload[name] is None or "k" in payload[name]


# ---------------------------------------------------------------------------
# build_spectral_decomposition
# ---------------------------------------------------------------------------
def test_build_tiny_artifact_provenance_and_metadata(tiny_z_matrix, tiny_payload) -> None:
    decomposition, report = build_spectral_decomposition(
        tiny_z_matrix, payload=tiny_payload, source_artifact=Path("x/z_matrix.json")
    )
    assert not report.has_errors and not report.warnings
    assert decomposition.config.resolved_method == "svd"
    assert decomposition.config.source == "effective"
    assert decomposition.config.k_resolved == decomposition.k
    assert decomposition.config.symmetric_input is False
    metadata = decomposition.metadata
    assert metadata["method"] == "svd"
    assert metadata["source"] == "effective"
    assert metadata["n_neurons"] == 3
    assert metadata["n_modes_total"] == 3
    assert metadata["n_modes_retained"] == decomposition.k
    assert 1 <= decomposition.k <= 3
    assert metadata["generator"] == "src.spectral.spectral_decomposition"
    payload = decomposition.to_dict()
    assert payload["provenance"]["phase"] == "03"
    assert payload["provenance"]["source_artifact"] == "x/z_matrix.json"
    assert payload["provenance"]["z_matrix_config_hash"] == tiny_z_matrix.config.config_hash()
    assert payload["provenance"]["z_matrix_config"] == tiny_z_matrix.config.to_dict()
    assert payload["neuron_order"] == tiny_z_matrix.neuron_order
    assert decomposition.loadings_left.shape == (3, decomposition.k)
    assert decomposition.loadings_right.shape == (3, decomposition.k)


def test_build_from_the_symmetric_source_uses_eigen(tiny_z_matrix, tiny_payload) -> None:
    decomposition, report = build_spectral_decomposition(
        tiny_z_matrix, payload=tiny_payload, config=SpectralConfig(source="symmetric")
    )
    assert not report.has_errors
    assert decomposition.config.resolved_method == "eigen"
    assert decomposition.config.symmetric_input is True
    assert decomposition.metadata["matrix_is_symmetric"] is True
    # the eigen path puts the same vector on both axes
    assert np.array_equal(decomposition.loadings_left, decomposition.loadings_right)


def test_select_source_matrix_effective_vs_symmetric(tiny_z_matrix) -> None:
    effective = select_source_matrix(tiny_z_matrix, "effective")
    symmetric = select_source_matrix(tiny_z_matrix, "symmetric")
    assert np.array_equal(effective, tiny_z_matrix.matrix)
    assert np.array_equal(symmetric, tiny_z_matrix.matrix_symmetric)
    assert matrix_is_symmetric(symmetric)
    assert not matrix_is_symmetric(effective)


def test_build_full_k_reconstructs_exactly(tiny_z_matrix, tiny_payload) -> None:
    decomposition, report = build_spectral_decomposition(
        tiny_z_matrix, payload=tiny_payload, config=SpectralConfig(k_requested="0")
    )
    assert not report.has_errors
    assert decomposition.k == decomposition.n_neurons
    assert decomposition.metadata["reconstruction_error"] < 1e-12
    assert decomposition.metadata["max_mode_residual"] < 1e-12
    assert decomposition.metadata["orthogonality_error"] < 1e-12


def test_build_truncated_k_reports_a_larger_truncation_error(tiny_z_matrix, tiny_payload) -> None:
    full, _ = build_spectral_decomposition(
        tiny_z_matrix, payload=tiny_payload, config=SpectralConfig(k_requested="0")
    )
    truncated, _ = build_spectral_decomposition(
        tiny_z_matrix, payload=tiny_payload, config=SpectralConfig(k_requested="1")
    )
    assert truncated.k == 1
    assert truncated.metadata["reconstruction_error"] > full.metadata["reconstruction_error"]
    # a truncated decomposition is informational, never a validation error


def test_build_rejects_eigen_on_a_directed_source(tiny_z_matrix, tiny_payload) -> None:
    with pytest.raises(SpectralValidationError, match="symmetric") as error:
        build_spectral_decomposition(
            tiny_z_matrix, payload=tiny_payload, config=SpectralConfig(method="eigen")
        )
    report = error.value.report
    assert report is not None
    assert CODE_METHOD_INCOMPATIBLE in report.codes()


def test_build_rejects_an_invalid_config(tiny_z_matrix, tiny_payload) -> None:
    with pytest.raises(SpectralValidationError) as error:
        build_spectral_decomposition(
            tiny_z_matrix, payload=tiny_payload, config=SpectralConfig(gap_window=1)
        )
    assert CODE_CONFIG in error.value.report.codes()
    with pytest.raises(SpectralValidationError):
        build_spectral_decomposition(
            tiny_z_matrix, payload=tiny_payload, config=SpectralConfig(k_requested="99")
        )


def test_build_rejects_a_malformed_phase02_payload(tiny_z_matrix) -> None:
    with pytest.raises(SpectralValidationError) as error:
        build_spectral_decomposition(tiny_z_matrix, payload={"nonsense": True})
    assert CODE_INPUT_SCHEMA in error.value.report.codes()


def test_build_rejects_a_non_finite_or_non_square_matrix(tiny_z_matrix, tiny_payload) -> None:
    tampered = tiny_z_matrix
    original = tampered.matrix
    try:
        tampered.matrix = np.full((3, 3), np.nan)
        with pytest.raises(SpectralValidationError, match="non-finite"):
            build_spectral_decomposition(tampered, payload=tiny_payload)
        tampered.matrix = np.zeros((2, 3))
        with pytest.raises(SpectralValidationError, match="square"):
            build_spectral_decomposition(tampered, payload=tiny_payload)
    finally:
        tampered.matrix = original


def test_build_warns_on_an_all_zero_matrix(tiny_z_matrix, tiny_payload) -> None:
    tampered = tiny_z_matrix
    original = tampered.matrix
    try:
        tampered.matrix = np.zeros((3, 3))
        # auto picks the eigen path for a symmetric (all-zero) matrix
        automatic, auto_report = build_spectral_decomposition(tampered, payload=tiny_payload)
        decomposition, report = build_spectral_decomposition(
            tampered, payload=tiny_payload, config=SpectralConfig(method="svd")
        )
    finally:
        tampered.matrix = original
    assert automatic.metadata["method"] == "eigen"
    assert automatic.metadata["numerical_rank"] is None
    assert CODE_TRIVIAL_MATRIX in auto_report.codes()
    assert CODE_TRIVIAL_MATRIX in report.codes()
    assert decomposition.metadata["abs_value_max"] == 0.0
    assert decomposition.metadata["numerical_rank"] == 0


def test_build_is_deterministic_under_a_fixed_source_date_epoch(
    tiny_z_matrix, tiny_payload, monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "1700000000")
    first, _ = build_spectral_decomposition(tiny_z_matrix, payload=tiny_payload)
    second, _ = build_spectral_decomposition(tiny_z_matrix, payload=tiny_payload)
    assert first.to_dict() == second.to_dict()

    write_spectral_artifacts(first, tmp_path / "first")
    write_spectral_artifacts(second, tmp_path / "second")
    for name in ("eigen.json", "eigen.npz"):
        first_bytes = (tmp_path / "first" / TINY.stem / name).read_bytes()
        second_bytes = (tmp_path / "second" / TINY.stem / name).read_bytes()
        assert first_bytes == second_bytes, name


def test_build_does_not_mutate_the_phase02_object(tiny_z_matrix, tiny_payload) -> None:
    before = tiny_z_matrix.to_dict()
    build_spectral_decomposition(tiny_z_matrix, payload=tiny_payload)
    assert tiny_z_matrix.to_dict() == before


# ---------------------------------------------------------------------------
# Artifact naming (variant lineage)
# ---------------------------------------------------------------------------
def _build_tiny(tiny_z_matrix, tiny_payload, **kwargs):
    decomposition, report = build_spectral_decomposition(
        tiny_z_matrix, payload=tiny_payload, config=SpectralConfig(**kwargs)
    )
    assert not report.has_errors
    return decomposition


def test_input_variant_inherits_the_phase02_suffix() -> None:
    assert input_variant(None) == ""
    assert input_variant("data/processed/x/z_matrix.json") == ""
    assert input_variant("data/processed/x/z_matrix.f8652585.json") == ".f8652585"
    assert input_variant("data/processed/x/z_matrix.filtered.json") == ".filtered"
    assert input_variant("data/processed/x/other.json") == ""


def test_variant_stem_defaults_and_inheritance() -> None:
    config = SpectralConfig()
    assert config.is_default()
    assert variant_stem(config) == "eigen"
    assert variant_stem(config, source_artifact="x/z_matrix.json") == "eigen"
    assert variant_stem(config, source_artifact="x/z_matrix.f8652585.json") == "eigen.f8652585"
    assert variant_stem(config, source_artifact="x/z_matrix.filtered.json") == "eigen.filtered"
    assert variant_stem(config, force_config_hash=True) == f"eigen.{config.config_hash()}"


def test_variant_stem_marks_non_default_configs() -> None:
    config = SpectralConfig(method="eigen")
    assert not config.is_default()
    assert variant_stem(config) == f"eigen.{config.config_hash()}"
    assert (
        variant_stem(config, source_artifact="x/z_matrix.f8652585.json")
        == f"eigen.f8652585.{config.config_hash()}"
    )


def test_config_hash_ignores_presentation_only_fields() -> None:
    base = SpectralConfig()
    assert SpectralConfig(resolved_method="svd", k_resolved=21).is_default()
    assert SpectralConfig(resolved_method="svd").config_hash() == base.config_hash()
    assert SpectralConfig(symmetric_input=True).config_hash() == base.config_hash()
    assert SpectralConfig(k_requested="3").config_hash() != base.config_hash()
    assert SpectralConfig(rank_by="value").config_hash() != base.config_hash()
    assert SpectralConfig().hash_fields().keys() == set(MATRIX_CONFIG_FIELDS)


def test_artifact_paths_follow_the_gv_stem_and_variant(tiny_z_matrix, tiny_payload) -> None:
    decomposition = _build_tiny(tiny_z_matrix, tiny_payload)
    paths = artifact_paths(decomposition, "out")
    directory = Path("out") / Path(decomposition.source_file).stem
    assert paths["json"] == directory / "eigen.json"
    assert paths["npz"] == directory / "eigen.npz"
    assert paths["png"] == directory / "eigen.png"
    assert paths["modes"] == directory / "eigen.modes.png"
    assert paths["data"] == directory / "eigen.data.json"
    assert sidecar_path(paths["json"]) == paths["npz"]


def test_artifact_paths_preserve_the_input_variant(tiny_z_matrix, tiny_payload) -> None:
    decomposition = _build_tiny(tiny_z_matrix, tiny_payload)
    decomposition.source_artifact = Path("data/processed/x/z_matrix.f8652585.json")
    paths = artifact_paths(decomposition, "out")
    assert paths["json"].name == "eigen.f8652585.json"
    assert paths["npz"].name == "eigen.f8652585.npz"

    decomposition.source_artifact = Path("data/processed/x/z_matrix.filtered.json")
    assert artifact_paths(decomposition, "out")["json"].name == "eigen.filtered.json"

    forced = artifact_paths(decomposition, "out", force_config_hash=True)
    assert forced["json"].name == f"eigen.filtered.{decomposition.config.config_hash()}.json"


# ---------------------------------------------------------------------------
# Sidecar cache and loading
# ---------------------------------------------------------------------------
SIDECAR_KEYS = {
    "values",
    "loadings_left",
    "loadings_right",
    "explained_variance_ratio",
    "rank_indices",
    "neuron_order",
    "source_json_sha256",
    "numpy_version",
}


def test_sidecar_bundle_and_digest(tiny_z_matrix, tiny_payload, tmp_path) -> None:
    decomposition = _build_tiny(tiny_z_matrix, tiny_payload)
    written = write_spectral_artifacts(decomposition, tmp_path)
    arrays = load_sidecar_arrays(written["npz"])
    assert set(arrays) == SIDECAR_KEYS
    assert np.array_equal(arrays["values"], decomposition.spectrum)
    assert np.array_equal(arrays["loadings_left"], decomposition.loadings_left)
    assert np.array_equal(arrays["loadings_right"], decomposition.loadings_right)
    assert np.array_equal(
        arrays["explained_variance_ratio"], decomposition.explained_variance_ratio
    )
    assert [int(index) for index in arrays["rank_indices"]] == [
        value.original_index for value in decomposition.values
    ]
    assert list(arrays["neuron_order"]) == decomposition.neuron_order
    assert str(arrays["source_json_sha256"][0]) == hashlib.sha256(
        written["json"].read_bytes()
    ).hexdigest()
    assert str(arrays["numpy_version"][0]) == np.__version__
    # the npz path is also accepted by the fast loader
    assert np.array_equal(load_sidecar_arrays(written["json"])["values"], decomposition.spectrum)


def test_sidecar_zip_entries_use_a_fixed_timestamp(tiny_z_matrix, tiny_payload, tmp_path) -> None:
    import zipfile

    decomposition = _build_tiny(tiny_z_matrix, tiny_payload)
    written = write_spectral_artifacts(decomposition, tmp_path)
    with zipfile.ZipFile(written["npz"]) as archive:
        for info in archive.infolist():
            assert info.date_time == (1980, 1, 1, 0, 0, 0)


def test_sidecar_arrays_are_built_from_the_json_content(tiny_z_matrix, tiny_payload) -> None:
    decomposition = _build_tiny(tiny_z_matrix, tiny_payload)
    arrays = build_sidecar_arrays(decomposition, source_json_sha256="a" * 64)
    assert set(arrays) == SIDECAR_KEYS
    assert str(arrays["source_json_sha256"][0]) == "a" * 64
    assert arrays["loadings_left"].dtype == np.dtype("<f8")
    assert arrays["rank_indices"].dtype == np.dtype("<i8")


def test_load_spectrum_round_trips_to_dict(tiny_z_matrix, tiny_payload, tmp_path) -> None:
    decomposition = _build_tiny(tiny_z_matrix, tiny_payload)
    written = write_spectral_artifacts(decomposition, tmp_path)
    loaded = load_spectrum(written["json"])
    assert loaded.to_dict() == decomposition.to_dict()
    assert loaded.loaded_from == written["json"]
    assert loaded.loadings_left.shape == decomposition.loadings_left.shape
    assert loaded.index == decomposition.index


def test_load_spectrum_from_the_npz_path(tiny_z_matrix, tiny_payload, tmp_path) -> None:
    decomposition = _build_tiny(tiny_z_matrix, tiny_payload)
    written = write_spectral_artifacts(decomposition, tmp_path)
    loaded = load_spectrum(written["npz"])
    assert loaded.to_dict() == decomposition.to_dict()
    assert loaded.loaded_from == written["json"]


def test_load_spectrum_without_a_sidecar(tiny_z_matrix, tiny_payload, tmp_path) -> None:
    decomposition = _build_tiny(tiny_z_matrix, tiny_payload)
    written = write_spectral_artifacts(decomposition, tmp_path, sidecar=False)
    assert set(written) == {"json"}
    assert not sidecar_path(written["json"]).exists()
    assert load_spectrum(written["json"]).to_dict() == decomposition.to_dict()


def test_load_spectrum_from_an_orphan_npz_raises(tmp_path) -> None:
    orphan = tmp_path / "eigen.npz"
    orphan.write_bytes(b"not really an npz")
    with pytest.raises(FileNotFoundError, match="sibling JSON"):
        load_spectrum(orphan)


def test_stale_sidecar_is_ignored_with_a_warning(
    tiny_z_matrix, tiny_payload, tmp_path, caplog
) -> None:
    decomposition = _build_tiny(tiny_z_matrix, tiny_payload)
    written = write_spectral_artifacts(decomposition, tmp_path)
    # changing the JSON bytes (a comment-free extra newline) invalidates the cache
    written["json"].write_bytes(written["json"].read_bytes() + b"\n")
    with caplog.at_level(logging.WARNING, logger="src.spectral.spectral_decomposition"):
        loaded = load_spectrum(written["json"])
    assert loaded.to_dict() == decomposition.to_dict()
    assert any("ignoring sidecar" in record.message for record in caplog.records)


def test_tampered_sidecar_raises_when_read_directly(tiny_z_matrix, tiny_payload, tmp_path) -> None:
    decomposition = _build_tiny(tiny_z_matrix, tiny_payload)
    written = write_spectral_artifacts(decomposition, tmp_path)
    arrays = load_sidecar_arrays(written["npz"])
    arrays["values"] = arrays["values"] + 1.0
    write_npz_atomic(written["npz"], arrays)
    with pytest.raises(SpectralValidationError, match="values"):
        load_spectrum(written["npz"])
    # the JSON path falls back to the canonical artifact instead of failing
    assert load_spectrum(written["json"]).to_dict() == decomposition.to_dict()


def test_default_run_writes_exactly_json_and_npz(tiny_z_matrix, tiny_payload, tmp_path) -> None:
    decomposition = _build_tiny(tiny_z_matrix, tiny_payload)
    written = write_spectral_artifacts(decomposition, tmp_path)
    assert set(written) == {"json", "npz"}
    assert written["json"].name == "eigen.json"
    assert written["npz"].name == "eigen.npz"
    assert written["json"].parent == tmp_path / TINY.stem
    # the JSON is canonical: `load_spectrum` succeeds even after the cache is removed
    written["npz"].unlink()
    assert load_spectrum(written["json"]).to_dict() == decomposition.to_dict()


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def _png_size(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    assert data[:8] == PNG_MAGIC
    return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")


def test_plot_spectrum_writes_a_valid_png(tiny_z_matrix, tiny_payload, tmp_path) -> None:
    decomposition = _build_tiny(tiny_z_matrix, tiny_payload)
    written = plot_spectrum(decomposition, tmp_path / "eigen.png", popup=False)
    assert written == tmp_path / "eigen.png"
    width, height = _png_size(written)
    assert width > 100 and height > 100


def test_plot_spectrum_covers_the_eigen_branch(tiny_z_matrix, tiny_payload, tmp_path) -> None:
    decomposition = _build_tiny(tiny_z_matrix, tiny_payload, source="symmetric")
    assert decomposition.metadata["method"] == "eigen"
    written = plot_spectrum(
        decomposition, tmp_path / "eigen.png", popup=False, title="custom title"
    )
    assert _png_size(written)[0] > 100


def test_plot_mode_heatmaps_writes_a_valid_png(tiny_z_matrix, tiny_payload, tmp_path) -> None:
    decomposition = _build_tiny(tiny_z_matrix, tiny_payload)
    written = plot_mode_heatmaps(decomposition, tmp_path / "eigen.modes.png", count=2)
    assert _png_size(written)[0] > 100
    single = plot_mode_heatmaps(decomposition, tmp_path / "single.png", count=1)
    assert _png_size(single)[0] > 100


def test_plot_popup_only_when_a_gui_backend_is_available(
    tiny_z_matrix, tiny_payload, tmp_path, monkeypatch
) -> None:
    decomposition = _build_tiny(tiny_z_matrix, tiny_payload)
    calls: list[float] = []
    monkeypatch.setattr(plt, "show", lambda *_args, **_kwargs: calls.append(1.0))

    monkeypatch.setattr("src.spectral.spectral_decomposition.can_popup", lambda: False)
    plot_spectrum(decomposition, tmp_path / "hidden.png", popup=True)
    assert calls == []  # no GUI backend: the PNG is still written

    monkeypatch.setattr("src.spectral.spectral_decomposition.can_popup", lambda: True)
    plot_spectrum(decomposition, tmp_path / "suppressed.png", popup=False)
    assert calls == []  # --no-popup even on a GUI backend

    plot_spectrum(decomposition, tmp_path / "shown.png", popup=True)
    assert calls == [1.0]  # exactly one window


def test_write_artifact_set_plot_and_save_data(tiny_z_matrix, tiny_payload, tmp_path) -> None:
    decomposition = _build_tiny(tiny_z_matrix, tiny_payload)
    written = write_spectral_artifacts(
        decomposition, tmp_path, plot=True, plot_modes=2, save_data=True, popup=False
    )
    assert set(written) == {"json", "npz", "png", "modes", "data"}
    assert _png_size(written["png"])[0] > 100
    assert _png_size(written["modes"])[0] > 100
    saved = json.loads(written["data"].read_text())
    assert set(saved) == set(SAVE_DATA_KEYS)


def test_write_artifact_set_skips_the_mode_figure_when_disabled(
    tiny_z_matrix, tiny_payload, tmp_path
) -> None:
    decomposition = _build_tiny(tiny_z_matrix, tiny_payload)
    written = write_spectral_artifacts(
        decomposition, tmp_path, plot=True, plot_modes=0, popup=False
    )
    assert set(written) == {"json", "npz", "png"}


def test_write_artifact_set_reports_a_failed_plot(
    tiny_z_matrix, tiny_payload, tmp_path, monkeypatch
) -> None:
    decomposition = _build_tiny(tiny_z_matrix, tiny_payload)
    calls: list[str] = []

    def _explode(*_args, **_kwargs):
        calls.append("called")
        raise RuntimeError("no display")

    monkeypatch.setattr("src.spectral.spectral_decomposition.plot_spectrum", _explode)
    written = write_spectral_artifacts(decomposition, tmp_path, plot=True)
    assert calls == ["called"]
    assert set(written) == {"json", "npz"}  # the JSON still lands
    assert written["json"].exists()


# ---------------------------------------------------------------------------
# Terminal rendering
# ---------------------------------------------------------------------------
def test_render_statistics_contains_the_headline_labels(tiny_z_matrix, tiny_payload) -> None:
    decomposition = _build_tiny(tiny_z_matrix, tiny_payload)
    text = render_statistics(decomposition)
    for needle in (
        "Phase 03 -- spectral statistics",
        "method / source",
        "shape",
        "top 5",
        "heuristics",
        "recommended k",
        "frobenius norm",
        "reconstruction error",
        "orthogonality error",
        "max mode residual",
        "degenerate clusters",
    ):
        assert needle in text, needle


def test_render_summary_box_written_variant(tiny_z_matrix, tiny_payload, tmp_path) -> None:
    decomposition = _build_tiny(tiny_z_matrix, tiny_payload)
    paths = artifact_paths(decomposition, tmp_path)
    text = render_summary_box(decomposition, paths)
    assert "Phase 03 complete" in text
    assert "JSON written: eigen.json" in text
    assert "eigen.npz" in text
    assert "not requested (--plot)" in text
    assert "not requested (--save-data)" in text
    assert "not requested (--stats)" in text
    assert text.startswith("+") and text.endswith("+")


def test_render_summary_box_dry_run_and_no_sidecar(tiny_z_matrix, tiny_payload, tmp_path) -> None:
    decomposition = _build_tiny(tiny_z_matrix, tiny_payload)
    paths = artifact_paths(decomposition, tmp_path)
    dry = render_summary_box(decomposition, paths, written=False)
    assert "Phase 03 dry-run (nothing written)" in dry
    assert "skipped (--dry-run)" in dry
    no_sidecar = render_summary_box(decomposition, paths, sidecar=False)
    assert "skipped (--no-sidecar)" in no_sidecar
    assert "eigen.json" in no_sidecar


def test_render_summary_box_plot_and_save_data_lines(tiny_z_matrix, tiny_payload, tmp_path) -> None:
    decomposition = _build_tiny(tiny_z_matrix, tiny_payload)
    paths = artifact_paths(decomposition, tmp_path)
    text = render_summary_box(
        decomposition, paths, plot=True, plot_modes=4, save_data=True, stats=True
    )
    assert "eigen.png" in text and "eigen.modes.png" in text
    assert "eigen.data.json" in text
    assert "printed above (--stats)" in text
    without_modes = render_summary_box(decomposition, paths, plot=True, plot_modes=0)
    assert "eigen.modes.png" not in without_modes


# ---------------------------------------------------------------------------
# resolve_inputs
# ---------------------------------------------------------------------------
def test_resolve_inputs_directory_recursion_and_exclusions(tmp_path) -> None:
    for relative in (
        "a/z_matrix.json",
        "a/z_matrix.data.json",
        "a/eigen.json",
        "b/z_matrix.f8652585.json",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    (tmp_path / "a" / "notes.txt").write_text("x", encoding="utf-8")

    default = resolve_inputs([tmp_path])
    assert [path.name for path in default] == ["z_matrix.json"]
    assert default[0].parent.name == "a"

    with_variants = resolve_inputs([tmp_path], include_variants=True)
    assert sorted(path.name for path in with_variants) == [
        "z_matrix.f8652585.json",
        "z_matrix.json",
    ]


def test_resolve_inputs_explicit_files_and_errors(tmp_path, caplog) -> None:
    good = tmp_path / "z_matrix.json"
    good.write_text("{}", encoding="utf-8")
    wrong_suffix = tmp_path / "z_matrix.txt"
    wrong_suffix.write_text("{}", encoding="utf-8")
    save_data = tmp_path / "z_matrix.data.json"
    save_data.write_text("{}", encoding="utf-8")

    assert resolve_inputs([good]) == [good]
    with caplog.at_level(logging.ERROR, logger="src.spectral.spectral_decomposition"):
        assert resolve_inputs([wrong_suffix]) == []
        assert resolve_inputs([save_data]) == []
        assert resolve_inputs([tmp_path / "missing.json"]) == []
    messages = " ".join(record.message for record in caplog.records)
    assert "unsupported suffix" in messages
    assert "save-data artifact" in messages
    assert "no such file or directory" in messages


def test_resolve_inputs_is_deterministic_and_deduplicated(tmp_path) -> None:
    for name in ("a", "b"):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "z_matrix.json").write_text("{}", encoding="utf-8")
    first = resolve_inputs([tmp_path, tmp_path / "a" / "z_matrix.json"])
    second = resolve_inputs([tmp_path, tmp_path / "a" / "z_matrix.json"])
    assert first == second
    assert len(first) == len(set(first)) == 2


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _artifact(workspace: Path) -> Path:
    return workspace / TINY.stem / "z_matrix.json"


def _snapshot(root: Path) -> set[str]:
    return {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()}


def test_cli_end_to_end(tiny_workspace, capsys) -> None:
    code = main(
        [
            "-i", str(_artifact(tiny_workspace)),
            "-o", str(tiny_workspace),
            "--stats",
            "--plot",
            "--no-popup",
            "--save-data",
        ]
    )
    assert code == 0
    output = capsys.readouterr().out
    assert "Phase 03 -- spectral statistics" in output
    assert "Phase 03 complete" in output
    directory = tiny_workspace / TINY.stem
    for name in ("eigen.json", "eigen.npz", "eigen.png", "eigen.modes.png", "eigen.data.json"):
        assert (directory / name).is_file(), name
    payload = json.loads((directory / "eigen.json").read_text())
    assert payload["provenance"]["phase"] == "03"
    assert load_spectrum(directory / "eigen.json").to_dict() == payload


def test_cli_dry_run_writes_nothing(tiny_workspace, capsys) -> None:
    before = _snapshot(tiny_workspace)
    code = main(["-i", str(_artifact(tiny_workspace)), "-o", str(tiny_workspace), "--dry-run"])
    assert code == 0
    assert "Phase 03 dry-run (nothing written)" in capsys.readouterr().out
    assert _snapshot(tiny_workspace) == before


def test_cli_no_sidecar(tiny_workspace) -> None:
    code = main(["-i", str(_artifact(tiny_workspace)), "-o", str(tiny_workspace), "--no-sidecar"])
    assert code == 0
    directory = tiny_workspace / TINY.stem
    assert (directory / "eigen.json").is_file()
    assert not (directory / "eigen.npz").exists()
    assert load_spectrum(directory / "eigen.json").n_neurons == 3


def test_cli_strict_is_clean_on_a_valid_input(tiny_workspace) -> None:
    assert main(["-i", str(_artifact(tiny_workspace)), "-o", str(tiny_workspace), "--strict"]) == 0


def test_cli_returns_2_when_nothing_matches(tmp_path) -> None:
    assert main(["-i", str(tmp_path / "empty"), "-o", str(tmp_path)]) == 2


def test_cli_returns_1_for_an_invalid_config(tiny_workspace) -> None:
    code = main(
        ["-i", str(_artifact(tiny_workspace)), "-o", str(tiny_workspace), "--gap-window", "1"]
    )
    assert code == 1


def test_cli_returns_1_for_an_unreadable_input(tmp_path) -> None:
    broken = tmp_path / "z_matrix.json"
    broken.write_text("{ not json", encoding="utf-8")
    assert main(["-i", str(broken), "-o", str(tmp_path)]) == 1


def test_cli_processes_a_directory_with_variants(tiny_workspace) -> None:
    source = _artifact(tiny_workspace)
    variant = source.with_name("z_matrix.f8652585.json")
    variant.write_bytes(source.read_bytes())
    code = main(["-i", str(tiny_workspace), "-o", str(tiny_workspace), "--include-variants"])
    assert code == 0
    directory = tiny_workspace / TINY.stem
    assert (directory / "eigen.json").is_file()
    assert (directory / "eigen.f8652585.json").is_file()
    # the save-data file is never treated as an input
    assert not (directory / "eigen.data.json").exists()


def test_cli_custom_k_and_config_hash(tiny_workspace) -> None:
    code = main(
        [
            "-i", str(_artifact(tiny_workspace)),
            "-o", str(tiny_workspace),
            "--k", "2",
            "--config-hash",
            "--no-sidecar",
        ]
    )
    assert code == 0
    expected = tiny_workspace / TINY.stem / f"eigen.{SpectralConfig(k_requested='2').config_hash()}.json"
    assert expected.is_file()
    assert load_spectrum(expected).k == 2


def test_cli_writes_no_csv(tiny_workspace) -> None:
    assert main(["-i", str(_artifact(tiny_workspace)), "-o", str(tiny_workspace)]) == 0
    assert not list(tiny_workspace.rglob("*.csv"))


def test_cli_module_is_runnable_as_a_subprocess(tiny_workspace) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m", "src.spectral.spectral_decomposition",
            "-i", str(_artifact(tiny_workspace)),
            "-o", str(tiny_workspace),
            "--dry-run",
            "--log-level", "ERROR",
        ],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "Phase 03 dry-run" in completed.stdout


# ---------------------------------------------------------------------------
# Reference dataset (deselect with -m "not slow")
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def reference():
    """The real Phase 02 artifact, decomposed once for the slow integration tests."""
    if not REFERENCE_ARTIFACT.is_file():
        pytest.skip("reference Phase 02 artifact is not available")
    from src.matrices.build_square_matrix import load_z_matrix

    z_matrix = load_z_matrix(REFERENCE_ARTIFACT)
    decomposition, report = build_spectral_decomposition(
        z_matrix, payload=z_matrix.to_dict(), source_artifact=REFERENCE_ARTIFACT
    )
    assert not report.has_errors and not report.warnings
    return z_matrix, decomposition, report


@pytest.mark.slow
def test_reference_directed_svd_spectrum(reference) -> None:
    z_matrix, decomposition, _ = reference
    metadata = decomposition.metadata
    assert decomposition.n_neurons == 113
    assert decomposition.config.resolved_method == "svd"
    assert metadata["matrix_is_symmetric"] is False
    assert metadata["n_modes_total"] == 113

    expected = [4.441772842851828, 2.650443500279925, 2.3707819941036536, 2.3242901665143907]
    for value, want in zip(decomposition.values[:4], expected):
        assert value.value == pytest.approx(want, rel=1e-9)
    # Phase 02's informational numbers are reproduced
    assert metadata["abs_value_max"] == pytest.approx(
        z_matrix.metadata["largest_singular_value"], rel=1e-12
    )
    assert metadata["frobenius_norm"] == pytest.approx(
        z_matrix.metadata["frobenius_norm"], rel=1e-15
    )
    assert metadata["sum_squares"] == pytest.approx(metadata["frobenius_norm"] ** 2, rel=1e-12)
    assert [round(value.explained_variance_ratio, 6) for value in decomposition.values[:4]] == [
        0.307823,
        0.109604,
        0.087694,
        0.084289,
    ]
    assert decomposition.values[3].cumulative_variance_ratio == pytest.approx(0.589409, abs=1e-6)


@pytest.mark.slow
def test_reference_heuristics(reference) -> None:
    _, decomposition, _ = reference
    heuristics = decomposition.heuristics
    assert heuristics["variance_threshold"] == {"k": 22, "threshold": 0.9}
    assert heuristics["elbow"]["k"] == 21
    assert heuristics["spectral_gap"]["k"] == 1
    assert heuristics["spectral_gap"]["metric"] == "ratio"
    assert heuristics["spectral_gap"]["window"] == DEFAULT_GAP_WINDOW
    assert heuristics["spectral_gap"]["value"] == pytest.approx(1.6758602258009712, rel=1e-9)
    assert heuristics["participation_ratio"] == pytest.approx(35.6323231094758, rel=1e-9)
    assert heuristics["recommended_k"] == 21
    assert decomposition.k == 21
    assert decomposition.metadata["cumulative_at_k"] == pytest.approx(0.896063, abs=1e-6)


@pytest.mark.slow
def test_reference_numerical_health(reference) -> None:
    _, decomposition, _ = reference
    metadata = decomposition.metadata
    assert metadata["numerical_rank"] == 109
    assert metadata["condition_number"] == pytest.approx(409126.97892414214, rel=1e-6)
    assert metadata["orthogonality_error"] < 1e-12
    assert metadata["max_mode_residual"] < 1e-12
    assert metadata["degenerate_groups"]["count"] >= 1
    assert metadata["degenerate_groups"]["sizes"] == [4]


@pytest.mark.slow
def test_reference_symmetric_eigen_spectrum(reference) -> None:
    z_matrix, _, _ = reference
    decomposition, report = build_spectral_decomposition(
        z_matrix,
        payload=z_matrix.to_dict(),
        source_artifact=REFERENCE_ARTIFACT,
        config=SpectralConfig(source="symmetric"),
    )
    assert not report.has_errors
    metadata = decomposition.metadata
    assert decomposition.config.resolved_method == "eigen"
    assert metadata["matrix_is_symmetric"] is True
    # the spectral radius matches Phase 02's symmetrized_spectral_radius exactly
    assert metadata["abs_value_max"] == pytest.approx(
        z_matrix.metadata["symmetrized_spectral_radius"], rel=1e-12
    )
    assert decomposition.values[0].value == pytest.approx(-2.4333138382394823, rel=1e-12)
    assert metadata["trace"] == pytest.approx(0.0, abs=1e-12)
    assert metadata["numerical_rank"] is None
    assert metadata["condition_number"] is None
    assert metadata["max_mode_residual"] < 1e-12


@pytest.mark.slow
def test_reference_eigen_and_svd_agree_on_a_symmetric_input(reference) -> None:
    z_matrix, _, _ = reference
    payload = z_matrix.to_dict()
    eigen, _ = build_spectral_decomposition(
        z_matrix, payload=payload, config=SpectralConfig(source="symmetric", method="eigen")
    )
    singular, _ = build_spectral_decomposition(
        z_matrix, payload=payload, config=SpectralConfig(source="symmetric", method="svd")
    )
    assert np.allclose(eigen.abs_spectrum, singular.spectrum, rtol=1e-12, atol=1e-14)
    assert eigen.metadata["sum_squares"] == pytest.approx(
        singular.metadata["sum_squares"], rel=1e-12
    )


@pytest.mark.slow
def test_reference_variant_uses_the_eigen_path() -> None:
    if not VARIANT_ARTIFACT.is_file():
        pytest.skip("reference Phase 02 variant artifact is not available")
    from src.matrices.build_square_matrix import load_z_matrix

    z_matrix = load_z_matrix(VARIANT_ARTIFACT)
    decomposition, report = build_spectral_decomposition(
        z_matrix, payload=z_matrix.to_dict(), source_artifact=VARIANT_ARTIFACT
    )
    assert not report.has_errors
    assert decomposition.config.resolved_method == "eigen"
    assert decomposition.metadata["abs_value_max"] == pytest.approx(
        z_matrix.metadata["spectral_radius"], rel=1e-12
    )


@pytest.mark.slow
def test_reference_cli_run(tmp_path) -> None:
    code = main(
        [
            "-i", str(REFERENCE_ARTIFACT),
            "-o", str(tmp_path),
            "--no-sidecar",
            "--strict",
            "--stats",
        ]
    )
    assert code == 0
    written = tmp_path / REFERENCE_GV.stem / "eigen.json"
    assert written.is_file()
    loaded = load_spectrum(written)
    assert loaded.n_neurons == 113
    assert loaded.metadata["method"] == "svd"
    assert loaded.k == 21
    assert validate_spectral_payload_schema(loaded.to_dict()) == []



