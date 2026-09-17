"""Unit tests for the Phase 02 Z-matrix builder.

The synthetic fixtures in ``tests/fixtures`` drive the fast path; the integration
test against the real 727 KB dataset is marked ``slow`` and can be deselected with
``-m "not slow"``.  The matplotlib backend is pinned to ``Agg`` for the whole
module so ``--plot`` never opens (or blocks on) a GUI window.
"""

from __future__ import annotations

import builtins
import io
import json
import logging
import math
import struct
import subprocess
import sys
import zipfile
from dataclasses import fields
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: --plot must save the PNG without a popup

import numpy as np  # noqa: E402  (after the backend pin, deliberately)
import pytest  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402  (after matplotlib.use("Agg"))

from src.matrices import build_square_matrix as builder  # noqa: E402
from src.matrices.build_square_matrix import (  # noqa: E402
    DEFAULT_CMAP,
    DEFAULT_HIST_BINS,
    INTERACTIVE_ANNOTATION_KWARGS,
    MATRIX_CONFIG_FIELDS,
    ZMatrixConfig,
    ZMatrixValidationError,
    artifact_paths,
    attach_cell_cursor,
    build_argument_parser,
    build_interactive_figure,
    build_z_matrix,
    build_z_matrix_from_file,
    can_popup,
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
    variant_stem,
    write_artifact_set,
    write_npz_atomic,
)
from src.parsing.parse_graphviz import (  # noqa: E402
    build_parsed_graph,
    parse_graphviz_file,
    write_artifact,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
TINY = FIXTURES / "tiny_valid.gv"
ZERO_Z = FIXTURES / "zero_z.gv"
REFERENCE_GV = PROJECT_ROOT / "data" / "raw_dot" / "FB4Yaffect_FB45_999prePost_001_all.gv"
REFERENCE_ARTIFACT = (
    PROJECT_ROOT / "data" / "processed" / REFERENCE_GV.stem / "parsed_graph.json"
)

FIXED_EPOCH = "1700000000"

#: ``zscore_context.md`` section 5 -- the documented worked examples.
DOCUMENTED_EXAMPLES = [
    ((0.2, 0.9), 0.44),    # strong gain
    ((0.6, 0.7), 0.0),     # neutral (post == pre + eps)
    ((0.9, 0.2), -0.51),   # strong attenuation
    ((0.1, 0.5), 0.22),    # weak but high gain
    ((0.05, 0.08), -0.036),  # very weak
]


def expected_weight(pre_z: float, post_z: float, eps: float = 0.1, alpha: float = 1.0) -> float:
    """Independent stdlib implementation of the unification rule."""
    strength = (abs(pre_z) + abs(post_z)) / 2.0
    return strength * math.tanh(alpha * math.log(post_z / (pre_z + eps)))


@pytest.fixture(scope="module")
def tiny_payload() -> dict:
    parsed, report = build_parsed_graph(TINY)
    assert not report.issues, report.to_dict()
    return parsed.to_dict()


@pytest.fixture(scope="module")
def tiny_z(tiny_payload: dict):
    z_matrix, report = build_z_matrix(tiny_payload)
    assert not report.issues, report.to_dict()
    return z_matrix


@pytest.fixture()
def tiny_artifact(tmp_path: Path) -> Path:
    """Write the Phase 01 artifact for ``tiny_valid.gv`` and return its path."""
    return write_artifact(parse_graphviz_file(TINY), tmp_path / "processed")


# ---------------------------------------------------------------------------
# Unification rule
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(("z_scores", "documented"), DOCUMENTED_EXAMPLES)
def test_unified_weight_reproduces_documented_examples(z_scores, documented) -> None:
    # the doc quotes 2 significant decimals, so compare with that tolerance
    assert unified_weight(*z_scores) == pytest.approx(documented, abs=0.005)


@pytest.mark.parametrize(
    "z_scores",
    [(0.0, 0.0), (0.1, 0.1), (0.2, 0.9), (1.0, 0.1), (0.33, 0.77), (0.05, 0.08), (0.5, 0.0), (0.0, 0.4)],
)
def test_unified_weight_matches_an_independent_implementation(z_scores: tuple[float, float]) -> None:
    pre_z, post_z = z_scores
    if pre_z == 0 or post_z == 0:
        assert unified_weight(pre_z, post_z) == 0.0  # zero-in-zero-out policy
    else:
        assert unified_weight(pre_z, post_z) == pytest.approx(expected_weight(pre_z, post_z), rel=1e-12)


def test_unified_weights_vectorized_matches_scalar() -> None:
    pre = np.array([0.2, 0.6, 0.9, 0.1, 0.05])
    post = np.array([0.9, 0.7, 0.2, 0.5, 0.08])
    vectorized = unified_weights(pre, post)
    assert np.allclose(vectorized, [expected_weight(p, q) for p, q in zip(pre, post)], rtol=0, atol=1e-15)
    assert vectorized.dtype == np.float64


def test_zero_in_zero_out_policy_vs_raw_formula() -> None:
    # 0.5 / 0.0 -> the raw formula hits the attenuation limit -s = -0.25
    assert unified_weight(0.5, 0.0) == 0.0
    assert unified_weight(0.5, 0.0, zero_policy="formula") == pytest.approx(-0.25, rel=1e-12)
    assert unified_weight(0.0, 0.0) == 0.0
    assert unified_weight(0.0, 0.4) == 0.0


def test_negative_zero_is_canonicalized() -> None:
    values = unified_weights(np.array([0.0]), np.array([0.0]))
    assert not np.signbit(values).any()
    assert json.dumps(float(values[0])) == "0.0"


def test_unified_weights_never_emit_runtime_warnings() -> None:
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        unified_weights(np.array([0.0, 0.5, 0.0]), np.array([0.0, 0.0, 0.4]))


def test_unified_weight_is_bounded_and_monotone() -> None:
    for pre_z in (0.1, 0.5, 1.0):
        for post_z in (0.1, 0.5, 1.0):
            weight = unified_weight(pre_z, post_z)
            strength = (abs(pre_z) + abs(post_z)) / 2.0
            assert abs(weight) <= strength + 1e-12
        # increasing post (gain) and decreasing pre (easier gain) both raise w
        assert unified_weight(pre_z, 0.2) < unified_weight(pre_z, 0.9)
        assert unified_weight(0.9, 0.5) < unified_weight(0.2, 0.5)


def test_alpha_controls_sensitivity() -> None:
    weak = unified_weight(0.1, 0.5, alpha=0.5)
    strong = unified_weight(0.1, 0.5, alpha=2.0)
    assert 0 < weak < strong


# ---------------------------------------------------------------------------
# Matrix assembly
# ---------------------------------------------------------------------------
def test_tiny_matrix_is_exact(tiny_z) -> None:
    weights = [expected_weight(0.8, 0.6), expected_weight(0.4, 0.2)]
    assert tiny_z.neuron_order == ["N1", "N2", "N3"]
    assert tiny_z.matrix.shape == (3, 3)
    assert tiny_z.matrix[0, 1] == pytest.approx(weights[0], rel=1e-12)
    assert tiny_z.matrix[1, 2] == pytest.approx(weights[1], rel=1e-12)
    assert tiny_z.matrix[0, 1] == pytest.approx(-0.2692307692307692, rel=1e-12)
    assert tiny_z.matrix.sum() == pytest.approx(sum(weights), rel=1e-12)
    # missing edges and the diagonal are exactly zero
    assert tiny_z.matrix[1, 0] == 0.0 and tiny_z.matrix[2, 0] == 0.0
    assert np.all(np.diag(tiny_z.matrix) == 0.0)


def test_tiny_matrix_symmetric_key_is_the_mean_rule(tiny_z) -> None:
    directed = tiny_z.matrix
    assert tiny_z.matrix_symmetric[0, 1] == pytest.approx(directed[0, 1] / 2, rel=1e-12)
    assert np.array_equal(tiny_z.matrix_symmetric, tiny_z.matrix_symmetric.T)


def test_matrix_equals_the_scatter_of_pair_weights(tiny_z) -> None:
    index = tiny_z.index
    rebuilt = np.zeros((tiny_z.n_neurons, tiny_z.n_neurons))
    for pair in tiny_z.pairs:
        rebuilt[index[pair.source], index[pair.target]] = pair.weight
    assert np.array_equal(rebuilt, tiny_z.matrix)


def test_pair_weights_are_pre_normalization(tiny_payload: dict) -> None:
    z_matrix, report = build_z_matrix(tiny_payload, config=ZMatrixConfig(normalize="unit"))
    assert not report.issues
    # weights stay raw even though the matrix was scaled
    assert [pair.weight for pair in z_matrix.pairs] == pytest.approx([-0.2692307692307692, -0.21724137931034482])
    assert np.abs(z_matrix.matrix).max() == pytest.approx(1.0, rel=1e-12)


def test_symmetric_mode_makes_matrix_and_symmetric_key_identical(tiny_payload: dict) -> None:
    directed, _ = build_z_matrix(tiny_payload)
    z_matrix, report = build_z_matrix(tiny_payload, config=ZMatrixConfig(symmetric=True))
    assert not report.issues
    assert np.array_equal(z_matrix.matrix, z_matrix.matrix_symmetric)
    assert np.array_equal(
        z_matrix.matrix,
        (directed.matrix + directed.matrix.T) / 2,
    )
    # symmetric matrices have real eigenvalues and orthogonal eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(z_matrix.matrix)
    assert np.allclose(eigenvalues.imag, 0.0)
    assert np.allclose(eigenvectors.T @ eigenvectors, np.eye(3), atol=1e-12)


@pytest.mark.parametrize("method", ["mean", "sum", "max-abs", "min-abs"])
def test_symmetrize_methods(method: str) -> None:
    values = np.array([[0.0, -0.5, 0.25], [0.75, 0.0, 0.0], [0.0, 0.0, 0.0]])
    result = symmetrize(values, method=method)
    assert np.array_equal(result, result.T)
    if method == "mean":
        assert np.allclose(result, (values + values.T) / 2)
    if method == "sum":
        assert np.allclose(result, values + values.T)
    if method == "max-abs":
        assert result[0, 1] == 0.75 and result[1, 0] == 0.75
    if method == "min-abs":
        assert result[0, 1] == -0.5 and result[1, 0] == -0.5


def test_symmetrize_rejects_unknown_method() -> None:
    with pytest.raises(ZMatrixValidationError):
        symmetrize(np.zeros((2, 2)), method="bogus")


@pytest.mark.parametrize(
    ("method", "check"),
    [
        ("rows", lambda out: np.allclose(np.abs(out).sum(axis=1), [1.0, 1.0, 0.0])),
        ("cols", lambda out: np.allclose(np.abs(out).sum(axis=0), [1.0, 1.0, 1.0])),
        ("unit", lambda out: np.isclose(np.abs(out).max(), 1.0)),
        ("spectral", lambda out: np.isclose(np.linalg.svd(out, compute_uv=False)[0], 1.0)),
        ("none", lambda out: np.allclose(out, [[0.0, -0.5, 0.25], [0.75, 0.0, 0.0], [0.0, 0.0, 0.0]])),
    ],
)
def test_normalize_properties(method: str, check) -> None:
    values = np.array([[0.0, -0.5, 0.25], [0.75, 0.0, 0.0], [0.0, 0.0, 0.0]])
    normalized, info = normalize_matrix(values, method=method)
    assert info["method"] == method
    assert check(normalized)


def test_normalize_zscore_nonzero_keeps_zeros_and_standardizes_stored_weights() -> None:
    values = np.array([[0.0, -0.5, 0.25], [0.75, 0.0, 0.0], [0.0, 0.0, 0.0]])
    normalized, info = normalize_matrix(values, method="zscore-nonzero")
    stored = normalized[normalized != 0]
    assert np.isclose(stored.mean(), 0.0, atol=1e-12)
    assert np.isclose(stored.std(), 1.0, atol=1e-12)
    assert (normalized == 0).sum() == (values == 0).sum()
    assert info["scale"] is not None


@pytest.mark.parametrize("method", ["rows", "cols", "unit", "spectral", "zscore-nonzero"])
def test_normalize_all_zero_matrix_is_a_noop(method: str) -> None:
    normalized, _ = normalize_matrix(np.zeros((4, 4)), method=method)
    assert np.array_equal(normalized, np.zeros((4, 4)))


def test_normalize_rejects_unknown_method() -> None:
    with pytest.raises(ZMatrixValidationError):
        normalize_matrix(np.zeros((2, 2)), method="bogus")


# ---------------------------------------------------------------------------
# Configuration hashing and artifact naming
# ---------------------------------------------------------------------------
def test_config_hash_is_eight_hex_characters_and_stable() -> None:
    config = ZMatrixConfig()
    assert len(config.config_hash()) == 8
    assert all(character in "0123456789abcdef" for character in config.config_hash())
    assert config.config_hash() == ZMatrixConfig().config_hash()
    assert config.is_default() is True


@pytest.mark.parametrize(
    "overrides",
    [
        {"eps": 0.2},
        {"alpha": 2.0},
        {"zero_policy": "formula"},
        {"symmetric": True},
        {"symmetrize": "sum"},
        {"normalize": "rows"},
        {"filters": [{"type": "min_pre", "value": 0.2, "removed": 3}]},
    ],
)
def test_config_hash_incorporates_matrix_defining_fields(overrides: dict) -> None:
    assert ZMatrixConfig(**overrides).config_hash() != ZMatrixConfig().config_hash()


def test_filters_are_hashed_but_do_not_force_a_hash_segment() -> None:
    filtered = ZMatrixConfig(filters=[{"type": "min_pre", "value": 0.2, "removed": 3}])
    assert filtered.is_default() is True  # `.filtered` already names this variant
    assert filtered.config_hash() != ZMatrixConfig().config_hash()


def test_variant_stem_naming_policy() -> None:
    canonical = Path("data/processed/x/parsed_graph.json")
    filtered = Path("data/processed/x/parsed_graph.filtered.json")
    default_config = ZMatrixConfig()
    variant_config = ZMatrixConfig(normalize="rows")

    assert variant_stem(default_config, source_artifact=canonical) == "z_matrix"
    assert variant_stem(default_config, source_artifact=filtered) == "z_matrix.filtered"
    assert variant_stem(variant_config, source_artifact=canonical) == f"z_matrix.{variant_config.config_hash()}"
    assert (
        variant_stem(default_config, source_artifact=filtered, force_config_hash=True)
        == f"z_matrix.filtered.{default_config.config_hash()}"
    )
    assert variant_stem(default_config, source_artifact=canonical, force_config_hash=True) == (
        f"z_matrix.{default_config.config_hash()}"
    )


def test_artifact_paths_sit_next_to_the_phase_one_artifact(tiny_z) -> None:
    paths = artifact_paths(tiny_z, Path("data/processed"))
    assert set(paths) == {"json", "npz", "png", "data"}
    assert paths["json"] == Path("data/processed/tiny_valid/z_matrix.json")
    assert paths["npz"] == Path("data/processed/tiny_valid/z_matrix.npz")
    assert paths["png"] == Path("data/processed/tiny_valid/z_matrix.png")
    assert paths["data"] == Path("data/processed/tiny_valid/z_matrix.data.json")
    assert sidecar_path(paths["json"]) == paths["npz"]


# ---------------------------------------------------------------------------
# Artifacts: JSON + NPZ sidecar
# ---------------------------------------------------------------------------
def test_write_artifact_set_writes_the_canonical_pair(tiny_z, tmp_path: Path) -> None:
    written = write_artifact_set(tiny_z, tmp_path)
    assert sorted(path.name for path in (tmp_path / "tiny_valid").iterdir()) == [
        "z_matrix.json",
        "z_matrix.npz",
    ]
    assert set(written) == {"json", "npz"}


def test_sidecar_array_contract(tiny_z, tmp_path: Path) -> None:
    write_artifact_set(tiny_z, tmp_path)
    json_path = tmp_path / "tiny_valid" / "z_matrix.json"

    arrays = load_sidecar_arrays(json_path)
    assert set(arrays) == {
        "matrix",
        "matrix_symmetric",
        "neuron_order",
        "edge_rows",
        "edge_cols",
        "edge_weights",
        "source_json_sha256",
        "numpy_version",
    }
    assert arrays["matrix"].dtype == np.dtype("<f8")
    assert arrays["neuron_order"].dtype.kind == "U"
    assert arrays["edge_rows"].dtype == np.dtype("<i8")
    assert np.array_equal(arrays["matrix"], tiny_z.matrix)
    assert np.array_equal(arrays["matrix_symmetric"], tiny_z.matrix_symmetric)
    assert list(arrays["neuron_order"]) == tiny_z.neuron_order
    assert list(arrays["edge_weights"]) == [pair.weight for pair in tiny_z.pairs]
    # edges align 1:1 with `pairs`
    index = tiny_z.index
    assert [int(value) for value in arrays["edge_rows"]] == [index[pair.source] for pair in tiny_z.pairs]
    assert [int(value) for value in arrays["edge_cols"]] == [index[pair.target] for pair in tiny_z.pairs]
    # the cache is keyed by the canonical JSON's bytes
    import hashlib

    assert str(arrays["source_json_sha256"][0]) == hashlib.sha256(json_path.read_bytes()).hexdigest()
    assert str(arrays["numpy_version"][0]) == np.__version__


def test_sidecar_loads_without_pickle(tiny_z, tmp_path: Path) -> None:
    write_artifact_set(tiny_z, tmp_path)
    npz_path = tmp_path / "tiny_valid" / "z_matrix.npz"
    with np.load(npz_path, allow_pickle=False) as archive:
        assert "matrix" in archive.files
    assert set(load_sidecar_arrays(npz_path)) >= {"matrix", "neuron_order"}


def test_sidecar_archives_are_written_with_a_fixed_timestamp(tiny_z, tmp_path: Path) -> None:
    # numpy pins the zip entry date, which is what makes the cache reproducible
    write_artifact_set(tiny_z, tmp_path)
    npz_path = tmp_path / "tiny_valid" / "z_matrix.npz"
    with zipfile.ZipFile(io.BytesIO(npz_path.read_bytes())) as archive:
        assert {info.date_time for info in archive.infolist()} == {(1980, 1, 1, 0, 0, 0)}


def test_both_artifacts_are_byte_reproducible(tiny_artifact: Path, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SOURCE_DATE_EPOCH", FIXED_EPOCH)
    outdir = tmp_path / "out"
    assert main(["-i", str(tiny_artifact), "-o", str(outdir)]) == 0
    first = (outdir / "tiny_valid" / "z_matrix.json").read_bytes()
    first_npz = (outdir / "tiny_valid" / "z_matrix.npz").read_bytes()

    assert main(["-i", str(tiny_artifact), "-o", str(outdir)]) == 0
    assert (outdir / "tiny_valid" / "z_matrix.json").read_bytes() == first
    assert (outdir / "tiny_valid" / "z_matrix.npz").read_bytes() == first_npz


def test_load_z_matrix_round_trips_and_accepts_both_paths(tiny_z, tmp_path: Path) -> None:
    from src.utils.io import dumps_json

    write_artifact_set(tiny_z, tmp_path)
    json_path = tmp_path / "tiny_valid" / "z_matrix.json"

    from_json = load_z_matrix(json_path)
    from_npz = load_z_matrix(sidecar_path(json_path))
    assert from_json.to_dict() == from_npz.to_dict()
    # provenance always describes the Phase 01 input, so re-serialization is identical
    assert dumps_json(from_json.to_dict()) == json_path.read_text(encoding="utf-8")
    assert from_json.loaded_from == json_path
    # this object was built in memory, so provenance has no input path
    assert from_json.source_artifact == Path("<in-memory>")
    assert from_json.source_file == "tiny_valid.gv"


def test_stale_sidecar_is_ignored_with_a_warning(tiny_z, tmp_path: Path, caplog) -> None:
    write_artifact_set(tiny_z, tmp_path)
    npz_path = tmp_path / "tiny_valid" / "z_matrix.npz"

    tampered = load_sidecar_arrays(npz_path)
    tampered["matrix"] = tampered["matrix"] * 2.0
    write_npz_atomic(npz_path, tampered)

    with caplog.at_level(logging.WARNING, logger="src.matrices.build_square_matrix"):
        loaded = load_z_matrix(tmp_path / "tiny_valid" / "z_matrix.json")
    assert "ignoring sidecar" in caplog.text
    assert np.array_equal(loaded.matrix, tiny_z.matrix)  # the JSON wins


def test_load_z_matrix_rejects_a_tampered_cache_when_read_directly(tiny_z, tmp_path: Path) -> None:
    write_artifact_set(tiny_z, tmp_path)
    npz_path = tmp_path / "tiny_valid" / "z_matrix.npz"
    tampered = load_sidecar_arrays(npz_path)
    tampered["matrix"] = tampered["matrix"] * 2.0
    write_npz_atomic(npz_path, tampered)
    with pytest.raises(ZMatrixValidationError):
        load_z_matrix(npz_path)


def test_load_sidecar_arrays_raises_when_missing(tiny_z, tmp_path: Path) -> None:
    write_artifact_set(tiny_z, tmp_path, sidecar=False)
    with pytest.raises(FileNotFoundError):
        load_sidecar_arrays(tmp_path / "tiny_valid" / "z_matrix.json")


def test_no_sidecar_writes_only_the_json(tiny_artifact: Path, tmp_path: Path) -> None:
    outdir = tmp_path / "out"
    assert main(["-i", str(tiny_artifact), "-o", str(outdir), "--no-sidecar"]) == 0
    files = sorted(path.name for path in (outdir / "tiny_valid").iterdir())
    assert files == ["z_matrix.json"]
    loaded = load_z_matrix(outdir / "tiny_valid" / "z_matrix.json")
    assert loaded.n_neurons == 3


def test_write_npz_atomic_leaves_no_partial_file(tmp_path: Path) -> None:
    destination = tmp_path / "bundle.npz"
    with pytest.raises(TypeError):
        write_npz_atomic(destination, {1: np.zeros(2)})  # np.savez needs string keys
    assert list(tmp_path.iterdir()) == []


def test_sidecar_carries_the_symmetric_matrix_in_symmetric_mode(tiny_payload: dict, tmp_path: Path) -> None:
    z_matrix, report = build_z_matrix(tiny_payload, config=ZMatrixConfig(symmetric=True))
    assert not report.issues
    write_artifact_set(z_matrix, tmp_path)
    # a non-default config gets the `<config_hash8>` filename segment
    json_path = artifact_paths(z_matrix, tmp_path)["json"]
    assert json_path.name == f"z_matrix.{ZMatrixConfig(symmetric=True).config_hash()}.json"
    arrays = load_sidecar_arrays(json_path)
    assert np.array_equal(arrays["matrix"], arrays["matrix_symmetric"])
    assert np.array_equal(arrays["matrix"], z_matrix.matrix)


# ---------------------------------------------------------------------------
# --save-data, --stats and the completion box
# ---------------------------------------------------------------------------
def test_save_data_payload_contract(tiny_z) -> None:
    payload = save_data_payload(tiny_z)
    assert set(payload) == set(builder.SAVE_DATA_KEYS)
    assert payload["symmetric"] is False
    assert payload["normalize"] == "none"
    assert payload["pairs"] == [pair.to_dict() for pair in tiny_z.pairs]
    assert payload["z_stats"]["min"] == pytest.approx(-0.2692307692307692, rel=1e-12)
    assert payload["z_stats"]["max"] == pytest.approx(-0.21724137931034482, rel=1e-12)
    assert payload["sparsity"] == pytest.approx(7 / 9)
    assert (payload["n_positive"], payload["n_negative"], payload["n_zero"]) == (0, 2, 0)
    assert len(payload["row_norms"]) == 3 and len(payload["col_norms"]) == 3
    assert payload["row_norms"][0] == pytest.approx(0.2692307692307692, rel=1e-12)
    assert payload["col_norms"][2] == pytest.approx(0.21724137931034482, rel=1e-12)
    histogram = payload["histogram"]
    assert histogram["n_bins"] == DEFAULT_HIST_BINS
    assert len(histogram["bins"]) == DEFAULT_HIST_BINS + 1
    assert sum(histogram["counts"]) == 2
    assert payload["spectral_radius"] is None  # only computed in symmetric mode
    assert payload["density"] == pytest.approx(2 / 6)
    assert payload["reciprocity"] == 0


def test_save_data_records_symmetric_flag_normalization_and_spectral_radius(tiny_payload: dict) -> None:
    z_matrix, _ = build_z_matrix(tiny_payload, config=ZMatrixConfig(symmetric=True, normalize="rows"))
    payload = save_data_payload(z_matrix)
    assert payload["symmetric"] is True
    assert payload["normalize"] == "rows"
    assert payload["spectral_radius"] is not None
    assert payload["spectral_radius"] == pytest.approx(
        float(np.abs(np.linalg.eigvalsh(z_matrix.matrix)).max()), rel=1e-12
    )


def test_save_data_file_is_written_by_the_cli(tiny_artifact: Path, tmp_path: Path) -> None:
    outdir = tmp_path / "out"
    assert main(["-i", str(tiny_artifact), "-o", str(outdir), "--save-data"]) == 0
    data_path = outdir / "tiny_valid" / "z_matrix.data.json"
    assert data_path.is_file()
    payload = json.loads(data_path.read_text(encoding="utf-8"))
    assert len(payload["pairs"]) == 2
    assert payload["config"]["symmetric"] is False


def test_render_statistics_reports_every_requested_quantity(tiny_z) -> None:
    text = render_statistics(tiny_z)
    for fragment in (
        "min / max",
        "mean / std",
        "sparsity",
        "positive / negative",
        "row norms",
        "col norms",
        "spectral radius",
        "n/a (run with --symmetric)",
        "density / reciprocity",
    ):
        assert fragment in text


def test_render_statistics_prints_the_spectral_radius_in_symmetric_mode(tiny_payload: dict) -> None:
    z_matrix, _ = build_z_matrix(tiny_payload, config=ZMatrixConfig(symmetric=True))
    text = render_statistics(z_matrix)
    assert "n/a (run with --symmetric)" not in text
    assert "spectral radius     :" in text


def test_render_summary_box_contains_every_requested_line(tiny_z) -> None:
    paths = artifact_paths(tiny_z, Path("data/processed"))
    box = render_summary_box(tiny_z, paths, plot=False, save_data=False, stats=True)
    lines = box.splitlines()
    assert lines[0].startswith("+---") and lines[-1].startswith("+---")
    assert all(line.startswith("| ") and line.endswith(" |") for line in lines[1:-1])
    for fragment in (
        "Phase 02 complete",
        "Z matrix shape: 3x3",
        "Unified weights: OK",
        "Symmetric mode: no",
        "Normalization: none",
        "JSON written: z_matrix.json",
        "NPZ written:  z_matrix.npz",
        "Plot:        not requested (--plot)",
        "Save-data:   not requested (--save-data)",
        "Stats: min/max/mean/std printed above (--stats)",
    ):
        assert fragment in box


def test_render_summary_box_reflects_skips_and_writes(tiny_z) -> None:
    paths = artifact_paths(tiny_z, Path("data/processed"))
    box = render_summary_box(tiny_z, paths, written=False, sidecar=False, plot=True, save_data=True)
    assert "Phase 02 dry-run (nothing written)" in box
    assert "NPZ written:  skipped (--no-sidecar)" in box
    assert "JSON written: skipped (--dry-run)" in box


def test_cli_stats_prints_the_box_and_statistics(tiny_artifact: Path, tmp_path: Path, capsys) -> None:
    outdir = tmp_path / "out"
    assert main(["-i", str(tiny_artifact), "-o", str(outdir), "--stats"]) == 0
    captured = capsys.readouterr().out
    assert "Phase 02 -- Z-matrix statistics" in captured
    assert "Phase 02 complete" in captured


def test_cli_dry_run_writes_nothing_and_still_prints_the_box(
    tiny_artifact: Path, tmp_path: Path, capsys
) -> None:
    outdir = tmp_path / "out"
    assert main(["-i", str(tiny_artifact), "-o", str(outdir), "--dry-run", "--stats"]) == 0
    assert not outdir.exists()
    assert "Phase 02 dry-run (nothing written)" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# --plot (PNG + optional GUI popup)
# ---------------------------------------------------------------------------
def test_can_popup_is_false_under_the_agg_backend() -> None:
    assert can_popup() is False  # module pins Agg for the test suite


@pytest.mark.parametrize(
    "backend", ["qtagg", "QtAgg", "tkagg", "gtk3agg", "gtk3cairo", "QtCairo", "wxagg", "MacOSX"]
)
def test_can_popup_accepts_gui_backends(backend: str, monkeypatch) -> None:
    # regression guard: `"agg" in "qtagg"` used to classify the desktop backend
    # as headless and silently suppress every popup
    monkeypatch.setattr(matplotlib, "get_backend", lambda: backend)
    assert can_popup() is True


@pytest.mark.parametrize("backend", ["agg", "Agg", "pdf", "ps", "svg", "template", "pgf", "cairo"])
def test_can_popup_rejects_non_interactive_backends(backend: str, monkeypatch) -> None:
    monkeypatch.setattr(matplotlib, "get_backend", lambda: backend)
    assert can_popup() is False


def test_can_popup_falls_back_to_the_name_set_without_the_backend_registry(
    monkeypatch,
) -> None:
    """matplotlib < 3.9 has no backend registry: classify by exact backend name."""

    def _unavailable(_backend):
        raise ImportError("no backend registry in this matplotlib")

    monkeypatch.setattr(builder, "_backend_gui_framework", _unavailable)
    monkeypatch.setattr(matplotlib, "get_backend", lambda: "qtagg")
    assert can_popup() is True
    monkeypatch.setattr(matplotlib, "get_backend", lambda: "agg")
    assert can_popup() is False


def test_plot_matrix_writes_a_valid_png(tiny_z, tmp_path: Path) -> None:
    destination = tmp_path / "heatmap.png"
    plot_matrix(tiny_z.matrix, destination, cmap=DEFAULT_CMAP, popup=True)
    raw = destination.read_bytes()
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"
    width, height = struct.unpack(">II", raw[16:24])
    assert width > 100 and height > 100


def test_cli_plot_writes_the_png_and_skips_the_popup(tiny_artifact: Path, tmp_path: Path) -> None:
    outdir = tmp_path / "out"
    assert main(["-i", str(tiny_artifact), "-o", str(outdir), "--plot"]) == 0
    png = outdir / "tiny_valid" / "z_matrix.png"
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert sorted(path.name for path in (outdir / "tiny_valid").iterdir()) == [
        "z_matrix.json",
        "z_matrix.npz",
        "z_matrix.png",
    ]


def test_cli_plot_uses_the_additional_variant_filenames(tiny_artifact: Path, tmp_path: Path) -> None:
    outdir = tmp_path / "out"
    config = ZMatrixConfig(symmetric=True)
    assert main(["-i", str(tiny_artifact), "-o", str(outdir), "--symmetric", "--plot", "--save-data"]) == 0
    names = sorted(path.name for path in (outdir / "tiny_valid").iterdir())
    assert names == [
        f"z_matrix.{config.config_hash()}.data.json",
        f"z_matrix.{config.config_hash()}.json",
        f"z_matrix.{config.config_hash()}.npz",
        f"z_matrix.{config.config_hash()}.png",
    ]


# ---------------------------------------------------------------------------
# Validation paths
# ---------------------------------------------------------------------------
def test_strict_escalates_warnings(tiny_payload: dict) -> None:
    empty = json.loads(json.dumps(tiny_payload))
    empty["pairs"] = []
    _, lenient = build_z_matrix(empty)
    assert not lenient.has_errors and lenient.warnings
    _, strict = build_z_matrix(empty, strict=True)
    assert strict.has_errors
    assert builder.CODE_EMPTY_PAIRS in strict.codes()


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (lambda payload: payload["pairs"].append(dict(payload["pairs"][0], source="N2", target="N3")), builder.CODE_DUPLICATE_COORDINATE),
        (lambda payload: payload["pairs"].__setitem__(0, dict(payload["pairs"][0], target="NOPE")), builder.CODE_ENDPOINT_NOT_NEURON),
        (lambda payload: payload["pairs"].__setitem__(0, dict(payload["pairs"][0], target="N1")), builder.CODE_SELF_LOOP),
        (lambda payload: payload["pairs"].__setitem__(0, dict(payload["pairs"][0], post_z=-0.5)), builder.CODE_Z_INVALID),
        (lambda payload: payload["neurons"].__setitem__(0, dict(payload["neurons"][0], neuron_id="HUB--NAME")), builder.CODE_NEURON_ID),
    ],
)
def test_input_validation_errors(tiny_payload: dict, mutate, code: str) -> None:
    payload = json.loads(json.dumps(tiny_payload))
    mutate(payload)
    _, report = build_z_matrix(payload)
    assert code in report.codes(), report.to_dict()


@pytest.mark.parametrize("overrides", [{"eps": 0.0}, {"eps": -0.1}, {"alpha": 0.0}, {"zero_policy": "nope"}, {"normalize": "nope"}, {"symmetrize": "nope"}, {"dtype": "float32"}])
def test_config_validation_errors(tiny_payload: dict, overrides: dict) -> None:
    _, report = build_z_matrix(tiny_payload, config=ZMatrixConfig(**overrides))
    assert builder.CODE_CONFIG in report.codes()


def test_zero_z_fixture_produces_a_zero_weight_without_warnings(capsys) -> None:
    parsed, parse_report = build_parsed_graph(ZERO_Z)
    assert not parse_report.issues
    z_matrix, report = build_z_matrix(parsed.to_dict())
    assert not report.issues
    assert z_matrix.matrix[0, 1] == 0.0   # A -> B had post_z = 0.0
    assert z_matrix.matrix[1, 2] == 0.0   # B -> C had pre_z = 0.0
    assert not np.signbit(z_matrix.matrix).any()


def test_zero_z_fixture_cli_has_no_runtime_warning(tmp_path: Path) -> None:
    artifact = write_artifact(parse_graphviz_file(ZERO_Z), tmp_path / "processed")
    completed = subprocess.run(
        [
            sys.executable, "-m", "src.matrices.build_square_matrix",
            "-i", str(artifact), "-o", str(tmp_path / "out"), "--save-data", "--stats",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    combined = completed.stdout + completed.stderr
    assert "RuntimeWarning" not in combined
    assert "invalid value encountered" not in combined
    assert "divide by zero" not in combined


def test_diagnostics_conventions(tiny_z) -> None:
    diagnostics = tiny_z.diagnostics
    assert diagnostics["n_entries"] == 9
    assert diagnostics["n_stored_nonzero"] == 2
    assert diagnostics["n_matrix_zeros"] == 7
    assert diagnostics["sparsity"] == pytest.approx(7 / 9)
    # matrix stats exclude the absent-edge zeros; sparsity summarizes them
    assert diagnostics["matrix_stats"]["mean"] == pytest.approx(-0.243236074270557, rel=1e-9)
    # weight stats include every pair weight (zeros included)
    assert len(tiny_z.pairs) == 2


def test_compute_diagnostics_spectral_radius_only_in_symmetric_mode(tiny_payload: dict) -> None:
    directed, _ = build_z_matrix(tiny_payload)
    symmetric, _ = build_z_matrix(tiny_payload, config=ZMatrixConfig(symmetric=True))
    assert directed.diagnostics["spectral_radius"] is None
    assert directed.diagnostics["symmetrized_spectral_radius"] is not None
    assert symmetric.diagnostics["spectral_radius"] == pytest.approx(
        symmetric.diagnostics["symmetrized_spectral_radius"], rel=1e-12
    )


# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------
def test_resolve_inputs_is_recursive_and_skips_filtered_by_default(tiny_artifact: Path) -> None:
    filtered = tiny_artifact.with_name("parsed_graph.filtered.json")
    filtered.write_text(tiny_artifact.read_text(encoding="utf-8"), encoding="utf-8")
    root = tiny_artifact.parents[1]
    assert resolve_inputs([root]) == [tiny_artifact]
    # matches are returned in deterministic (alphabetical) order
    assert resolve_inputs([root], include_filtered=True) == [filtered, tiny_artifact]


def test_resolve_inputs_rejects_non_json_files(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("ignored", encoding="utf-8")
    assert resolve_inputs([tmp_path / "notes.txt"]) == []


def test_cli_returns_two_when_nothing_matches(tmp_path: Path) -> None:
    assert main(["-i", str(tmp_path), "-o", str(tmp_path)]) == 2


def test_cli_returns_one_for_a_non_conforming_artifact(tiny_artifact: Path, tmp_path: Path) -> None:
    broken = tmp_path / "broken.json"
    payload = json.loads(tiny_artifact.read_text(encoding="utf-8"))
    del payload["metadata"]["reciprocity"]  # Phase 01 schema violation
    broken.write_text(json.dumps(payload), encoding="utf-8")
    assert main(["-i", str(broken), "-o", str(tmp_path / "out")]) == 1


def test_cli_module_invocation(tiny_artifact: Path, tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable, "-m", "src.matrices.build_square_matrix",
            "-i", str(tiny_artifact), "-o", str(tmp_path / "out"), "--no-sidecar",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    assert "Phase 02 complete" in completed.stdout


def test_validate_z_payload_schema_accepts_a_fresh_payload(tiny_z) -> None:
    assert validate_z_payload_schema(tiny_z.to_dict()) == []


# ---------------------------------------------------------------------------
# Integration against the real dataset (deselect with -m "not slow")
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def reference_z(tmp_path_factory: pytest.TempPathFactory):
    """Build, write and reload the reference Phase 02 artifact once per module."""
    if not REFERENCE_ARTIFACT.is_file():  # pragma: no cover - dataset not checked out
        pytest.skip(f"reference artifact missing: {REFERENCE_ARTIFACT}")
    built, report = build_z_matrix_from_file(REFERENCE_ARTIFACT)
    assert not report.issues, report.to_dict()
    paths = write_artifact_set(built, tmp_path_factory.mktemp("phase02"))
    # reload through the written artifacts: this also verifies the npz cache
    return load_z_matrix(paths["json"]), built


@pytest.mark.slow
def test_reference_matrix_reference_numbers(reference_z) -> None:
    z_matrix, built = reference_z
    assert np.array_equal(z_matrix.matrix, built.matrix)
    metadata = z_matrix.metadata

    assert z_matrix.n_neurons == 113
    assert z_matrix.n_edges == 1355
    assert z_matrix.matrix.shape == (113, 113)
    assert metadata["n_stored_nonzero"] == 1350
    assert metadata["n_matrix_zeros"] == 11419
    assert metadata["sparsity"] == pytest.approx(0.894275197745, rel=1e-9)
    assert (metadata["n_weight_positive"], metadata["n_weight_negative"], metadata["n_weight_zero"]) == (302, 1048, 5)
    assert metadata["matrix_stats"]["min"] == pytest.approx(-0.563605108055, rel=1e-9)
    assert metadata["matrix_stats"]["max"] == pytest.approx(0.507692307692, rel=1e-9)
    assert metadata["matrix_stats"]["mean"] == pytest.approx(-0.096311952675, rel=1e-9)
    assert metadata["weight_stats"]["mean"] == pytest.approx(-0.095956558015220, rel=1e-9)
    assert metadata["weight_stats"]["sum"] == pytest.approx(-130.021136110623, rel=1e-9)
    assert metadata["frobenius_norm"] == pytest.approx(8.005821952321897, rel=1e-9)
    assert metadata["largest_singular_value"] == pytest.approx(4.441772842851828, rel=1e-9)
    assert metadata["symmetrized_spectral_radius"] == pytest.approx(2.433313838239482, rel=1e-9)
    assert metadata["spectral_radius"] is None
    assert metadata["row_norm_stats"]["max"] == pytest.approx(7.021678574677, rel=1e-9)
    assert metadata["col_norm_stats"]["max"] == pytest.approx(7.592547154392, rel=1e-9)
    assert metadata["density"] == pytest.approx(0.1070638432364096, rel=1e-12)
    assert metadata["reciprocity"] == 574
    assert validate_z_payload_schema(z_matrix.to_dict()) == []

    index = z_matrix.index
    assert z_matrix.matrix[index["ExR7(ring)_L_1"], index["ExR7(ring)_L_2"]] == pytest.approx(-0.102144, rel=1e-12)
    assert z_matrix.matrix_symmetric[index["ExR7(ring)_L_1"], index["ExR7(ring)_L_2"]] == pytest.approx(
        -0.09997766037735847, rel=1e-12
    )
    assert np.all(np.diag(z_matrix.matrix) == 0.0)
    # no negative zero ever reaches the artifact
    assert not np.any(np.signbit(z_matrix.matrix) & (z_matrix.matrix == 0.0))


@pytest.mark.slow
def test_reference_symmetric_variant_supports_eigen_decomposition(reference_z) -> None:
    z_matrix, _ = reference_z
    symmetric, report = build_z_matrix_from_file(
        REFERENCE_ARTIFACT, config=ZMatrixConfig(symmetric=True)
    )
    assert not report.issues
    assert np.array_equal(symmetric.matrix, symmetric.matrix_symmetric)
    assert np.array_equal(symmetric.matrix, (z_matrix.matrix + z_matrix.matrix.T) / 2)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric.matrix)
    assert np.isclose(float(np.abs(eigenvalues).max()), symmetric.metadata["spectral_radius"], rtol=1e-12)
    assert np.allclose(eigenvectors.T @ eigenvectors, np.eye(113), atol=1e-10)


# ---------------------------------------------------------------------------
# --interactive (matplotlib window + mplcursors hover tooltips)
# ---------------------------------------------------------------------------
def test_format_cell_tooltip_reports_names_weight_and_indices(tiny_z) -> None:
    row, col = tiny_z.index["N1"], tiny_z.index["N2"]
    text = format_cell_tooltip(tiny_z, row, col)
    lines = text.splitlines()
    assert lines[0] == "source: N1"
    assert lines[1] == "target: N2"
    assert lines[2] == f"unified weight: {tiny_z.matrix[row, col]:.6f}"
    assert lines[3] == f"(i, j) = ({row}, {col})"
    # default directed config: the matrix value *is* the unified weight
    assert len(lines) == 4
    assert "absent" not in text


def test_format_cell_tooltip_marks_absent_cells_and_out_of_range_indices(tiny_z) -> None:
    absent = format_cell_tooltip(tiny_z, tiny_z.index["N2"], tiny_z.index["N1"])
    assert "unified weight: 0.000000 (absent edge)" in absent
    assert "stored edge: none (absent)" in absent
    assert "source: N2" in absent and "target: N1" in absent

    # an out-of-range index degrades gracefully: this text is produced from a GUI callback
    outside = format_cell_tooltip(tiny_z, -1, 99)
    assert "source: <row -1>" in outside
    assert "target: <col 99>" in outside
    assert "(i, j) = (-1, 99)" in outside


def test_format_cell_tooltip_adds_the_raw_weight_only_when_needed(tiny_payload: dict) -> None:
    normalized, _ = build_z_matrix(tiny_payload, config=ZMatrixConfig(normalize="rows"))
    raw = {(pair.source, pair.target): pair.weight for pair in normalized.pairs}
    row, col = normalized.index["N1"], normalized.index["N2"]
    text = format_cell_tooltip(normalized, row, col)
    assert f"unified weight: {normalized.matrix[row, col]:.6f}" in text
    assert f"raw unified weight: {raw[('N1', 'N2')]:.6f}" in text

    symmetric, _ = build_z_matrix(tiny_payload, config=ZMatrixConfig(symmetric=True))
    forward = format_cell_tooltip(symmetric, symmetric.index["N1"], symmetric.index["N2"])
    assert "raw unified weight: " in forward
    reverse = format_cell_tooltip(symmetric, symmetric.index["N2"], symmetric.index["N1"])
    assert "raw unified weight (reverse direction): " in reverse
    assert "absent" not in reverse  # in symmetric mode the cell is filled from the pair


def test_format_cell_tooltip_uses_a_supplied_lookup(tiny_z) -> None:
    text = format_cell_tooltip(tiny_z, tiny_z.index["N2"], tiny_z.index["N1"], raw_weights={})
    assert "stored edge: none (absent)" in text


def test_stored_zero_weight_is_not_an_absent_edge() -> None:
    parsed, report = build_parsed_graph(ZERO_Z)
    assert not report.issues, report.to_dict()
    z_matrix, build_report = build_z_matrix(parsed.to_dict())
    assert not build_report.issues
    index = z_matrix.index
    text = format_cell_tooltip(z_matrix, index["A"], index["B"])
    assert "unified weight: 0.000000" in text
    assert "absent" not in text  # stored edge with a zero-in-zero-out weight


def test_build_interactive_figure_renders_the_effective_matrix(tiny_z) -> None:
    figure, axes, image = build_interactive_figure(tiny_z, cmap=DEFAULT_CMAP)
    try:
        assert len(figure.axes) == 2  # heatmap + colorbar
        assert axes.images and axes.images[0] is image
        assert np.array_equal(image.get_array(), tiny_z.matrix)
        assert image.cmap.name == DEFAULT_CMAP
        assert image.axes is axes
    finally:
        plt.close(figure)


def test_build_interactive_figure_follows_normalization_and_symmetric_mode(tiny_payload: dict) -> None:
    z_matrix, report = build_z_matrix(
        tiny_payload, config=ZMatrixConfig(normalize="unit", symmetric=True)
    )
    assert not report.issues
    figure, _axes, image = build_interactive_figure(z_matrix, title="custom")
    try:
        assert np.array_equal(image.get_array(), z_matrix.matrix)
        assert np.array_equal(z_matrix.matrix, z_matrix.matrix_symmetric)
        assert figure.axes[0].get_title() == "custom"
    finally:
        plt.close(figure)


def test_attach_cell_cursor_annotates_the_real_selection_path(tiny_z) -> None:
    mplcursors = pytest.importorskip("mplcursors")
    figure, axes, image = build_interactive_figure(tiny_z)
    try:
        cursor = attach_cell_cursor(image, tiny_z)
        assert isinstance(cursor, mplcursors.Cursor)
        index = tiny_z.index
        row, col = index["N1"], index["N2"]
        # select_at() dispatches through mplcursors' real image-picking machinery;
        # imshow data coordinates are (x=target index, y=source index)
        selection = cursor.select_at(axes, (col, row))
        assert selection is not None
        assert (int(selection.index[0]), int(selection.index[1])) == (row, col)
        assert selection.annotation.get_text() == format_cell_tooltip(tiny_z, row, col)
    finally:
        plt.close(figure)


def test_attach_cell_cursor_raises_a_helpful_error_without_mplcursors(tiny_z, monkeypatch) -> None:
    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "mplcursors":
            raise ModuleNotFoundError("No module named 'mplcursors'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    figure, _axes, image = build_interactive_figure(tiny_z)
    try:
        with pytest.raises(ZMatrixValidationError) as info:
            attach_cell_cursor(image, tiny_z)
        assert "mplcursors is required for --interactive" in str(info.value)
        assert "conda install -c conda-forge mplcursors" in str(info.value)
    finally:
        plt.close(figure)


def test_show_interactive_matrix_shows_only_on_an_interactive_backend(tiny_z, monkeypatch) -> None:
    shown_calls: list[int] = []
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: shown_calls.append(1))

    assert can_popup() is False  # the module pins Agg
    assert show_interactive_matrix(tiny_z) is False
    assert shown_calls == []

    monkeypatch.setattr(builder, "can_popup", lambda: True)
    assert show_interactive_matrix(tiny_z, cmap=DEFAULT_CMAP) is True
    assert shown_calls == [1]


def test_render_summary_box_hides_the_interactive_line_by_default(tiny_z) -> None:
    paths = artifact_paths(tiny_z, Path("data/processed"))
    assert "Interactive:" not in render_summary_box(tiny_z, paths)
    shown = render_summary_box(tiny_z, paths, interactive=True, interactive_shown=True)
    assert "Interactive: window shown (--interactive)" in shown
    skipped = render_summary_box(tiny_z, paths, interactive=True)
    assert "Interactive: skipped (non-interactive backend)" in skipped
    dry = render_summary_box(tiny_z, paths, written=False, interactive=True)
    assert "Interactive: skipped (--dry-run)" in dry



def test_cli_interactive_writes_the_same_artifacts_as_a_plain_run(
    tiny_artifact: Path, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("SOURCE_DATE_EPOCH", FIXED_EPOCH)
    plain, interactive = tmp_path / "plain", tmp_path / "interactive"
    assert main(["-i", str(tiny_artifact), "-o", str(plain)]) == 0
    assert main(["-i", str(tiny_artifact), "-o", str(interactive), "--interactive"]) == 0

    def _snapshot(root: Path) -> dict[str, bytes]:
        return {path.name: path.read_bytes() for path in (root / "tiny_valid").iterdir()}

    assert _snapshot(interactive) == _snapshot(plain)  # byte-identical: nothing new is written


def test_cli_interactive_keeps_the_canonical_filenames(tiny_artifact: Path, tmp_path: Path) -> None:
    outdir = tmp_path / "out"
    assert main(["-i", str(tiny_artifact), "-o", str(outdir), "--interactive"]) == 0
    assert sorted(path.name for path in (outdir / "tiny_valid").iterdir()) == [
        "z_matrix.json",
        "z_matrix.npz",
    ]


def test_cli_interactive_suppresses_the_plain_plot_popup(
    tiny_artifact: Path, tmp_path: Path, monkeypatch
) -> None:
    outdir = tmp_path / "out"
    calls: list[bool] = []

    def _spy(matrix, out_path, **kwargs):
        calls.append(kwargs["popup"])
        return plot_matrix(matrix, out_path, **kwargs)  # the PNG is still written

    monkeypatch.setattr(builder, "plot_matrix", _spy)
    assert main(["-i", str(tiny_artifact), "-o", str(outdir), "--plot", "--interactive"]) == 0
    assert calls == [False]  # one window only: the --interactive one
    assert (outdir / "tiny_valid" / "z_matrix.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

    # --plot alone keeps its existing behaviour
    calls.clear()
    assert main(["-i", str(tiny_artifact), "-o", str(outdir), "--plot"]) == 0
    assert calls == [True]


def test_cli_interactive_dry_run_opens_no_window(
    tiny_artifact: Path, tmp_path: Path, monkeypatch, capsys
) -> None:
    outdir = tmp_path / "out"
    opened: list[int] = []
    monkeypatch.setattr(builder, "show_interactive_matrix", lambda *a, **k: opened.append(1))
    assert main(["-i", str(tiny_artifact), "-o", str(outdir), "--dry-run", "--interactive"]) == 0
    assert opened == []
    assert not outdir.exists()
    captured = capsys.readouterr().out
    assert "Interactive: skipped (--dry-run)" in captured


def test_cli_interactive_reports_the_window_in_the_summary_box(
    tiny_artifact: Path, tmp_path: Path, monkeypatch, capsys
) -> None:
    outdir = tmp_path / "out"
    monkeypatch.setattr(builder, "show_interactive_matrix", lambda *a, **k: True)
    assert main(["-i", str(tiny_artifact), "-o", str(outdir), "--interactive"]) == 0
    assert "Interactive: window shown (--interactive)" in capsys.readouterr().out


def test_cli_interactive_warns_for_multiple_inputs(
    tiny_artifact: Path, tmp_path: Path, caplog
) -> None:
    second = tmp_path / "second" / "parsed_graph.json"
    second.parent.mkdir(parents=True)
    second.write_bytes(tiny_artifact.read_bytes())
    with caplog.at_level(logging.WARNING, logger="src.matrices.build_square_matrix"):
        assert main(["-i", str(tiny_artifact), str(second), "-o", str(tmp_path / "out"), "--interactive"]) == 0
    assert any(
        "--interactive resolves to 2 input(s)" in record.getMessage() for record in caplog.records
    )


def test_cli_interactive_without_mplcursors_fails_with_exit_code_one(
    tiny_artifact: Path, tmp_path: Path, monkeypatch
) -> None:
    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "mplcursors":
            raise ModuleNotFoundError("No module named 'mplcursors'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    outdir = tmp_path / "out"
    assert main(["-i", str(tiny_artifact), "-o", str(outdir), "--interactive"]) == 1
    # the artifacts themselves are unaffected by the failed window
    assert (outdir / "tiny_valid" / "z_matrix.json").is_file()


def test_interactive_is_not_part_of_the_config_hash() -> None:
    args = build_argument_parser().parse_args(["-i", "x", "--interactive"])
    assert args.interactive is True
    # presentation-only: it must never change the matrix, the hash or the filenames
    assert "interactive" not in MATRIX_CONFIG_FIELDS
    assert "interactive" not in {config_field.name for config_field in fields(ZMatrixConfig)}
    assert ZMatrixConfig().is_default()
    assert set(INTERACTIVE_ANNOTATION_KWARGS) == {"bbox", "arrowprops"}  # tooltip style only

