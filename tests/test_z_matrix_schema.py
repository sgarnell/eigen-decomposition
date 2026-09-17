"""Schema conformance tests for the canonical Phase 02 artifact.

``jsonschema`` / ``pydantic`` are not installed in this environment, so the
expected schema is asserted with a small hand-rolled helper plus targeted tests on
the single producer (:func:`validate_z_payload_schema`), mirroring
``tests/test_parsed_graph_schema.py`` from Phase 01.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # keep the suite headless (no GUI popup from --plot)

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from src.matrices.build_square_matrix import (  # noqa: E402
    CONFIG_KEYS,
    METADATA_KEYS,
    PROVENANCE_KEYS,
    SAVE_DATA_KEYS,
    STATS_KEYS,
    TOP_LEVEL_KEYS,
    WEIGHT_STATS_KEYS,
    WEIGHTED_PAIR_KEYS,
    ZMatrixConfig,
    build_z_matrix_from_file,
    build_z_matrix,
    main,
    save_data_payload,
    validate_z_payload_schema,
    write_artifact_set,
)
from src.parsing.parse_graphviz import build_parsed_graph, parse_graphviz_file, write_artifact  # noqa: E402
from src.utils.io import dumps_json  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
TINY = FIXTURES / "tiny_valid.gv"
REFERENCE_GV = PROJECT_ROOT / "data" / "raw_dot" / "FB4Yaffect_FB45_999prePost_001_all.gv"
REFERENCE_ARTIFACT = PROJECT_ROOT / "data" / "processed" / REFERENCE_GV.stem / "parsed_graph.json"

RFC3339 = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

EXPECTED_TOP_LEVEL = {"provenance", "config", "neuron_order", "pairs", "matrix", "matrix_symmetric", "metadata"}
EXPECTED_PAIR_KEYS = {"source", "target", "pre_z", "post_z", "weight"}
NORMALIZATIONS = {"none", "rows", "cols", "unit", "spectral", "zscore-nonzero"}


def assert_z_schema(payload: dict) -> None:
    """Hand-rolled schema check for the Phase 02 JSON artifact."""
    assert set(payload) == EXPECTED_TOP_LEVEL
    assert set(TOP_LEVEL_KEYS) == EXPECTED_TOP_LEVEL

    provenance = payload["provenance"]
    assert set(provenance) == set(PROVENANCE_KEYS)
    assert isinstance(provenance["source_artifact"], str)
    assert re.fullmatch(r"[0-9a-f]{64}", provenance["source_artifact_sha256"])
    assert isinstance(provenance["source_file"], str)
    assert re.fullmatch(r"[0-9a-f]{64}", provenance["source_file_sha256"])
    assert RFC3339.match(provenance["parsed_created_utc"])
    assert provenance["phase"] == "02"

    config = payload["config"]
    assert set(config) == set(CONFIG_KEYS)
    assert isinstance(config["symmetric"], bool)
    assert isinstance(config["eps"], float) and isinstance(config["alpha"], float)
    assert config["normalize"] in NORMALIZATIONS
    assert config["symmetrize"] in {"mean", "sum", "max-abs", "min-abs"}
    assert config["dtype"] == "float64"
    assert isinstance(config["filters"], list)
    assert re.fullmatch(r"[0-9a-f]{8}", config["config_hash"])
    assert "tanh" in config["unification"] and "pre_z" in config["unification"]

    # a LIST, not a dict: `dumps_json` sorts object keys, which would destroy the
    # index ordering if this were an object
    order = payload["neuron_order"]
    assert isinstance(order, list) and order
    assert all(isinstance(name, str) for name in order)
    assert len(set(order)) == len(order)

    pairs = payload["pairs"]
    assert isinstance(pairs, list) and pairs
    for pair in pairs:
        assert set(pair) == EXPECTED_PAIR_KEYS
        assert set(pair) == set(WEIGHTED_PAIR_KEYS)
        assert pair["source"] in order and pair["target"] in order
        assert pair["source"] != pair["target"]
        assert all(isinstance(pair[key], float) for key in ("pre_z", "post_z", "weight"))
    assert pairs == sorted(pairs, key=lambda item: (item["source"], item["target"]))

    size = len(order)
    for key in ("matrix", "matrix_symmetric"):
        matrix = payload[key]
        assert isinstance(matrix, list) and len(matrix) == size
        for row in matrix:
            assert isinstance(row, list) and len(row) == size
            assert all(isinstance(value, float) for value in row)
        assert all(matrix[index][index] == 0.0 for index in range(size))
    symmetric = payload["matrix_symmetric"]
    assert all(
        symmetric[row][column] == symmetric[column][row]
        for row in range(size)
        for column in range(size)
    )

    metadata = payload["metadata"]
    assert set(metadata) == set(METADATA_KEYS)
    for key in (
        "n_neurons",
        "n_edges",
        "n_stored_nonzero",
        "n_matrix_zeros",
        "n_weight_positive",
        "n_weight_negative",
        "n_weight_zero",
        "n_self_loops",
        "reciprocity",
    ):
        assert isinstance(metadata[key], int) and not isinstance(metadata[key], bool)
    for key in ("sparsity", "density", "frobenius_norm", "largest_singular_value"):
        assert isinstance(metadata[key], float)
    assert metadata["normalization_scale"] is None or isinstance(metadata["normalization_scale"], float)
    assert metadata["spectral_radius"] is None or isinstance(metadata["spectral_radius"], float)
    assert metadata["symmetrized_spectral_radius"] is None or isinstance(
        metadata["symmetrized_spectral_radius"], float
    )
    assert re.fullmatch(r"[0-9a-f]{8}", metadata["config_hash"])
    assert isinstance(metadata["numpy_version"], str)
    assert isinstance(metadata["generator"], str)
    assert RFC3339.match(metadata["created_utc"])
    for key in ("matrix_stats", "row_norm_stats", "col_norm_stats"):
        assert set(metadata[key]) == set(STATS_KEYS)
        for stat in STATS_KEYS:
            assert metadata[key][stat] is None or isinstance(metadata[key][stat], float)
    assert set(metadata["weight_stats"]) == set(WEIGHT_STATS_KEYS)
    assert metadata["n_edges"] == len(pairs)
    assert metadata["n_neurons"] == size


@pytest.fixture(scope="module")
def phase_one_artifact(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A real Phase 01 artifact, so provenance carries a true SHA-256."""
    return write_artifact(parse_graphviz_file(TINY), tmp_path_factory.mktemp("processed"))


@pytest.fixture(scope="module")
def payload(phase_one_artifact: Path) -> dict:
    z_matrix, report = build_z_matrix_from_file(phase_one_artifact)
    assert not report.issues, report.to_dict()
    return z_matrix.to_dict()


@pytest.fixture(scope="module")
def symmetric_payload(phase_one_artifact: Path) -> dict:
    z_matrix, report = build_z_matrix_from_file(
        phase_one_artifact, config=ZMatrixConfig(symmetric=True)
    )
    assert not report.issues, report.to_dict()
    return z_matrix.to_dict()


def test_payload_matches_schema(payload: dict) -> None:
    assert_z_schema(payload)


def test_symmetric_payload_matches_schema(symmetric_payload: dict) -> None:
    assert_z_schema(symmetric_payload)
    assert symmetric_payload["config"]["symmetric"] is True
    assert symmetric_payload["matrix"] == symmetric_payload["matrix_symmetric"]


def test_schema_helper_accepts_the_payload(payload: dict) -> None:
    assert validate_z_payload_schema(payload) == []


def test_metadata_keys_are_exact(payload: dict) -> None:
    assert set(payload["metadata"]) == set(METADATA_KEYS)


def test_neuron_order_matches_the_phase_one_order(payload: dict) -> None:
    phase_one, _ = build_parsed_graph(TINY)
    assert payload["neuron_order"] == [neuron["neuron_id"] for neuron in phase_one.to_dict()["neurons"]]


def test_phase_one_pairs_are_not_touched(payload: dict) -> None:
    """Phase 02 must not add a `weight` key to the Phase 01 artifact."""
    parsed, _ = build_parsed_graph(TINY)
    for pair in parsed.to_dict()["pairs"]:
        assert set(pair) == {"source", "target", "pre_z", "post_z"}
        assert "weight" not in pair


def test_matrix_and_symmetric_matrix_relationship(payload: dict) -> None:
    matrix = np.array(payload["matrix"])
    other = np.array(payload["matrix_symmetric"])
    assert matrix.shape == other.shape == (3, 3)
    assert np.array_equal(other, other.T)
    assert np.array_equal(other, (matrix + matrix.T) / 2)


def test_save_data_keys_are_exact() -> None:
    z_matrix, _ = build_z_matrix(build_parsed_graph(TINY)[0].to_dict())
    data = save_data_payload(z_matrix)
    assert set(data) == set(SAVE_DATA_KEYS)
    assert data["symmetric"] is False
    assert data["normalize"] == "none"
    assert set(data["histogram"]) >= {"bins", "counts", "n_bins", "range", "zero_count", "degenerate"}
    assert len(data["row_norms"]) == len(data["col_norms"]) == len(z_matrix.neuron_order)
    assert data["density"] == pytest.approx(2 / 6)
    assert data["reciprocity"] == 0


def test_hash_filename_segment_for_a_variant_config(tmp_path: Path) -> None:
    artifact = write_artifact(parse_graphviz_file(TINY), tmp_path / "processed")
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    config = ZMatrixConfig(symmetric=True)
    z_matrix, report = build_z_matrix(payload, source_artifact=artifact, config=config)
    assert not report.issues
    write_artifact_set(z_matrix, tmp_path / "processed")
    written = sorted(path.name for path in artifact.parent.iterdir())
    assert written == [
        "parsed_graph.json",
        f"z_matrix.{config.config_hash()}.json",
        f"z_matrix.{config.config_hash()}.npz",
    ]


def test_force_config_hash_segment_on_the_canonical_config(tmp_path: Path) -> None:
    artifact = write_artifact(parse_graphviz_file(TINY), tmp_path / "processed")
    outdir = tmp_path / "out"
    assert main(["-i", str(artifact), "-o", str(outdir), "--config-hash"]) == 0
    default_hash = ZMatrixConfig().config_hash()
    assert sorted(path.name for path in (outdir / "tiny_valid").iterdir()) == [
        f"z_matrix.{default_hash}.json",
        f"z_matrix.{default_hash}.npz",
    ]


def test_no_csv_anywhere_in_the_output_tree(tmp_path: Path) -> None:
    artifact = write_artifact(parse_graphviz_file(TINY), tmp_path / "processed")
    outdir = tmp_path / "out"
    assert main(["-i", str(artifact), "-o", str(outdir), "--save-data", "--plot"]) == 0
    assert not list(tmp_path.rglob("*.csv"))


def test_sort_keys_round_trip_is_stable(payload: dict) -> None:
    text = dumps_json(payload)
    assert json.loads(text) == payload
    assert dumps_json(json.loads(text)) == text
    assert text.endswith("\n")
    assert text.index('"config"') < text.index('"metadata"')  # keys are sorted
    assert '"config_hash"' in text


def test_created_utc_parses_as_rfc3339(payload: dict) -> None:
    created = payload["metadata"]["created_utc"]
    assert RFC3339.match(created)
    moment = datetime.strptime(created, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    assert moment.tzinfo is timezone.utc


# ---------------------------------------------------------------------------
# Negative cases: the schema helper must catch what Phases 03+ rely on
# ---------------------------------------------------------------------------
def _tampered(payload: dict) -> dict:
    return json.loads(dumps_json(payload))


def test_schema_helper_rejects_extra_top_level_key(payload: dict) -> None:
    tampered = _tampered(payload)
    tampered["extra"] = 1
    assert validate_z_payload_schema(tampered)


def test_schema_helper_rejects_missing_metadata_key(payload: dict) -> None:
    tampered = _tampered(payload)
    del tampered["metadata"]["sparsity"]
    problems = validate_z_payload_schema(tampered)
    assert any("sparsity" in problem for problem in problems)


def test_schema_helper_rejects_a_pair_without_a_weight(payload: dict) -> None:
    tampered = _tampered(payload)
    del tampered["pairs"][0]["weight"]
    problems = validate_z_payload_schema(tampered)
    assert any("weight" in problem for problem in problems)


def test_schema_helper_rejects_a_dict_neuron_index(payload: dict) -> None:
    tampered = _tampered(payload)
    tampered["neuron_order"] = {"N1": 0, "N2": 1, "N3": 2}
    problems = validate_z_payload_schema(tampered)
    assert any("neuron_order" in problem for problem in problems)


def test_schema_helper_rejects_a_pair_endpoint_outside_the_order(payload: dict) -> None:
    tampered = _tampered(payload)
    tampered["pairs"][0]["target"] = "NOPE"
    problems = validate_z_payload_schema(tampered)
    assert any("not in neuron_order" in problem for problem in problems)


def test_schema_helper_rejects_a_wrongly_shaped_matrix(payload: dict) -> None:
    tampered = _tampered(payload)
    tampered["matrix"] = [[0.0, 1.0], [1.0, 0.0]]
    assert validate_z_payload_schema(tampered)


def test_schema_helper_rejects_a_non_numeric_matrix_entry(payload: dict) -> None:
    tampered = _tampered(payload)
    tampered["matrix"][0][1] = "not-a-number"
    problems = validate_z_payload_schema(tampered)
    assert any("non-numeric" in problem for problem in problems)


def test_schema_helper_rejects_a_bad_config_hash(payload: dict) -> None:
    tampered = _tampered(payload)
    tampered["config"]["config_hash"] = "nothex"
    problems = validate_z_payload_schema(tampered)
    assert any("config_hash" in problem for problem in problems)


def test_schema_helper_rejects_a_non_boolean_symmetric_flag(payload: dict) -> None:
    tampered = _tampered(payload)
    tampered["config"]["symmetric"] = "yes"
    problems = validate_z_payload_schema(tampered)
    assert any("symmetric" in problem for problem in problems)


def test_schema_helper_rejects_an_unknown_normalization(payload: dict) -> None:
    tampered = _tampered(payload)
    tampered["config"]["normalize"] = "wishful"
    problems = validate_z_payload_schema(tampered)
    assert any("normalize" in problem for problem in problems)


@pytest.mark.slow
def test_reference_payload_matches_schema() -> None:
    if not REFERENCE_ARTIFACT.is_file():  # pragma: no cover - dataset not checked out
        pytest.skip(f"reference artifact missing: {REFERENCE_ARTIFACT}")

    z_matrix, report = build_z_matrix_from_file(REFERENCE_ARTIFACT)
    assert not report.issues, report.to_dict()
    payload = z_matrix.to_dict()
    assert_z_schema(payload)
    assert len(payload["neuron_order"]) == 113
    assert len(payload["pairs"]) == 1355
    assert payload["metadata"]["n_neurons"] == len(payload["neuron_order"])
    assert payload["metadata"]["density"] == pytest.approx(0.1070638432364096, rel=1e-12)
