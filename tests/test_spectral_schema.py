"""Schema conformance tests for the canonical Phase 03 artifact.

``jsonschema`` / ``pydantic`` are not installed in this environment, so the
expected schema is asserted with a small hand-rolled helper plus targeted tests
on the single producer (:func:`validate_spectral_payload_schema`), mirroring
``tests/test_parsed_graph_schema.py`` (Phase 01) and
``tests/test_z_matrix_schema.py`` (Phase 02).
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # keep the suite headless

import pytest  # noqa: E402

from src.matrices.build_square_matrix import (  # noqa: E402
    ZMatrixConfig,
    build_z_matrix_from_file,
)
from src.parsing.parse_graphviz import build_parsed_graph, write_artifact  # noqa: E402
from src.spectral.spectral_decomposition import (  # noqa: E402
    CONFIG_KEYS,
    HEURISTICS_KEYS,
    METADATA_KEYS,
    MODE_KEYS,
    PROVENANCE_KEYS,
    SAVE_DATA_KEYS,
    TOP_LEVEL_KEYS,
    VALUE_KEYS,
    SpectralConfig,
    build_spectral_decomposition,
    save_data_payload,
    validate_spectral_payload_schema,
)
from src.utils.io import dumps_json  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"
TINY = FIXTURES / "tiny_valid.gv"

RFC3339 = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

EXPECTED_TOP_LEVEL = {
    "provenance",
    "config",
    "neuron_order",
    "values",
    "modes",
    "heuristics",
    "metadata",
}
EXPECTED_CONFIG = {
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
EXPECTED_VALUE = {
    "rank",
    "original_index",
    "value",
    "abs_value",
    "energy",
    "explained_variance_ratio",
    "cumulative_variance_ratio",
}
EXPECTED_MODE = EXPECTED_VALUE | {
    "left",
    "right",
    "left_norm",
    "right_norm",
    "sign_flipped",
    "residual",
}
EXPECTED_PROVENANCE = {
    "source_artifact",
    "source_artifact_sha256",
    "source_file",
    "source_file_sha256",
    "parsed_created_utc",
    "z_matrix_config_hash",
    "z_matrix_config",
    "phase",
}


@pytest.fixture(scope="module")
def tiny_z_matrix(tmp_path_factory):
    """A minimal, valid Phase 02 object built from the shared tiny fixture.

    Built through the real Phase 01 -> Phase 02 file path so the provenance
    digests (``source_artifact_sha256`` / ``source_file_sha256``) are populated,
    exactly as they are in a CLI run.
    """
    outdir = tmp_path_factory.mktemp("phase03_schema")
    parsed, _ = build_parsed_graph(TINY)
    write_artifact(parsed, outdir)
    artifact = outdir / TINY.stem / "parsed_graph.json"
    z_matrix, report = build_z_matrix_from_file(artifact, config=ZMatrixConfig())
    assert not report.has_errors
    return z_matrix


@pytest.fixture(scope="module")
def tiny_payload(tiny_z_matrix):
    """The Phase 02 JSON payload Phase 03 consumes."""
    return tiny_z_matrix.to_dict()


@pytest.fixture(scope="module")
def tiny_artifact(tiny_z_matrix, tiny_payload: dict):
    """A Phase 03 decomposition of the tiny matrix (auto -> SVD path)."""
    decomposition, report = build_spectral_decomposition(tiny_z_matrix, payload=tiny_payload)
    assert not report.has_errors
    return decomposition


def assert_spectral_schema(payload: dict) -> None:
    """Hand-rolled schema check for the Phase 03 JSON artifact."""
    assert set(payload) == EXPECTED_TOP_LEVEL
    assert set(TOP_LEVEL_KEYS) == EXPECTED_TOP_LEVEL

    provenance = payload["provenance"]
    assert set(provenance) == EXPECTED_PROVENANCE
    assert set(PROVENANCE_KEYS) == EXPECTED_PROVENANCE
    assert isinstance(provenance["source_artifact"], str)
    assert re.fullmatch(r"[0-9a-f]{64}", provenance["source_artifact_sha256"])
    assert isinstance(provenance["source_file"], str)
    assert re.fullmatch(r"[0-9a-f]{64}", provenance["source_file_sha256"])
    assert RFC3339.match(provenance["parsed_created_utc"])
    assert re.fullmatch(r"[0-9a-f]{8}", provenance["z_matrix_config_hash"])
    assert isinstance(provenance["z_matrix_config"], dict)
    assert provenance["phase"] == "03"

    config = payload["config"]
    assert set(config) == EXPECTED_CONFIG
    assert set(CONFIG_KEYS) == EXPECTED_CONFIG
    assert config["method"] in {"auto", "eigen", "svd"}
    assert config["resolved_method"] in {"eigen", "svd"}
    assert config["source"] in {"effective", "symmetric"}
    assert config["rank_by"] in {"magnitude", "value"}
    assert config["sign_convention"] in {"max-abs-positive", "none"}
    assert isinstance(config["elbow"], bool) and isinstance(config["spectral_gap"], bool)
    assert isinstance(config["symmetric_input"], bool)
    assert isinstance(config["k_resolved"], int) and config["k_resolved"] >= 1
    assert re.fullmatch(r"[0-9a-f]{8}", config["config_hash"])

    # a LIST, not a dict: `dumps_json` sorts object keys, which would destroy the
    # index ordering if this were an object
    order = payload["neuron_order"]
    assert isinstance(order, list) and order
    assert all(isinstance(name, str) for name in order)
    assert len(set(order)) == len(order)

    values = payload["values"]
    assert isinstance(values, list) and len(values) == len(order)
    assert [entry["rank"] for entry in values] == list(range(1, len(order) + 1))
    assert sorted(entry["original_index"] for entry in values) == list(range(len(order)))
    for entry in values:
        assert set(entry) == EXPECTED_VALUE
        assert set(VALUE_KEYS) == EXPECTED_VALUE
        assert entry["abs_value"] >= 0.0
        assert entry["energy"] >= 0.0
        assert entry["explained_variance_ratio"] >= 0.0
        assert entry["cumulative_variance_ratio"] >= entry["explained_variance_ratio"] - 1e-12

    modes = payload["modes"]
    assert isinstance(modes, list) and 1 <= len(modes) <= len(order)
    assert [entry["rank"] for entry in modes] == list(range(1, len(modes) + 1))
    for entry in modes:
        assert set(entry) == EXPECTED_MODE
        assert len(entry["left"]) == len(order)
        assert len(entry["right"]) == len(order)
        # every retained vector is unit-norm after the sign convention
        assert abs(entry["left_norm"] - 1.0) < 1e-9
        assert abs(entry["right_norm"] - 1.0) < 1e-9

    heuristics = payload["heuristics"]
    assert set(heuristics) == HEURISTICS_KEYS
    assert heuristics["recommended_k"] >= 1
    assert isinstance(heuristics["rule"], str) and heuristics["rule"]
    assert heuristics["participation_ratio"] >= 0.0
    for name in ("variance_threshold", "elbow", "spectral_gap"):
        entry = heuristics[name]
        if entry is not None:
            assert isinstance(entry["k"], int) and 1 <= entry["k"] <= len(order)

    metadata = payload["metadata"]
    assert set(metadata) == METADATA_KEYS
    assert metadata["n_neurons"] == len(order)
    assert metadata["n_modes_total"] == len(order)
    assert metadata["n_modes_retained"] == len(modes)
    assert metadata["method"] == config["resolved_method"]
    assert metadata["source"] == config["source"]
    assert metadata["generator"] == "src.spectral.spectral_decomposition"
    assert RFC3339.match(metadata["created_utc"])
    assert metadata["heuristics"] == heuristics
    assert metadata["config_hash"] == config["config_hash"]
    assert metadata["reconstruction_error"] >= 0.0
    assert metadata["orthogonality_error"] >= 0.0


def test_tiny_artifact_is_schema_conformant(tiny_artifact) -> None:
    assert_spectral_schema(tiny_artifact.to_dict())
    assert validate_spectral_payload_schema(tiny_artifact.to_dict()) == []


def test_serialized_artifact_is_schema_conformant(tiny_artifact) -> None:
    payload = json.loads(dumps_json(tiny_artifact.to_dict()))
    assert_spectral_schema(payload)
    assert validate_spectral_payload_schema(payload) == []


@pytest.mark.parametrize(
    ("mutate", "needle"),
    [
        (lambda p: p.pop("modes"), "top level: missing key(s) ['modes']"),
        (lambda p: p.update({"extra": 1}), "top level: unexpected key(s) ['extra']"),
        (lambda p: p.pop("provenance"), "top level: missing key(s) ['provenance']"),
        (
            lambda p: p["config"].pop("resolved_method"),
            "config: missing key(s) ['resolved_method']",
        ),
        (lambda p: p["config"].update({"resolved_method": "pca"}), "config.resolved_method"),
        (lambda p: p["config"].update({"rank_by": "vibes"}), "config.rank_by"),
        (lambda p: p["config"].update({"config_hash": "zz"}), "config.config_hash"),
        (
            lambda p: p["values"][0].update({"rank": 99}),
            "values: ranks are not 1..N in order",
        ),
        (lambda p: p["values"][0].pop("abs_value"), "values[0]: missing key(s) ['abs_value']"),
        (
            lambda p: p["values"][0].update({"spectral_radius": 1.0}),
            "values[0]: unexpected key(s) ['spectral_radius']",
        ),
        (lambda p: p["modes"][0].update({"left": [0.0]}), "modes[0].left: expected"),
        (lambda p: p["modes"][0].pop("left_norm"), "modes[0]: missing key(s) ['left_norm']"),
        (
            lambda p: p["heuristics"].update({"participation_ratio": "high"}),
            "heuristics.participation_ratio",
        ),
        (
            lambda p: p["heuristics"].update({"variance_threshold": {"threshold": 0.9}}),
            "heuristics.variance_threshold",
        ),
        (lambda p: p["metadata"].pop("n_neurons"), "metadata: missing key(s) ['n_neurons']"),
        (lambda p: p["provenance"].update({"extra": 1}), "provenance: unexpected key(s) ['extra']"),
    ],
)
def test_schema_negative_cases(tiny_artifact, mutate, needle: str) -> None:
    payload = copy.deepcopy(tiny_artifact.to_dict())
    mutate(payload)
    problems = validate_spectral_payload_schema(payload)
    assert any(needle in problem for problem in problems), problems


def test_non_object_payload() -> None:
    assert validate_spectral_payload_schema(["not", "an", "object"]) == ["payload is not an object"]


def test_key_sets_are_exact() -> None:
    assert set(VALUE_KEYS) == EXPECTED_VALUE
    assert set(MODE_KEYS) == EXPECTED_MODE
    assert set(CONFIG_KEYS) == EXPECTED_CONFIG
    assert set(PROVENANCE_KEYS) == EXPECTED_PROVENANCE
    assert set(TOP_LEVEL_KEYS) == EXPECTED_TOP_LEVEL
    assert set(HEURISTICS_KEYS) == {
        "variance_threshold",
        "elbow",
        "spectral_gap",
        "participation_ratio",
        "recommended_k",
        "rule",
    }
    assert set(SpectralConfig().to_dict()) == EXPECTED_CONFIG


def test_save_data_keys_are_exact(tiny_artifact) -> None:
    payload = save_data_payload(tiny_artifact)
    assert set(payload) == set(SAVE_DATA_KEYS)
    assert payload["n_modes_total"] == tiny_artifact.n_neurons
    assert payload["n_modes_retained"] == tiny_artifact.k
    assert len(payload["values"]) == tiny_artifact.n_neurons
    assert len(payload["modes"]) == tiny_artifact.k
    assert len(payload["energy_share"]) == tiny_artifact.k
    assert sum(value["energy_share"] for value in payload["modes"]) == pytest.approx(1.0, abs=1e-9)
    # per-mode detail the canonical JSON deliberately does not carry
    for entry in payload["modes"]:
        assert set(entry["left_stats"]) == {"min", "max", "mean", "std"}
        assert set(entry["right_stats"]) == {"min", "max", "mean", "std"}
        assert 0 <= entry["left_argmax"] < tiny_artifact.n_neurons
        assert 0 <= entry["right_argmax"] < tiny_artifact.n_neurons


def test_no_csv_anywhere_in_the_schema(tiny_artifact) -> None:
    text = dumps_json(tiny_artifact.to_dict()) + dumps_json(save_data_payload(tiny_artifact))
    assert "csv" not in text.lower()


def test_neuron_order_is_a_list_not_an_object(tiny_artifact) -> None:
    payload = tiny_artifact.to_dict()
    assert isinstance(payload["neuron_order"], list)
    assert tiny_artifact.index == {
        name: position for position, name in enumerate(tiny_artifact.neuron_order)
    }


