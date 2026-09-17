"""Schema conformance tests for the canonical Phase 01 artifact.

``jsonschema`` / ``pydantic`` are not installed in this environment, so the
expected schema is asserted with a small hand-rolled helper plus targeted tests
on the one producer (:func:`validate_payload_schema`).
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.parsing.parse_graphviz import (
    METADATA_KEYS,
    NEURON_KEYS,
    PAIR_KEYS,
    RAW_EDGE_KEYS,
    STATS_KEYS,
    TOP_LEVEL_KEYS,
    build_parsed_graph,
    validate_payload_schema,
    write_artifact,
)
from src.utils.io import dumps_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
TINY = FIXTURES / "tiny_valid.gv"
REFERENCE = PROJECT_ROOT / "data" / "raw_dot" / "FB4Yaffect_FB45_999prePost_001_all.gv"

RFC3339 = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

#: The six documented top-level keys of ``parsed_graph.json``.
EXPECTED_TOP_LEVEL = {"source_file", "file_sha256", "neurons", "pairs", "raw_edges", "metadata"}


def assert_schema(payload: dict) -> None:
    """Hand-rolled schema check for the Phase 01 JSON artifact."""
    assert set(payload) == EXPECTED_TOP_LEVEL
    assert set(TOP_LEVEL_KEYS) == EXPECTED_TOP_LEVEL

    assert isinstance(payload["source_file"], str)
    assert re.fullmatch(r"[0-9a-f]{64}", payload["file_sha256"])

    assert isinstance(payload["neurons"], list) and payload["neurons"]
    for neuron in payload["neurons"]:
        assert set(neuron) == set(NEURON_KEYS)
        assert isinstance(neuron["neuron_id"], str)
        assert neuron["cent"] is None or isinstance(neuron["cent"], float)
        assert isinstance(neuron["out_degree"], int) and not isinstance(neuron["out_degree"], bool)
        assert isinstance(neuron["in_degree"], int)
        assert isinstance(neuron["pre_strength"], float)
        assert isinstance(neuron["post_strength"], float)

    assert isinstance(payload["pairs"], list) and payload["pairs"]
    for pair in payload["pairs"]:
        assert set(pair) == {"source", "target", "pre_z", "post_z"}
        assert set(pair) == set(PAIR_KEYS)
        assert "weight" not in pair  # Phase 02 owns the unified weight
        assert isinstance(pair["source"], str) and isinstance(pair["target"], str)
        assert isinstance(pair["pre_z"], float) and isinstance(pair["post_z"], float)

    assert isinstance(payload["raw_edges"], list) and payload["raw_edges"]
    for edge in payload["raw_edges"]:
        assert set(edge) == set(RAW_EDGE_KEYS)
        assert isinstance(edge["src"], str) and isinstance(edge["dst"], str)
        assert isinstance(edge["is_hub"], bool)
        assert edge["z_score"] is None or isinstance(edge["z_score"], float)
        assert edge["penwidth"] is None or isinstance(edge["penwidth"], float)
        assert edge["color"] is None or isinstance(edge["color"], str)

    metadata = payload["metadata"]
    assert set(metadata) == set(METADATA_KEYS)
    assert all(
        isinstance(metadata[key], int) and not isinstance(metadata[key], bool)
        for key in ("n_neurons", "n_hubs", "n_raw_edges", "n_pairs", "reciprocity")
    )
    assert isinstance(metadata["density"], float)
    for key in ("pre_stats", "post_stats"):
        assert set(metadata[key]) == set(STATS_KEYS)
        for stat in STATS_KEYS:
            assert metadata[key][stat] is None or isinstance(metadata[key][stat], float)
    assert isinstance(metadata["cent_distribution"], dict)
    assert all(isinstance(key, str) for key in metadata["cent_distribution"])
    assert isinstance(metadata["filters"], list)
    assert isinstance(metadata["parser"], str)
    assert isinstance(metadata["parser_version"], str)
    assert RFC3339.match(metadata["created_utc"])


@pytest.fixture(scope="module")
def payload() -> dict:
    parsed, report = build_parsed_graph(TINY)
    assert not report.issues, report.to_dict()
    return parsed.to_dict()


def test_payload_matches_schema(payload: dict) -> None:
    assert_schema(payload)


def test_schema_helper_accepts_the_payload(payload: dict) -> None:
    assert validate_payload_schema(payload) == []


def test_top_level_keys_are_exact(payload: dict) -> None:
    assert set(payload) == EXPECTED_TOP_LEVEL


def test_pair_keys_contain_no_weight(payload: dict) -> None:
    assert all(set(pair) == {"source", "target", "pre_z", "post_z"} for pair in payload["pairs"])


def test_metadata_keys_are_exact(payload: dict) -> None:
    assert set(payload["metadata"]) == {
        "n_neurons",
        "n_hubs",
        "n_raw_edges",
        "n_pairs",
        "density",
        "reciprocity",
        "pre_stats",
        "post_stats",
        "cent_distribution",
        "filters",
        "parser",
        "parser_version",
        "created_utc",
    }


def test_cent_distribution_keys_are_strings(payload: dict) -> None:
    for key in payload["metadata"]["cent_distribution"]:
        assert isinstance(key, str)


def test_created_utc_parses_as_rfc3339(payload: dict) -> None:
    created = payload["metadata"]["created_utc"]
    assert RFC3339.match(created)
    parsed = datetime.strptime(created, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    assert parsed.tzinfo is timezone.utc


def test_sort_keys_round_trip_is_stable(payload: dict) -> None:
    text = dumps_json(payload)
    reloaded = json.loads(text)
    assert dumps_json(reloaded) == text
    assert text.endswith("\n")
    assert text.index('"file_sha256"') < text.index('"metadata"')  # keys are sorted


# ---------------------------------------------------------------------------
# Negative cases: the schema helper must catch the things Phase 02 depends on
# ---------------------------------------------------------------------------
def test_schema_helper_rejects_a_weight_key_in_pairs(payload: dict) -> None:
    tampered = json.loads(dumps_json(payload))
    tampered["pairs"][0]["weight"] = 0.42
    problems = validate_payload_schema(tampered)
    assert any("weight" in problem for problem in problems)


def test_schema_helper_rejects_extra_top_level_key(payload: dict) -> None:
    tampered = json.loads(dumps_json(payload))
    tampered["extra"] = 1
    assert validate_payload_schema(tampered)


def test_schema_helper_rejects_missing_metadata_key(payload: dict) -> None:
    tampered = json.loads(dumps_json(payload))
    del tampered["metadata"]["reciprocity"]
    problems = validate_payload_schema(tampered)
    assert any("reciprocity" in problem for problem in problems)


def test_schema_helper_rejects_bad_timestamp(payload: dict) -> None:
    tampered = json.loads(dumps_json(payload))
    tampered["metadata"]["created_utc"] = "2026-09-15 23:00:00"
    problems = validate_payload_schema(tampered)
    assert any("created_utc" in problem for problem in problems)


def test_schema_helper_rejects_non_string_cent_keys(payload: dict) -> None:
    tampered = json.loads(dumps_json(payload))
    tampered["metadata"]["cent_distribution"] = {"0.1": 3}
    assert validate_payload_schema(tampered) == []
    tampered["metadata"]["cent_distribution"] = {"0.1": "three"}
    assert validate_payload_schema(tampered)


def test_schema_helper_rejects_invalid_sha256(payload: dict) -> None:
    tampered = json.loads(dumps_json(payload))
    tampered["file_sha256"] = "not-a-digest"
    assert validate_payload_schema(tampered)


# ---------------------------------------------------------------------------
# The artifact is the Phase 01 -> Phase 02 boundary: it must round-trip
# ---------------------------------------------------------------------------
def test_artifact_round_trips_without_post_processing(tmp_path: Path) -> None:
    parsed = build_parsed_graph(TINY)[0]
    artifact = write_artifact(parsed, tmp_path)
    assert artifact.name == "parsed_graph.json"
    assert sorted(path.name for path in artifact.parent.iterdir()) == ["parsed_graph.json"]

    reloaded = json.loads(artifact.read_text(encoding="utf-8"))
    assert_schema(reloaded)

    # Phase 02 inputs are directly usable: no CSV, no extra transforms.
    pairs = [(pair["source"], pair["target"], pair["pre_z"], pair["post_z"]) for pair in reloaded["pairs"]]
    assert pairs == [("N1", "N2", 0.8, 0.6), ("N2", "N3", 0.4, 0.2)]
    assert [neuron["neuron_id"] for neuron in reloaded["neurons"]] == ["N1", "N2", "N3"]
    assert not list(tmp_path.rglob("*.csv"))


@pytest.mark.slow
def test_reference_payload_matches_schema() -> None:
    if not REFERENCE.is_file():  # pragma: no cover - dataset not checked out
        pytest.skip(f"reference dataset missing: {REFERENCE}")
    parsed, report = build_parsed_graph(REFERENCE)
    assert not report.issues, report.to_dict()
    payload = parsed.to_dict()
    assert_schema(payload)
    assert len(payload["pairs"]) == 1355
    assert len(payload["neurons"]) == 113
    assert len(payload["raw_edges"]) == 2710
    assert payload["metadata"]["n_neurons"] == len(payload["neurons"])