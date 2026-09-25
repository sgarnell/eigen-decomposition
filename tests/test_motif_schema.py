"""Schema tests for the frozen Phase 04 payload (``motifs.json``).

These pin the **complete** key set declared in sub-phase 04A -- including the link,
family, group, pathway, node, edge and top-mode shapes owned by 04B-04D -- so a later
sub-phase can populate the neutral values without ever touching the constants or the
validator.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from src.clustering.functional_motifs import (
    CONFIG_KEYS,
    EDGE_KEYS,
    FAMILY_KEYS,
    GRAPHML_EDGE_KEYS,
    GRAPHML_NODE_KEYS,
    GROUP_KEYS,
    LINK_CLASSES,
    LINK_KEYS,
    MEMBER_KEYS,
    METADATA_KEYS,
    MOTIF_KEYS,
    NEUTRAL_PATHWAY,
    NODE_KEYS,
    OCCURRENCE_KEYS,
    PATHWAY_KEYS,
    PROVENANCE_KEYS,
    STAGES,
    TOP_LEVEL_KEYS,
    TOP_MODE_KEYS,
    build_motif_analysis,
    load_motifs,
    load_sidecar_arrays,
    validate_motif_payload_schema,
    write_artifact_set,
)
from test_functional_motifs import make_spectrum


@pytest.fixture(scope="module")
def payload() -> dict:
    analysis, report = build_motif_analysis(make_spectrum())
    assert not report.has_errors
    return analysis.to_dict()


# ---------------------------------------------------------------------------
# The frozen key sets
# ---------------------------------------------------------------------------
def test_key_sets_are_exactly_the_frozen_ones() -> None:
    assert TOP_LEVEL_KEYS == {
        "provenance",
        "config",
        "neuron_order",
        "motifs",
        "links",
        "families",
        "groups",
        "pathway",
        "metadata",
    }
    assert len(PROVENANCE_KEYS) == 11
    assert len(MOTIF_KEYS) == 22
    assert len(MEMBER_KEYS) == 10
    assert len(CONFIG_KEYS) == 30  # 26 hashed + k_resolved + n_motifs + stage + config_hash
    assert set(LINK_KEYS) == {
        "source_mode",
        "target_mode",
        "source_label",
        "target_label",
        "shared_members",
        "n_shared",
        "jaccard",
        "polarity_agreement",
        "classification",
    }
    assert OCCURRENCE_KEYS == {"mode", "participation", "signed_participation"}
    assert FAMILY_KEYS == {
        "family_id",
        "label",
        "modes",
        "n_modes",
        "size",
        "n_members",
        "members",
        "polarity",
        "occurrences",
    }
    assert "is_background" in GROUP_KEYS and "coherence" in GROUP_KEYS
    assert "weight_concentration" in PATHWAY_KEYS
    assert STAGES == ("04A", "04B", "04C", "04D")


def test_valid_payload_conforms(payload: dict) -> None:
    assert validate_motif_payload_schema(payload) == []
    assert set(payload) == set(TOP_LEVEL_KEYS)
    assert set(payload["provenance"]) == set(PROVENANCE_KEYS)
    assert set(payload["config"]) == set(CONFIG_KEYS)
    assert set(payload["metadata"]) == set(METADATA_KEYS)
    assert set(payload["pathway"]) == set(PATHWAY_KEYS)
    for motif in payload["motifs"]:
        assert set(motif) == set(MOTIF_KEYS)
        for member in motif["members"]:
            assert set(member) == set(MEMBER_KEYS)


def test_populated_blocks_are_accepted(payload: dict) -> None:
    # 04B populates tracking, 04C the default group block, 04D the pathway block.
    assert isinstance(payload["links"], list) and payload["links"]
    assert isinstance(payload["groups"], list) and payload["groups"]
    assert payload["pathway"] != NEUTRAL_PATHWAY
    assert payload["pathway"]["n_nodes"] == len(payload["groups"])
    assert payload["pathway"]["n_edges"] == len(payload["pathway"]["edges"])
    assert payload["families"]  # a partition of all modes
    metadata = payload["metadata"]
    for key in (
        "n_links",
        "link_classes",
        "n_families",
        "n_multi_mode_families",
    ):
        assert metadata[key] is not None, key
    assert metadata["n_links"] == len(payload["links"])
    assert metadata["n_families"] == len(payload["families"])
    assert sum(metadata["link_classes"].values()) == metadata["n_links"]
    for key in (
        "n_groups",
        "group_sizes",
        "n_background_groups",
        "n_singleton_groups",
        "largest_group",
        "smallest_group",
        "mean_group_size",
        "n_polarity_split_groups",
        "n_pathway_nodes",
        "n_pathway_edges",
        "pathway_abs_max",
        "pathway_positive_edges",
        "pathway_negative_edges",
        "pathway_intra_edges",
        "pathway_cross_edges",
        "pathway_weight_concentration",
    ):
        assert metadata[key] is not None, key
    # the neutral-value convention: the type set includes the neutral value
    assert isinstance(metadata["n_groups"], (int, type(None)))
    assert isinstance(metadata["n_members_total"], int)
    assert metadata["n_pathway_nodes"] == len(payload["pathway"]["nodes"])
    assert (
        metadata["pathway_positive_edges"] + metadata["pathway_negative_edges"]
        == metadata["n_pathway_edges"]
    )
    assert (
        metadata["pathway_intra_edges"] + metadata["pathway_cross_edges"]
        == metadata["n_pathway_edges"]
    )


def test_link_entries_conform(payload: dict) -> None:
    assert payload["links"]
    for link in payload["links"]:
        assert set(link) == set(LINK_KEYS)
        assert link["classification"] in LINK_CLASSES
        assert 0.0 <= link["jaccard"] <= 1.0
        assert -1.0 <= link["polarity_agreement"] <= 1.0
        assert link["n_shared"] == len(link["shared_members"])
        assert link["source_mode"] < link["target_mode"]


def test_family_entries_conform_and_partition_the_modes(payload: dict) -> None:
    k = len(payload["motifs"])
    covered: list[int] = []
    for position, family in enumerate(payload["families"]):
        assert set(family) == set(FAMILY_KEYS)
        assert family["family_id"] == position + 1
        assert family["label"] == f"F{family['family_id']:02d}"
        assert family["n_modes"] == len(family["modes"]) == family["size"]
        assert family["n_members"] == len(family["members"])
        for occurrence in family["occurrences"]:
            assert occurrence["neuron_id"] in family["members"]
            for item in occurrence["occurrences"]:
                assert set(item) == set(OCCURRENCE_KEYS)
        covered.extend(family["modes"])
    assert sorted(covered) == list(range(1, k + 1))
    assert len(covered) == len(set(covered))


def test_metadata_link_block_types(payload: dict) -> None:
    metadata = payload["metadata"]
    assert isinstance(metadata["n_links"], int)
    assert isinstance(metadata["n_families"], int)
    assert isinstance(metadata["n_multi_mode_families"], int)
    assert set(metadata["link_classes"]) == set(LINK_CLASSES)
    assert sum(metadata["link_classes"].values()) == metadata["n_links"]
    assert metadata["n_links"] == len(payload["links"])
    assert metadata["n_families"] == len(payload["families"])


def test_npz_carries_the_tracking_arrays(payload: dict, tmp_path: Path) -> None:
    analysis, _ = build_motif_analysis(make_spectrum())
    written = write_artifact_set(analysis, tmp_path)
    arrays = load_sidecar_arrays(written["npz"])
    k = len(payload["motifs"])
    assert arrays["family_of_mode"].shape == (k,)
    assert arrays["link_jaccard"].shape == (k, k)
    assert arrays["link_polarity"].shape == (k, k)
    # the arrays are consistent with the JSON links/families
    assert arrays["family_of_mode"].tolist() == analysis.family_of_mode.tolist()
    assert load_motifs(written["json"]).to_dict() == analysis.to_dict()


def test_non_object_payload_is_rejected() -> None:
    assert validate_motif_payload_schema([]) == ["payload is not an object"]  # type: ignore[arg-type]


def test_json_and_npz_preserve_the_schema(payload: dict, tmp_path: Path) -> None:
    analysis, _ = build_motif_analysis(make_spectrum())
    written = write_artifact_set(analysis, tmp_path)
    reloaded = json.loads(written["json"].read_text(encoding="utf-8"))
    assert validate_motif_payload_schema(reloaded) == []
    assert load_motifs(written["json"]).to_dict() == payload


# ---------------------------------------------------------------------------
# 04D -- the pathway block, its npz arrays and the GraphML artifact
# ---------------------------------------------------------------------------
def test_pathway_entries_conform(payload: dict) -> None:
    pathway = payload["pathway"]
    assert set(pathway) == set(PATHWAY_KEYS)
    assert pathway["n_nodes"] == len(pathway["nodes"])
    assert pathway["n_edges"] == len(pathway["edges"]) == len(pathway["weights"])
    assert pathway["source"] in {"mode", "matrix", "both"}
    assert pathway["abs_max"] is None or pathway["abs_max"] > 0
    node_ids = [node["node_id"] for node in pathway["nodes"]]
    assert len(set(node_ids)) == len(node_ids)
    assert pathway["n_nodes"] == len(payload["groups"])
    for node in pathway["nodes"]:
        assert set(node) == set(NODE_KEYS)
        assert node["size"] == len(node["members"])
        for entry in node["top_modes"]:
            assert set(entry) == set(TOP_MODE_KEYS)
            assert 0.0 <= entry["share"] <= 1.0
            assert entry["abs_weight"] == abs(entry["signed_weight"])
    seen = set()
    for edge in pathway["edges"]:
        assert set(edge) == set(EDGE_KEYS)
        assert edge["source"] in node_ids and edge["target"] in node_ids
        assert (edge["source"], edge["target"]) not in seen
        seen.add((edge["source"], edge["target"]))
        assert edge["polarity"] in (-1, 0, 1)
        assert edge["abs_weight"] == abs(edge["weight"])
        assert edge["n_modes"] == len(edge["modes"])
        assert edge["is_intra"] == (edge["source"] == edge["target"])
        assert all(
            abs(edge["weight"]) >= pathway["edge_threshold"] * pathway["abs_max"] - 1e-12
            for _ in (0,)
        )
        for entry in edge["top_modes"]:
            assert set(entry) == set(TOP_MODE_KEYS)
    for edge in pathway["edges"]:
        if edge["is_intra"]:
            assert pathway["intra"] is True


def test_npz_carries_the_pathway_arrays(payload: dict, tmp_path: Path) -> None:
    analysis, _ = build_motif_analysis(make_spectrum())
    written = write_artifact_set(analysis, tmp_path)
    arrays = load_sidecar_arrays(written["npz"])
    assert len(arrays) == 24
    pathway = payload["pathway"]
    node_ids = [node["node_id"] for node in pathway["nodes"]]
    assert arrays["pathway_adjacency"].shape == (len(node_ids), len(node_ids))
    assert arrays["pathway_adjacency"].dtype == np.float64
    assert int(np.count_nonzero(arrays["pathway_adjacency"])) == pathway["n_edges"]
    assert np.allclose(arrays["pathway_adjacency_abs"], np.abs(arrays["pathway_adjacency"]))
    assert arrays["mode_outer_products"].shape == (
        len(payload["motifs"]),
        len(payload["neuron_order"]),
        len(payload["neuron_order"]),
    )
    assert str(arrays["stage"][0]) == "04D"
    assert load_motifs(written["json"]).to_dict() == payload


def test_graphml_matches_the_payload(payload: dict, tmp_path: Path) -> None:
    nx = pytest.importorskip("networkx")
    analysis, _ = build_motif_analysis(make_spectrum())
    written = write_artifact_set(analysis, tmp_path)
    graph = nx.read_graphml(written["graphml"])
    pathway = payload["pathway"]
    assert graph.number_of_nodes() == pathway["n_nodes"]
    assert graph.number_of_edges() == pathway["n_edges"]
    assert graph.graph["n_nodes"] == pathway["n_nodes"]
    assert graph.graph["n_edges"] == pathway["n_edges"]
    assert graph.graph["weight_normalization"] == "abs-max"
    assert graph.graph["pathway_source"] == pathway["source"]
    assert graph.graph["pathway_weight_rule"] == pathway["weight"]
    assert graph.graph["threshold"] == pathway["edge_threshold"]
    assert graph.graph["topN"] == pathway["top_edges"]
    for node in pathway["nodes"]:
        attributes = graph.nodes[node["node_id"]]
        assert set(attributes) <= set(GRAPHML_NODE_KEYS)
        assert attributes["group_id"] == node["group_id"]
        assert attributes["size"] == node["size"]
        assert json.loads(attributes["region_composition"]) == node["region_composition"]
    for edge in pathway["edges"]:
        attributes = graph.edges[edge["source"], edge["target"]]
        assert set(attributes) <= set(GRAPHML_EDGE_KEYS)
        assert attributes["weight"] == edge["weight"]
        assert attributes["polarity"] == edge["polarity"]
        assert attributes["intra_flag"] == edge["is_intra"]
        assert json.loads(attributes["contribution_by_mode"]) == edge["top_modes"]


# ---------------------------------------------------------------------------
# Negative cases -- every level of the frozen schema
# ---------------------------------------------------------------------------
def _drop_family_key(payload: dict) -> None:
    payload["families"][0].pop("occurrences")


def _family_modes_not_ints(payload: dict) -> None:
    payload["families"][0]["modes"] = ["1"]


def _family_occurrence_bad_keys(payload: dict) -> None:
    payload["families"][0]["occurrences"][0]["occurrences"][0].pop("participation")


def _group_entry_missing_key(payload: dict) -> None:
    payload["groups"] = [{"group_id": "G01", "label": "G01", "members": [], "n_members": 0}]


def _link_entry_missing_key(payload: dict) -> None:
    payload["links"] = [{"source_mode": 1, "target_mode": 2}]


def _node_top_modes_bad_keys(payload: dict) -> None:
    payload["pathway"]["nodes"] = [
        {
            "node_id": "N1",
            "group_id": "G01",
            "label": "G01",
            "size": 1,
            "members": [],
            "cent_mean": None,
            "region_composition": {},
            "dominant_mode": None,
            "top_modes": [{"mode": 1}],
        }
    ]
    payload["pathway"]["n_nodes"] = 1


def _edge_missing_key(payload: dict) -> None:
    payload["pathway"]["edges"] = [{"source": "G01", "target": "G02", "weight": -0.5}]
    payload["pathway"]["n_edges"] = 1


NEGATIVE_CASES = [
    ("non-object", lambda p: None, "payload is not an object", True),
    ("top-level missing motif block", lambda p: p.pop("motifs"), "top level: missing", False),
    ("top-level extra key", lambda p: p.__setitem__("extra", 1), "top level: unexpected", False),
    ("provenance missing stage", lambda p: p["provenance"].pop("stage"), "provenance: missing", False),
    ("provenance extra key", lambda p: p["provenance"].__setitem__("x", 1), "provenance: unexpected", False),
    ("provenance wrong phase", lambda p: p["provenance"].__setitem__("phase", "03"), "provenance.phase", False),
    ("provenance invalid stage", lambda p: p["provenance"].__setitem__("stage", "04E"), "provenance.stage", False),
    ("config missing field", lambda p: p["config"].pop("participation"), "config: missing", False),
    ("config extra field", lambda p: p["config"].__setitem__("mystery", 1), "config: unexpected", False),
    ("config hash length", lambda p: p["config"].__setitem__("config_hash", "abc"), "config.config_hash", False),
    ("config invalid participation", lambda p: p["config"].__setitem__("participation", "bogus"), "config.participation", False),
    ("config invalid method", lambda p: p["config"].__setitem__("threshold_method", "bogus"), "config.threshold_method", False),
    ("duplicate neuron ids", lambda p: p["neuron_order"].__setitem__(1, "n0"), "duplicate", False),
    ("neuron_order not a list", lambda p: p.__setitem__("neuron_order", {"n0": 0}), "neuron_order", False),
    ("empty motifs", lambda p: p.__setitem__("motifs", []), "motifs: expected", False),
    ("motif missing key", lambda p: p["motifs"][0].pop("share"), "motifs[0]: missing", False),
    ("motif extra key", lambda p: p["motifs"][0].__setitem__("mystery", 1), "motifs[0]: unexpected", False),
    ("motif member count mismatch", lambda p: p["motifs"][0].__setitem__("n_members", 99), "n_members", False),
    ("member missing key", lambda p: p["motifs"][0]["members"][0].pop("region"), "members[0]: missing", False),
    ("member invalid polarity", lambda p: p["motifs"][0]["members"][0].__setitem__("polarity", 0), "polarity", False),
    ("member invalid axis", lambda p: p["motifs"][0]["members"][0].__setitem__("dominant_axis", "up"), "dominant_axis", False),
    ("sender_members not a list", lambda p: p["motifs"][0].__setitem__("sender_members", None), "sender_members", False),
    ("links not a list", lambda p: p.__setitem__("links", None), "links: expected", False),
    ("link entry missing key", _link_entry_missing_key, "links[0]: missing", False),
    ("families not a list", lambda p: p.__setitem__("families", None), "families: expected", False),
    ("family entry missing key", _drop_family_key, "families[0]: missing", False),
    ("family modes not ints", _family_modes_not_ints, "families[0].modes", False),
    ("occurrence bad keys", _family_occurrence_bad_keys, "occurrences[0].occurrences", False),
    ("groups not a list", lambda p: p.__setitem__("groups", None), "groups: expected", False),
    ("group entry missing key", _group_entry_missing_key, "groups[0]: missing", False),
    ("pathway not an object", lambda p: p.__setitem__("pathway", []), "pathway: not an object", False),
    ("pathway missing weights", lambda p: p["pathway"].pop("weights"), "pathway: missing", False),
    ("pathway node bad top_modes", _node_top_modes_bad_keys, "top_modes[0]: missing", False),
    ("pathway node extra key", lambda p: p["pathway"]["nodes"][0].__setitem__("mystery", 1), "pathway.nodes[0]: unexpected", False),
    ("pathway edge missing key", _edge_missing_key, "pathway.edges[0]: missing", False),
    ("pathway edge extra key", lambda p: p["pathway"]["edges"][0].__setitem__("mystery", 1), "pathway.edges[0]: unexpected", False),
    ("metadata missing key", lambda p: p["metadata"].pop("n_groups"), "metadata: missing", False),
    ("metadata extra key", lambda p: p["metadata"].__setitem__("mystery", 1), "metadata: unexpected", False),
    ("metadata neuron mismatch", lambda p: p["metadata"].__setitem__("n_neurons", 99), "does not match", False),
]


@pytest.mark.parametrize(
    "label,mutator,fragment,is_not_object",
    NEGATIVE_CASES,
    ids=[case[0] for case in NEGATIVE_CASES],
)
def test_schema_negative_cases(payload, label, mutator, fragment, is_not_object) -> None:
    if is_not_object:
        problems = validate_motif_payload_schema([])  # type: ignore[arg-type]
    else:
        target = copy.deepcopy(payload)
        mutator(target)
        problems = validate_motif_payload_schema(target)
    assert any(fragment in problem for problem in problems), (label, problems)
