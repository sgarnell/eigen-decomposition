"""Unit tests for the Phase 04 participation matrix, motif extraction and 04B tracking.

The synthetic ``8 x 3`` spectrum in this module makes every threshold and member
count hand-computable; the integration tests against the real 113 x 21 artifact are
marked ``slow`` and can be deselected with ``-m "not slow"``.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.clustering.functional_motifs import (
    CANONICAL_STEM,
    CODE_CONFIG,
    CODE_MATRIX_MISSING,
    CODE_MODE_WEIGHTS,
    CODE_PATHWAY_SKIPPED,
    DEFAULT_MAX_MEMBERS,
    DEFAULT_MIN_MEMBERS,
    DEFAULT_PARTICIPATION,
    DEFAULT_PATHWAY_WEIGHT,
    DEFAULT_RELATIVE_THRESHOLD,
    DEFAULT_THRESHOLD_METHOD,
    EDGE_KEYS,
    GRAPHML_EDGE_KEYS,
    GRAPHML_GRAPH_KEYS,
    GRAPHML_NODE_KEYS,
    LINK_CLASSES,
    MOTIF_CONFIG_FIELDS,
    NEURON_PARTICIPATION_KEYS,
    NODE_KEYS,
    PATHWAY_KEYS,
    PATHWAY_TOP_MODES,
    SAVE_DATA_KEYS,
    STAGE,
    TOP_MODE_KEYS,
    MotifConfig,
    MotifValidationError,
    PathwayDiagram,
    artifact_paths,
    assign_groups,
    build_argument_parser,
    build_motif_analysis,
    build_pathway,
    filter_pathway_edges,
    graphml_edge_metadata,
    graphml_node_metadata,
    group_coherence,
    group_centroid,
    group_pair_contributions,
    loadings_clustering,
    initial_group_labels,
    build_sidecar_arrays,
    classify_link,
    compute_participation,
    derive_region,
    extract_motif,
    extract_motifs,
    family_members,
    initial_families,
    input_variant,
    load_anatomy,
    load_motifs,
    load_node_metadata,
    load_sidecar_arrays,
    main,
    merge_families,
    mode_outer_product,
    mode_weights,
    motif_links,
    motif_membership,
    motif_statistics,
    motif_thresholds,
    pathway_matrix_path,
    region_composition,
    split_large_groups,
    recurrence_table,
    render_statistics,
    render_summary_box,
    resolve_inputs,
    save_data_payload,
    sidecar_path,
    validate_motif_payload_schema,
    variant_stem,
    write_artifact_set,
    write_graphml_pathway,
    write_npz_atomic,
)
from src.spectral.spectral_decomposition import (
    DEFAULT_DEGENERATE_TOL,
    METADATA_KEYS as SPECTRAL_METADATA_KEYS,
    SpectralConfig,
    SpectralDecomposition,
    SpectralMode,
    SpectralValue,
    load_spectrum,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_JSON = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "FB4Yaffect_FB45_999prePost_001_all"
    / "eigen.json"
)

NEURON_IDS = ["n0", "n1", "n2", "n3", "n4", "n5", "n6", "n7"]

#: Hand-designed so that the rms participation and every threshold are exact.
LEFT = np.array(
    [
        [0.80, 0.00, 0.00],
        [0.40, 0.10, 0.00],
        [0.20, 0.20, 0.10],
        [0.00, 0.40, 0.00],
        [0.00, 0.00, 0.50],
        [0.10, 0.00, 0.00],
        [0.00, 0.00, 0.00],
        [0.05, 0.05, 0.05],
    ],
    dtype=np.float64,
)
RIGHT = np.array(
    [
        [0.60, 0.00, 0.00],
        [0.50, -0.30, 0.00],
        [-0.10, 0.30, -0.20],
        [0.00, 0.20, 0.00],
        [0.00, 0.00, -0.30],
        [0.30, 0.00, -0.50],
        [0.00, 0.00, 0.00],
        [-0.05, 0.05, 0.05],
    ],
    dtype=np.float64,
)
#: The exact rms participation matrix for the fixture above.
RMS = np.sqrt((LEFT * LEFT + RIGHT * RIGHT) / 2.0)
SYNTHETIC_VALUES = np.array([3.0, 2.0, 1.0, 0.5, 0.4, 0.3, 0.2, 0.1])


def make_spectrum(
    *,
    neuron_ids: list[str] | None = None,
    left: np.ndarray | None = None,
    right: np.ndarray | None = None,
    values: np.ndarray | None = None,
    source_file: str = "synthetic.gv",
) -> SpectralDecomposition:
    """Build a conforming in-memory Phase 03 spectrum for the 8 x 3 fixture."""
    names = list(neuron_ids or NEURON_IDS)
    matrix = np.asarray(LEFT if left is None else left, dtype=np.float64)
    other = np.asarray(RIGHT if right is None else right, dtype=np.float64)
    spectrum_values = np.asarray(SYNTHETIC_VALUES if values is None else values, dtype=np.float64)
    n_neurons = len(names)
    n_modes = int(matrix.shape[1])
    energy = spectrum_values**2
    total = float(energy.sum())
    ratio = energy / total if total else np.zeros_like(energy)
    cumulative = np.cumsum(ratio)

    ranked = [
        SpectralValue(
            rank=index + 1,
            original_index=index,
            value=float(spectrum_values[index]),
            abs_value=float(abs(spectrum_values[index])),
            energy=float(energy[index]),
            explained_variance_ratio=float(ratio[index]),
            cumulative_variance_ratio=float(cumulative[index]),
        )
        for index in range(n_neurons)
    ]
    modes = [
        SpectralMode.from_value(
            ranked[index],
            left=tuple(float(entry) for entry in matrix[:, index]),
            right=tuple(float(entry) for entry in other[:, index]),
            left_norm=float(np.linalg.norm(matrix[:, index])),
            right_norm=float(np.linalg.norm(other[:, index])),
            sign_flipped=False,
            residual=0.0,
        )
        for index in range(n_modes)
    ]
    heuristics = {
        "variance_threshold": {"k": n_modes, "threshold": 0.9},
        "elbow": {"k": n_modes, "curve": "cumulative", "method": "l-method"},
        "spectral_gap": {"k": 1, "window": 25, "metric": "ratio", "value": 1.5},
        "participation_ratio": float(n_modes),
        "recommended_k": int(n_modes),
        "rule": "median of the enabled heuristic k values",
    }
    metadata: dict[str, object] = {
        "n_neurons": n_neurons,
        "n_modes_total": n_neurons,
        "n_modes_retained": n_modes,
        "method": "svd",
        "source": "effective",
        "matrix_is_symmetric": False,
        "n_positive": int(np.count_nonzero(matrix > 0)),
        "n_negative": int(np.count_nonzero(matrix < 0)),
        "n_zero": int(np.count_nonzero(matrix == 0)),
        "value_min": float(spectrum_values.min()),
        "value_max": float(spectrum_values.max()),
        "abs_value_max": float(np.abs(spectrum_values).max()),
        "smallest_value": float(np.abs(spectrum_values).min()),
        "sum_values": float(spectrum_values.sum()),
        "sum_squares": float(total),
        "frobenius_norm": float(np.sqrt(total)),
        "trace": 0.0,
        "numerical_rank": n_neurons,
        "condition_number": 1.0,
        "explained_variance_top1": float(ratio[0]),
        "explained_variance_topk": float(ratio[:n_modes].sum()),
        "cumulative_at_k": float(cumulative[n_modes - 1]),
        "participation_ratio": float(n_modes),
        "reconstruction_error": 0.0,
        "reconstruction_error_abs": 0.0,
        "orthogonality_error": 1e-16,
        "max_mode_residual": 1e-16,
        "degenerate_groups": {"count": 0, "sizes": [], "tolerance": DEFAULT_DEGENERATE_TOL},
        "n_sign_flipped": 0,
        "heuristics": heuristics,
        "config_hash": "00000000",
        "scipy_version": "1.17.1",
        "numpy_version": np.__version__,
        "generator": "src.spectral.spectral_decomposition",
        "created_utc": "2026-09-17T00:00:00Z",
    }
    assert set(metadata) == set(SPECTRAL_METADATA_KEYS)
    return SpectralDecomposition(
        source_artifact=Path("data/processed/synthetic/z_matrix.json"),
        source_artifact_sha256="0" * 64,
        source_file=source_file,
        source_file_sha256="1" * 64,
        parsed_created_utc="2026-09-16T00:00:00Z",
        z_matrix_config_hash="1aa02137",
        z_matrix_config={"config_hash": "1aa02137"},
        config=SpectralConfig(resolved_method="svd", k_resolved=n_modes),
        neuron_order=names,
        values=ranked,
        modes=modes,
        heuristics=heuristics,
        metadata=metadata,
        diagnostics=dict(metadata),
    )


@pytest.fixture
def synthetic_spectrum() -> SpectralDecomposition:
    return make_spectrum()


@pytest.fixture
def synthetic_workspace(tmp_path: Path, synthetic_spectrum: SpectralDecomposition) -> Path:
    """Persist the synthetic spectrum as ``eigen.json`` and return its directory."""
    from src.utils.io import write_json_atomic

    directory = tmp_path / "synthetic.gv"
    directory.mkdir(parents=True, exist_ok=True)
    write_json_atomic(directory / "eigen.json", synthetic_spectrum.to_dict())
    return tmp_path


def _participation(spectrum: SpectralDecomposition, variant: str = DEFAULT_PARTICIPATION) -> np.ndarray:
    return compute_participation(spectrum.loadings_left, spectrum.loadings_right, variant)


def _membership(matrix: np.ndarray, method: str, **kwargs: object) -> np.ndarray:
    thresholds = motif_thresholds(matrix, method, **kwargs)  # type: ignore[arg-type]
    return motif_membership(matrix, thresholds, min_members=0, max_members=0)


# ---------------------------------------------------------------------------
# Participation
# ---------------------------------------------------------------------------
def test_participation_variants_match_their_definition(synthetic_spectrum) -> None:
    left = synthetic_spectrum.loadings_left
    right = synthetic_spectrum.loadings_right
    assert np.allclose(compute_participation(left, right, "left"), np.abs(left))
    assert np.allclose(compute_participation(left, right, "right"), np.abs(right))
    assert np.allclose(compute_participation(left, right, "max"), np.maximum(np.abs(left), np.abs(right)))
    assert np.allclose(
        compute_participation(left, right, "rms"), np.sqrt((left * left + right * right) / 2.0)
    )


def test_participation_default_is_rms(synthetic_spectrum) -> None:
    assert DEFAULT_PARTICIPATION == "rms"
    assert np.allclose(_participation(synthetic_spectrum), RMS)


def test_participation_is_finite_non_negative_float64(synthetic_spectrum) -> None:
    matrix = _participation(synthetic_spectrum)
    assert matrix.dtype == np.float64
    assert matrix.shape == (len(NEURON_IDS), 3)
    assert np.all(np.isfinite(matrix))
    assert np.all(matrix >= 0.0)


def test_participation_rejects_mismatched_shapes() -> None:
    with pytest.raises(MotifValidationError):
        compute_participation(np.zeros((3, 2)), np.zeros((4, 2)))


def test_participation_rejects_unknown_variant(synthetic_spectrum) -> None:
    with pytest.raises(MotifValidationError):
        compute_participation(
            synthetic_spectrum.loadings_left, synthetic_spectrum.loadings_right, "bogus"
        )


def test_rms_sign_selection_uses_the_dominant_axis(synthetic_spectrum) -> None:
    analysis, report = build_motif_analysis(synthetic_spectrum, config=MotifConfig(min_members=0))
    assert not report.has_errors
    motif = analysis.motifs[0]
    by_id = {member.neuron_id: member for member in motif.members}
    # |L| > |R| -> sender axis, polarity from L
    assert by_id["n0"].dominant_axis == "left" and by_id["n0"].polarity == 1
    # |R| > |L| -> receiver axis, polarity from R
    assert by_id["n1"].dominant_axis == "right" and by_id["n1"].polarity == 1
    # L == 0 -> fallback to R (mode 3, n5: L = 0, R = -0.5)
    motif2 = analysis.motifs[2]
    member = {entry.neuron_id: entry for entry in motif2.members}["n5"]
    assert member.dominant_axis == "right" and member.polarity == -1
    assert member.signed_participation < 0.0


def test_rms_sign_selection_defaults_to_positive_when_both_axes_are_zero(synthetic_spectrum) -> None:
    membership = np.zeros((len(NEURON_IDS), 3), dtype=bool)
    membership[6, 0] = True  # n6 has L = R = 0
    motif = extract_motif(
        _participation(synthetic_spectrum),
        synthetic_spectrum.loadings_left,
        synthetic_spectrum.loadings_right,
        0,
        threshold=0.0,
        threshold_method="relative",
        membership=membership,
        neuron_ids=NEURON_IDS,
        regions=[derive_region(name) for name in NEURON_IDS],
        value=1.0,
        abs_value=1.0,
        explained_variance_ratio=0.5,
        cumulative_variance_ratio=0.5,
    )
    assert motif.n_members == 1
    assert motif.members[0].polarity == 1
    assert motif.members[0].dominant_axis == "left"


def test_zero_row_is_an_isolated_neuron(synthetic_spectrum) -> None:
    analysis, _ = build_motif_analysis(synthetic_spectrum)
    assert analysis.metadata["n_isolated_neurons"] == 1
    assert all("n6" not in motif.sender_members + motif.receiver_members for motif in analysis.motifs)


# ---------------------------------------------------------------------------
# Thresholds and membership
# ---------------------------------------------------------------------------
def test_relative_threshold_is_a_fraction_of_the_mode_maximum() -> None:
    thresholds = motif_thresholds(RMS, "relative", relative_threshold=0.25)
    assert np.allclose(thresholds, 0.25 * RMS.max(axis=0))
    assert _membership(RMS, "relative").sum(axis=0).tolist() == [3, 3, 3]


def test_absolute_threshold_is_constant(synthetic_spectrum) -> None:
    thresholds = motif_thresholds(RMS, "absolute", absolute_threshold=0.05)
    assert np.allclose(thresholds, 0.05)
    assert _membership(RMS, "absolute", absolute_threshold=0.05).sum(axis=0).tolist() == [5, 4, 4]


def test_quantile_threshold_matches_numpy() -> None:
    thresholds = motif_thresholds(RMS, "quantile", quantile=0.8)
    expected = [float(np.quantile(RMS[:, index], 0.8)) for index in range(RMS.shape[1])]
    assert np.allclose(thresholds, expected)
    assert np.allclose(thresholds, [0.36110357, 0.24241320, 0.27537767], atol=1e-6)
    assert _membership(RMS, "quantile", quantile=0.8).sum(axis=0).tolist() == [2, 2, 2]


def test_participation_method_uses_the_global_threshold() -> None:
    thresholds = motif_thresholds(RMS, "participation", participation_threshold=0.1)
    assert np.allclose(thresholds, 0.1)
    assert _membership(RMS, "participation", participation_threshold=0.1).sum(axis=0).tolist() == [4, 3, 3]
    assert DEFAULT_THRESHOLD_METHOD == "relative"
    assert DEFAULT_RELATIVE_THRESHOLD == 0.25


def test_all_zero_mode_column_gets_an_infinite_threshold() -> None:
    matrix = np.array([[0.0, 0.5], [0.0, 0.25]], dtype=np.float64)
    assert np.isinf(motif_thresholds(matrix, "relative")[0])
    assert np.isinf(motif_thresholds(matrix, "quantile", quantile=0.8)[0])
    assert (
        motif_membership(
            matrix, motif_thresholds(matrix, "relative"), min_members=0, max_members=0
        )
        .sum(axis=0)
        .tolist()
        == [0, 2]
    )


def test_unknown_threshold_method_raises() -> None:
    with pytest.raises(MotifValidationError):
        motif_thresholds(RMS, "bogus")


def test_min_members_tops_a_motif_up() -> None:
    thresholds = motif_thresholds(RMS, "relative")
    assert motif_membership(RMS, thresholds, min_members=1, max_members=0).sum(axis=0).tolist() == [3, 3, 3]
    assert motif_membership(RMS, thresholds, min_members=4, max_members=0).sum(axis=0).tolist() == [4, 4, 4]


def test_max_members_truncates_a_motif() -> None:
    thresholds = motif_thresholds(RMS, "relative")
    assert motif_membership(RMS, thresholds, min_members=0, max_members=2).sum(axis=0).tolist() == [2, 2, 2]
    # the strongest member always survives the truncation
    membership = motif_membership(RMS, thresholds, min_members=0, max_members=1)
    assert membership[0, 0] and membership[3, 1] and membership[4, 2]
    assert DEFAULT_MIN_MEMBERS == 3 and DEFAULT_MAX_MEMBERS == 0


def test_membership_rejects_a_wrong_threshold_length() -> None:
    with pytest.raises(MotifValidationError):
        motif_membership(RMS, np.zeros(5))


# ---------------------------------------------------------------------------
# Members, strengths and identities
# ---------------------------------------------------------------------------
def test_member_ordering_is_descending_participation(synthetic_spectrum) -> None:
    analysis, _ = build_motif_analysis(synthetic_spectrum, config=MotifConfig(min_members=0))
    for motif in analysis.motifs:
        values = [member.participation for member in motif.members]
        assert values == sorted(values, reverse=True)
        assert [member.rank for member in motif.members] == list(range(1, motif.n_members + 1))


def test_strength_share_and_polarity_balance_identities(synthetic_spectrum) -> None:
    analysis, _ = build_motif_analysis(synthetic_spectrum, config=MotifConfig(min_members=0))
    total = sum(motif.strength_energy for motif in analysis.motifs)
    assert total > 0.0
    assert np.isclose(sum(motif.share for motif in analysis.motifs), 1.0)
    for motif in analysis.motifs:
        l1 = sum(member.participation for member in motif.members)
        energy = sum(member.participation**2 for member in motif.members)
        assert np.isclose(motif.strength_l1, l1)
        assert np.isclose(motif.strength_energy, energy)
        assert np.isclose(motif.share, energy / total)
        assert motif.n_positive + motif.n_negative == motif.n_members
        assert 0 <= motif.n_mixed <= motif.n_members
        expected = (motif.n_positive - motif.n_negative) / motif.n_members
        assert np.isclose(motif.polarity_balance, expected)
        assert all(member.polarity == np.sign(member.signed_participation) for member in motif.members)


def test_sender_and_receiver_members_partition_the_members(synthetic_spectrum) -> None:
    analysis, _ = build_motif_analysis(synthetic_spectrum)
    for motif in analysis.motifs:
        assert set(motif.sender_members).isdisjoint(motif.receiver_members)
        assert set(motif.sender_members) | set(motif.receiver_members) == {
            member.neuron_id for member in motif.members
        }


def test_region_composition_and_dominant_region(synthetic_spectrum) -> None:
    analysis, _ = build_motif_analysis(synthetic_spectrum)
    motif = analysis.motifs[0]
    assert sum(motif.region_composition.values()) == motif.n_members
    assert motif.dominant_region in motif.region_composition


# ---------------------------------------------------------------------------
# Regions and anatomy
@pytest.mark.parametrize(
    "neuron_id,expected",
    [
        ("hDeltaA_12_C10_1", "hDeltaA"),
        ("hDeltaI_03_C3_2", "hDeltaI"),
        ("FB4P_a_R_1", "FB4P_a"),
        ("DNa03_R_1", "DNa03"),
        ("ExR7(ring)_L_2", "ExR7(ring)"),
        ("MBON09(y3B'1)(AVM17)_L_1", "MBON09(y3B'1)(AVM17)"),
        ("plainname", "plainname"),
    ],
)
def test_derive_region_strips_trailing_tokens(neuron_id: str, expected: str) -> None:
    assert derive_region(neuron_id) == expected


def test_anatomy_override_and_mismatch_warning(synthetic_spectrum, tmp_path: Path) -> None:
    anatomy_path = tmp_path / "anatomy.json"
    anatomy_path.write_text(
        json.dumps({"n0": "custom_region", "n6": "custom_region"}), encoding="utf-8"
    )
    anatomy = load_anatomy(anatomy_path)
    analysis, report = build_motif_analysis(synthetic_spectrum, anatomy=anatomy)
    assert analysis.metadata["n_anatomy_matches"] == 0
    assert analysis.metadata["n_anatomy_conflicts"] == 2
    assert analysis.metadata["n_anatomy_missing"] == 6
    assert "custom_region" in analysis.metadata["regions"]
    codes = report.codes()
    assert "anatomy_mismatch" in codes
    assert any(member.region == "custom_region" for member in analysis.motifs[0].members)


def test_anatomy_accepts_record_lists_and_parsed_graph_shape(synthetic_spectrum, tmp_path: Path) -> None:
    records = tmp_path / "anatomy_records.json"
    records.write_text(
        json.dumps({"neurons": [{"neuron_id": "n0", "region": "n0", "cent": 0.2}]}),
        encoding="utf-8",
    )
    anatomy = load_anatomy(records)
    assert anatomy["n0"]["region"] == "n0"
    analysis, report = build_motif_analysis(synthetic_spectrum, anatomy=anatomy)
    assert analysis.metadata["n_anatomy_matches"] == 1
    assert not report.has_errors


def test_load_anatomy_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(MotifValidationError):
        load_anatomy(tmp_path / "nope.json")


def test_parsed_graph_enrichment(synthetic_spectrum, tmp_path: Path) -> None:
    parsed = tmp_path / "parsed_graph.json"
    parsed.write_text(
        json.dumps(
            {
                "neurons": [
                    {"neuron_id": name, "cent": 0.2, "out_degree": 3, "in_degree": 4}
                    for name in NEURON_IDS
                ]
            }
        ),
        encoding="utf-8",
    )
    metadata = load_node_metadata(parsed)
    assert metadata["n0"] == {"cent": 0.2, "out_degree": 3, "in_degree": 4}
    analysis, _ = build_motif_analysis(synthetic_spectrum, node_metadata=metadata)
    payload = save_data_payload(analysis)
    assert payload["neurons"][0]["cent"] == 0.2
    assert payload["neurons"][0]["out_degree"] == 3
    assert payload["neurons"][0]["in_degree"] == 4


def test_save_data_payload_keys(synthetic_spectrum) -> None:
    analysis, _ = build_motif_analysis(synthetic_spectrum)
    payload = save_data_payload(analysis)
    assert set(payload) == set(SAVE_DATA_KEYS)
    assert len(payload["neurons"]) == len(NEURON_IDS)
    assert set(payload["neurons"][0]) == set(NEURON_PARTICIPATION_KEYS) | {"group_id"}
    assert payload["neurons"][0]["participation"] == list(analysis.participation[0])


# ---------------------------------------------------------------------------
# Families
# ---------------------------------------------------------------------------
def test_initial_families_partition_every_mode(synthetic_spectrum) -> None:
    analysis, _ = build_motif_analysis(synthetic_spectrum)
    families = initial_families(analysis.motifs)
    assert len(families) == analysis.n_modes
    covered = [mode for family in families for mode in family.modes]
    assert sorted(covered) == list(range(1, analysis.n_modes + 1))
    assert all(family.n_modes == 1 and family.size == 1 for family in families)
    assert families[0].label == "F01"
    assert set(families[0].members) == {member.neuron_id for member in analysis.motifs[0].members}
    # 04B: family_of_mode is derived from the merged families, so it covers exactly
    # the family indices (the identity when every mode is a singleton).
    assert set(analysis.family_of_mode.tolist()) == set(range(len(analysis.families)))
    for position, family in enumerate(analysis.families):
        for mode in family["modes"]:
            assert analysis.family_of_mode[mode - 1] == position


def test_families_are_serialized_with_occurrences(synthetic_spectrum) -> None:
    analysis, _ = build_motif_analysis(synthetic_spectrum)
    assert analysis.families
    entry = analysis.families[0]
    assert entry["family_id"] == 1 and entry["n_modes"] == len(entry["modes"])
    assert entry["occurrences"][0]["occurrences"][0]["mode"] in entry["modes"]


# ---------------------------------------------------------------------------
# The builder, config and provenance
def test_config_has_26_hashed_fields() -> None:
    assert len(MOTIF_CONFIG_FIELDS) == 26
    assert len(set(MOTIF_CONFIG_FIELDS)) == 26
    assert "participation" in MOTIF_CONFIG_FIELDS and "diagram_layout" in MOTIF_CONFIG_FIELDS


def test_config_hash_is_stable_and_default_aware() -> None:
    assert MotifConfig().is_default()
    assert MotifConfig().config_hash() == MotifConfig().config_hash()
    assert len(MotifConfig().config_hash()) == 8
    other = MotifConfig(participation="max")
    assert not other.is_default()
    assert other.config_hash() != MotifConfig().config_hash()
    assert MotifConfig(k_resolved=9).config_hash() == MotifConfig().config_hash()


def test_build_provenance_and_stage(synthetic_spectrum) -> None:
    analysis, report = build_motif_analysis(synthetic_spectrum)
    assert not report.has_errors
    payload = analysis.to_dict()
    provenance = payload["provenance"]
    assert provenance["phase"] == "04" and provenance["stage"] == STAGE
    assert provenance["spectral_config_hash"] == synthetic_spectrum.config.config_hash()
    assert provenance["source_file"] == "synthetic.gv"
    assert payload["config"]["k_resolved"] == analysis.n_modes
    assert payload["config"]["n_motifs"] == analysis.n_modes


def test_build_rejects_a_broken_config(synthetic_spectrum) -> None:
    with pytest.raises(MotifValidationError):
        build_motif_analysis(synthetic_spectrum, config=MotifConfig(participation="bogus"))
    _, report = build_motif_analysis(
        synthetic_spectrum, config=MotifConfig(relative_threshold=2.0)
    )
    assert report.has_errors
    assert CODE_CONFIG in report.codes()


def test_phase03_object_is_not_mutated(synthetic_spectrum) -> None:
    before = synthetic_spectrum.to_dict()
    build_motif_analysis(synthetic_spectrum)
    assert synthetic_spectrum.to_dict() == before


def test_payload_schema_accepts_the_built_analysis(synthetic_spectrum) -> None:
    analysis, _ = build_motif_analysis(synthetic_spectrum)
    assert validate_motif_payload_schema(analysis.to_dict()) == []


def test_small_graph_warns() -> None:
    one = make_spectrum(
        neuron_ids=["solo"],
        left=LEFT[:1, :1],
        right=RIGHT[:1, :1],
        values=SYNTHETIC_VALUES[:1],
    )
    _, report = build_motif_analysis(one, config=MotifConfig(min_members=0))
    assert "small_graph" in report.codes()


def test_variant_stem_inheritance_and_config_hash_segment() -> None:
    assert input_variant(Path("x/eigen.json")) == ""
    assert input_variant(Path("x/eigen.f8652585.json")) == ".f8652585"
    assert variant_stem(MotifConfig(), source_artifact="x/eigen.f8652585.json") == "motifs.f8652585"
    stem = variant_stem(MotifConfig(participation="left"), source_artifact="x/eigen.json")
    assert stem.startswith("motifs.") and len(stem.split(".")[-1]) == 8
    assert CANONICAL_STEM == "motifs"


def test_resolve_inputs_rejects_save_data_and_finds_variants(synthetic_workspace: Path) -> None:
    directory = synthetic_workspace / "synthetic.gv"
    (directory / "eigen.data.json").write_text("{}", encoding="utf-8")
    (directory / "eigen.f8652585.json").write_text(synthetic_workspace_text(directory), encoding="utf-8")
    found = resolve_inputs([synthetic_workspace], include_variants=True)
    names = sorted(path.name for path in found)
    assert names == ["eigen.f8652585.json", "eigen.json"]
    rejected = resolve_inputs([directory / "eigen.data.json"])
    assert rejected == []


def synthetic_workspace_text(directory: Path) -> str:
    return (directory / "eigen.json").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Artifacts, sidecar and determinism
# ---------------------------------------------------------------------------
def test_write_artifact_set_names_and_roundtrip(synthetic_spectrum, tmp_path: Path) -> None:
    analysis, _ = build_motif_analysis(synthetic_spectrum)
    written = write_artifact_set(analysis, tmp_path, save_data=True)
    assert set(written) == {"json", "npz", "data", "graphml"}
    assert written["json"].name == "motifs.json"
    assert written["npz"].name == "motifs.npz"
    assert written["data"].name == "motifs.data.json"
    for path in written.values():
        assert path.is_file()
    loaded = load_motifs(written["json"])
    assert loaded.to_dict() == analysis.to_dict()
    assert loaded.loaded_from == written["json"]


def test_artifact_paths_follow_the_variant_stem(synthetic_spectrum, tmp_path: Path) -> None:
    analysis, _ = build_motif_analysis(synthetic_spectrum, source_artifact="x/eigen.f8652585.json")
    paths = artifact_paths(analysis, tmp_path)
    assert paths["json"].name == "motifs.f8652585.json"
    assert sidecar_path(paths["json"]) == paths["npz"]


def test_sidecar_bundle_dtypes_and_digest(synthetic_spectrum, tmp_path: Path) -> None:
    analysis, _ = build_motif_analysis(synthetic_spectrum)
    written = write_artifact_set(analysis, tmp_path)
    arrays = load_sidecar_arrays(written["npz"])
    assert len(arrays) == 24
    assert str(arrays["stage"][0]) == STAGE
    assert arrays["participation"].shape == (len(NEURON_IDS), 3)
    assert arrays["membership"].dtype == np.int64
    assert arrays["family_of_mode"].tolist() == analysis.family_of_mode.tolist()
    assert arrays["link_jaccard"].shape == (3, 3)
    assert arrays["link_polarity"].shape == (3, 3)
    digest = hashlib.sha256(written["json"].read_bytes()).hexdigest()
    assert str(arrays["source_json_sha256"][0]) == digest
    assert str(arrays["numpy_version"][0]) == np.__version__
    assert np.allclose(arrays["participation"], analysis.participation)
    assert np.allclose(arrays["thresholds"], analysis.thresholds)


def test_sidecar_zip_entries_use_a_fixed_timestamp(synthetic_spectrum, tmp_path: Path) -> None:
    import zipfile

    analysis, _ = build_motif_analysis(synthetic_spectrum)
    written = write_artifact_set(analysis, tmp_path)
    with zipfile.ZipFile(written["npz"]) as archive:
        for info in archive.infolist():
            assert info.date_time == (1980, 1, 1, 0, 0, 0)


def test_artifacts_are_byte_identical_under_source_date_epoch(
    synthetic_spectrum, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "1700000000")
    first, _ = build_motif_analysis(synthetic_spectrum)
    second, _ = build_motif_analysis(synthetic_spectrum)
    write_artifact_set(first, tmp_path / "first", save_data=True)
    write_artifact_set(second, tmp_path / "second", save_data=True)
    for name in ("motifs.json", "motifs.npz", "motifs.data.json", "motifs.pathway.graphml"):
        left = (tmp_path / "first" / "synthetic" / name).read_bytes()
        right = (tmp_path / "second" / "synthetic" / name).read_bytes()
        assert left == right, name


def test_stale_cache_is_ignored_with_a_warning(
    synthetic_spectrum, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    analysis, _ = build_motif_analysis(synthetic_spectrum)
    written = write_artifact_set(analysis, tmp_path)
    arrays = load_sidecar_arrays(written["npz"])
    arrays["source_json_sha256"] = np.asarray(["0" * 64], dtype="<U64")
    write_npz_atomic(written["npz"], arrays)
    with caplog.at_level("WARNING"):
        loaded = load_motifs(written["json"])
    assert loaded.to_dict() == analysis.to_dict()
    assert any("ignoring sidecar" in record.message for record in caplog.records)


def test_tampered_cache_raises_when_read_directly(synthetic_spectrum, tmp_path: Path) -> None:
    analysis, _ = build_motif_analysis(synthetic_spectrum)
    written = write_artifact_set(analysis, tmp_path)
    arrays = load_sidecar_arrays(written["npz"])
    arrays["thresholds"] = arrays["thresholds"] + 1.0
    write_npz_atomic(written["npz"], arrays)
    with pytest.raises(MotifValidationError):
        load_motifs(written["npz"])


def test_no_sidecar_leaves_the_json_and_the_graphml(synthetic_spectrum, tmp_path: Path) -> None:
    analysis, _ = build_motif_analysis(synthetic_spectrum)
    written = write_artifact_set(analysis, tmp_path, sidecar=False)
    assert set(written) == {"json", "graphml"}
    directory = written["json"].parent
    assert sorted(path.name for path in directory.iterdir()) == [
        "motifs.json",
        "motifs.pathway.graphml",
    ]
    loaded = load_motifs(written["json"])
    assert loaded.to_dict() == analysis.to_dict()


def test_no_graphml_leaves_only_the_json(synthetic_spectrum, tmp_path: Path) -> None:
    analysis, _ = build_motif_analysis(synthetic_spectrum)
    written = write_artifact_set(analysis, tmp_path, sidecar=False, graphml=False)
    assert set(written) == {"json"}
    directory = written["json"].parent
    assert sorted(path.name for path in directory.iterdir()) == ["motifs.json"]


def test_sidecar_requires_in_memory_arrays() -> None:
    analysis, _ = build_motif_analysis(make_spectrum())
    analysis.arrays = {}
    with pytest.raises(MotifValidationError):
        build_sidecar_arrays(analysis, source_json_sha256="0" * 64)


# ---------------------------------------------------------------------------
# CLI
def test_cli_writes_the_artifact_set(synthetic_workspace: Path, capsys) -> None:
    eigen = synthetic_workspace / "synthetic.gv" / "eigen.json"
    outdir = synthetic_workspace / "out"
    assert main(["-i", str(eigen), "-o", str(outdir), "--stats", "--save-data"]) == 0
    captured = capsys.readouterr().out
    assert "Phase 04D complete" in captured
    assert "participation       : rms" in captured
    assert "families" in captured
    assert "groups" in captured
    target = outdir / "synthetic"
    assert sorted(path.name for path in target.iterdir()) == [
        "motifs.data.json",
        "motifs.json",
        "motifs.npz",
        "motifs.pathway.graphml",
    ]
    assert not list(target.rglob("*.csv"))


def test_cli_dry_run_writes_nothing(synthetic_workspace: Path) -> None:
    eigen = synthetic_workspace / "synthetic.gv" / "eigen.json"
    outdir = synthetic_workspace / "dry"
    assert main(["-i", str(eigen), "-o", str(outdir), "--dry-run", "--stats"]) == 0
    assert not outdir.exists()


def test_cli_no_sidecar(synthetic_workspace: Path) -> None:
    eigen = synthetic_workspace / "synthetic.gv" / "eigen.json"
    outdir = synthetic_workspace / "nosidecar"
    assert main(["-i", str(eigen), "-o", str(outdir), "--no-sidecar"]) == 0
    assert sorted(path.name for path in (outdir / "synthetic").iterdir()) == [
        "motifs.json",
        "motifs.pathway.graphml",
    ]


def test_cli_exit_codes(synthetic_workspace: Path, tmp_path: Path, caplog) -> None:
    eigen = synthetic_workspace / "synthetic.gv" / "eigen.json"
    # 0 on success
    assert main(["-i", str(eigen), "-o", str(tmp_path / "ok")]) == 0
    # 2 when nothing matches
    assert main(["-i", str(tmp_path / "missing"), "-o", str(tmp_path / "x")]) == 2
    # 2 when the input is a save-data artifact
    save_data = synthetic_workspace / "synthetic.gv" / "eigen.data.json"
    save_data.write_text("{}", encoding="utf-8")
    assert main(["-i", str(save_data), "-o", str(tmp_path / "y")]) == 2
    # 1 when the spectrum is unusable
    broken_dir = tmp_path / "broken" / "synthetic.gv"
    broken_dir.mkdir(parents=True)
    (broken_dir / "eigen.json").write_text("{}", encoding="utf-8")
    with caplog.at_level("ERROR"):
        assert main(["-i", str(broken_dir / "eigen.json"), "-o", str(tmp_path / "z")]) == 1


def test_cli_strict_escalates_warnings(synthetic_workspace: Path, tmp_path: Path) -> None:
    # a clean spectrum passes --strict
    eigen = synthetic_workspace / "synthetic.gv" / "eigen.json"
    assert main(["-i", str(eigen), "-o", str(tmp_path / "strict"), "--strict"]) == 0
    # a one-neuron spectrum warns (small_graph) and --strict turns that into a failure
    small_dir = tmp_path / "small" / "synthetic.gv"
    small_dir.mkdir(parents=True)
    from src.utils.io import write_json_atomic

    write_json_atomic(
        small_dir / "eigen.json",
        make_spectrum(
            neuron_ids=["solo"], left=LEFT[:1, :1], right=RIGHT[:1, :1], values=SYNTHETIC_VALUES[:1]
        ).to_dict(),
    )
    assert main(["-i", str(small_dir / "eigen.json"), "-o", str(tmp_path / "smallout")]) == 0
    assert main(
        ["-i", str(small_dir / "eigen.json"), "-o", str(tmp_path / "smallstrict"), "--strict"]
    ) == 1


def test_python_m_subprocess_smoke(synthetic_workspace: Path, tmp_path: Path) -> None:
    eigen = synthetic_workspace / "synthetic.gv" / "eigen.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.clustering.functional_motifs",
            "-i",
            str(eigen),
            "-o",
            str(tmp_path / "sub"),
        ],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "sub" / "synthetic" / "motifs.json").is_file()


def test_render_summary_box_reflects_the_outcome(synthetic_spectrum, tmp_path: Path) -> None:
    analysis, _ = build_motif_analysis(synthetic_spectrum)
    paths = artifact_paths(analysis, tmp_path)
    box = render_summary_box(analysis, paths, written=True, sidecar=False, save_data=False, stats=True)
    assert "Phase 04D complete" in box
    assert "Pathway:" in box and "GraphML:" in box
    assert "skipped (--no-sidecar)" in box
    assert "not requested (--save-data)" in box
    dry = render_summary_box(analysis, paths, written=False, stats=False)
    assert "dry-run" in dry


def test_render_statistics_reports_every_layer(synthetic_spectrum) -> None:
    analysis, _ = build_motif_analysis(synthetic_spectrum)
    text = render_statistics(analysis)
    for marker in (
        "Phase 04D -- motif statistics",
        "links",
        "families",
        "groups",
        "pathway",
        "pathway threshold",
        "pathway polarity",
        "pathway concentration",
    ):
        assert marker in text
    assert "not computed yet (04D)" not in text
    assert len(analysis.links) <= analysis.n_modes * (analysis.n_modes - 1) // 2


def test_extract_motifs_and_motif_statistics_work_from_arrays(synthetic_spectrum) -> None:
    """The two lower-level entry points are usable without the full builder."""
    matrix = _participation(synthetic_spectrum)
    regions = [derive_region(name) for name in NEURON_IDS]
    thresholds = motif_thresholds(matrix, "relative")
    membership = motif_membership(matrix, thresholds, min_members=0, max_members=0)
    motifs = extract_motifs(
        matrix,
        synthetic_spectrum.loadings_left,
        synthetic_spectrum.loadings_right,
        thresholds=thresholds,
        threshold_method="relative",
        membership=membership,
        neuron_ids=NEURON_IDS,
        regions=regions,
        values=synthetic_spectrum.spectrum,
        abs_values=synthetic_spectrum.abs_spectrum,
        explained_variance_ratio=synthetic_spectrum.explained_variance_ratio,
        cumulative_variance_ratio=synthetic_spectrum.cumulative_variance_ratio,
    )
    assert len(motifs) == 3
    metadata = motif_statistics(
        matrix,
        membership,
        motifs,
        MotifConfig(k_resolved=3),
        neuron_ids=NEURON_IDS,
        regions=regions,
    )
    assert metadata["n_motifs"] == 3
    assert metadata["n_members_total"] == int(membership.sum())
    assert metadata["n_regions"] == len(NEURON_IDS)


# ---------------------------------------------------------------------------
# Cross-mode tracking (04B): links, families and their statistics
# ---------------------------------------------------------------------------
def test_04c_assign_groups_argmax_ties_and_background() -> None:
    participation = np.array([[1.0, 2.0], [3.0, 3.0], [0.0, 0.0]])
    assert assign_groups(participation).tolist() == [2, 1, 0]
    assert initial_group_labels(participation).tolist() == [2, 1, 0]


def test_04c_partition_helpers_and_group_math() -> None:
    participation = np.array([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.0, 0.0]])
    labels = assign_groups(participation)
    assert np.array_equal(group_centroid([0, 1], participation), np.array([0.9, 0.1]))
    assert region_composition([2, 0, 1], ["B", "A", "A", "Z"]) == {"A": 2, "B": 1}
    coherence = group_coherence(participation, labels)
    assert coherence[0] is None and coherence[1] == pytest.approx(0.9701425001)


def test_04c_polarity_minority_guard() -> None:
    labels = np.array([1, 1, 1, 1, 2, 2])
    signs = np.array([[1], [1], [1], [-1], [1], [1]])
    unchanged, count = split_large_groups(labels, signs, polarity_min_members=2)
    assert count == 0 and unchanged.tolist() == labels.tolist()
    signs[2, 0] = -1
    split, count = split_large_groups(labels, signs, polarity_min_members=2)
    assert count == 1 and split[2] != split[0]


def test_04c_loadings_and_none_modes(synthetic_spectrum) -> None:
    analysis, report = build_motif_analysis(synthetic_spectrum, config=MotifConfig(grouping="loadings", n_groups="2"))
    assert not report.has_errors and len(analysis.groups) == 3
    assert analysis.arrays["group_labels"].shape == (len(NEURON_IDS),)
    none, report = build_motif_analysis(synthetic_spectrum, config=MotifConfig(grouping="none"))
    assert not report.has_errors and none.groups == []
    assert not {"group_labels", "group_centroids", "group_sizes"}.intersection(none.arrays)


def test_04c_reference_group_values() -> None:
    spectrum = load_spectrum(REFERENCE_JSON)
    analysis, report = build_motif_analysis(spectrum, source_artifact=REFERENCE_JSON)
    assert not report.has_errors
    assert analysis.metadata["n_groups"] == 22
    assert analysis.metadata["group_sizes"] == [1, 6, 10, 7, 7, 7, 10, 9, 2, 7, 1, 1, 5, 1, 4, 5, 5, 4, 6, 7, 4, 4]
    values = [group["coherence"] for group in analysis.groups if group["coherence"] is not None]
    assert min(values) == pytest.approx(-0.1555307908)
    assert max(values) == pytest.approx(0.8764502786)
    assert np.mean(values) == pytest.approx(0.1478265824)


def test_link_classes_constant() -> None:
    assert LINK_CLASSES == ("stable", "flipped", "composite", "weak")


@pytest.mark.parametrize(
    "jaccard,agreement,expected",
    [
        (0.0, 0.0, "weak"),
        (0.0999, 1.0, "weak"),
        (0.1, 0.0, "composite"),  # link_min_jaccard is an inclusive lower bound
        (0.1, -1.0, "composite"),
        (0.1999, 0.0, "composite"),
        (0.2, -0.5, "flipped"),  # family_jaccard is inclusive ...
        (0.2, 0.5, "stable"),    # ... and so is the polarity band
        (0.2, 0.4999, "composite"),
        (0.2, -0.4999, "composite"),
        (0.5, -1.0, "flipped"),
        (1.0, 1.0, "stable"),
    ],
)
def test_classify_link_boundaries(jaccard: float, agreement: float, expected: str) -> None:
    assert classify_link(jaccard, agreement) == expected


def test_classify_link_honours_custom_thresholds() -> None:
    assert classify_link(0.3, 0.2, family_polarity=0.9) == "composite"
    assert classify_link(0.3, 0.95, family_polarity=0.9) == "stable"
    assert classify_link(0.3, -0.95, family_polarity=0.9) == "flipped"
    assert classify_link(0.05, 1.0, link_min_jaccard=0.01) == "composite"
    assert classify_link(0.4, 1.0, family_jaccard=0.5) == "composite"


def test_motif_links_covers_every_shared_pair(synthetic_spectrum) -> None:
    analysis, _ = build_motif_analysis(synthetic_spectrum)
    links = analysis.links
    # the three synthetic modes share exactly one member each -> one link per pair
    assert [tuple(sorted((link["source_mode"], link["target_mode"]))) for link in links] == [
        (1, 2),
        (1, 3),
        (2, 3),
    ]
    assert all(link["n_shared"] == 1 for link in links)
    assert all(link["jaccard"] == pytest.approx(0.2) for link in links)
    assert all(link["polarity_agreement"] == pytest.approx(-1.0) for link in links)
    assert all(link["classification"] == "flipped" for link in links)
    assert links[0]["source_label"] == "M01" and links[0]["target_label"] == "M02"


def test_motif_links_orders_shared_members_by_neuron_index(synthetic_spectrum) -> None:
    analysis, _ = build_motif_analysis(synthetic_spectrum)
    index = {name: position for position, name in enumerate(NEURON_IDS)}
    for link in analysis.links:
        positions = [index[name] for name in link["shared_members"]]
        assert positions == sorted(positions)


def test_no_links_when_modes_are_disjoint() -> None:
    left = np.zeros((8, 3))
    right = np.zeros((8, 3))
    left[0, 0] = 1.0
    left[1, 1] = 0.5
    left[2, 2] = 0.25
    right[3, 0] = 1.0
    right[4, 1] = 0.5
    right[5, 2] = 0.25
    spectrum = make_spectrum(left=left, right=right)
    analysis, report = build_motif_analysis(spectrum, config=MotifConfig(min_members=0))
    assert not report.has_errors
    assert analysis.links == []
    assert analysis.metadata["n_links"] == 0
    assert analysis.metadata["link_classes"] == {
        "stable": 0,
        "flipped": 0,
        "composite": 0,
        "weak": 0,
    }
    assert analysis.metadata["n_families"] == analysis.n_modes
    assert analysis.metadata["n_multi_mode_families"] == 0
    assert analysis.family_of_mode.tolist() == list(range(analysis.n_modes))


def test_flipped_links_merge_into_one_family(synthetic_spectrum) -> None:
    analysis, _ = build_motif_analysis(synthetic_spectrum)
    assert len(analysis.families) == 1
    family = analysis.families[0]
    assert family["modes"] == [1, 2, 3]
    assert family["label"] == "F01" and family["family_id"] == 1
    assert family["n_modes"] == 3
    assert analysis.family_of_mode.tolist() == [0, 0, 0]
    assert analysis.metadata["n_multi_mode_families"] == 1


def test_composite_links_do_not_merge_families(synthetic_spectrum) -> None:
    """The giant-component regression: only stable/flipped links may merge."""
    analysis, _ = build_motif_analysis(synthetic_spectrum)
    links = motif_links(analysis.motifs, family_jaccard=0.21)
    assert {link.classification for link in links} == {"composite"}
    families = merge_families(analysis.motifs, links)
    assert len(families) == analysis.n_modes
    assert all(family.n_modes == 1 for family in families)
    assert [family.modes for family in families] == [(1,), (2,), (3,)]


def test_merge_families_uses_only_the_merging_links(synthetic_spectrum) -> None:
    analysis, _ = build_motif_analysis(synthetic_spectrum)
    links = motif_links(analysis.motifs)
    stable = merge_families(analysis.motifs, links)
    assert len(stable) == 1 and stable[0].modes == (1, 2, 3)
    # dropping the merging links restores one singleton family per mode
    none = merge_families(analysis.motifs, [])
    assert [family.modes for family in none] == [(1,), (2,), (3,)]
    assert [family.label for family in none] == ["F01", "F02", "F03"]


def test_flipped_family_aligns_shared_member_signs(synthetic_spectrum) -> None:
    analysis, _ = build_motif_analysis(synthetic_spectrum)
    occurrences = {
        entry["neuron_id"]: entry["occurrences"] for entry in analysis.families[0]["occurrences"]
    }
    # n1 is shared by modes 1 and 2 with opposite raw polarity; the family flip makes
    # its occurrences sign-consistent.
    n1 = occurrences["n1"]
    assert {entry["mode"] for entry in n1} == {1, 2}
    assert len({entry["signed_participation"] > 0 for entry in n1}) == 1


def test_family_members_union_and_occurrences(synthetic_spectrum) -> None:
    analysis, _ = build_motif_analysis(synthetic_spectrum)
    by_mode = {motif.mode: motif for motif in analysis.motifs}
    members, occurrences = family_members((1, 2), by_mode, {1: 1, 2: -1})
    assert set(members) == {"n0", "n1", "n5", "n3", "n2"}
    assert len(occurrences) == len(members)
    for entry in occurrences:
        assert set(entry) == {"neuron_id", "occurrences"}
        for item in entry["occurrences"]:
            assert set(item) == {"mode", "participation", "signed_participation"}
            assert item["participation"] >= 0.0


def test_recurrence_table_counts_modes_and_families(synthetic_spectrum) -> None:
    analysis, _ = build_motif_analysis(synthetic_spectrum)
    table = recurrence_table(analysis.motifs, analysis.families)
    assert set(table) == {"n0", "n1", "n2", "n3", "n4", "n5"}
    assert table["n1"]["n_modes"] == 2 and table["n1"]["modes"] == [1, 2]
    assert table["n0"]["n_modes"] == 1 and table["n0"]["modes"] == [1]
    assert all(entry["n_families"] == 1 for entry in table.values())
    assert table["n1"]["max_participation"] > 0.0
    # without families the family columns stay empty
    alone = recurrence_table(analysis.motifs)
    assert all(entry["n_families"] == 0 and entry["families"] == [] for entry in alone.values())


def test_link_matrices_are_symmetric_with_unit_diagonal(synthetic_spectrum) -> None:
    analysis, _ = build_motif_analysis(synthetic_spectrum)
    jaccard = analysis.arrays["link_jaccard"]
    polarity = analysis.arrays["link_polarity"]
    assert jaccard.shape == (3, 3) and polarity.shape == (3, 3)
    assert np.array_equal(jaccard, jaccard.T) and np.array_equal(polarity, polarity.T)
    assert np.allclose(jaccard.diagonal(), 1.0) and np.allclose(polarity.diagonal(), 1.0)
    assert jaccard[0, 1] == pytest.approx(0.2)
    assert polarity[0, 1] == pytest.approx(-1.0)


def test_tracking_metadata_and_save_data(synthetic_spectrum) -> None:
    analysis, _ = build_motif_analysis(synthetic_spectrum)
    metadata = analysis.metadata
    assert metadata["n_links"] == len(analysis.links) == 3
    assert metadata["link_classes"] == {"stable": 0, "flipped": 3, "composite": 0, "weak": 0}
    assert metadata["n_families"] == len(analysis.families) == 1
    assert metadata["n_multi_mode_families"] == 1
    payload = save_data_payload(analysis)
    assert payload["links"] == analysis.links
    assert payload["families"] == analysis.families


@pytest.mark.parametrize(
    "kwargs",
    [
        {"family_jaccard": 1.5},
        {"family_jaccard": -0.1},
        {"family_polarity": 1.5},
        {"link_min_jaccard": 2.0},
        {"family_jaccard": 0.1, "link_min_jaccard": 0.5},
    ],
)
def test_config_rejects_bad_tracking_thresholds(synthetic_spectrum, kwargs) -> None:
    _, report = build_motif_analysis(synthetic_spectrum, config=MotifConfig(**kwargs))
    assert report.has_errors
    assert CODE_CONFIG in report.codes()


@pytest.mark.parametrize(
    "config",
    [
        MotifConfig(),
        MotifConfig(link_min_jaccard=0.0),
        MotifConfig(link_min_jaccard=0.0, family_jaccard=0.0, family_polarity=0.0),
        MotifConfig(link_min_jaccard=0.0, family_jaccard=0.05, family_polarity=0.9),
    ],
)
def test_family_partition_invariant_for_any_config(synthetic_spectrum, config) -> None:
    analysis, report = build_motif_analysis(synthetic_spectrum, config=config)
    assert not report.has_errors
    covered = [mode for family in analysis.families for mode in family["modes"]]
    assert sorted(covered) == list(range(1, analysis.n_modes + 1))
    assert len(covered) == len(set(covered))
    assert all(family["label"] == f"F{family['family_id']:02d}" for family in analysis.families)
    assert analysis.family_of_mode.shape == (analysis.n_modes,)
    assert analysis.family_of_mode.tolist() == [
        next(
            position
            for position, family in enumerate(analysis.families)
            if mode in family["modes"]
        )
        for mode in range(1, analysis.n_modes + 1)
    ]


def test_tracking_cli_flags_reach_the_config() -> None:
    parser = build_argument_parser()
    args = parser.parse_args(
        [
            "-i",
            "x",
            "--family-jaccard",
            "0.5",
            "--family-polarity",
            "0.3",
            "--link-min-jaccard",
            "0.05",
        ]
    )
    assert (args.family_jaccard, args.family_polarity, args.link_min_jaccard) == (0.5, 0.3, 0.05)
    assert MotifConfig(family_jaccard=0.5).config_hash() != MotifConfig().config_hash()
    assert MotifConfig(link_min_jaccard=0.05).config_hash() != MotifConfig().config_hash()


def test_tracking_artifacts_are_deterministic(synthetic_spectrum, tmp_path: Path) -> None:
    first, _ = build_motif_analysis(synthetic_spectrum)
    second, _ = build_motif_analysis(synthetic_spectrum)
    write_artifact_set(first, tmp_path / "a", save_data=True)
    write_artifact_set(second, tmp_path / "b", save_data=True)
    for name in ("motifs.json", "motifs.npz", "motifs.data.json"):
        assert (tmp_path / "a" / "synthetic" / name).read_bytes() == (
            tmp_path / "b" / "synthetic" / name
        ).read_bytes(), name


# ---------------------------------------------------------------------------
# 04D -- pathway graph: mode weights, contributions, filtering
# ---------------------------------------------------------------------------
def test_04d_mode_weights_rules() -> None:
    values = np.array([3.0, 2.0, 1.0])
    evr = mode_weights(values, "evr")
    energy = mode_weights(values, "energy")
    assert np.allclose(evr, energy)  # documented equivalence (energy == evr)
    assert np.isclose(evr.sum(), 1.0)
    assert np.allclose(evr, values**2 / (values**2).sum())
    uniform = mode_weights(values, "uniform")
    assert np.allclose(uniform, 1.0 / 3.0) and np.isclose(uniform.sum(), 1.0)
    absolute = mode_weights(values, "abs_value")
    assert np.allclose(absolute, values / np.abs(values).sum())
    signed = mode_weights(np.array([3.0, -1.0, -1.0]), "value")
    assert np.isclose(np.abs(signed).sum(), 1.0)
    assert signed[0] > 0 and signed[1] < 0
    assert DEFAULT_PATHWAY_WEIGHT == "evr"


def test_04d_mode_weights_reject_degenerate_input() -> None:
    with pytest.raises(MotifValidationError) as zeros:
        mode_weights(np.zeros(3), "evr")
    assert CODE_MODE_WEIGHTS in str(zeros.value)
    with pytest.raises(MotifValidationError) as negative:
        mode_weights(np.array([-3.0, -2.0, -1.0]), "value")
    assert CODE_MODE_WEIGHTS in str(negative.value)
    with pytest.raises(MotifValidationError):
        mode_weights(np.array([1.0, 2.0]), "bogus")


def test_04d_mode_outer_product_is_rank_one(synthetic_spectrum) -> None:
    left = synthetic_spectrum.loadings_left
    right = synthetic_spectrum.loadings_right
    values = synthetic_spectrum.spectrum[: len(synthetic_spectrum.modes)]
    for mode in (1, 2, 3):
        product = mode_outer_product(left, right, values, mode)
        assert product.shape == (len(NEURON_IDS), len(NEURON_IDS))
        assert np.allclose(
            product, values[mode - 1] * np.outer(left[:, mode - 1], right[:, mode - 1])
        )
    with pytest.raises(MotifValidationError):
        mode_outer_product(left, right, values, 4)


def test_04d_group_pair_contributions_match_brute_force() -> None:
    left = np.array([[1.0, 0.5], [0.0, -1.0], [2.0, 0.0], [-1.0, 1.0]])
    right = np.array([[0.5, 1.0], [1.0, 0.0], [-1.0, 0.5], [0.0, -2.0]])
    values = np.array([2.0, 3.0])
    labels = np.array([0, 0, 1, 2], dtype="<i8")
    weights = mode_weights(values, "evr")
    stack = np.asarray(
        [values[index] * np.outer(left[:, index], right[:, index]) for index in range(2)]
    )
    matrix = group_pair_contributions(stack, labels, weights)
    assert matrix.shape == (3, 3)
    for source in range(3):
        for target in range(3):
            expected = 0.0
            for index in range(2):
                for row in np.flatnonzero(labels == source):
                    for column in np.flatnonzero(labels == target):
                        expected += weights[index] * stack[index][row, column]
            assert np.isclose(matrix[source, target], expected)


def test_04d_group_pair_contributions_validate_shapes() -> None:
    with pytest.raises(MotifValidationError):
        group_pair_contributions(np.zeros((2, 3, 4)), np.zeros(3, dtype="<i8"), np.ones(2))
    with pytest.raises(MotifValidationError):
        group_pair_contributions(np.zeros((2, 3, 3)), np.zeros(4, dtype="<i8"), np.ones(2))
    with pytest.raises(MotifValidationError):
        group_pair_contributions(np.zeros((2, 3, 3)), np.zeros(3, dtype="<i8"), np.ones(3))
    with pytest.raises(MotifValidationError):
        group_pair_contributions(np.zeros((1, 2, 2)), np.array([0, -1]), np.ones(1))


def test_04d_filter_threshold_is_relative_to_abs_max() -> None:
    matrix = np.array([[0.0, 2.0, 0.4], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    kept, stats = filter_pathway_edges(matrix, edge_threshold=0.5, top_edges=10)
    assert stats["abs_max"] == 2.0
    assert stats["cutoff"] == 1.0
    assert [(source, target) for source, target, _ in kept] == [(0, 1)]
    kept_half, _ = filter_pathway_edges(matrix, edge_threshold=0.1, top_edges=10)
    assert len(kept_half) == 2  # 2.0 and 0.4 both survive a 10% threshold
    # the sweep is a threshold-only diagnostic (0.05/0.1/0.2/0.25 of abs_max)
    assert stats["threshold_sweep"] == {"0.05": 2, "0.1": 2, "0.2": 2, "0.25": 1}


def test_04d_filter_top_edges_is_per_source_with_ties() -> None:
    matrix = np.zeros((3, 3))
    matrix[0, 1] = 3.0
    matrix[0, 2] = 3.0  # exact tie with (0, 1)
    matrix[1, 0] = 2.5
    matrix[1, 2] = 2.0
    matrix[2, 0] = -2.0
    kept, stats = filter_pathway_edges(matrix, edge_threshold=0.1, top_edges=1)
    assert stats["n_candidates"] == 5
    # one edge per source; the (0, *) tie breaks on the lower target index
    assert kept == [(0, 1, 3.0), (1, 0, 2.5), (2, 0, -2.0)]
    assert stats["per_source_counts"] == {0: 1, 1: 1, 2: 1}


def test_04d_filter_intra_switch_and_zero_matrix() -> None:
    matrix = np.zeros((2, 2))
    matrix[0, 0] = -1.0
    matrix[0, 1] = 0.5
    _, stats = filter_pathway_edges(matrix, edge_threshold=0.1, top_edges=5)
    assert stats["n_intra"] == 1 and stats["n_cross"] == 1
    without_intra, stats_off = filter_pathway_edges(
        matrix, edge_threshold=0.1, top_edges=5, intra=False
    )
    assert stats_off["n_intra"] == 0 and len(without_intra) == 1
    empty, empty_stats = filter_pathway_edges(np.zeros((3, 3)), edge_threshold=0.1, top_edges=5)
    assert empty == []
    assert empty_stats["abs_max"] == 0.0 and empty_stats["weight_concentration"] == 0.0
    with pytest.raises(MotifValidationError):
        filter_pathway_edges(np.zeros((2, 3)))


def test_04d_filter_reports_polarity_and_concentration() -> None:
    matrix = np.array([[0.0, -2.0, 0.5], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    kept, stats = filter_pathway_edges(matrix, edge_threshold=0.1, top_edges=5)
    assert stats["n_positive"] == 2 and stats["n_negative"] == 1
    assert stats["n_zeros"] == 0
    assert np.isclose(stats["weight_concentration"], 2.0 / 3.5)
    assert stats["weights"] == [weight for _, _, weight in kept]


# ---------------------------------------------------------------------------
# 04D -- build_pathway, the pipeline block and the npz arrays
# ---------------------------------------------------------------------------
def _pathway_inputs(spectrum):
    analysis, report = build_motif_analysis(spectrum)
    assert not report.has_errors
    return analysis


def test_04d_build_pathway_structure(synthetic_spectrum) -> None:
    analysis = _pathway_inputs(synthetic_spectrum)
    diagram, diagnostics = build_pathway(
        analysis.groups,
        participation=analysis.participation,
        loadings_left=analysis.loadings_left,
        loadings_right=analysis.loadings_right,
        values=analysis.retained_values,
        neuron_ids=analysis.neuron_order,
        config=analysis.config,
    )
    payload = diagram.to_dict()
    assert set(payload) == set(PATHWAY_KEYS)
    assert payload["n_nodes"] == len(analysis.groups) == len(payload["nodes"])
    assert payload["n_edges"] == len(payload["edges"]) == len(payload["weights"])
    assert payload["source"] == "mode" and payload["weight"] == "evr"
    assert payload["intra"] is True
    assert payload["n_intra"] + payload["n_cross"] == payload["n_edges"]
    assert np.isclose(payload["abs_max"], max(abs(weight) for weight in payload["weights"]))
    for node in payload["nodes"]:
        assert set(node) == set(NODE_KEYS)
        assert node["node_id"] == node["group_id"]
        assert node["size"] == len(node["members"])
        shares = [entry["share"] for entry in node["top_modes"]]
        assert shares == sorted(shares, reverse=True)
        assert sum(shares) <= 1.0 + 1e-12
        for rank, entry in enumerate(node["top_modes"], start=1):
            assert set(entry) == set(TOP_MODE_KEYS)
            assert entry["rank"] == rank
            assert np.isclose(entry["abs_weight"], abs(entry["signed_weight"]))
    order = [(edge["source"], edge["target"]) for edge in payload["edges"]]
    assert order == sorted(order)
    for edge in payload["edges"]:
        assert set(edge) == set(EDGE_KEYS)
        assert edge["is_intra"] == (edge["source"] == edge["target"])
        assert edge["polarity"] == int(np.sign(edge["weight"]))
        assert np.isclose(edge["abs_weight"], abs(edge["weight"]))
        assert edge["n_modes"] == len(edge["modes"])
        assert edge["modes"] == sorted(edge["modes"])
        assert len(edge["top_modes"]) <= PATHWAY_TOP_MODES
        assert sum(entry["share"] for entry in edge["top_modes"]) <= 1.0 + 1e-12
    assert set(diagnostics) == {
        "abs_max",
        "cutoff",
        "n_candidates",
        "threshold_sweep",
        "per_source_counts",
        "mode_weights",
        "z_contributions",
    }
    assert np.isclose(sum(abs(value) for value in diagnostics["mode_weights"]), 1.0)


def test_04d_build_pathway_is_deterministic(synthetic_spectrum) -> None:
    analysis = _pathway_inputs(synthetic_spectrum)
    kwargs = dict(
        participation=analysis.participation,
        loadings_left=analysis.loadings_left,
        loadings_right=analysis.loadings_right,
        values=analysis.retained_values,
        neuron_ids=analysis.neuron_order,
        config=analysis.config,
    )
    first, first_diagnostics = build_pathway(analysis.groups, **kwargs)
    second, second_diagnostics = build_pathway(analysis.groups, **kwargs)
    assert first.to_dict() == second.to_dict()
    assert first_diagnostics == second_diagnostics


def test_04d_build_pathway_without_groups_is_neutral(synthetic_spectrum) -> None:
    analysis = _pathway_inputs(synthetic_spectrum)
    diagram, diagnostics = build_pathway(
        [],
        participation=analysis.participation,
        loadings_left=analysis.loadings_left,
        loadings_right=analysis.loadings_right,
        values=analysis.retained_values,
        neuron_ids=analysis.neuron_order,
        config=analysis.config,
    )
    assert isinstance(diagram, PathwayDiagram)
    assert diagram.to_dict()["n_nodes"] is None
    assert diagram.to_dict()["nodes"] == [] and diagram.to_dict()["edges"] == []
    assert diagnostics["mode_weights"] == [] and diagnostics["z_contributions"] == {}


# ---------------------------------------------------------------------------
# 04D -- pathway pipeline block, npz arrays and GraphML
# ---------------------------------------------------------------------------
def test_04d_pathway_config_changes_the_graph(synthetic_spectrum) -> None:
    analysis = _pathway_inputs(synthetic_spectrum)
    strict = MotifConfig(pathway_edge_threshold=0.9, pathway_top_edges=1, intra=False)
    diagram, _ = build_pathway(
        analysis.groups,
        participation=analysis.participation,
        loadings_left=analysis.loadings_left,
        loadings_right=analysis.loadings_right,
        values=analysis.retained_values,
        neuron_ids=analysis.neuron_order,
        config=strict,
    )
    payload = diagram.to_dict()
    assert payload["intra"] is False
    assert payload["edge_threshold"] == 0.9 and payload["top_edges"] == 1
    assert all(edge["is_intra"] is False for edge in payload["edges"])
    assert len(payload["edges"]) <= 1
    assert all(
        abs(edge["weight"]) >= 0.9 * payload["abs_max"] - 1e-12 for edge in payload["edges"]
    )


def test_04d_pipeline_populates_the_pathway_block(synthetic_spectrum) -> None:
    analysis = _pathway_inputs(synthetic_spectrum)
    assert analysis.config.stage == "04D"
    assert validate_motif_payload_schema(analysis.to_dict()) == []
    metadata = analysis.metadata
    assert metadata["n_pathway_nodes"] == len(analysis.pathway["nodes"])
    assert metadata["n_pathway_edges"] == len(analysis.pathway["edges"])
    assert metadata["pathway_abs_max"] == analysis.pathway["abs_max"]
    assert metadata["pathway_positive_edges"] == analysis.pathway["n_positive"]
    assert metadata["pathway_negative_edges"] == analysis.pathway["n_negative"]
    assert metadata["pathway_intra_edges"] == analysis.pathway["n_intra"]
    assert metadata["pathway_cross_edges"] == analysis.pathway["n_cross"]
    assert metadata["pathway_weight_concentration"] == analysis.pathway["weight_concentration"]
    assert analysis.diagnostics["pathway"]["threshold_sweep"]
    saved = save_data_payload(analysis)
    assert saved["pathway"] == analysis.pathway
    assert saved["stage"] == "04D"


def test_04d_grouping_none_keeps_the_pathway_neutral(synthetic_spectrum, caplog) -> None:
    with caplog.at_level("INFO"):
        analysis, report = build_motif_analysis(
            synthetic_spectrum, config=MotifConfig(grouping="none")
        )
    assert not report.has_errors
    assert analysis.groups == []
    assert analysis.pathway["nodes"] == [] and analysis.pathway["edges"] == []
    assert analysis.pathway["n_nodes"] is None and analysis.pathway["abs_max"] is None
    assert analysis.metadata["n_pathway_nodes"] is None
    assert "pathway_adjacency" not in analysis.arrays
    assert any(CODE_PATHWAY_SKIPPED in record.message for record in caplog.records)
    assert "not computed yet (grouping none)" in render_statistics(analysis)


def test_04d_npz_carries_the_pathway_arrays(synthetic_spectrum, tmp_path: Path) -> None:
    analysis = _pathway_inputs(synthetic_spectrum)
    written = write_artifact_set(analysis, tmp_path)
    arrays = load_sidecar_arrays(written["npz"])
    assert len(arrays) == 24
    k = analysis.n_modes
    n_neurons = analysis.n_neurons
    n_groups = len(analysis.groups)
    assert arrays["mode_outer_products"].shape == (k, n_neurons, n_neurons)
    assert arrays["pathway_adjacency"].shape == (n_groups, n_groups)
    assert arrays["pathway_adjacency_abs"].shape == (n_groups, n_groups)
    assert np.allclose(arrays["pathway_adjacency_abs"], np.abs(arrays["pathway_adjacency"]))
    node_ids = [node["node_id"] for node in analysis.pathway["nodes"]]
    expected = np.zeros((n_groups, n_groups))
    for edge in analysis.pathway["edges"]:
        expected[node_ids.index(edge["source"]), node_ids.index(edge["target"])] = edge["weight"]
    assert np.allclose(arrays["pathway_adjacency"], expected)
    assert int(np.count_nonzero(arrays["pathway_adjacency"])) == len(analysis.pathway["edges"])
    left = analysis.loadings_left
    right = analysis.loadings_right
    for index in range(k):
        assert np.allclose(
            arrays["mode_outer_products"][index],
            analysis.retained_values[index] * np.outer(left[:, index], right[:, index]),
        )
    assert load_motifs(written["json"]).to_dict() == analysis.to_dict()


# ---------------------------------------------------------------------------
# 04D -- GraphML artifact, CLI and the matrix source
# ---------------------------------------------------------------------------
def test_04d_graphml_metadata_helpers() -> None:
    node = {
        "node_id": "G01",
        "group_id": "G01",
        "label": "G01",
        "size": 6,
        "members": ["a"],
        "cent_mean": 0.2,
        "region_composition": {"FB4K": 3},
        "dominant_mode": 1,
        "top_modes": [],
    }
    group = {"group_id": "G01", "coherence": -0.155531, "is_background": False, "is_singleton": False}
    metadata = graphml_node_metadata(node, group, np.array([0.1, 0.2, 0.3]))
    assert set(metadata) <= set(GRAPHML_NODE_KEYS)
    assert metadata["group_id"] == "G01" and metadata["size"] == 6
    assert metadata["region_composition"] == '{"FB4K": 3}'
    assert metadata["centroid"] == "[0.1, 0.2, 0.3]"
    assert metadata["coherence"] == -0.155531
    # the background group has no dominant mode; singletons have no coherence
    background = graphml_node_metadata(
        {**node, "group_id": "G00", "dominant_mode": None},
        {"is_background": True, "is_singleton": True},
    )
    assert "dominant_mode" not in background and "coherence" not in background
    assert background["is_background"] is True and background["is_singleton"] is True
    assert "centroid" not in graphml_node_metadata(node)
    edge = {
        "source": "G01",
        "target": "G02",
        "weight": -1.5,
        "abs_weight": 1.5,
        "polarity": -1,
        "is_intra": False,
        "top_modes": [
            {"mode": 1, "rank": 1, "share": 1.0, "signed_weight": -1.5, "abs_weight": 1.5}
        ],
    }
    plain = graphml_edge_metadata(edge)
    assert set(plain) <= set(GRAPHML_EDGE_KEYS)
    assert plain["threshold_flag"] is True and plain["topN_flag"] is True
    assert plain["intra_flag"] is False and plain["polarity"] == -1
    assert "z_contribution" not in plain
    assert graphml_edge_metadata(edge, z_contribution=0.42)["z_contribution"] == 0.42


def test_04d_graphml_round_trips(synthetic_spectrum, tmp_path: Path) -> None:
    nx = pytest.importorskip("networkx")
    analysis = _pathway_inputs(synthetic_spectrum)
    paths = artifact_paths(analysis, tmp_path)
    written = write_graphml_pathway(
        paths["graphml"],
        analysis.pathway,
        groups=analysis.groups,
        config=analysis.config,
        centroids=analysis.arrays.get("group_centroids"),
    )
    assert written == paths["graphml"] and written.is_file()
    text = written.read_text(encoding="utf-8")
    assert text.startswith('<?xml version="1.0" encoding="UTF-8"?>')
    graph = nx.read_graphml(written)
    assert graph.number_of_nodes() == analysis.pathway["n_nodes"]
    assert graph.number_of_edges() == analysis.pathway["n_edges"]
    assert graph.graph["n_nodes"] == analysis.pathway["n_nodes"]
    assert graph.graph["n_edges"] == analysis.pathway["n_edges"]
    assert graph.graph["weight_normalization"] == "abs-max"
    assert graph.graph["pathway_source"] == "mode"
    assert graph.graph["pathway_weight_rule"] == "evr"
    assert graph.graph["threshold"] == analysis.config.pathway_edge_threshold
    assert graph.graph["topN"] == analysis.config.pathway_top_edges
    assert set(graph.graph) - {"node_default", "edge_default"} <= set(GRAPHML_GRAPH_KEYS)
    for node_id, attributes in graph.nodes(data=True):
        assert set(attributes) <= set(GRAPHML_NODE_KEYS)
        assert attributes["group_id"] == node_id
    for source, target, attributes in graph.edges(data=True):
        assert set(attributes) <= set(GRAPHML_EDGE_KEYS)
        assert attributes["source_group"] == source and attributes["target_group"] == target
        assert attributes["polarity"] == int(np.sign(attributes["weight"]))
        assert np.isclose(attributes["abs_weight"], abs(attributes["weight"]))
        assert attributes["threshold_flag"] is True and attributes["topN_flag"] is True
        assert attributes["intra_flag"] == (source == target)
        assert isinstance(json.loads(attributes["contribution_by_mode"]), list)
        assert "z_contribution" not in attributes


def test_04d_graphml_is_deterministic(synthetic_spectrum, tmp_path: Path) -> None:
    analysis = _pathway_inputs(synthetic_spectrum)
    first = write_graphml_pathway(
        tmp_path / "a" / "motifs.pathway.graphml",
        analysis.pathway,
        groups=analysis.groups,
        config=analysis.config,
    )
    second = write_graphml_pathway(
        tmp_path / "b" / "motifs.pathway.graphml",
        analysis.pathway,
        groups=analysis.groups,
        config=analysis.config,
    )
    assert first.read_bytes() == second.read_bytes()


def test_04d_graphml_rejects_unknown_keys(
    synthetic_spectrum, tmp_path: Path, monkeypatch
) -> None:
    import src.clustering.functional_motifs as fm

    analysis = _pathway_inputs(synthetic_spectrum)
    original = fm.graphml_node_metadata

    def leaky(*args, **kwargs):
        return {**original(*args, **kwargs), "unexpected": 1}

    monkeypatch.setattr(fm, "graphml_node_metadata", leaky)
    with pytest.raises(MotifValidationError):
        write_graphml_pathway(
            tmp_path / "bad.graphml", analysis.pathway, config=analysis.config
        )


def test_04d_graphml_write_failure_is_reported(
    synthetic_spectrum, tmp_path: Path, monkeypatch, caplog
) -> None:
    import src.clustering.functional_motifs as fm

    analysis = _pathway_inputs(synthetic_spectrum)

    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(fm, "_write_text_atomic", boom)
    with caplog.at_level("ERROR"):
        written = write_artifact_set(analysis, tmp_path)
    assert "graphml" not in written
    assert (tmp_path / "synthetic" / "motifs.json").is_file()


def test_04d_cli_pathway_flags_reach_the_config() -> None:
    parser = build_argument_parser()
    args = parser.parse_args(
        [
            "-i",
            "x/eigen.json",
            "--pathway-source",
            "matrix",
            "--pathway-weight",
            "uniform",
            "--pathway-edge-threshold",
            "0.25",
            "--pathway-top-edges",
            "3",
            "--no-intra",
            "--no-graphml",
        ]
    )
    assert args.pathway_source == "matrix" and args.pathway_weight == "uniform"
    assert args.pathway_edge_threshold == 0.25 and args.pathway_top_edges == 3
    assert args.intra is False and args.graphml is False
    defaults = parser.parse_args(["-i", "x/eigen.json"])
    assert defaults.pathway_source == "mode" and defaults.pathway_weight == "evr"
    assert defaults.pathway_edge_threshold == 0.1 and defaults.pathway_top_edges == 5
    assert defaults.intra is True and defaults.graphml is True


@pytest.mark.parametrize(
    "config",
    [
        MotifConfig(pathway_edge_threshold=-0.5),
        MotifConfig(pathway_edge_threshold=1.5),
        MotifConfig(pathway_top_edges=0),
    ],
)
def test_04d_config_rejects_bad_pathway_ranges(synthetic_spectrum, config: MotifConfig) -> None:
    _, report = build_motif_analysis(synthetic_spectrum, config=config)
    assert report.has_errors
    assert report.codes("error") == [CODE_CONFIG]


def test_04d_matrix_source_requires_the_phase02_artifact(
    synthetic_workspace: Path, tmp_path: Path, caplog
) -> None:
    eigen = synthetic_workspace / "synthetic.gv" / "eigen.json"
    outdir = tmp_path / "matrix"
    with caplog.at_level("ERROR"):
        assert main(["-i", str(eigen), "-o", str(outdir), "--pathway-source", "matrix"]) == 1
    assert any(CODE_MATRIX_MISSING in record.message for record in caplog.records)
    assert not outdir.exists()
    assert pathway_matrix_path(eigen).name == "z_matrix.json"


def test_04d_both_source_records_z_contributions(
    synthetic_workspace: Path, tmp_path: Path, monkeypatch
) -> None:
    import src.clustering.functional_motifs as fm

    nx = pytest.importorskip("networkx")
    eigen = synthetic_workspace / "synthetic.gv" / "eigen.json"
    z = np.full((len(NEURON_IDS), len(NEURON_IDS)), -0.25)
    monkeypatch.setattr(fm, "load_pathway_matrix", lambda _source: (z, list(NEURON_IDS)))
    outdir = tmp_path / "both"
    assert main(["-i", str(eigen), "-o", str(outdir), "--pathway-source", "both"]) == 0
    payload = json.loads((outdir / "synthetic" / "motifs.json").read_text(encoding="utf-8"))
    assert payload["pathway"]["source"] == "both"
    sizes = {node["node_id"]: node["size"] for node in payload["pathway"]["nodes"]}
    graph = nx.read_graphml(outdir / "synthetic" / "motifs.pathway.graphml")
    assert graph.graph["pathway_source"] == "both"
    assert graph.number_of_edges() == payload["pathway"]["n_edges"]
    for source, target, attributes in graph.edges(data=True):
        assert np.isclose(attributes["z_contribution"], -0.25 * sizes[source] * sizes[target])


def test_04d_no_graphml_cli_flag(synthetic_workspace: Path, tmp_path: Path) -> None:
    eigen = synthetic_workspace / "synthetic.gv" / "eigen.json"
    outdir = tmp_path / "nogml"
    assert main(["-i", str(eigen), "-o", str(outdir), "--no-graphml"]) == 0
    assert not (outdir / "synthetic" / "motifs.pathway.graphml").exists()
    assert (outdir / "synthetic" / "motifs.json").is_file()


# ---------------------------------------------------------------------------
# Integration against the real artifact (deselect with -m "not slow")
# ---------------------------------------------------------------------------
REFERENCE_MEMBER_COUNTS = [
    33, 23, 18, 30, 16, 12, 19, 22, 11, 43, 41, 19, 60, 16, 17, 18, 32, 39, 55, 39, 60
]


@pytest.mark.slow
def test_reference_motif_numbers() -> None:
    spectrum = load_spectrum(REFERENCE_JSON)
    analysis, report = build_motif_analysis(spectrum, source_artifact=REFERENCE_JSON)
    assert not report.has_errors, [issue.message for issue in report.errors]
    assert report.codes("warning") == []
    metadata = analysis.metadata
    assert (metadata["n_neurons"], metadata["n_modes"], metadata["n_motifs"]) == (113, 21, 21)
    assert [motif.n_members for motif in analysis.motifs] == REFERENCE_MEMBER_COUNTS
    assert metadata["n_members_total"] == 623
    assert (metadata["n_members_min"], metadata["n_members_max"]) == (11, 60)
    assert metadata["n_members_median"] == 23.0
    assert metadata["n_members_mean"] == pytest.approx(29.666666666666668)
    assert metadata["n_unique_members"] == 107
    assert metadata["n_neurons_in_no_motif"] == 6
    assert metadata["membership_density"] == pytest.approx(0.26253687315634217)
    assert metadata["n_isolated_neurons"] == 1
    assert metadata["recurrence_max"] == 13
    assert (metadata["n_recurrence_ge_1"], metadata["n_recurrence_ge_2"], metadata["n_recurrence_ge_5"]) == (
        107,
        99,
        70,
    )
    assert metadata["n_regions"] == 31
    top = [(entry["neuron_id"], entry["recurrence"]) for entry in metadata["top_recurrent"]]
    assert top[:5] == [
        ("FB4Z_R_1", 13),
        ("hDeltaA_12_C10_1", 13),
        ("FB4Z_R_2", 12),
        ("hDeltaA_08_C7_1", 12),
        ("hDeltaA_10_C8_1", 12),
    ]
    assert metadata["n_links"] == 187 and metadata["n_groups"] == 22
    assert metadata["n_pathway_nodes"] == 22 and metadata["n_pathway_edges"] == 39
    assert validate_motif_payload_schema(analysis.to_dict()) == []
    assert len(analysis.families) == 18
    assert sorted(mode for family in analysis.families for mode in family["modes"]) == list(range(1, 22))


@pytest.mark.slow
def test_reference_tracking_numbers() -> None:
    """04B owns these numbers: 187 links, 1/2/133/51 classes, 18 families."""
    spectrum = load_spectrum(REFERENCE_JSON)
    analysis, report = build_motif_analysis(spectrum, source_artifact=REFERENCE_JSON)
    assert not report.has_errors, [issue.message for issue in report.errors]
    metadata = analysis.metadata
    assert metadata["n_links"] == 187
    assert metadata["link_classes"] == {
        "stable": 1,
        "flipped": 2,
        "composite": 133,
        "weak": 51,
    }
    assert metadata["n_families"] == 18
    assert metadata["n_multi_mode_families"] == 3
    multi = [family["modes"] for family in analysis.families if family["n_modes"] > 1]
    assert multi == [[2, 15], [8, 13], [11, 21]]
    # the families partition all 21 modes and family_of_mode agrees with them
    covered = [mode for family in analysis.families for mode in family["modes"]]
    assert sorted(covered) == list(range(1, 22))
    assert analysis.family_of_mode.tolist() == [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 7, 12, 1, 13, 14, 15, 16, 17, 10
    ]
    # the merged components use exactly the stable/flipped links
    merging = [
        (link["source_mode"], link["target_mode"])
        for link in analysis.links
        if link["classification"] in ("stable", "flipped")
    ]
    assert sorted(merging) == [(2, 15), (8, 13), (11, 21)]
    # the two link matrices are symmetric with a unit diagonal
    jaccard = analysis.arrays["link_jaccard"]
    polarity = analysis.arrays["link_polarity"]
    assert jaccard.shape == (21, 21) and polarity.shape == (21, 21)
    assert np.array_equal(jaccard, jaccard.T) and np.array_equal(polarity, polarity.T)
    assert np.allclose(jaccard.diagonal(), 1.0) and np.allclose(polarity.diagonal(), 1.0)
    assert validate_motif_payload_schema(analysis.to_dict()) == []


@pytest.mark.slow
def test_reference_eigen_path_variant() -> None:
    variant = REFERENCE_JSON.parent / "eigen.f8652585.json"
    if not variant.is_file():
        pytest.skip("the eigen-path variant artifact is not present")
    spectrum = load_spectrum(variant)
    analysis, report = build_motif_analysis(spectrum, source_artifact=variant)
    assert not report.has_errors
    assert analysis.n_modes == 25
    # eigen path: both bases are the same eigenvector matrix
    assert np.allclose(spectrum.loadings_left, spectrum.loadings_right)
    assert validate_motif_payload_schema(analysis.to_dict()) == []


@pytest.mark.slow
def test_reference_pathway_numbers() -> None:
    """04D owns these numbers: 22 nodes / 39 edges, 0/39 polarity, 7/32 intra."""
    spectrum = load_spectrum(REFERENCE_JSON)
    analysis, report = build_motif_analysis(spectrum, source_artifact=REFERENCE_JSON)
    assert not report.has_errors, [issue.message for issue in report.errors]
    pathway = analysis.pathway
    metadata = analysis.metadata
    assert pathway["n_nodes"] == 22 and pathway["n_edges"] == 39
    assert metadata["n_pathway_nodes"] == 22 and metadata["n_pathway_edges"] == 39
    assert pathway["source"] == "mode" and pathway["weight"] == "evr"
    assert pathway["edge_threshold"] == 0.1 and pathway["top_edges"] == 5
    assert pathway["intra"] is True
    assert pathway["abs_max"] == pytest.approx(2.345299756658418)
    assert metadata["pathway_abs_max"] == pytest.approx(2.345300)
    assert pathway["n_positive"] == 0 and pathway["n_negative"] == 39
    assert pathway["n_intra"] == 7 and pathway["n_cross"] == 32
    assert pathway["weight_concentration"] == pytest.approx(0.055842135916593295)
    # the smallest kept edge is the 39th largest contribution
    assert min(abs(weight) for weight in pathway["weights"]) == pytest.approx(0.23597480790052852)
    # the threshold sweep is a threshold-only diagnostic on the abs-max scale
    assert analysis.diagnostics["pathway"]["threshold_sweep"] == {
        "0.05": 91,
        "0.1": 46,
        "0.2": 17,
        "0.25": 14,
    }
    # per-source top-N counts and the intra edges
    assert analysis.diagnostics["pathway"]["per_source_counts"] == {
        1: 2, 2: 2, 3: 2, 4: 5, 6: 2, 7: 5, 8: 5, 9: 5, 11: 1, 12: 1, 17: 1, 18: 5, 20: 2, 21: 1
    }
    intra = sorted(edge["source"] for edge in pathway["edges"] if edge["is_intra"])
    assert intra == ["G01", "G02", "G03", "G04", "G06", "G09", "G18"]
    strongest = max(pathway["edges"], key=lambda edge: abs(edge["weight"]))
    assert (strongest["source"], strongest["target"]) == ("G04", "G01")
    assert strongest["weight"] == pytest.approx(-2.345299756658418)
    assert strongest["top_modes"][0]["mode"] == 1
    assert strongest["top_modes"][0]["share"] == pytest.approx(0.9911640200384335)
    # every kept edge carries all 21 mode contributions on this dataset
    assert all(edge["n_modes"] == 21 for edge in pathway["edges"])
    isolated = sorted(
        node["node_id"]
        for node in pathway["nodes"]
        if not any(
            edge["source"] == node["node_id"] or edge["target"] == node["node_id"]
            for edge in pathway["edges"]
        )
    )
    assert isolated == ["G00", "G10", "G13", "G15", "G16", "G19"]
    assert validate_motif_payload_schema(analysis.to_dict()) == []
    # the npz arrays agree with the payload
    adjacency = analysis.arrays["pathway_adjacency"]
    assert adjacency.shape == (22, 22)
    assert int(np.count_nonzero(adjacency)) == 39
    assert np.abs(adjacency).max() == pytest.approx(2.345299756658418)
    assert np.allclose(analysis.arrays["pathway_adjacency_abs"], np.abs(adjacency))


@pytest.mark.slow
def test_reference_pathway_graphml(tmp_path: Path) -> None:
    nx = pytest.importorskip("networkx")
    spectrum = load_spectrum(REFERENCE_JSON)
    analysis, report = build_motif_analysis(spectrum, source_artifact=REFERENCE_JSON)
    assert not report.has_errors
    written = write_artifact_set(analysis, tmp_path)
    assert sorted(path.name for path in written["graphml"].parent.iterdir()) == [
        "motifs.json",
        "motifs.npz",
        "motifs.pathway.graphml",
    ]
    graph = nx.read_graphml(written["graphml"])
    assert graph.number_of_nodes() == 22 and graph.number_of_edges() == 39
    assert graph.graph["pathway_source"] == "mode"
    assert graph.graph["threshold"] == 0.1 and graph.graph["topN"] == 5
    assert graph.nodes["G00"]["is_background"] is True
    assert "dominant_mode" not in graph.nodes["G00"]
    assert graph.nodes["G01"]["size"] == 6
    edge = graph.edges["G04", "G01"]
    assert edge["weight"] == pytest.approx(-2.345299756658418)
    assert edge["intra_flag"] is False and edge["polarity"] == -1
    assert "z_contribution" not in edge
    assert len(json.loads(edge["contribution_by_mode"])) == PATHWAY_TOP_MODES
    centroid = json.loads(graph.nodes["G01"]["centroid"])
    assert centroid == pytest.approx(analysis.arrays["group_centroids"][1].tolist())
    assert np.allclose(json.loads(graph.nodes["G00"]["centroid"]), 0.0)
    # the whole artifact set round-trips through load_motifs
    loaded = load_motifs(written["json"])
    assert loaded.to_dict() == analysis.to_dict()
