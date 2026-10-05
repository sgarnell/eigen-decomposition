"""Unit tests for the Phase 04 Update B unified pathway scorer (S-score).

The component functions are tested on hand-computed fixtures; the integration tests build
real ``motifs.*`` artifacts from the synthetic spectrum and the reference numbers against
the committed artifacts are marked ``slow`` (deselect with ``-m "not slow"``).
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

import numpy as np

from src.clustering.motif_analysis import (
    CODE_SCORE_CONFIG,
    CODE_SCORE_EMPTY,
    CODE_SCORE_INPUT,
    CODE_SCORE_MATRIX,
    CSV_COLUMNS,
    DEFAULT_W_BALANCE,
    DEFAULT_W_CONCENTRATION,
    DEFAULT_W_DISAGREEMENT,
    DEFAULT_W_ENTROPY,
    SCORE_STEM,
    SCORE_VERSION,
    PathwayScore,
    ScoreConfig,
    ScoreValidationError,
    build_pathway_matrix_contributions,
    discover_motif_artifacts,
    load_motif_artifact,
    main,
    matrix_artifact_candidates,
    matrix_artifact_path,
    mode_matrix_disagreement,
    polarity_balance,
    polarity_entropy,
    rank_records,
    recover_mode_contribution,
    render_ranking,
    render_score_table,
    s_score,
    score_directory,
    score_pathway,
    weight_concentration_term,
    write_summary,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = PROJECT_ROOT / "data" / "processed" / "FB4Yaffect_FB45_999prePost_001_all"
REFERENCE_MOTIFS = REFERENCE_DIR / "motifs.json"
OUTDIR_DIR = PROJECT_ROOT / "outdir" / "FB4Yaffect_FB45_999prePost_001_all"


def _payload(
    *,
    edges: list[dict[str, object]] | None = None,
    weight_concentration: float | None = 0.05,
    rule: str = "signed",
    source: str = "mode",
    alpha: float = 0.5,
    n_nodes: int = 2,
    declared: tuple[int | None, int | None] = (None, None),
) -> dict[str, object]:
    """A minimal, schema-shaped motif payload for the unit tests."""
    edge_records = list(edges if edges is not None else [])
    return {
        "config": {
            "config_hash": "deadbeef",
            "stage": "04D",
            "pathway_weight_rule": rule,
            "pathway_polarity_rule": "sign",
            "pathway_hybrid_alpha": alpha,
            "pathway_source": source,
        },
        "neuron_order": ["n0", "n1"],
        "groups": [
            {"group_id": "G00", "members": ["n0"]},
            {"group_id": "G01", "members": ["n1"]},
        ],
        "pathway": {
            "nodes": [{"node_id": "G00"}, {"node_id": "G01"}],
            "edges": edge_records,
            "n_nodes": n_nodes,
            "n_edges": len(edge_records),
            "n_positive": declared[0],
            "n_negative": declared[1],
            "n_intra": 0,
            "n_cross": len(edge_records),
            "abs_max": 1.0,
            "weight_concentration": weight_concentration,
            "source": source,
            "weight": "evr",
            "weights": [edge.get("weight") for edge in edge_records],
        },
    }


def _edge(source: str, target: str, weight: float, polarity: int) -> dict[str, object]:
    return {
        "source": source,
        "target": target,
        "weight": weight,
        "abs_weight": abs(weight),
        "polarity": polarity,
        "n_modes": 1,
        "modes": [1],
        "is_intra": source == target,
        "top_modes": [],
    }


# ---------------------------------------------------------------------------
# ScoreConfig
# ---------------------------------------------------------------------------
def test_score_config_defaults_and_hash() -> None:
    config = ScoreConfig()
    assert config.is_default()
    assert config.hash_fields() == {
        "w_balance": DEFAULT_W_BALANCE,
        "w_entropy": DEFAULT_W_ENTROPY,
        "w_disagreement": DEFAULT_W_DISAGREEMENT,
        "w_concentration": DEFAULT_W_CONCENTRATION,
    }
    assert (DEFAULT_W_BALANCE, DEFAULT_W_ENTROPY) == (1.0, 1.0)
    assert (DEFAULT_W_DISAGREEMENT, DEFAULT_W_CONCENTRATION) == (1.0, 1.0)
    assert config.score_hash() == "7835a30e"
    assert len(config.score_hash()) == 8


def test_score_config_to_dict_records_the_definition() -> None:
    block = ScoreConfig(w_balance=2.0).to_dict()
    assert block["w_balance"] == 2.0
    assert block["score_hash"] == ScoreConfig(w_balance=2.0).score_hash()
    assert "S = w_B*B" in block["definition"]


def test_score_config_hash_changes_with_every_weight() -> None:
    hashes = {
        ScoreConfig().score_hash(),
        ScoreConfig(w_balance=2.0).score_hash(),
        ScoreConfig(w_entropy=2.0).score_hash(),
        ScoreConfig(w_disagreement=2.0).score_hash(),
        ScoreConfig(w_concentration=2.0).score_hash(),
    }
    assert len(hashes) == 5
    assert not ScoreConfig(w_balance=2.0).is_default()


# ---------------------------------------------------------------------------
# polarity balance / entropy / concentration
# ---------------------------------------------------------------------------
def test_polarity_balance_zero_edges_is_none() -> None:
    assert polarity_balance(0, 0, 0) is None


def test_polarity_balance_hand_computed_cases() -> None:
    assert polarity_balance(0, 39, 39) == 0.0
    assert polarity_balance(39, 0, 39) == 0.0
    assert polarity_balance(20, 20, 40) == 1.0
    assert polarity_balance(30, 10, 40) == pytest.approx(0.5)
    assert polarity_balance(13, 65, 78) == pytest.approx(1 - 52 / 78)


def test_polarity_balance_counts_zero_polarity_edges_in_the_denominator() -> None:
    # 2 positive, 2 negative, 2 zero -> |2-2|/6 = 0
    assert polarity_balance(2, 2, 6) == 1.0


def test_polarity_entropy_no_edges_is_none() -> None:
    assert polarity_entropy(0, 0, 0) is None


def test_polarity_entropy_single_class_is_zero() -> None:
    assert polarity_entropy(0, 39, 0) == 0.0
    assert polarity_entropy(16, 0, 0) == 0.0


def test_polarity_entropy_two_classes_is_log_two() -> None:
    assert polarity_entropy(20, 20, 0) == pytest.approx(math.log(2.0))


def test_polarity_entropy_three_equal_classes_is_log_three() -> None:
    assert polarity_entropy(1, 1, 1) == pytest.approx(math.log(3.0))


def test_polarity_entropy_skips_absent_classes() -> None:
    # p_zero = 0 must not contribute a NaN from 0 * log 0.
    value = polarity_entropy(12, 63, 0)
    assert value is not None and math.isfinite(value)
    assert value == pytest.approx(
        -(12 / 75) * math.log(12 / 75) - (63 / 75) * math.log(63 / 75)
    )


def test_weight_concentration_term_reads_the_payload() -> None:
    assert weight_concentration_term({"weight_concentration": 0.25}) == 0.25
    assert weight_concentration_term({"weight_concentration": None}) is None
    assert weight_concentration_term({}) is None


def test_weight_concentration_term_rejects_non_finite() -> None:
    with pytest.raises(ScoreValidationError):
        weight_concentration_term({"weight_concentration": float("nan")})



# ---------------------------------------------------------------------------
# W_mode recovery and the disagreement metric
# ---------------------------------------------------------------------------
def test_recover_mode_contribution_signed_both() -> None:
    mode, reason = recover_mode_contribution(-1.5, -0.5, rule="signed", source="both")
    assert mode == pytest.approx(-1.0)
    assert reason is None


def test_recover_mode_contribution_mode_only() -> None:
    mode, reason = recover_mode_contribution(-1.5, None, rule="signed", source="mode")
    assert mode is None
    assert reason == "mode-only: no matrix contribution"


def test_recover_mode_contribution_matrix_only() -> None:
    mode, reason = recover_mode_contribution(-1.5, -1.5, rule="matrix-only", source="matrix")
    assert mode is None
    assert "no mode contribution stored" in str(reason)


def test_recover_mode_contribution_hybrid_alpha_boundaries() -> None:
    # hybrid alpha=1 is exactly the mode-only recipe.
    mode, _ = recover_mode_contribution(-1.0, -4.0, rule="hybrid", source="hybrid", alpha=1.0)
    assert mode == pytest.approx(-1.0)
    # alpha = 0.5: W = 0.5*W_mode + 0.5*W_matrix
    mode, _ = recover_mode_contribution(-2.0, -4.0, rule="hybrid", source="hybrid", alpha=0.5)
    assert mode == pytest.approx(0.0)
    # alpha = 0 degenerates to matrix-only.
    mode, reason = recover_mode_contribution(
        -4.0, -4.0, rule="hybrid", source="hybrid", alpha=0.0
    )
    assert mode is None
    assert "hybrid alpha 0" in str(reason)


def test_recover_mode_contribution_unknown_rule() -> None:
    mode, reason = recover_mode_contribution(1.0, 1.0, rule="nonsense", source="mode")
    assert mode is None
    assert "unknown pathway_weight_rule" in str(reason)


def test_mode_matrix_disagreement_hand_computed() -> None:
    edges = [
        _edge("G00", "G01", -2.0, -1),  # W_mode = -1, W_matrix = -1 -> agree
        _edge("G01", "G00", 1.0, 1),    # W_mode = +2, W_matrix = -1 -> disagree
        _edge("G00", "G00", 0.0, 0),    # W_mode =  0, W_matrix = -1 -> disagree
        _edge("G01", "G01", 3.0, 1),    # W_mode = +3, W_matrix =  0 -> disagree
    ]
    contributions = {
        ("G00", "G01"): -1.0,
        ("G01", "G00"): -1.0,
        ("G00", "G00"): -1.0,
        ("G01", "G01"): 0.0,
    }
    value, reason = mode_matrix_disagreement(
        edges, contributions, rule="signed", source="both"
    )
    assert value == pytest.approx(3 / 4)
    assert reason is None


def test_mode_matrix_disagreement_no_edges() -> None:
    value, reason = mode_matrix_disagreement([], {}, rule="signed", source="both")
    assert value is None
    assert reason == "no pathway edges"


def test_mode_matrix_disagreement_is_none_for_matrix_only() -> None:
    edges = [_edge("G00", "G01", -1.0, -1)]
    value, reason = mode_matrix_disagreement(
        edges, {("G00", "G01"): -1.0}, rule="matrix-only", source="matrix"
    )
    assert value is None
    assert "matrix-only" in str(reason)


def test_mode_matrix_disagreement_missing_matrix_is_none() -> None:
    edges = [_edge("G00", "G01", -1.0, -1)]
    value, reason = mode_matrix_disagreement(edges, None, rule="signed", source="both")
    assert value is None
    assert reason == "z_matrix artifact unavailable"


# ---------------------------------------------------------------------------
# W_matrix rebuild
# ---------------------------------------------------------------------------
def test_build_pathway_matrix_contributions_matches_brute_force() -> None:
    matrix = np.array(
        [
            [0.0, 1.0, -2.0, 0.0],
            [0.5, 0.0, 3.0, -1.0],
            [-1.0, 2.0, 0.0, 4.0],
            [0.0, -3.0, 1.0, 0.0],
        ]
    )
    order = ["n0", "n1", "n2", "n3"]
    groups = [
        {"group_id": "G00", "members": ["n0", "n2"]},
        {"group_id": "G01", "members": ["n1"]},
        {"group_id": "G02", "members": ["n3"]},
    ]
    contributions = build_pathway_matrix_contributions(groups, matrix, order)
    expected = {
        ("G00", "G00"): matrix[np.ix_([0, 2], [0, 2])].sum(),
        ("G00", "G01"): matrix[np.ix_([0, 2], [1])].sum(),
        ("G00", "G02"): matrix[np.ix_([0, 2], [3])].sum(),
        ("G01", "G00"): matrix[np.ix_([1], [0, 2])].sum(),
        ("G02", "G02"): matrix[np.ix_([3], [3])].sum(),
    }
    for key, value in expected.items():
        assert contributions[key] == pytest.approx(float(value))
    assert len(contributions) == 9


def test_build_pathway_matrix_contributions_rejects_a_bad_shape() -> None:
    with pytest.raises(ScoreValidationError):
        build_pathway_matrix_contributions([], np.zeros((2, 3)), ["n0", "n1"])
    with pytest.raises(ScoreValidationError):
        build_pathway_matrix_contributions(
            [{"group_id": "G00", "members": ["n0"]}], np.zeros((3, 3)), ["n0"]
        )


# ---------------------------------------------------------------------------
# s_score
# ---------------------------------------------------------------------------
def test_s_score_definition() -> None:
    assert s_score(0.4, 0.3, 0.2, 0.1) == pytest.approx(0.4 + 0.3 + 0.2 - 0.1)


def test_s_score_drops_undefined_terms() -> None:
    assert s_score(0.5, 0.5, None, 0.0) == pytest.approx(1.0)
    # dropping D differs from substituting zero only when a run really disagrees
    assert s_score(0.5, 0.5, None, 0.0) != pytest.approx(0.5 + 0.5 + 0.5)


def test_s_score_is_none_when_nothing_is_defined() -> None:
    assert s_score(None, None, None, None) is None


def test_s_score_honours_custom_weights() -> None:
    config = ScoreConfig(w_balance=2.0, w_entropy=0.0, w_disagreement=0.0, w_concentration=3.0)
    assert s_score(0.5, 0.9, 0.9, 0.2, config) == pytest.approx(2.0 * 0.5 - 3.0 * 0.2)


def test_s_score_canonicalises_negative_zero() -> None:
    assert math.copysign(1.0, s_score(0.0, 0.0, None, 0.0) or 0.0) == 1.0



# ---------------------------------------------------------------------------
# score_pathway
# ---------------------------------------------------------------------------
def test_score_pathway_mode_only_run() -> None:
    payload = _payload(
        edges=[_edge("G00", "G01", -0.6, -1), _edge("G01", "G00", -0.4, -1)],
        weight_concentration=0.05584213591659331,
        declared=(0, 2),
    )
    record = score_pathway(payload, artifact="motifs.json")
    assert record.balance == 0.0
    assert record.entropy == 0.0
    assert record.disagreement is None
    assert record.d_unavailable_reason == "mode-only: no matrix contribution"
    assert record.concentration == pytest.approx(0.05584213591659331)
    assert record.s_score == pytest.approx(-0.05584213591659331)
    assert record.weights_used == ScoreConfig().hash_fields()
    assert record.config_hash == "deadbeef"
    assert record.stage == "04D"


def test_score_pathway_records_the_resolved_rules_and_counts() -> None:
    payload = _payload(
        edges=[
            _edge("G00", "G01", 1.0, 1),
            _edge("G01", "G00", -1.0, -1),
            _edge("G00", "G00", 0.0, 0),
        ],
        rule="signed",
        source="both",
        declared=(1, 1),
    )
    record = score_pathway(
        payload,
        matrix_contributions={("G00", "G01"): 2.0, ("G01", "G00"): -1.0, ("G00", "G00"): 1.0},
        artifact="motifs.json",
    )
    assert (record.n_positive, record.n_negative, record.n_zero) == (1, 1, 1)
    assert record.balance == pytest.approx(1.0)
    assert record.entropy == pytest.approx(math.log(3.0))
    # W_mode = 1-2 = -1 (vs +2), -1+1 = 0 (vs -1), 0-1 = -1 (vs +1) -> all three disagree
    assert record.disagreement == pytest.approx(1.0)
    assert record.s_score == pytest.approx(1.0 + math.log(3.0) + 1.0 - 0.05)


def test_score_pathway_rejects_a_declared_count_mismatch() -> None:
    payload = _payload(edges=[_edge("G00", "G01", -1.0, -1)], declared=(1, 0))
    with pytest.raises(ScoreValidationError, match=CODE_SCORE_INPUT):
        score_pathway(payload, artifact="motifs.json")


def test_score_pathway_empty_pathway_is_unscored() -> None:
    payload = _payload(edges=[], weight_concentration=0.0)
    record = score_pathway(payload, artifact="motifs.json")
    assert record.s_score is None
    assert record.balance is None
    assert record.d_unavailable_reason == "no pathway edges"
    assert any(CODE_SCORE_EMPTY in issue for issue in record.issues)


def test_score_pathway_warns_when_the_matrix_is_missing() -> None:
    from src.parsing.parse_graphviz import ValidationReport

    payload = _payload(edges=[_edge("G00", "G01", -1.0, -1)], rule="signed", source="both")
    report = ValidationReport()
    record = score_pathway(
        payload, matrix_contributions=None, artifact="motifs.json", report=report
    )
    assert record.disagreement is None
    assert "z_matrix artifact not found" in str(record.d_unavailable_reason)
    assert report.codes() == [CODE_SCORE_MATRIX]


def test_score_pathway_rejects_negative_weights() -> None:
    payload = _payload(edges=[_edge("G00", "G01", -1.0, -1)])
    with pytest.raises(ScoreValidationError, match=CODE_SCORE_CONFIG):
        score_pathway(payload, config=ScoreConfig(w_balance=-1.0), artifact="motifs.json")


def test_score_pathway_flags_an_ill_conditioned_hybrid_alpha() -> None:
    payload = _payload(
        edges=[_edge("G00", "G01", -1.0, -1)], rule="hybrid", source="hybrid", alpha=0.05
    )
    record = score_pathway(
        payload, matrix_contributions={("G00", "G01"): -1.0}, artifact="motifs.json"
    )
    assert record.disagreement is not None
    assert any("ill-conditioned" in issue for issue in record.issues)



# ---------------------------------------------------------------------------
# discovery and artifact paths
# ---------------------------------------------------------------------------
def test_matrix_artifact_candidates_try_the_variant_before_the_canonical() -> None:
    assert [p.name for p in matrix_artifact_candidates("d/motifs.json")] == ["z_matrix.json"]
    assert [p.name for p in matrix_artifact_candidates("d/motifs.f8652585.json")] == [
        "z_matrix.f8652585.json",
        "z_matrix.json",
    ]
    assert [p.name for p in matrix_artifact_candidates("d/motifs.f8652585.e973fece.json")] == [
        "z_matrix.f8652585.e973fece.json",
        "z_matrix.f8652585.json",
        "z_matrix.json",
    ]


def test_matrix_artifact_path_resolves_by_existence(tmp_path: Path) -> None:
    motif = tmp_path / "motifs.f8652585.e973fece.json"
    motif.write_text("{}", encoding="utf-8")
    # nothing on disk yet -> the best (fully variant-preserving) candidate
    assert matrix_artifact_path(motif).name == "z_matrix.f8652585.e973fece.json"
    real = tmp_path / "z_matrix.f8652585.json"
    real.write_text("{}", encoding="utf-8")
    assert matrix_artifact_path(motif) == real


def test_discover_motif_artifacts_accepts_a_run_directory(tmp_path: Path) -> None:
    for name in (
        "motifs.json",
        "motifs.deadbeef.json",
        "motifs.data.json",
        "motifs.deadbeef.data.json",
        "motifs.npz",
        "motifs.pathway.graphml",
        "z_matrix.json",
        "eigen.json",
    ):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    found = [path.name for path in discover_motif_artifacts(tmp_path)]
    assert found == ["motifs.deadbeef.json", "motifs.json"]
    with_data = [path.name for path in discover_motif_artifacts(tmp_path, include_data=True)]
    assert "motifs.data.json" in with_data
    assert "motifs.deadbeef.data.json" in with_data
    assert "z_matrix.json" not in with_data


def test_discover_motif_artifacts_searches_recursively(tmp_path: Path) -> None:
    nested = tmp_path / "stem"
    nested.mkdir()
    (nested / "motifs.json").write_text("{}", encoding="utf-8")
    assert [path.name for path in discover_motif_artifacts(tmp_path)] == ["motifs.json"]


def test_discover_motif_artifacts_accepts_a_single_file(tmp_path: Path) -> None:
    target = tmp_path / "motifs.deadbeef.json"
    target.write_text("{}", encoding="utf-8")
    assert discover_motif_artifacts(target) == [target]


def test_discover_motif_artifacts_rejects_a_foreign_file(tmp_path: Path) -> None:
    target = tmp_path / "eigen.json"
    target.write_text("{}", encoding="utf-8")
    with pytest.raises(ScoreValidationError, match=CODE_SCORE_INPUT):
        discover_motif_artifacts(target)


def test_load_motif_artifact_validates_the_required_blocks(tmp_path: Path) -> None:
    good = tmp_path / "motifs.json"
    good.write_text(json.dumps(_payload()), encoding="utf-8")
    assert load_motif_artifact(good)["pathway"]["n_edges"] == 0
    bad = tmp_path / "motifs.bad.json"
    bad.write_text(json.dumps({"config": {}}), encoding="utf-8")
    with pytest.raises(ScoreValidationError, match=CODE_SCORE_INPUT):
        load_motif_artifact(bad)
    mismatched = tmp_path / "motifs.mismatch.json"
    payload = _payload()
    payload["pathway"]["n_edges"] = 7  # type: ignore[index]
    mismatched.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ScoreValidationError, match="declares n_edges"):
        load_motif_artifact(mismatched)



# ---------------------------------------------------------------------------
# ranking and the summary document
# ---------------------------------------------------------------------------
def test_rank_records_orders_by_score_and_breaks_ties_on_the_name() -> None:
    records = [
        PathwayScore(artifact="b.json", s_score=1.0),
        PathwayScore(artifact="a.json", s_score=1.0),
        PathwayScore(artifact="c.json", s_score=2.0),
        PathwayScore(artifact="d.json", s_score=None),
    ]
    ranked = rank_records(records)
    assert [record.artifact for record in ranked] == ["c.json", "a.json", "b.json", "d.json"]
    assert [record.rank for record in ranked] == [1, 2, 3, None]


def test_render_ranking_and_table_agree() -> None:
    record = PathwayScore(
        artifact="motifs.json",
        n_edges=39,
        balance=0.0,
        entropy=0.0,
        disagreement=None,
        concentration=0.05584213591659331,
        s_score=-0.05584213591659331,
        rank=1,
    )
    table = render_score_table([record])
    assert table == render_ranking([record.to_dict()])
    assert "motifs.json" in table
    assert "-0.055842" in table
    assert "n/a" in table
    assert "no motif artifact matched" in render_ranking([])


def test_score_directory_ranks_a_synthetic_run(tmp_path: Path) -> None:
    (tmp_path / "motifs.json").write_text(
        json.dumps(
            _payload(
                edges=[_edge("G00", "G01", -1.0, -1), _edge("G01", "G00", -1.0, -1)],
                declared=(0, 2),
            )
        ),
        encoding="utf-8",
    )
    (tmp_path / "motifs.deadbeef.json").write_text(
        json.dumps(
            _payload(
                edges=[_edge("G00", "G01", -1.0, -1), _edge("G01", "G00", 1.0, 1)],
                declared=(1, 1),
            )
        ),
        encoding="utf-8",
    )
    summary = score_directory(tmp_path)
    rankings = summary["rankings"]
    assert summary["summary"]["n_scored"] == 2
    assert rankings[0]["artifact"] == "motifs.deadbeef.json"
    assert rankings[0]["s_score"] > rankings[1]["s_score"]
    assert rankings[0]["rank"] == 1
    assert rankings[1]["artifact"] == "motifs.json"
    assert summary["metadata"]["n_errors"] == 0
    assert set(summary) == {"provenance", "scorer_config", "summary", "rankings", "metadata"}
    assert summary["provenance"]["scorer_version"] == SCORE_VERSION


def test_score_directory_reports_a_broken_artifact_without_aborting(tmp_path: Path) -> None:
    (tmp_path / "motifs.good.json").write_text(json.dumps(_payload()), encoding="utf-8")
    (tmp_path / "motifs.bad.json").write_text("{not json", encoding="utf-8")
    summary = score_directory(tmp_path)
    assert summary["summary"]["n_artifacts"] == 2
    assert summary["metadata"]["n_errors"] == 1
    assert summary["metadata"]["issues"][0]["severity"] == "error"


def test_score_directory_is_empty_for_an_empty_directory(tmp_path: Path) -> None:
    summary = score_directory(tmp_path)
    assert summary["summary"]["n_artifacts"] == 0
    assert summary["rankings"] == []
    assert summary["summary"]["s_max"] is None


# ---------------------------------------------------------------------------
# write_summary
# ---------------------------------------------------------------------------
def test_write_summary_writes_json_only_by_default(tmp_path: Path) -> None:
    (tmp_path / "motifs.json").write_text(json.dumps(_payload()), encoding="utf-8")
    summary = score_directory(tmp_path)
    paths = write_summary(tmp_path, summary)
    assert set(paths) == {"json"}
    assert paths["json"].name == f"{SCORE_STEM}.json"
    assert not list(tmp_path.rglob("*.csv"))
    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert payload["scorer_config"]["score_hash"] == ScoreConfig().score_hash()


def test_write_summary_csv_is_opt_in_and_has_the_fixed_columns(tmp_path: Path) -> None:
    (tmp_path / "motifs.json").write_text(json.dumps(_payload()), encoding="utf-8")
    summary = score_directory(tmp_path)
    paths = write_summary(tmp_path, summary, csv=True)
    assert set(paths) == {"json", "csv"}
    rows = paths["csv"].read_text(encoding="utf-8").splitlines()
    assert rows[0] == ",".join(CSV_COLUMNS)
    assert len(rows) == 1 + len(summary["rankings"])
    assert rows[1].split(",")[1] == "motifs.json"


def test_write_summary_is_byte_deterministic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "1700000000")
    (tmp_path / "motifs.json").write_text(json.dumps(_payload()), encoding="utf-8")
    first = write_summary(tmp_path, score_directory(tmp_path))["json"].read_bytes()
    second = write_summary(tmp_path, score_directory(tmp_path))["json"].read_bytes()
    assert first == second
    tokens = first.replace(b",", b" ").replace(b"\n", b" ").split()
    assert b"-0.0" not in tokens
    assert b"NaN" not in first and b"Infinity" not in first



# ---------------------------------------------------------------------------
# integration with the real Phase 04 pipeline (synthetic spectrum)
# ---------------------------------------------------------------------------
@pytest.fixture
def scored_workspace(tmp_path: Path) -> Path:
    """Build real ``motifs.*`` artifacts from the synthetic spectrum and score them."""
    from src.clustering.functional_motifs import (
        MotifConfig,
        build_motif_analysis,
        write_artifact_set,
    )
    from src.spectral.spectral_decomposition import load_spectrum
    from src.utils.io import write_json_atomic
    from tests.test_functional_motifs import make_spectrum

    spectrum = make_spectrum()
    source = tmp_path / "synthetic.gv"
    source.mkdir(parents=True, exist_ok=True)
    write_json_atomic(source / "eigen.json", spectrum.to_dict())
    analysis, report = build_motif_analysis(
        load_spectrum(source / "eigen.json"),
        source_artifact=source / "eigen.json",
        config=MotifConfig(),
    )
    assert not report.has_errors
    write_artifact_set(analysis, tmp_path, save_data=True, graphml=True)
    return tmp_path


def test_scoring_a_real_motif_run_does_not_mutate_its_artifacts(scored_workspace: Path) -> None:
    before = {
        path.name: path.read_bytes()
        for path in sorted(scored_workspace.iterdir())
        if path.is_file()
    }
    summary = score_directory(scored_workspace)
    write_summary(scored_workspace, summary, csv=True)
    after = {
        path.name: path.read_bytes()
        for path in sorted(scored_workspace.iterdir())
        if path.is_file() and path.name.startswith("motifs")
    }
    for name, content in after.items():
        assert content == before[name], f"{name} was modified by the scorer"
    assert summary["summary"]["n_artifacts"] == 1
    assert summary["rankings"][0]["pathway_source"] == "mode"
    assert summary["rankings"][0]["d_unavailable_reason"] == "mode-only: no matrix contribution"
    assert (scored_workspace / f"{SCORE_STEM}.json").is_file()
    assert (scored_workspace / f"{SCORE_STEM}.csv").is_file()


def test_scoring_two_runs_ranks_the_richer_one_first(scored_workspace: Path) -> None:
    from src.clustering.functional_motifs import MotifConfig, build_motif_analysis, write_artifact_set
    from src.spectral.spectral_decomposition import load_spectrum
    from tests.test_functional_motifs import make_spectrum, NEURON_IDS

    spectrum = make_spectrum()
    source_artifact = scored_workspace / "synthetic.gv" / "eigen.json"
    # a second, differently-configured run of the same spectrum (absolute thresholds)
    analysis, report = build_motif_analysis(
        load_spectrum(source_artifact),
        source_artifact=source_artifact,
        config=MotifConfig(threshold_method="absolute", absolute_threshold=0.05),
    )
    assert not report.has_errors
    write_artifact_set(analysis, scored_workspace, force_config_hash=True)
    summary = score_directory(scored_workspace)
    assert summary["summary"]["n_artifacts"] == 2
    assert [record["rank"] for record in summary["rankings"]] == [1, 2]
    assert len({record["s_score"] for record in summary["rankings"]}) >= 1
    assert len(NEURON_IDS) == 8
    assert spectrum.neuron_order == NEURON_IDS


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def test_cli_writes_the_summary_and_exits_zero(scored_workspace: Path) -> None:
    assert main(["-i", str(scored_workspace)]) == 0
    payload = json.loads((scored_workspace / f"{SCORE_STEM}.json").read_text(encoding="utf-8"))
    assert payload["summary"]["n_scored"] == 1
    assert not (scored_workspace / f"{SCORE_STEM}.csv").exists()


def test_cli_dry_run_writes_nothing(scored_workspace: Path) -> None:
    assert main(["-i", str(scored_workspace), "--dry-run", "--stats"]) == 0
    assert not list(scored_workspace.rglob(f"{SCORE_STEM}*"))


def test_cli_csv_flag_writes_the_csv(scored_workspace: Path) -> None:
    assert main(["-i", str(scored_workspace), "--csv"]) == 0
    assert (scored_workspace / f"{SCORE_STEM}.csv").is_file()


def test_cli_no_input_returns_two(tmp_path: Path) -> None:
    assert main(["-i", str(tmp_path)]) == 2


def test_cli_missing_path_returns_two(tmp_path: Path) -> None:
    assert main(["-i", str(tmp_path / "nowhere")]) == 2


def test_cli_rejects_a_negative_weight(tmp_path: Path) -> None:
    (tmp_path / "motifs.json").write_text(json.dumps(_payload()), encoding="utf-8")
    assert main(["-i", str(tmp_path), "--w-balance", "-1"]) == 1


def test_cli_strict_escalates_the_matrix_warning(tmp_path: Path) -> None:
    (tmp_path / "motifs.json").write_text(
        json.dumps(
            _payload(edges=[_edge("G00", "G01", -1.0, -1)], rule="signed", source="both")
        ),
        encoding="utf-8",
    )
    assert main(["-i", str(tmp_path)]) == 0
    assert (tmp_path / f"{SCORE_STEM}.json").is_file()
    (tmp_path / f"{SCORE_STEM}.json").unlink()
    assert main(["-i", str(tmp_path), "--strict"]) == 1
    assert not (tmp_path / f"{SCORE_STEM}.json").exists()


def test_cli_subprocess_smoke(scored_workspace: Path) -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "src.clustering.motif_analysis", "-i", str(scored_workspace)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "Phase 04B -- pathway ranking" in completed.stdout
    assert (scored_workspace / f"{SCORE_STEM}.json").is_file()



# ---------------------------------------------------------------------------
# Integration against the real artifacts (slow)
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_reference_canonical_motif_score() -> None:
    """The default reference run: all-negative pathway, so B = H = 0 and S = -C."""
    if not REFERENCE_MOTIFS.is_file():
        pytest.skip("the reference motifs.json artifact is unavailable")
    summary = score_directory(REFERENCE_DIR)
    assert summary["summary"]["n_artifacts"] == 1
    record = summary["rankings"][0]
    assert record["artifact"] == "motifs.json"
    assert record["artifact_kind"] == "motifs"
    assert record["stage"] == "04D"
    assert record["n_nodes"] == 22
    assert record["n_edges"] == 39
    assert (record["n_positive"], record["n_negative"], record["n_zero"]) == (0, 39, 0)
    assert record["pathway_source"] == "mode"
    assert record["pathway_weight_rule"] == "signed"
    assert record["balance"] == 0.0
    assert record["entropy"] == 0.0
    assert record["disagreement"] is None
    assert record["d_unavailable_reason"] == "mode-only: no matrix contribution"
    assert record["concentration"] == pytest.approx(0.05584213591659331)
    assert record["s_score"] == pytest.approx(-0.05584213591659331)
    assert summary["metadata"]["n_errors"] == 0


@pytest.mark.slow
def test_reference_matrix_only_and_both_scores_rank_above_the_canonical_run(
    tmp_path: Path,
) -> None:
    """The outdir variants: matrix-only rows are unscored on D, the `both` row is scored."""
    variant = OUTDIR_DIR / "motifs.4e9b4ea8.json"
    matrix_only = OUTDIR_DIR / "motifs.086daabf.json"
    matrix = REFERENCE_DIR / "z_matrix.json"
    if not (variant.is_file() and matrix_only.is_file() and matrix.is_file()):
        pytest.skip("the outdir variant artifacts are unavailable")
    for source in (variant, matrix_only):
        (tmp_path / source.name).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "z_matrix.json").write_text(matrix.read_text(encoding="utf-8"), encoding="utf-8")
    summary = score_directory(tmp_path)
    assert summary["summary"]["n_scored"] == 2
    by_name = {record["artifact"]: record for record in summary["rankings"]}

    both = by_name[variant.name]
    assert both["n_edges"] == 75
    assert (both["n_positive"], both["n_negative"]) == (12, 63)
    assert both["balance"] == pytest.approx(0.32)
    assert both["entropy"] == pytest.approx(0.439669879401343)
    assert both["disagreement"] == pytest.approx(0.013333333333333334)
    assert both["d_unavailable_reason"] is None
    assert both["s_score"] == pytest.approx(0.7399899397697458)

    only = by_name[matrix_only.name]
    assert only["n_edges"] == 78
    assert only["disagreement"] is None
    assert "matrix-only" in str(only["d_unavailable_reason"])
    assert only["s_score"] == pytest.approx(0.75231781823471)
    assert summary["rankings"][0]["artifact"] == matrix_only.name


@pytest.mark.slow
def test_reference_scoring_is_byte_deterministic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    if not REFERENCE_MOTIFS.is_file():
        pytest.skip("the reference motifs.json artifact is unavailable")
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "1700000000")
    target = tmp_path / "run"
    target.mkdir()
    (target / "motifs.json").write_text(
        REFERENCE_MOTIFS.read_text(encoding="utf-8"), encoding="utf-8"
    )
    first = write_summary(target, score_directory(target), csv=True)
    snapshot = {key: path.read_bytes() for key, path in first.items()}
    second = write_summary(target, score_directory(target), csv=True)
    for key, path in second.items():
        assert path.read_bytes() == snapshot[key]

