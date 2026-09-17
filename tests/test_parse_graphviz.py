"""Unit tests for the Phase 01 Graphviz parser.

The synthetic fixtures in ``tests/fixtures`` each target one behaviour (happy
path, one validation rule, one warning case).  The integration test against the
real 727 KB dataset is marked ``slow`` and can be deselected with
``-m "not slow"``.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.parsing import parse_graphviz as parser
from src.parsing.parse_graphviz import (
    DEFAULT_ARTIFACT_NAME,
    FILTERED_ARTIFACT_NAME,
    GraphvizValidationError,
    NeuronPair,
    apply_filters,
    artifact_path,
    build_parsed_graph,
    is_hub_name,
    main,
    normalize_label,
    parse_cent,
    parse_graphviz_file,
    parse_z_score,
    resolve_inputs,
    split_hub_name,
    strip_quotes,
    validate_payload_schema,
    write_artifact,
)
from src.utils.io import dumps_json, utc_timestamp, write_json_atomic

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
TINY = FIXTURES / "tiny_valid.gv"
REFERENCE = PROJECT_ROOT / "data" / "raw_dot" / "FB4Yaffect_FB45_999prePost_001_all.gv"

FIXED_EPOCH = "1700000000"
FIXED_TIMESTAMP = "2023-11-14T22:13:20Z"

#: ``(fixture, expected validation error code)`` -- one fixture per error rule.
VALIDATION_CASES = [
    ("missing_post_edge.gv", parser.CODE_HUB_DEGREE),
    ("mismatched_hub_name.gv", parser.CODE_HUB_ENDPOINT_MISMATCH),
    ("self_loop.gv", parser.CODE_SELF_LOOP),
    ("duplicate_pair.gv", parser.CODE_DUPLICATE_PAIR),
    ("hub_in_neurons.gv", parser.CODE_HUB_IN_NEURONS),
    ("endpoint_not_neuron.gv", parser.CODE_ENDPOINT_NOT_NEURON),
    ("negative_z.gv", parser.CODE_Z_INVALID),
]


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("A--B", True),
        ('"A--B"', True),
        ("A", False),
        ("FB4Y(EB/NO1)_R_2", False),
        ("", False),
    ],
)
def test_is_hub_name(name: str, expected: bool) -> None:
    assert is_hub_name(name) is expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ('"A--B"', "A--B"),
        ("A--B", "A--B"),
        ('" A--B "', "A--B"),
        ('"FB4Y(EB/NO1)_R_2"', "FB4Y(EB/NO1)_R_2"),
        ("", ""),
        (None, None),
    ],
)
def test_strip_quotes(value: str | None, expected: str | None) -> None:
    assert strip_quotes(value) == expected


def test_normalize_label_expands_literal_newline_and_escapes() -> None:
    assert normalize_label('"N1\ncent= 0.1"') == "N1\ncent= 0.1"
    assert normalize_label('"N1\\ncent= 0.1"') == "N1\ncent= 0.1"
    assert normalize_label(None) is None


@pytest.mark.parametrize(
    ("name", "expected"),
    [("A--B", ("A", "B")), ("A--B--C", ("A", "B--C")), ("A", None)],
)
def test_split_hub_name(name: str, expected: tuple[str, str] | None) -> None:
    assert split_hub_name(name) == expected


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("N1\ncent= 0.1", (0.1, "0.1")),
        ("N2\ncent=0.2", (0.2, "0.2")),
        ("N3\ncent= 0.10", (0.1, "0.10")),  # raw token is preserved verbatim
        ("N4", (None, None)),
        ("N5\ncent= abc", (None, "abc")),
        (None, (None, None)),
    ],
)
def test_parse_cent(label: str | None, expected: tuple[float | None, str | None]) -> None:
    assert parse_cent(label) == expected


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("Zscore = 0.8", 0.8),
        ("Zscore=0.33", 0.33),
        ('  "Zscore = 0.1"  ', 0.1),
        ("Zscore = 1e-2", 0.01),
        ("Zscore = -0.5", -0.5),
        ("no score here", None),
        (None, None),
    ],
)
def test_parse_z_score(label: str | None, expected: float | None) -> None:
    assert parse_z_score(label) == expected


# ---------------------------------------------------------------------------
# tiny_valid.gv: the happy path
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def tiny_parsed():
    parsed, report = build_parsed_graph(TINY)
    assert not report.issues, report.to_dict()
    return parsed


@pytest.fixture(scope="module")
def tiny_payload(tiny_parsed):
    return tiny_parsed.to_dict()


def test_tiny_metadata_counts(tiny_payload: dict) -> None:
    metadata = tiny_payload["metadata"]
    assert metadata["n_neurons"] == 3
    assert metadata["n_hubs"] == 2
    assert metadata["n_raw_edges"] == 4
    assert metadata["n_pairs"] == 2
    assert metadata["density"] == pytest.approx(2 / (3 * 2))
    assert metadata["reciprocity"] == 0
    assert metadata["filters"] == []
    assert metadata["parser"] == "pydot"
    assert metadata["parser_version"]


def test_tiny_hub_collapse_pairs(tiny_payload: dict) -> None:
    assert tiny_payload["pairs"] == [
        {"source": "N1", "target": "N2", "pre_z": 0.8, "post_z": 0.6},
        {"source": "N2", "target": "N3", "pre_z": 0.4, "post_z": 0.2},
    ]


def test_tiny_neuron_records(tiny_payload: dict) -> None:
    neurons = {neuron["neuron_id"]: neuron for neuron in tiny_payload["neurons"]}
    assert sorted(neurons) == ["N1", "N2", "N3"]
    assert neurons["N1"]["cent"] == 0.1
    assert neurons["N2"]["cent"] == 0.2
    # degrees / strengths are derived from the collapsed pairs
    assert (neurons["N1"]["out_degree"], neurons["N1"]["in_degree"]) == (1, 0)
    assert (neurons["N2"]["out_degree"], neurons["N2"]["in_degree"]) == (1, 1)
    assert (neurons["N3"]["out_degree"], neurons["N3"]["in_degree"]) == (0, 1)
    assert neurons["N2"]["pre_strength"] == pytest.approx(0.4)
    assert neurons["N2"]["post_strength"] == pytest.approx(0.6)
    assert neurons["N1"]["pre_strength"] == pytest.approx(0.8)
    assert neurons["N3"]["post_strength"] == pytest.approx(0.2)


def test_tiny_cent_distribution_uses_raw_tokens(tiny_payload: dict) -> None:
    assert tiny_payload["metadata"]["cent_distribution"] == {"0.1": 2, "0.2": 1}


def test_tiny_raw_edges_are_verbatim(tiny_payload: dict) -> None:
    # sorted by (src, dst), all hub attributes kept verbatim
    assert tiny_payload["raw_edges"] == [
        {
            "src": "N1",
            "dst": "N1--N2",
            "is_hub": True,
            "z_score": 0.8,
            "penwidth": 1.6,
            "color": "blue",
        },
        {
            "src": "N1--N2",
            "dst": "N2",
            "is_hub": True,
            "z_score": 0.6,
            "penwidth": 1.2,
            "color": "gold",
        },
        {
            "src": "N2",
            "dst": "N2--N3",
            "is_hub": True,
            "z_score": 0.4,
            "penwidth": 0.8,
            "color": "red",
        },
        {
            "src": "N2--N3",
            "dst": "N3",
            "is_hub": True,
            "z_score": 0.2,
            "penwidth": 0.4,
            "color": "red",
        },
    ]


def test_tiny_sha256_matches_source_bytes(tiny_payload: dict) -> None:
    digest = hashlib.sha256(TINY.read_bytes()).hexdigest()
    assert tiny_payload["file_sha256"] == digest
    assert tiny_payload["source_file"] == TINY.name


def test_tiny_payload_passes_schema(tiny_payload: dict) -> None:
    assert validate_payload_schema(tiny_payload) == []


def test_embedded_newline_label_is_parsed_as_single_node(tiny_parsed) -> None:
    """Regression guard: a literal newline inside a quoted label must not split a
    node statement into two nodes."""
    assert sorted(neuron.neuron_id for neuron in tiny_parsed.neurons) == ["N1", "N2", "N3"]


# ---------------------------------------------------------------------------
# Validation rules
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(("fixture", "code"), VALIDATION_CASES)
def test_validation_rule_is_reported(fixture: str, code: str) -> None:
    _, report = build_parsed_graph(FIXTURES / fixture)
    assert code in report.codes(parser.SEVERITY_ERROR)


@pytest.mark.parametrize(("fixture", "code"), VALIDATION_CASES)
def test_validation_rule_raises_with_report(fixture: str, code: str) -> None:
    with pytest.raises(GraphvizValidationError) as excinfo:
        parse_graphviz_file(FIXTURES / fixture)
    assert excinfo.value.report is not None
    assert code in excinfo.value.report.codes(parser.SEVERITY_ERROR)


def test_valid_fixture_raises_nothing() -> None:
    assert parse_graphviz_file(TINY) is not None


def test_malformed_dot_reports_parse_failure() -> None:
    with pytest.raises(GraphvizValidationError) as excinfo:
        parse_graphviz_file(FIXTURES / "malformed.gv")
    report = excinfo.value.report
    assert report is not None
    assert parser.CODE_PARSE_FAILURE in report.codes(parser.SEVERITY_ERROR)


def test_missing_file_raises_file_not_found() -> None:
    with pytest.raises(FileNotFoundError):
        parse_graphviz_file(FIXTURES / "does_not_exist.gv")


def test_hub_degree_message_names_the_hub() -> None:
    _, report = build_parsed_graph(FIXTURES / "missing_post_edge.gv")
    message = next(
        issue.message for issue in report.errors if issue.code == parser.CODE_HUB_DEGREE
    )
    assert "A--B" in message
    assert "1 pre edge(s) and 0 post edge(s)" in message


def test_invalid_z_scores_never_reach_the_pairs() -> None:
    parsed, _ = build_parsed_graph(FIXTURES / "negative_z.gv")
    assert parsed.pairs == []
    assert all(edge.z_score is not None for edge in parsed.raw_edges)


def test_duplicate_pair_keeps_both_pairs_in_lenient_mode() -> None:
    parsed, report = build_parsed_graph(FIXTURES / "duplicate_pair.gv")
    assert len(parsed.pairs) == 2
    assert report.codes(parser.SEVERITY_ERROR) == [parser.CODE_DUPLICATE_PAIR]


# ---------------------------------------------------------------------------
# Warnings (lenient by default, escalated by --strict)
# ---------------------------------------------------------------------------
def test_missing_cent_is_a_warning_not_an_error() -> None:
    parsed, report = build_parsed_graph(FIXTURES / "no_cent.gv")
    assert not report.has_errors
    assert report.codes(parser.SEVERITY_WARNING) == [
        parser.CODE_CENT_MISSING,
        parser.CODE_CENT_MISSING,
    ]
    assert all(neuron.cent is None for neuron in parsed.neurons)
    assert parsed.metadata["cent_distribution"] == {}


def test_implicit_endpoints_are_collected_with_a_warning() -> None:
    parsed, report = build_parsed_graph(FIXTURES / "implicit_endpoints.gv")
    assert not report.has_errors
    assert parser.CODE_CENT_MISSING in report.codes(parser.SEVERITY_WARNING)
    assert sorted(neuron.neuron_id for neuron in parsed.neurons) == ["A", "B"]


def test_unexpected_attributes_are_warnings() -> None:
    _, report = build_parsed_graph(FIXTURES / "unexpected_attribute.gv")
    assert not report.has_errors
    assert report.codes(parser.SEVERITY_WARNING) == [parser.CODE_UNEXPECTED_ATTRIBUTE] * 2


def test_strict_escalates_warnings_to_errors() -> None:
    with pytest.raises(GraphvizValidationError) as excinfo:
        parse_graphviz_file(FIXTURES / "no_cent.gv", strict=True)
    report = excinfo.value.report
    assert report is not None
    assert report.codes(parser.SEVERITY_WARNING)


def test_lenient_mode_accepts_warnings() -> None:
    parsed = parse_graphviz_file(FIXTURES / "no_cent.gv")
    assert parsed.metadata["n_pairs"] == 1


def test_multi_graph_is_merged_with_a_warning() -> None:
    parsed, report = build_parsed_graph(FIXTURES / "multi_graph.gv")
    assert parser.CODE_MULTI_GRAPH in report.codes(parser.SEVERITY_WARNING)
    assert parsed.metadata["n_neurons"] == 4
    assert parsed.metadata["n_hubs"] == 2
    assert parsed.metadata["n_pairs"] == 2


def test_multi_graph_is_an_error_under_strict() -> None:
    _, report = build_parsed_graph(FIXTURES / "multi_graph.gv", strict=True)
    assert parser.CODE_MULTI_GRAPH in report.codes(parser.SEVERITY_ERROR)


def test_validation_report_helpers() -> None:
    report = parser.ValidationReport()
    report.error("boom", "an error")
    report.warning("hmm", "a warning")
    assert report.has_errors and not report.ok
    assert report.codes() == ["boom", "hmm"]
    assert report.codes(parser.SEVERITY_WARNING) == ["hmm"]
    assert report.summary() == "1 error(s), 1 warning(s)"
    assert report.to_dict()["errors"][0] == {
        "code": "boom",
        "severity": "error",
        "message": "an error",
    }


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------
def test_min_pre_filter_arithmetic() -> None:
    parsed, report = build_parsed_graph(TINY, min_pre=0.5)
    assert not report.has_errors
    assert parsed.metadata["filters"] == [{"type": "min_pre", "value": 0.5, "removed": 1}]
    assert parsed.metadata["n_pairs"] == 1
    assert [pair.source for pair in parsed.pairs] == ["N1"]
    # raw edges and neurons are untouched by filtering
    assert parsed.metadata["n_raw_edges"] == 4
    assert parsed.metadata["n_neurons"] == 3
    assert parsed.metadata["density"] == pytest.approx(1 / 6)


def test_min_post_filter_arithmetic() -> None:
    parsed, _ = build_parsed_graph(TINY, min_post=0.5)
    assert parsed.metadata["filters"] == [{"type": "min_post", "value": 0.5, "removed": 1}]
    assert [pair.target for pair in parsed.pairs] == ["N2"]
    assert parsed.metadata["pre_stats"] == {"min": 0.8, "mean": 0.8, "max": 0.8}


def test_top_k_filter_ranks_by_combined_z() -> None:
    parsed, _ = build_parsed_graph(TINY, top_k=1)
    assert parsed.metadata["filters"] == [{"type": "top_k", "value": 1, "removed": 1}]
    assert [(pair.source, pair.target) for pair in parsed.pairs] == [("N1", "N2")]


def test_filters_are_applied_sequentially_and_accounted() -> None:
    parsed, _ = build_parsed_graph(TINY, min_pre=0.4, min_post=0.3, top_k=1)
    assert parsed.metadata["filters"] == [
        {"type": "min_pre", "value": 0.4, "removed": 0},
        {"type": "min_post", "value": 0.3, "removed": 1},
        {"type": "top_k", "value": 1, "removed": 0},
    ]
    assert parsed.metadata["n_pairs"] == 1


def test_top_k_zero_removes_every_pair_and_nulls_the_stats() -> None:
    parsed, _ = build_parsed_graph(TINY, top_k=0)
    assert parsed.pairs == []
    assert parsed.metadata["n_pairs"] == 0
    assert parsed.metadata["density"] == 0.0
    assert parsed.metadata["pre_stats"] == {"min": None, "mean": None, "max": None}
    assert parsed.metadata["reciprocity"] == 0


def test_apply_filters_is_pure() -> None:
    pairs = [
        NeuronPair("A", "B", 0.2, 0.9),
        NeuronPair("A", "C", 0.9, 0.1),
        NeuronPair("B", "C", 0.5, 0.5),
    ]
    kept, filters = apply_filters(pairs, min_pre=0.2, min_post=0.2, top_k=2)
    assert len(pairs) == 3  # input untouched
    assert [f"{pair.source}->{pair.target}" for pair in kept] == ["A->B", "B->C"]
    assert [entry["type"] for entry in filters] == ["min_pre", "min_post", "top_k"]


def test_filtered_artifacts_use_a_separate_file_name(tmp_path: Path) -> None:
    parsed, _ = build_parsed_graph(TINY, min_pre=0.5)
    assert artifact_path(parsed, tmp_path).name == FILTERED_ARTIFACT_NAME
    canonical, _ = build_parsed_graph(TINY)
    assert artifact_path(canonical, tmp_path).name == DEFAULT_ARTIFACT_NAME
    assert artifact_path(canonical, tmp_path) == tmp_path / TINY.stem / DEFAULT_ARTIFACT_NAME


# ---------------------------------------------------------------------------
# Determinism and reproducibility
# ---------------------------------------------------------------------------
def test_repeated_runs_are_byte_identical(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SOURCE_DATE_EPOCH", FIXED_EPOCH)
    first, _ = build_parsed_graph(TINY)
    second, _ = build_parsed_graph(TINY)
    assert dumps_json(first.to_dict()) == dumps_json(second.to_dict())
    assert first.metadata["created_utc"] == FIXED_TIMESTAMP

    write_artifact(first, tmp_path / "a")
    write_artifact(second, tmp_path / "b")
    assert (tmp_path / "a" / TINY.stem / DEFAULT_ARTIFACT_NAME).read_bytes() == (
        tmp_path / "b" / TINY.stem / DEFAULT_ARTIFACT_NAME
    ).read_bytes()


def test_created_utc_is_rfc3339(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SOURCE_DATE_EPOCH", raising=False)
    parsed, _ = build_parsed_graph(TINY)
    assert parser.RFC3339_PATTERN.match(parsed.metadata["created_utc"])
    assert utc_timestamp().endswith("Z")


def test_lists_are_deterministically_ordered() -> None:
    payload = build_parsed_graph(TINY)[0].to_dict()
    assert payload["neurons"] == sorted(payload["neurons"], key=lambda item: item["neuron_id"])
    assert payload["pairs"] == sorted(
        payload["pairs"], key=lambda item: (item["source"], item["target"])
    )
    assert payload["raw_edges"] == sorted(
        payload["raw_edges"], key=lambda item: (item["src"], item["dst"])
    )


# ---------------------------------------------------------------------------
# Atomic JSON writing
# ---------------------------------------------------------------------------
def test_write_json_atomic_style(tmp_path: Path) -> None:
    destination = tmp_path / "nested" / "artifact.json"
    write_json_atomic(destination, {"b": 1, "a": 2})
    text = destination.read_text(encoding="utf-8")
    assert text == '{\n  "a": 2,\n  "b": 1\n}\n'  # sorted keys, indent 2, trailing newline
    assert json.loads(text) == {"a": 2, "b": 1}


def test_write_json_atomic_leaves_no_partial_file_on_failure(tmp_path: Path) -> None:
    destination = tmp_path / "artifact.json"
    destination.write_text("previous content", encoding="utf-8")
    with pytest.raises(TypeError):
        write_json_atomic(destination, {"bad": {1, 2, 3}})
    assert destination.read_text(encoding="utf-8") == "previous content"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["artifact.json"]


def test_write_json_atomic_replaces_previous_content(tmp_path: Path) -> None:
    destination = tmp_path / "artifact.json"
    destination.write_text("previous content", encoding="utf-8")
    write_json_atomic(destination, {"a": 1})
    assert json.loads(destination.read_text(encoding="utf-8")) == {"a": 1}
    assert sorted(path.name for path in tmp_path.iterdir()) == ["artifact.json"]


def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    target = tmp_path / "blob.bin"
    target.write_bytes(b"phase-01" * 1024)
    assert parser.sha256_file(target) == hashlib.sha256(target.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def test_cli_writes_canonical_artifact(tmp_path: Path) -> None:
    assert main(["--input", str(TINY), "--outdir", str(tmp_path)]) == 0
    artifact = tmp_path / TINY.stem / DEFAULT_ARTIFACT_NAME
    assert artifact.is_file()
    assert json.loads(artifact.read_text(encoding="utf-8"))["metadata"]["n_pairs"] == 2


def test_cli_dry_run_writes_nothing(tmp_path: Path) -> None:
    assert main(["--input", str(TINY), "--outdir", str(tmp_path), "--dry-run"]) == 0
    assert list(tmp_path.iterdir()) == []


def test_cli_filtered_run_writes_filtered_artifact(tmp_path: Path) -> None:
    assert main(["--input", str(TINY), "--outdir", str(tmp_path), "--min-pre", "0.5"]) == 0
    assert (tmp_path / TINY.stem / FILTERED_ARTIFACT_NAME).is_file()
    assert not (tmp_path / TINY.stem / DEFAULT_ARTIFACT_NAME).exists()


def test_cli_returns_nonzero_for_invalid_fixture(tmp_path: Path) -> None:
    code = main(["--input", str(FIXTURES / "self_loop.gv"), "--outdir", str(tmp_path)])
    assert code == 1
    assert list(tmp_path.iterdir()) == []


def test_cli_strict_returns_nonzero_for_warnings(tmp_path: Path) -> None:
    assert main(["--input", str(FIXTURES / "no_cent.gv"), "--outdir", str(tmp_path)]) == 0
    assert (
        main(["--input", str(FIXTURES / "no_cent.gv"), "--outdir", str(tmp_path), "--strict"]) == 1
    )


def test_cli_returns_two_when_nothing_matches(tmp_path: Path) -> None:
    assert main(["--input", str(tmp_path), "--outdir", str(tmp_path)]) == 2


def test_cli_batch_over_a_directory(tmp_path: Path) -> None:
    source_dir = tmp_path / "raw"
    source_dir.mkdir()
    (source_dir / TINY.name).write_bytes(TINY.read_bytes())
    (source_dir / "second.dot").write_text(
        TINY.read_text(encoding="utf-8").replace("N", "M"), encoding="utf-8"
    )
    outdir = tmp_path / "processed"
    assert main(["--input", str(source_dir), "--outdir", str(outdir)]) == 0
    assert (outdir / TINY.stem / DEFAULT_ARTIFACT_NAME).is_file()
    assert (outdir / "second" / DEFAULT_ARTIFACT_NAME).is_file()


def test_resolve_inputs_ignores_unsupported_suffixes(tmp_path: Path) -> None:
    (tmp_path / "a.gv").write_text("digraph G {}", encoding="utf-8")
    (tmp_path / "b.dot").write_text("digraph G {}", encoding="utf-8")
    (tmp_path / "c.txt").write_text("ignored", encoding="utf-8")
    assert [path.name for path in resolve_inputs([tmp_path])] == ["a.gv", "b.dot"]


def test_cli_module_invocation(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.parsing.parse_graphviz",
            "--input",
            str(TINY),
            "--outdir",
            str(tmp_path),
            "--dry-run",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "neurons=3" in completed.stdout + completed.stderr
    assert "RuntimeWarning" not in completed.stderr
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# Integration against the real dataset (deselect with -m "not slow")
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_reference_file_reference_numbers() -> None:
    if not REFERENCE.is_file():  # pragma: no cover - dataset not checked out
        pytest.skip(f"reference dataset missing: {REFERENCE}")

    parsed, report = build_parsed_graph(REFERENCE)
    metadata = parsed.metadata
    assert not report.has_errors, report.to_dict()
    assert metadata["n_neurons"] == 113
    assert metadata["n_hubs"] == 1355
    assert metadata["n_raw_edges"] == 2710
    assert metadata["n_pairs"] == 1355
    assert round(metadata["density"], 6) == 0.107064
    assert metadata["reciprocity"] == 574
    assert round(metadata["pre_stats"]["mean"], 4) == 0.3492
    assert round(metadata["post_stats"]["mean"], 4) == 0.3023
    assert metadata["pre_stats"]["min"] == 0.1
    assert metadata["pre_stats"]["max"] == 1.0
    assert metadata["post_stats"]["min"] == 0.1
    assert metadata["post_stats"]["max"] == 1.0
    assert metadata["cent_distribution"] == {"0.1": 112, "0.2": 1}
    assert metadata["filters"] == []
    assert metadata["parser_version"]
    assert validate_payload_schema(parsed.to_dict()) == []
    assert all(edge.is_hub for edge in parsed.raw_edges)
    assert all(
        pair.source != pair.target and pair.pre_z >= 0 and pair.post_z >= 0
        for pair in parsed.pairs
    )


@pytest.mark.slow
def test_reference_file_artifact_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if not REFERENCE.is_file():  # pragma: no cover - dataset not checked out
        pytest.skip(f"reference dataset missing: {REFERENCE}")

    monkeypatch.setenv("SOURCE_DATE_EPOCH", FIXED_EPOCH)
    parsed = parse_graphviz_file(REFERENCE)
    write_artifact(parsed, tmp_path)
    artifact = tmp_path / REFERENCE.stem / DEFAULT_ARTIFACT_NAME
    assert artifact.read_text(encoding="utf-8") == dumps_json(parsed.to_dict())

    reloaded = json.loads(artifact.read_text(encoding="utf-8"))
    assert len(reloaded["pairs"]) == 1355
    assert len(reloaded["neurons"]) == 113
    pairs_by_key = {(pair["source"], pair["target"]): pair for pair in reloaded["pairs"]}
    assert pairs_by_key[("FB4Q_b_R_1", "FB4P_b_R_3")] == {
        "source": "FB4Q_b_R_1",
        "target": "FB4P_b_R_3",
        "pre_z": 0.16,
        "post_z": 0.12,
    }
    assert {neuron["neuron_id"] for neuron in reloaded["neurons"]} == {
        neuron.neuron_id for neuron in parsed.neurons
    }

    # byte-identical on a second run (created_utc derives from SOURCE_DATE_EPOCH)
    assert dumps_json(parse_graphviz_file(REFERENCE).to_dict()) == artifact.read_text(
        encoding="utf-8"
    )


@pytest.mark.slow
def test_cli_batch_over_the_real_dataset(tmp_path: Path) -> None:
    if not REFERENCE.is_file():  # pragma: no cover - dataset not checked out
        pytest.skip(f"reference dataset missing: {REFERENCE}")

    assert main(["--input", str(REFERENCE.parent), "--outdir", str(tmp_path)]) == 0
    artifact = tmp_path / REFERENCE.stem / DEFAULT_ARTIFACT_NAME
    assert artifact.is_file()
    assert json.loads(artifact.read_text(encoding="utf-8"))["metadata"]["n_pairs"] == 1355
