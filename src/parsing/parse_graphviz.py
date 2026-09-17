"""Phase 01 -- parse Graphviz ``.gv`` files into the canonical JSON artifact.

The raw files encode every functional connection through an intermediate *hub*
node::

    A -> "A--B" [label="Zscore = <pre_z>"]      # pre-synaptic z-score
    "A--B" -> B [label="Zscore = <post_z>"]     # post-synaptic z-score

So a hub named ``"A--B"`` carries exactly one inbound edge (whose source is the
real neuron ``A``) and exactly one outbound edge (whose target is the real neuron
``B``).  Collapsing the hubs therefore recovers one
``(source, target, pre_z, post_z)`` pair per hub.

Phase 01 is **purely descriptive**: ``pre_z`` and ``post_z`` are preserved
verbatim and no pre/post unification is performed.  The unified weight
``w = s * tanh(a * log(post_z / (pre_z + eps)))`` belongs to Phase 02.

Artifact layout (the single Phase 01 -> Phase 02 boundary, JSON only)::

    data/processed/<stem>/parsed_graph.json

Filtered runs (``--min-pre`` / ``--min-post`` / ``--top-k``) never mutate the
canonical artifact: they write ``parsed_graph.filtered.json`` instead and record
the applied filters in ``metadata.filters``.

CLI
---
::

    python -m src.parsing.parse_graphviz \\
        --input data/raw_dot/FB4Yaffect_FB45_999prePost_001_all.gv \\
        --outdir data/processed \\
        [--glob '*.gv'] [--strict] [--min-pre X] [--min-post X] \\
        [--top-k N] [--dry-run] [--log-level INFO]
"""

from __future__ import annotations

import argparse
import logging
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import pydot

from src.utils.io import dumps_json, sha256_file, utc_timestamp, write_json_atomic

__all__ = [
    "DEFAULT_ARTIFACT_NAME",
    "FILTERED_ARTIFACT_NAME",
    "FILTER_KEYS",
    "GraphvizValidationError",
    "METADATA_KEYS",
    "NEURON_KEYS",
    "Neuron",
    "NeuronPair",
    "PAIR_KEYS",
    "ParsedGraph",
    "RAW_EDGE_KEYS",
    "RawEdge",
    "STATS_KEYS",
    "TOP_LEVEL_KEYS",
    "ValidationIssue",
    "ValidationReport",
    "apply_filters",
    "artifact_path",
    "build_argument_parser",
    "build_parsed_graph",
    "is_hub_name",
    "main",
    "normalize_label",
    "parse_cent",
    "parse_graphviz_file",
    "parse_z_score",
    "pydot_version",
    "resolve_inputs",
    "split_hub_name",
    "strip_quotes",
    "validate_payload_schema",
    "write_artifact",
]

LOGGER = logging.getLogger("src.parsing.parse_graphviz")

# ---------------------------------------------------------------------------
# Format constants
# ---------------------------------------------------------------------------
HUB_SEPARATOR = "--"
PSEUDO_NODE_NAMES = frozenset({"graph", "node", "edge"})
SUPPORTED_SUFFIXES = (".gv", ".dot")
DEFAULT_ARTIFACT_NAME = "parsed_graph.json"
FILTERED_ARTIFACT_NAME = "parsed_graph.filtered.json"
DEFAULT_PARSER_NAME = "pydot"

SEVERITY_ERROR = "error"
SEVERITY_WARNING = "warning"

#: Tolerant ``Zscore = <float>`` matcher (labels are quoted by pydot and the file
#: mixes ``Zscore = 0.8`` with ``Zscore=0.8``-style spacing).
ZSCORE_PATTERN = re.compile(r"Zscore\s*=\s*([-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?)")
#: ``cent= 0.1`` -> raw token ``0.1`` (kept verbatim for ``cent_distribution``).
CENT_PATTERN = re.compile(r"cent\s*=\s*([^\s,;]+)")
RFC3339_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

# Validation issue codes (one per rule in the Phase 01 plan).
CODE_HUB_DEGREE = "hub_degree"
CODE_HUB_ENDPOINT_MISMATCH = "hub_endpoint_mismatch"
CODE_SELF_LOOP = "self_loop"
CODE_DUPLICATE_PAIR = "duplicate_pair"
CODE_ENDPOINT_NOT_NEURON = "endpoint_not_neuron"
CODE_HUB_IN_NEURONS = "hub_in_neurons"
CODE_Z_INVALID = "z_invalid"
CODE_CENT_MISSING = "cent_missing"
CODE_CENT_INVALID = "cent_invalid"
CODE_PENWIDTH_INVALID = "penwidth_invalid"
CODE_COLOR_MISSING = "color_missing"
CODE_UNEXPECTED_ATTRIBUTE = "unexpected_attribute"
CODE_MULTI_GRAPH = "multi_graph"
CODE_PARSE_FAILURE = "parse_failure"
CODE_SCHEMA = "schema"

#: Attributes understood by this parser; anything else is a warning.
NEURON_ATTRIBUTES = frozenset(
    {
        "label", "color", "fixedsize", "fontcolor", "fontname", "fontsize", "height",
        "width", "shape", "style", "penwidth", "peripheries", "tooltip", "class",
        "pos", "pin", "image", "fillcolor",
    }
)
EDGE_ATTRIBUTES = frozenset(
    {
        "label", "arrowhead", "arrowsize", "arrowtail", "color", "fontcolor",
        "fontname", "fontsize", "labelfloat", "labeldistance", "labelangle",
        "headlabel", "taillabel", "penwidth", "weight", "style", "dir",
        "constraint", "decorate", "minlen", "xlabel", "samehead", "sametail",
    }
)

# ---------------------------------------------------------------------------
# Output schema (asserted before the artifact is written)
# ---------------------------------------------------------------------------
TOP_LEVEL_KEYS = frozenset({"source_file", "file_sha256", "neurons", "pairs", "raw_edges", "metadata"})
NEURON_KEYS = frozenset(
    {"neuron_id", "cent", "out_degree", "in_degree", "pre_strength", "post_strength"}
)
PAIR_KEYS = frozenset({"source", "target", "pre_z", "post_z"})
RAW_EDGE_KEYS = frozenset({"src", "dst", "is_hub", "z_score", "penwidth", "color"})
STATS_KEYS = frozenset({"min", "mean", "max"})
FILTER_KEYS = frozenset({"type", "value", "removed"})
METADATA_KEYS = frozenset(
    {
        "n_neurons", "n_hubs", "n_raw_edges", "n_pairs", "density", "reciprocity",
        "pre_stats", "post_stats", "cent_distribution", "filters", "parser",
        "parser_version", "created_utc",
    }
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Neuron:
    """A real (non-hub) neuron and its ``cent`` metadata."""

    neuron_id: str
    cent: float | None = None
    cent_token: str | None = None  # raw source token, used for cent_distribution keys


@dataclass(frozen=True)
class NeuronPair:
    """A collapsed hub: one pre-synaptic and one post-synaptic z-score."""

    source: str
    target: str
    pre_z: float
    post_z: float


@dataclass(frozen=True)
class RawEdge:
    """A verbatim edge of the source file (hub edges included)."""

    src: str
    dst: str
    is_hub: bool
    z_score: float | None
    penwidth: float | None = None
    color: str | None = None


@dataclass(frozen=True)
class ValidationIssue:
    """A single validation finding."""

    code: str
    severity: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "severity": self.severity, "message": self.message}


@dataclass
class ValidationReport:
    """Collect-and-report accumulator for validation findings."""

    issues: list[ValidationIssue] = field(default_factory=list)

    def add(self, code: str, message: str, severity: str = SEVERITY_ERROR) -> ValidationIssue:
        issue = ValidationIssue(code=code, severity=severity, message=message)
        self.issues.append(issue)
        return issue

    def error(self, code: str, message: str) -> ValidationIssue:
        return self.add(code, message, SEVERITY_ERROR)

    def warning(self, code: str, message: str) -> ValidationIssue:
        return self.add(code, message, SEVERITY_WARNING)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == SEVERITY_ERROR]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == SEVERITY_WARNING]

    @property
    def has_errors(self) -> bool:
        return any(issue.severity == SEVERITY_ERROR for issue in self.issues)

    @property
    def ok(self) -> bool:
        return not self.issues

    def codes(self, severity: str | None = None) -> list[str]:
        """Return issue codes, optionally filtered by *severity*."""
        return [
            issue.code
            for issue in self.issues
            if severity is None or issue.severity == severity
        ]

    def summary(self) -> str:
        return f"{len(self.errors)} error(s), {len(self.warnings)} warning(s)"

    def to_dict(self) -> dict[str, list[dict[str, str]]]:
        return {
            "errors": [issue.to_dict() for issue in self.errors],
            "warnings": [issue.to_dict() for issue in self.warnings],
        }


class GraphvizValidationError(ValueError):
    """Raised when a ``.gv`` file violates a Phase 01 validation rule.

    The full :class:`ValidationReport` is available as :attr:`report`.
    """

    def __init__(self, message: str, report: ValidationReport | None = None) -> None:
        super().__init__(message)
        self.report = report


@dataclass
class ParsedGraph:
    """In-memory Phase 01 artifact (see :meth:`to_dict` for the JSON schema)."""

    source_file: Path
    file_sha256: str
    neurons: list[Neuron] = field(default_factory=list)
    pairs: list[NeuronPair] = field(default_factory=list)
    raw_edges: list[RawEdge] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the exact JSON payload written to ``parsed_graph.json``.

        ``neurons`` is sorted by ``neuron_id``, ``pairs`` by ``(source, target)``
        and ``raw_edges`` by ``(src, dst)``; degrees and strengths are derived
        from the (possibly filtered) pair list, so the artifact stays internally
        consistent.
        """
        out_degree: Counter[str] = Counter()
        in_degree: Counter[str] = Counter()
        pre_strength: defaultdict[str, float] = defaultdict(float)
        post_strength: defaultdict[str, float] = defaultdict(float)
        for pair in sorted(self.pairs, key=lambda item: (item.source, item.target)):
            out_degree[pair.source] += 1
            in_degree[pair.target] += 1
            pre_strength[pair.source] += pair.pre_z
            post_strength[pair.target] += pair.post_z

        neurons = [
            {
                "neuron_id": neuron.neuron_id,
                "cent": neuron.cent,
                "out_degree": out_degree.get(neuron.neuron_id, 0),
                "in_degree": in_degree.get(neuron.neuron_id, 0),
                "pre_strength": pre_strength.get(neuron.neuron_id, 0.0),
                "post_strength": post_strength.get(neuron.neuron_id, 0.0),
            }
            for neuron in sorted(self.neurons, key=lambda item: item.neuron_id)
        ]
        pairs = [
            {
                "source": pair.source,
                "target": pair.target,
                "pre_z": pair.pre_z,
                "post_z": pair.post_z,
            }
            for pair in sorted(self.pairs, key=lambda item: (item.source, item.target))
        ]
        raw_edges = [
            {
                "src": edge.src,
                "dst": edge.dst,
                "is_hub": edge.is_hub,
                "z_score": edge.z_score,
                "penwidth": edge.penwidth,
                "color": edge.color,
            }
            for edge in sorted(self.raw_edges, key=lambda item: (item.src, item.dst))
        ]
        return {
            "source_file": self.source_file.name,
            "file_sha256": self.file_sha256,
            "neurons": neurons,
            "pairs": pairs,
            "raw_edges": raw_edges,
            "metadata": dict(self.metadata),
        }


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------
def strip_quotes(value: str | None) -> str | None:
    """Strip Graphviz quoting/escaping from a node name, label or attribute.

    pydot keeps the surrounding double quotes, e.g. ``'"A--B"'``; real neuron
    ids may also contain parentheses and slashes (``FB4Y(EB/NO1)_R_2``), which is
    why the names are only unwrapped, never re-interpreted.  Surrounding
    whitespace is insignificant in Graphviz, so ``'"A--B "'`` normalizes to
    ``A--B`` as well.
    """
    if value is None:
        return None
    text = value.strip()
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        text = text[1:-1]
        text = text.replace('\\"', '"').replace("\\\\", "\\")
    return text.strip()


def normalize_label(value: str | None) -> str | None:
    """Return a label with quotes removed and escaped newlines expanded.

    Real-neuron labels embed a literal newline inside the quoted string
    (``"hDeltaI_04_C3_1\\ncent= 0.1"``); a literal ``\\n`` escape sequence is
    normalized to a real newline as well so both encodings split identically.
    """
    text = strip_quotes(value)
    if text is None:
        return None
    return text.replace("\\n", "\n")


def is_hub_name(name: str) -> bool:
    """A node is a hub if and only if its (normalized) name contains ``--``."""
    return HUB_SEPARATOR in name


def split_hub_name(name: str) -> tuple[str, str] | None:
    """Split a hub name into ``(lhs, rhs)`` on the first ``--``; ``None`` if absent."""
    if not is_hub_name(name):
        return None
    lhs, _, rhs = name.partition(HUB_SEPARATOR)
    return lhs, rhs


def parse_cent(label: str | None) -> tuple[float | None, str | None]:
    """Extract ``(cent_value, cent_token)`` from a real-neuron label.

    The raw source token is returned alongside the float so that
    ``cent_distribution`` keys never suffer float-repr drift.
    """
    if not label:
        return None, None
    match = CENT_PATTERN.search(label)
    if match is None:
        return None, None
    token = match.group(1)
    try:
        value = float(token)
    except ValueError:
        return None, token
    if not math.isfinite(value):
        return None, token
    return value, token


def parse_z_score(label: str | None) -> float | None:
    """Parse ``Zscore = <float>`` from an edge label; ``None`` when absent."""
    if not label:
        return None
    match = ZSCORE_PATTERN.search(label)
    if match is None:
        return None
    return float(match.group(1))


def parse_optional_float(token: str | None) -> float | None:
    """Parse *token* as a finite float; ``None`` when missing or malformed."""
    if token is None:
        return None
    text = strip_quotes(token) or ""
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


@dataclass(frozen=True)
class _EdgeRecord:
    """Internal bookkeeping: the raw token *and* the normalized endpoint."""

    raw_src: str
    raw_dst: str
    src: str
    dst: str
    z_score: float | None
    penwidth: float | None
    color: str | None


def _attributes(element: Any) -> dict[str, str]:
    """Return the element's attributes with normalized values."""
    return {
        key: (strip_quotes(value) or "")
        for key, value in (element.get_attributes() or {}).items()
    }


def _warn_unknown_attributes(
    report: ValidationReport,
    kind: str,
    name: str,
    attributes: dict[str, str],
    known: frozenset[str],
) -> None:
    unknown = sorted(set(attributes) - known)
    if unknown:
        report.warning(
            CODE_UNEXPECTED_ATTRIBUTE,
            f"unexpected {kind} attribute(s) {', '.join(unknown)} on {name!r}",
        )


def _z_is_valid(value: float | None) -> bool:
    """A usable z-score is a finite, non-negative number."""
    return value is not None and math.isfinite(value) and value >= 0


def _summary_stats(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "mean": None, "max": None}
    return {"min": min(values), "mean": sum(values) / len(values), "max": max(values)}


def pydot_version() -> str:
    """Installed ``pydot`` version, recorded in ``metadata.parser_version``."""
    try:
        from importlib.metadata import version

        return version("pydot")
    except Exception:  # pragma: no cover - defensive fallback
        return getattr(pydot, "__version__", "unknown")


# ---------------------------------------------------------------------------
# Loading and collection
# ---------------------------------------------------------------------------
def _load_graphs(path: Path, report: ValidationReport, strict: bool) -> list[pydot.Dot]:
    """Load every graph of *path* with pydot (the only newline-safe option).

    pydot returns ``None`` for syntactically broken DOT (it prints the parse
    error to stderr) and raises for unreadable files; both become a
    ``parse_failure`` error here.
    """
    try:
        graphs = pydot.graph_from_dot_file(str(path))
    except Exception as exc:  # pydot raises its own ParseException subclasses
        report.error(CODE_PARSE_FAILURE, f"pydot could not parse {path.name!r}: {exc}")
        raise GraphvizValidationError(
            f"failed to parse {path.name}: {exc}", report=report
        ) from exc

    graphs = [graph for graph in (graphs or []) if graph is not None]
    if not graphs:
        message = f"pydot could not parse {path.name!r} (malformed DOT)"
        report.error(CODE_PARSE_FAILURE, message)
        raise GraphvizValidationError(message, report=report)
    if len(graphs) > 1:
        message = f"{path.name!r} contains {len(graphs)} graphs; merging them"
        if strict:
            report.error(CODE_MULTI_GRAPH, message)
        else:
            report.warning(CODE_MULTI_GRAPH, message)
    return graphs


def _collect_node_statements(
    graph: Any,
    report: ValidationReport,
    neurons: dict[str, Neuron],
    hub_names: set[str],
    hub_tokens: set[str],
) -> None:
    """Extract real-neuron metadata and hub names from ``node`` statements."""
    for node in graph.get_nodes():
        raw_name = node.get_name()
        if raw_name is None:
            continue
        raw_token = raw_name.strip()
        name = strip_quotes(raw_name) or ""
        if not name or name in PSEUDO_NODE_NAMES:
            continue

        label = normalize_label(node.get_label())
        attributes = _attributes(node)

        if is_hub_name(name):
            hub_names.add(name)
            hub_tokens.add(raw_token)
            if label and CENT_PATTERN.search(label):
                report.error(
                    CODE_HUB_IN_NEURONS,
                    f"hub node {name!r} carries neuron metadata (cent); "
                    "hub names must never appear in the neuron list",
                )
            continue

        _warn_unknown_attributes(report, "node", name, attributes, NEURON_ATTRIBUTES)
        if name in neurons:
            continue  # pydot can repeat a node statement; keep the first
        if not label or not label.strip():
            report.warning(CODE_CENT_MISSING, f"neuron {name!r} has no label/cent metadata")
            neurons[name] = Neuron(name, None, None)
            continue
        cent, token = parse_cent(label)
        if token is None:
            report.warning(
                CODE_CENT_MISSING, f"neuron {name!r} label {label!r} lacks a 'cent=' token"
            )
        elif cent is None:
            report.warning(
                CODE_CENT_INVALID, f"neuron {name!r} has a non-numeric cent token {token!r}"
            )
        neurons[name] = Neuron(name, cent, token)


def _collect_edges(
    graph: Any,
    report: ValidationReport,
    hub_names: set[str],
    hub_tokens: set[str],
) -> list[_EdgeRecord]:
    """Extract every edge verbatim, validating its z-score label."""
    records: list[_EdgeRecord] = []
    for edge in graph.get_edges():
        raw_src = (edge.get_source() or "").strip()
        raw_dst = (edge.get_destination() or "").strip()
        src = strip_quotes(raw_src) or ""
        dst = strip_quotes(raw_dst) or ""
        label = normalize_label(edge.get_label())
        attributes = _attributes(edge)

        z_score = parse_z_score(label)
        if z_score is None:
            report.error(
                CODE_Z_INVALID,
                f"edge {src!r} -> {dst!r} has no parsable 'Zscore = <float>' label "
                f"(label={label!r})",
            )
        elif not _z_is_valid(z_score):
            report.error(
                CODE_Z_INVALID, f"edge {src!r} -> {dst!r} has an invalid z-score {z_score!r}"
            )

        penwidth_token = attributes.get("penwidth")
        penwidth = parse_optional_float(penwidth_token)
        if penwidth_token is not None and penwidth is None:
            report.warning(
                CODE_PENWIDTH_INVALID,
                f"edge {src!r} -> {dst!r} has a non-numeric penwidth {penwidth_token!r}",
            )
        color = attributes.get("color")
        if not color:
            report.warning(CODE_COLOR_MISSING, f"edge {src!r} -> {dst!r} has no color attribute")
        _warn_unknown_attributes(report, "edge", f"{src} -> {dst}", attributes, EDGE_ATTRIBUTES)

        if is_hub_name(src):
            hub_names.add(src)
            hub_tokens.add(raw_src)
        if is_hub_name(dst):
            hub_names.add(dst)
            hub_tokens.add(raw_dst)

        records.append(_EdgeRecord(raw_src, raw_dst, src, dst, z_score, penwidth, color))
    return records


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------
def apply_filters(
    pairs: Sequence[NeuronPair],
    *,
    min_pre: float | None = None,
    min_post: float | None = None,
    top_k: int | None = None,
) -> tuple[list[NeuronPair], list[dict[str, Any]]]:
    """Apply the optional Phase 01 filters *after* hub collapse.

    Filters run sequentially, each recording the number of pairs it removed, so
    ``metadata.filters`` documents the provenance of a filtered artifact.
    ``top_k`` ranks by ``pre_z + post_z`` (descending), ties broken by
    ``(source, target)`` for determinism.
    """
    result = list(pairs)
    filters: list[dict[str, Any]] = []

    if min_pre is not None:
        kept = [pair for pair in result if pair.pre_z >= min_pre]
        filters.append(
            {"type": "min_pre", "value": float(min_pre), "removed": len(result) - len(kept)}
        )
        result = kept
    if min_post is not None:
        kept = [pair for pair in result if pair.post_z >= min_post]
        filters.append(
            {"type": "min_post", "value": float(min_post), "removed": len(result) - len(kept)}
        )
        result = kept
    if top_k is not None:
        ranked = sorted(
            result, key=lambda pair: (-(pair.pre_z + pair.post_z), pair.source, pair.target)
        )
        kept = ranked[: max(int(top_k), 0)]
        kept.sort(key=lambda pair: (pair.source, pair.target))
        filters.append({"type": "top_k", "value": int(top_k), "removed": len(result) - len(kept)})
        result = kept

    return result, filters


# ---------------------------------------------------------------------------
# Schema validation (hand-rolled: no jsonschema/pydantic in this environment)
# ---------------------------------------------------------------------------
def _check_keys(problems: list[str], where: str, payload: Any, expected: frozenset[str]) -> bool:
    if not isinstance(payload, dict):
        problems.append(f"{where}: expected an object, got {type(payload).__name__}")
        return False
    actual = set(payload)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing:
        problems.append(f"{where}: missing key(s) {missing}")
    if extra:
        problems.append(f"{where}: unexpected key(s) {extra}")
    return not missing and not extra


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_payload_schema(payload: Any) -> list[str]:
    """Return a list of schema-conformance problems for a Phase 01 payload.

    An empty list means the payload matches the documented JSON schema exactly:
    top-level keys, ``neurons`` / ``pairs`` / ``raw_edges`` / ``metadata`` keys and
    their value types.  In particular ``pairs`` may only contain
    ``source`` / ``target`` / ``pre_z`` / ``post_z`` -- a ``weight`` key (which
    belongs to Phase 02) is reported as a problem.
    """
    problems: list[str] = []
    if not _check_keys(problems, "root", payload, TOP_LEVEL_KEYS):
        return problems

    if not isinstance(payload["source_file"], str):
        problems.append("source_file: expected a string")
    if not isinstance(payload["file_sha256"], str) or not re.fullmatch(
        r"[0-9a-f]{64}", payload["file_sha256"]
    ):
        problems.append("file_sha256: expected a 64-character lowercase hex digest")

    if not isinstance(payload["neurons"], list):
        problems.append("neurons: expected a list")
    else:
        for index, neuron in enumerate(payload["neurons"]):
            where = f"neurons[{index}]"
            if not _check_keys(problems, where, neuron, NEURON_KEYS):
                continue
            if not isinstance(neuron["neuron_id"], str):
                problems.append(f"{where}.neuron_id: expected a string")
            if neuron["cent"] is not None and not _is_number(neuron["cent"]):
                problems.append(f"{where}.cent: expected a number or null")
            for key in ("out_degree", "in_degree"):
                if not isinstance(neuron[key], int) or isinstance(neuron[key], bool):
                    problems.append(f"{where}.{key}: expected an integer")
            for key in ("pre_strength", "post_strength"):
                if not _is_number(neuron[key]):
                    problems.append(f"{where}.{key}: expected a number")

    if not isinstance(payload["pairs"], list):
        problems.append("pairs: expected a list")
    else:
        for index, pair in enumerate(payload["pairs"]):
            where = f"pairs[{index}]"
            if not _check_keys(problems, where, pair, PAIR_KEYS):
                continue
            for key in ("source", "target"):
                if not isinstance(pair[key], str):
                    problems.append(f"{where}.{key}: expected a string")
            for key in ("pre_z", "post_z"):
                if not _is_number(pair[key]):
                    problems.append(f"{where}.{key}: expected a number")

    if not isinstance(payload["raw_edges"], list):
        problems.append("raw_edges: expected a list")
    else:
        for index, edge in enumerate(payload["raw_edges"]):
            where = f"raw_edges[{index}]"
            if not _check_keys(problems, where, edge, RAW_EDGE_KEYS):
                continue
            for key in ("src", "dst"):
                if not isinstance(edge[key], str):
                    problems.append(f"{where}.{key}: expected a string")
            if not isinstance(edge["is_hub"], bool):
                problems.append(f"{where}.is_hub: expected a boolean")
            if edge["z_score"] is not None and not _is_number(edge["z_score"]):
                problems.append(f"{where}.z_score: expected a number or null")
            for key in ("penwidth", "color"):
                if edge[key] is not None and not isinstance(edge[key], (int, float, str)):
                    problems.append(f"{where}.{key}: expected a number, string or null")

    problems.extend(_validate_metadata_schema(payload["metadata"]))

    try:
        dumps_json(payload)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        problems.append(f"payload is not JSON serializable: {exc}")

    return problems


def _validate_metadata_schema(metadata: Any) -> list[str]:
    """Validate the ``metadata`` block (kept separate to stay readable)."""
    problems: list[str] = []
    if not _check_keys(problems, "metadata", metadata, METADATA_KEYS):
        return problems

    for key in ("n_neurons", "n_hubs", "n_raw_edges", "n_pairs", "reciprocity"):
        if not isinstance(metadata[key], int) or isinstance(metadata[key], bool):
            problems.append(f"metadata.{key}: expected an integer")
    if not _is_number(metadata["density"]):
        problems.append("metadata.density: expected a number")

    for key in ("pre_stats", "post_stats"):
        stats = metadata[key]
        if not _check_keys(problems, f"metadata.{key}", stats, STATS_KEYS):
            continue
        for stat_key in sorted(STATS_KEYS):
            if stats[stat_key] is not None and not _is_number(stats[stat_key]):
                problems.append(f"metadata.{key}.{stat_key}: expected a number or null")

    distribution = metadata["cent_distribution"]
    if not isinstance(distribution, dict) or not all(
        isinstance(key, str) for key in distribution
    ):
        problems.append("metadata.cent_distribution: expected an object with string keys")
    elif not all(
        isinstance(value, int) and not isinstance(value, bool)
        for value in distribution.values()
    ):
        problems.append("metadata.cent_distribution: expected integer counts")

    filters = metadata["filters"]
    if not isinstance(filters, list):
        problems.append("metadata.filters: expected a list")
    else:
        for index, entry in enumerate(filters):
            where = f"metadata.filters[{index}]"
            if not _check_keys(problems, where, entry, FILTER_KEYS):
                continue
            if not isinstance(entry["type"], str):
                problems.append(f"{where}.type: expected a string")
            if not _is_number(entry["value"]):
                problems.append(f"{where}.value: expected a number")
            if not isinstance(entry["removed"], int) or isinstance(entry["removed"], bool):
                problems.append(f"{where}.removed: expected an integer")

    if not isinstance(metadata["parser"], str):
        problems.append("metadata.parser: expected a string")
    if not isinstance(metadata["parser_version"], str):
        problems.append("metadata.parser_version: expected a string")
    if not isinstance(metadata["created_utc"], str) or not RFC3339_PATTERN.match(
        metadata["created_utc"]
    ):
        problems.append(
            "metadata.created_utc: expected an RFC 3339 UTC timestamp (YYYY-MM-DDThh:mm:ssZ)"
        )
    return problems


# ---------------------------------------------------------------------------
# Core: build the artifact
# ---------------------------------------------------------------------------
def build_parsed_graph(
    source_path: str | Path,
    *,
    strict: bool = False,
    min_pre: float | None = None,
    min_post: float | None = None,
    top_k: int | None = None,
    now: Any = None,
) -> tuple[ParsedGraph, ValidationReport]:
    """Parse *source_path* and return ``(ParsedGraph, ValidationReport)``.

    Nothing is written to disk here.  The caller decides whether to raise on
    findings (:func:`parse_graphviz_file`) or to persist the payload
    (:func:`write_artifact` / the CLI).

    Semantics of the returned artifact:

    * ``pairs`` always derives from the pre/post edge endpoints of the source
      file (the hub name is cross-checked, never trusted over the edges);
    * ``raw_edges`` is complete and unfiltered -- the optional filters only
      affect ``pairs`` (and, through them, degrees/strengths/density);
    * ``neurons`` keeps every real neuron of the file, even if a filter removed
      all of its pairs.
    """
    path = Path(source_path)
    if not path.is_file():
        raise FileNotFoundError(f"no such Graphviz file: {path}")

    report = ValidationReport()
    digest = sha256_file(path)
    graphs = _load_graphs(path, report, strict)

    neurons: dict[str, Neuron] = {}
    hub_names: set[str] = set()
    hub_tokens: set[str] = set()
    records: list[_EdgeRecord] = []
    for graph in graphs:
        _collect_node_statements(graph, report, neurons, hub_names, hub_tokens)
        records.extend(_collect_edges(graph, report, hub_names, hub_tokens))

    # Endpoints are mapped to raw node ids: Graphviz quoting variants (e.g.
    # `"A--B"` vs `"A--B "`) stay distinct hub nodes, which is what makes a
    # duplicated pair detectable below.
    pre_edges: defaultdict[str, list[_EdgeRecord]] = defaultdict(list)
    post_edges: defaultdict[str, list[_EdgeRecord]] = defaultdict(list)
    for record in records:
        if is_hub_name(record.dst):
            pre_edges[record.raw_dst].append(record)
        if is_hub_name(record.src):
            post_edges[record.raw_src].append(record)

    pairs: list[NeuronPair] = []
    seen_pairs: dict[tuple[str, str], str] = {}
    for token in sorted(hub_tokens):
        name = strip_quotes(token) or ""
        if not is_hub_name(name):
            continue
        pre = pre_edges.get(token, [])
        post = post_edges.get(token, [])
        if len(pre) != 1 or len(post) != 1:
            report.error(
                CODE_HUB_DEGREE,
                f"hub {name!r} has {len(pre)} pre edge(s) and {len(post)} post edge(s); "
                "expected exactly 1 of each",
            )
            continue
        pre_edge, post_edge = pre[0], post[0]
        pre_z = pre_edge.z_score
        post_z = post_edge.z_score
        if (
            pre_z is None
            or post_z is None
            or not math.isfinite(pre_z)
            or not math.isfinite(post_z)
            or pre_z < 0
            or post_z < 0
        ):
            continue  # z_invalid is already reported for the offending edge(s)

        pair = NeuronPair(pre_edge.src, post_edge.dst, pre_z, post_z)
        expected_name = f"{pair.source}{HUB_SEPARATOR}{pair.target}"
        if name != expected_name:
            report.error(
                CODE_HUB_ENDPOINT_MISMATCH,
                f"hub {name!r} does not match its edges "
                f"({pre_edge.src!r} -> {post_edge.dst!r}, expected {expected_name!r})",
            )
        if pair.source == pair.target:
            report.error(CODE_SELF_LOOP, f"hub {name!r} encodes the self-loop {pair.source!r}")
        key = (pair.source, pair.target)
        if key in seen_pairs:
            report.error(
                CODE_DUPLICATE_PAIR,
                f"duplicate pair ({pair.source!r}, {pair.target!r}) encoded by hubs "
                f"{seen_pairs[key]!r} and {name!r}",
            )
        else:
            seen_pairs[key] = name
        pairs.append(pair)

    # Real neurons that appear only as edge endpoints still belong to the graph
    # (they simply carry no `cent` metadata).
    for record in records:
        for endpoint in (record.src, record.dst):
            if is_hub_name(endpoint) or endpoint in PSEUDO_NODE_NAMES or endpoint in neurons:
                continue
            report.warning(
                CODE_CENT_MISSING,
                f"neuron {endpoint!r} appears in edges but has no node statement with cent metadata",
            )
            neurons[endpoint] = Neuron(endpoint, None, None)

    for pair in pairs:
        for role, endpoint in (("source", pair.source), ("target", pair.target)):
            if is_hub_name(endpoint) or endpoint not in neurons:
                report.error(
                    CODE_ENDPOINT_NOT_NEURON,
                    f"{role.title()} {endpoint!r} of pair ({pair.source!r}, {pair.target!r}) "
                    "is not a neuron",
                )

    filtered_pairs, filters = apply_filters(
        pairs, min_pre=min_pre, min_post=min_post, top_k=top_k
    )

    raw_edges = [
        RawEdge(
            src=record.src,
            dst=record.dst,
            is_hub=is_hub_name(record.src) or is_hub_name(record.dst),
            z_score=record.z_score,
            penwidth=record.penwidth,
            color=record.color,
        )
        for record in records
    ]

    pre_values = [pair.pre_z for pair in filtered_pairs]
    post_values = [pair.post_z for pair in filtered_pairs]
    pair_keys = {(pair.source, pair.target) for pair in filtered_pairs}
    reciprocity = sum(1 for pair in filtered_pairs if (pair.target, pair.source) in pair_keys)
    n_neurons = len(neurons)
    density = (
        len(filtered_pairs) / (n_neurons * (n_neurons - 1)) if n_neurons > 1 else 0.0
    )
    cent_distribution = Counter(
        neuron.cent_token for neuron in neurons.values() if neuron.cent_token is not None
    )

    metadata: dict[str, Any] = {
        "n_neurons": n_neurons,
        # distinct hub names (two raw tokens that normalize to the same name count once)
        "n_hubs": len(hub_names),
        "n_raw_edges": len(raw_edges),
        "n_pairs": len(filtered_pairs),
        "density": density,
        "reciprocity": reciprocity,
        "pre_stats": _summary_stats(pre_values),
        "post_stats": _summary_stats(post_values),
        "cent_distribution": dict(sorted(cent_distribution.items())),
        "filters": filters,
        "parser": DEFAULT_PARSER_NAME,
        "parser_version": pydot_version(),
        "created_utc": utc_timestamp(now),
    }

    parsed = ParsedGraph(
        source_file=path,
        file_sha256=digest,
        neurons=list(neurons.values()),
        pairs=filtered_pairs,
        raw_edges=raw_edges,
        metadata=metadata,
    )
    for problem in validate_payload_schema(parsed.to_dict()):
        report.error(CODE_SCHEMA, f"artifact does not conform to the Phase 01 schema: {problem}")
    return parsed, report


def parse_graphviz_file(
    source_path: str | Path,
    *,
    strict: bool = False,
    min_pre: float | None = None,
    min_post: float | None = None,
    top_k: int | None = None,
    now: Any = None,
) -> ParsedGraph:
    """Parse *source_path* into a :class:`ParsedGraph`.

    Raises :class:`GraphvizValidationError` (carrying the full
    :class:`ValidationReport`) when any validation *error* is found, or when
    *strict* is set and any *warning* was recorded.
    """
    parsed, report = build_parsed_graph(
        source_path,
        strict=strict,
        min_pre=min_pre,
        min_post=min_post,
        top_k=top_k,
        now=now,
    )
    if report.has_errors or (strict and report.warnings):
        raise GraphvizValidationError(
            f"{Path(source_path).name}: validation failed ({report.summary()})", report=report
        )
    return parsed


def artifact_path(parsed: ParsedGraph, outdir: str | Path, *, filtered: bool | None = None) -> Path:
    """Return ``<outdir>/<stem>/parsed_graph[.filtered].json`` for *parsed*.

    Filtered runs must never overwrite the canonical artifact, so the file name
    switches to ``parsed_graph.filtered.json`` whenever ``metadata.filters`` is
    non-empty (override with *filtered*).
    """
    if filtered is None:
        filtered = bool(parsed.metadata.get("filters"))
    name = FILTERED_ARTIFACT_NAME if filtered else DEFAULT_ARTIFACT_NAME
    return Path(outdir) / parsed.source_file.stem / name


def write_artifact(parsed: ParsedGraph, outdir: str | Path, *, filtered: bool | None = None) -> Path:
    """Atomically write *parsed* to :func:`artifact_path`."""
    return write_json_atomic(artifact_path(parsed, outdir, filtered=filtered), parsed.to_dict())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def resolve_inputs(paths: Sequence[str | Path], pattern: str = "*.gv") -> list[Path]:
    """Expand CLI inputs into a deterministic list of Graphviz files.

    A directory is globbed with *pattern* (``*.dot`` files are included as well
    when the default ``*.gv`` pattern is used); explicit files must carry a
    supported suffix.
    """
    patterns = [pattern, "*.dot"] if pattern == "*.gv" else [pattern]
    resolved: list[Path] = []
    for raw in paths:
        candidate = Path(raw)
        if candidate.is_dir():
            for item in sorted(
                {path for pat in patterns for path in candidate.glob(pat) if path.is_file()}
            ):
                if item.suffix.lower() in SUPPORTED_SUFFIXES:
                    resolved.append(item)
        elif candidate.is_file():
            if candidate.suffix.lower() not in SUPPORTED_SUFFIXES:
                LOGGER.error("%s: unsupported suffix (expected .gv or .dot)", candidate)
                continue
            resolved.append(candidate)
        else:
            LOGGER.error("%s: no such file or directory", candidate)
    unique: list[Path] = []
    for item in resolved:
        if item not in unique:
            unique.append(item)
    return unique


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.parsing.parse_graphviz",
        description="Parse Graphviz .gv files into data/processed/<stem>/parsed_graph.json",
    )
    parser.add_argument(
        "--input",
        "-i",
        nargs="+",
        required=True,
        metavar="PATH",
        help="one or more .gv/.dot files or directories (directories are globbed)",
    )
    parser.add_argument(
        "--outdir",
        "-o",
        default="data/processed",
        help="output root; the artifact lands in <outdir>/<stem>/ (default: data/processed)",
    )
    parser.add_argument(
        "--glob",
        dest="glob_pattern",
        default="*.gv",
        help="glob used for directory inputs (default: *.gv, plus *.dot)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="escalate validation warnings to errors (non-zero exit)",
    )
    parser.add_argument(
        "--min-pre", type=float, default=None, help="drop pairs with pre_z below this value"
    )
    parser.add_argument(
        "--min-post", type=float, default=None, help="drop pairs with post_z below this value"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="keep only the top-k pairs ranked by pre_z + post_z",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and report without writing any artifact",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="logging verbosity (default: INFO)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; returns the process exit code.

    ``0`` on success, ``1`` when any input failed parsing or validation and
    ``2`` when no input file matched.
    """
    args = build_argument_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(name)s: %(message)s",
    )

    inputs = resolve_inputs(args.input, args.glob_pattern)
    if not inputs:
        LOGGER.error("no input files matched: %s", ", ".join(str(item) for item in args.input))
        return 2

    failures = 0
    for source in inputs:
        try:
            parsed, report = build_parsed_graph(
                source,
                strict=args.strict,
                min_pre=args.min_pre,
                min_post=args.min_post,
                top_k=args.top_k,
            )
        except GraphvizValidationError as exc:
            LOGGER.error("%s: %s", source.name, exc)
            failures += 1
            continue

        for issue in report.warnings:
            LOGGER.warning("%s: [%s] %s", source.name, issue.code, issue.message)

        if report.has_errors or (args.strict and report.warnings):
            for issue in report.errors:
                LOGGER.error("%s: [%s] %s", source.name, issue.code, issue.message)
            if args.strict and report.warnings:
                LOGGER.error(
                    "%s: --strict escalates %d warning(s) to errors",
                    source.name,
                    len(report.warnings),
                )
            LOGGER.error("%s: validation failed (%s)", source.name, report.summary())
            failures += 1
            continue

        metadata = parsed.metadata
        LOGGER.info(
            "%s: neurons=%d hubs=%d raw_edges=%d pairs=%d density=%.6f reciprocity=%d",
            source.name,
            metadata["n_neurons"],
            metadata["n_hubs"],
            metadata["n_raw_edges"],
            metadata["n_pairs"],
            metadata["density"],
            metadata["reciprocity"],
        )

        artifact = artifact_path(parsed, args.outdir)
        if args.dry_run:
            LOGGER.info("%s: dry-run, would write %s", source.name, artifact)
            continue
        write_artifact(parsed, args.outdir)
        LOGGER.info("%s: wrote %s", source.name, artifact)

    if failures:
        LOGGER.error("%d of %d input file(s) failed", failures, len(inputs))
    return 1 if failures else 0


if __name__ == "__main__":  # pragma: no cover - exercised through subprocess tests
    sys.exit(main())