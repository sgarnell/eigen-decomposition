"""Phase 04 Update B -- unified pathway scoring (S-score).

Phase 04D produces one ``motifs.json`` (+ ``motifs.npz``, ``motifs.data.json`` and
``motifs.pathway.graphml``) per run, often several runs per dataset because a non-default
config gets its own ``.<config_hash8>`` filename segment.  Visual inspection of the
``.gv`` pathway graphs no longer scales, so this module reduces every pathway artifact to a
single scalar **S-score** that ranks them by structural richness.

The score is defined exactly as specified in ``execution-plans/04_update_a.md``::

    S = w_B * B + w_H * H + w_D * D - w_C * C

* ``B`` -- polarity balance        ``1 - |n_pos - n_neg| / n_edges``
* ``H`` -- polarity entropy        ``-sum_s p_s * log p_s`` over ``s in {-1, 0, +1}``
* ``D`` -- mode-matrix disagreement ``#(sign(W_mode) != sign(W_matrix)) / n_edges``
* ``C`` -- weight concentration    Phase 04D's ``pathway.weight_concentration``

The four weights default to ``1`` and are configurable on the command line.  A component
that is not defined for a run (``D`` on a mode-only or matrix-only run, or every component
when the pathway has zero edges) is reported as ``null`` and its term is **dropped** from
``S`` -- never silently replaced by zero -- and the row records why.

``W_matrix`` is rebuilt exactly as 04D builds it, from the sibling Phase 02
``z_matrix[.<variant>].json`` artifact (``sum_{i in A} sum_{j in B} Z[i, j]`` over the group
members), and ``W_mode`` is then recovered by inverting 04D's blend for the run's
``pathway_weight_rule``::

    signed/abs/positive/negative + source=both   W_mode = W - W_matrix
    hybrid (alpha > 0)                           W_mode = (W - (1 - alpha) * W_matrix) / alpha
    matrix-only / hybrid (alpha = 0)             not recoverable -> D = null
    mode-only / source=mode                      no matrix term -> D = null

This module is a **reader**: it never rewrites ``motifs.json``, ``motifs.npz``,
``motifs.data.json`` or ``motifs.pathway.graphml``, it never touches a Phase 01-03 artifact,
and it never changes ``MotifConfig`` or ``config_hash``.  It adds exactly two files to the
run directory (the CSV only when ``--csv`` is passed)::

    <outdir>/<stem>/motif_analysis_summary.json
    <outdir>/<stem>/motif_analysis_summary.csv      # --csv only

Because the raw pathway artifacts are all-negative on the reference dataset (Phase 02's
negative-weight bias), ``B = H = 0`` there and ``S = -C < 0``: a negative S on the canonical
run is the expected, correct consequence of the definition, not a defect.  ``S`` is not
normalised to ``[0, 1]``; compare scores only at equal weights.

CLI
---
::

    python -m src.clustering.motif_analysis \\
        -i outdir/FB4Yaffect_FB45_999prePost_001_all \\
        [--include-data] \\
        [--w-balance 1.0] [--w-entropy 1.0] \\
        [--w-disagreement 1.0] [--w-concentration 1.0] \\
        [--csv] [--stats] [--dry-run] [--strict] [--log-level INFO]
"""

from __future__ import annotations

import argparse
import csv as _csv
import hashlib
import io
import logging
import math
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.parsing.parse_graphviz import ValidationReport
from src.utils.io import (
    dumps_json,
    ensure_dir,
    read_json,
    sha256_file,
    utc_timestamp,
    write_json_atomic,
)

LOGGER = logging.getLogger("motif_analysis")

__all__ = [
    "CODE_SCORE_CONFIG",
    "CODE_SCORE_EMPTY",
    "CODE_SCORE_INPUT",
    "CODE_SCORE_MATRIX",
    "CODE_SCORE_WRITE",
    "CSV_COLUMNS",
    "DEFAULT_W_BALANCE",
    "DEFAULT_W_CONCENTRATION",
    "DEFAULT_W_DISAGREEMENT",
    "DEFAULT_W_ENTROPY",
    "GENERATOR",
    "MOTIF_DATA_SUFFIX",
    "MOTIF_STEM",
    "PHASE",
    "SCORE_STEM",
    "SCORE_VERSION",
    "PathwayScore",
    "ScoreConfig",
    "ScoreValidationError",
    "build_argument_parser",
    "build_pathway_matrix_contributions",
    "discover_motif_artifacts",
    "load_motif_artifact",
    "main",
    "matrix_artifact_candidates",
    "matrix_artifact_path",
    "mode_matrix_disagreement",
    "polarity_balance",
    "polarity_entropy",
    "rank_records",
    "recover_mode_contribution",
    "render_ranking",
    "render_score_table",
    "render_statistics",
    "s_score",
    "score_directory",
    "score_pathway",
    "weight_concentration_term",
    "write_summary",
]

PHASE = "04"
SCORE_VERSION = "1"
GENERATOR = "src.clustering.motif_analysis"

#: Canonical output stem, and the artifact prefixes handled as inputs.
SCORE_STEM = "motif_analysis_summary"
MOTIF_STEM = "motifs"
MATRIX_STEM = "z_matrix"
MATRIX_INPUT_PREFIX = "z_matrix"
MOTIF_DATA_SUFFIX = ".data.json"
SUPPORTED_SUFFIX = ".json"

#: Default S-score weights (all one, per the specification).
DEFAULT_W_BALANCE = 1.0
DEFAULT_W_ENTROPY = 1.0
DEFAULT_W_DISAGREEMENT = 1.0
DEFAULT_W_CONCENTRATION = 1.0

#: The exact score definition, recorded in the summary for reproducibility.
S_SCORE_DEFINITION = "S = w_B*B + w_H*H + w_D*D - w_C*C (an undefined term is dropped)"

#: ``motifs.json`` / ``motifs.<variant>[...].json`` (save-data handled separately).
MOTIF_JSON_RE = re.compile(r"^motifs(?:\.[A-Za-z0-9_]+)*\.json$")
MOTIF_DATA_RE = re.compile(r"^motifs(?:\.[A-Za-z0-9_]+)*\.data\.json$")
HASH8_RE = re.compile(r"^[0-9a-f]{8}$")

#: Rules whose stored ``weight`` is the raw ``W = W_mode + W_matrix`` when source is ``both``.
_SOURCE_BLEND_RULES = ("signed", "abs", "positive", "negative")

#: Weight rules for which the mode contribution cannot be recovered from the artifact.
_UNRECOVERABLE_RULES = ("mode-only", "matrix-only")

#: Blend alpha below which the hybrid inversion is numerically ill-conditioned.
HYBRID_ALPHA_WARN = 0.1

#: Tolerance for "this float is effectively zero" checks.
ZERO_TOL = 1e-12

# Validation issue codes (one per rule).
CODE_SCORE_INPUT = "score_input"
CODE_SCORE_MATRIX = "score_matrix"
CODE_SCORE_EMPTY = "score_empty"
CODE_SCORE_CONFIG = "score_config"
CODE_SCORE_WRITE = "score_write"


class ScoreValidationError(ValueError):
    """Raised when a scoring input/output violates a validation rule.

    The full :class:`src.parsing.parse_graphviz.ValidationReport` is available as
    :attr:`report`, mirroring ``GraphvizValidationError`` / ``ZMatrixValidationError`` /
    ``SpectralValidationError`` / ``MotifValidationError``.
    """

    def __init__(self, message: str, report: ValidationReport | None = None) -> None:
        super().__init__(message)
        self.report = report


@dataclass(frozen=True)
class ScoreConfig:
    """The S-score weights (user-configurable; never part of the Phase 04 config hash)."""

    w_balance: float = DEFAULT_W_BALANCE
    w_entropy: float = DEFAULT_W_ENTROPY
    w_disagreement: float = DEFAULT_W_DISAGREEMENT
    w_concentration: float = DEFAULT_W_CONCENTRATION

    def hash_fields(self) -> dict[str, float]:
        """The hashed subset: exactly the four weights."""
        return {
            "w_balance": float(self.w_balance),
            "w_entropy": float(self.w_entropy),
            "w_disagreement": float(self.w_disagreement),
            "w_concentration": float(self.w_concentration),
        }

    def score_hash(self) -> str:
        """``sha256`` of the canonical weight block, first 8 hex characters."""
        return hashlib.sha256(dumps_json(self.hash_fields()).encode("utf-8")).hexdigest()[:8]

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serializable ``scorer_config`` block."""
        return {
            "w_balance": float(self.w_balance),
            "w_entropy": float(self.w_entropy),
            "w_disagreement": float(self.w_disagreement),
            "w_concentration": float(self.w_concentration),
            "definition": S_SCORE_DEFINITION,
            "score_hash": self.score_hash(),
        }

    def is_default(self) -> bool:
        """True when every weight equals its documented default."""
        return all(
            math.isclose(value, default, rel_tol=0.0, abs_tol=ZERO_TOL)
            for value, default in (
                (self.w_balance, DEFAULT_W_BALANCE),
                (self.w_entropy, DEFAULT_W_ENTROPY),
                (self.w_disagreement, DEFAULT_W_DISAGREEMENT),
                (self.w_concentration, DEFAULT_W_CONCENTRATION),
            )
        )

# ---------------------------------------------------------------------------
# S-score components (pure functions of scalars / small arrays)
# ---------------------------------------------------------------------------
def _canonical(value: float) -> float:
    """Canonicalize ``-0.0`` to ``0.0`` (``json.dumps(-0.0)`` would break byte stability)."""
    number = float(value)
    return 0.0 if number == 0.0 else number


def _finite(value: Any, *, label: str) -> float:
    """Return *value* as a finite float or raise :class:`ScoreValidationError`."""
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ScoreValidationError(f"{label} is not a number: {value!r}") from exc
    if not math.isfinite(number):
        raise ScoreValidationError(f"{label} is not finite: {value!r}")
    return number


def _as_dict(value: Any) -> dict[str, Any]:
    """Return *value* when it is a mapping, else an empty dict (keeps the reads type-safe)."""
    return value if isinstance(value, dict) else {}


def polarity_balance(n_positive: int, n_negative: int, n_edges: int) -> float | None:
    """``B = 1 - |n_pos - n_neg| / n_edges`` -- ``None`` when there are no edges.

    ``B = 0`` means uniform polarity (all edges one sign), ``B = 1`` perfect balance.
    Zero-polarity edges make ``|n_pos - n_neg| < n_edges``, so they act as a mild bonus;
    that is the specified formula (the denominator is ``n_edges``, not ``n_pos + n_neg``).
    """
    if int(n_edges) <= 0:
        return None
    return _canonical(1.0 - abs(int(n_positive) - int(n_negative)) / float(int(n_edges)))


def polarity_entropy(n_positive: int, n_negative: int, n_zero: int) -> float | None:
    """``H = -sum_s p_s log p_s`` over ``s in {-1, 0, +1}`` -- ``None`` when no edges.

    Natural log; ``p_s = 0`` terms are skipped (the ``0 log 0`` limit), so ``H`` is
    computed even when ``p_zero = 0``.  ``H`` is 0 for a single polarity class and reaches
    ``log 3 ~ 1.0986`` only when all three classes are equally represented.
    """
    n_edges = int(n_positive) + int(n_negative) + int(n_zero)
    if n_edges <= 0:
        return None
    total = 0.0
    for count in (int(n_positive), int(n_negative), int(n_zero)):
        if count:
            probability = count / float(n_edges)
            total -= probability * math.log(probability)
    return _canonical(total)


def weight_concentration_term(pathway: dict[str, Any]) -> float | None:
    """``C`` -- Phase 04D's ``pathway.weight_concentration`` (``None`` when not computed)."""
    raw = pathway.get("weight_concentration")
    if raw is None:
        return None
    return _canonical(_finite(raw, label="pathway.weight_concentration"))




def recover_mode_contribution(
    weight: float,
    matrix_weight: float | None,
    *,
    rule: str,
    source: str,
    alpha: float = 0.0,
) -> tuple[float | None, str | None]:
    """Recover ``W_mode`` for one edge from the stored ``weight`` and ``W_matrix``.

    Returns ``(W_mode, reason)``.  When the mode contribution cannot be recovered from the
    artifact, ``W_mode`` is ``None`` and *reason* explains why (recorded in the summary as
    ``d_unavailable_reason``).
    """
    if rule in _UNRECOVERABLE_RULES or source == "matrix":
        label = rule if rule in _UNRECOVERABLE_RULES else "matrix-source"
        return None, f"{label}: no mode contribution stored"
    if rule in _SOURCE_BLEND_RULES:
        if source != "both":
            return None, "mode-only: no matrix contribution"
        if matrix_weight is None:
            return None, "z_matrix artifact unavailable"
        return _canonical(float(weight) - float(matrix_weight)), None
    if rule == "hybrid":
        if source != "hybrid":
            return None, "mode-only: no matrix contribution"
        if matrix_weight is None:
            return None, "z_matrix artifact unavailable"
        ratio = float(alpha)
        if ratio <= 0.0:
            return None, "matrix-only (hybrid alpha 0): no mode contribution stored"
        return (
            _canonical((float(weight) - (1.0 - ratio) * float(matrix_weight)) / ratio),
            None,
        )
    return None, f"unknown pathway_weight_rule {rule!r}"


def mode_matrix_disagreement(
    edges: Sequence[dict[str, Any]],
    matrix_contributions: dict[tuple[str, str], float] | None,
    *,
    rule: str,
    source: str,
    alpha: float = 0.0,
) -> tuple[float | None, str | None]:
    """``D`` -- fraction of kept edges where ``sign(W_mode) != sign(W_matrix)``.

    ``matrix_contributions`` maps ``(source_group, target_group)`` to ``W_matrix``; when it
    is ``None`` (and the rule could use it) the z_matrix artifact was unavailable.  Returns
    ``(D, reason)``; ``D`` is ``None`` when the mode contribution is not recoverable, when
    there are no edges, or when a matrix contribution is missing for some edge.
    """
    n_edges = len(edges)
    if n_edges == 0:
        return None, "no pathway edges"
    contributions = matrix_contributions or {}
    disagreements = 0
    for edge in edges:
        source_id = str(edge.get("source"))
        target_id = str(edge.get("target"))
        matrix_weight = contributions.get((source_id, target_id))
        mode_weight, reason = recover_mode_contribution(
            _finite(edge.get("weight", 0.0), label=f"edge {source_id}->{target_id} weight"),
            None if matrix_weight is None else float(matrix_weight),
            rule=str(rule),
            source=str(source),
            alpha=float(alpha),
        )
        if mode_weight is None:
            return None, reason
        if matrix_weight is None:
            return None, "z_matrix artifact unavailable"
        if np.sign(mode_weight) != np.sign(float(matrix_weight)):
            disagreements += 1
    return _canonical(disagreements / float(n_edges)), None


def s_score(
    balance: float | None,
    entropy: float | None,
    disagreement: float | None,
    concentration: float | None,
    config: ScoreConfig | None = None,
) -> float | None:
    """``S = w_B*B + w_H*H + w_D*D - w_C*C`` with undefined terms dropped.

    ``None`` when no term is defined at all (an edge-free pathway), so a degenerate run
    never masquerades as a score of 0.
    """
    settings = config or ScoreConfig()
    total = 0.0
    defined = False
    for value, weight in (
        (balance, settings.w_balance),
        (entropy, settings.w_entropy),
        (disagreement, settings.w_disagreement),
    ):
        if value is not None:
            total += float(weight) * float(value)
            defined = True
    if concentration is not None:
        total -= float(settings.w_concentration) * float(concentration)
        defined = True
    return _canonical(total) if defined else None


def build_pathway_matrix_contributions(
    groups: Sequence[dict[str, Any]],
    matrix: Any,
    neuron_order: Sequence[str],
) -> dict[tuple[str, str], float]:
    """Rebuild 04D's ``W_matrix[G, G]`` as ``{(source, target): value}``.

    Uses the same ``I.T @ Z @ I`` indicator reduction 04D's ``_group_pair_sum`` uses, so the
    group sums agree bit-for-bit with the pathway that was built.  Group ids keep their
    payload order (which is 04D's node order).
    """
    if matrix is None:
        raise ScoreValidationError("[score_matrix] no Phase 02 matrix was supplied")
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ScoreValidationError("[score_matrix] the Phase 02 matrix must be square")
    order = [str(name) for name in neuron_order]
    if values.shape[0] != len(order):
        raise ScoreValidationError(
            f"[score_matrix] the Phase 02 matrix is {values.shape[0]}x{values.shape[0]} but "
            f"the neuron order has {len(order)} entries"
        )
    position = {name: index for index, name in enumerate(order)}
    n_groups = len(groups)
    indicator = np.zeros((len(order), n_groups), dtype=np.float64)
    group_ids: list[str] = []
    for index, group in enumerate(groups):
        group_ids.append(str(group.get("group_id")))
        for member in group.get("members") or ():
            row = position.get(str(member))
            if row is not None:
                indicator[row, index] = 1.0
    contributions = indicator.T @ values @ indicator
    return {
        (group_ids[source], group_ids[target]): _canonical(contributions[source, target])
        for source in range(n_groups)
        for target in range(n_groups)
    }


# ---------------------------------------------------------------------------
# The per-artifact score record
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PathwayScore:
    """One scored motif artifact (a row of the summary ranking)."""

    artifact: str = "motifs.json"
    artifact_path: str = ""
    artifact_kind: str = "motifs"
    artifact_sha256: str = ""
    config_hash: str | None = None
    stage: str | None = None
    pathway_source: str | None = None
    pathway_weight_rule: str | None = None
    pathway_polarity_rule: str | None = None
    hybrid_alpha: float | None = None
    n_nodes: int | None = None
    n_edges: int = 0
    n_positive: int = 0
    n_negative: int = 0
    n_zero: int = 0
    n_intra: int = 0
    n_cross: int = 0
    abs_max: float | None = None
    weight_concentration: float | None = None
    balance: float | None = None
    entropy: float | None = None
    disagreement: float | None = None
    concentration: float | None = None
    s_score: float | None = None
    weights_used: dict[str, float] | None = None
    d_unavailable_reason: str | None = None
    issues: tuple[str, ...] = ()
    rank: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """The JSON-serializable record written into ``rankings[]``."""
        return {
            "rank": self.rank,
            "artifact": self.artifact,
            "artifact_path": self.artifact_path,
            "artifact_kind": self.artifact_kind,
            "artifact_sha256": self.artifact_sha256,
            "config_hash": self.config_hash,
            "stage": self.stage,
            "pathway_source": self.pathway_source,
            "pathway_weight_rule": self.pathway_weight_rule,
            "pathway_polarity_rule": self.pathway_polarity_rule,
            "hybrid_alpha": self.hybrid_alpha,
            "n_nodes": self.n_nodes,
            "n_edges": int(self.n_edges),
            "n_positive": int(self.n_positive),
            "n_negative": int(self.n_negative),
            "n_zero": int(self.n_zero),
            "n_intra": int(self.n_intra),
            "n_cross": int(self.n_cross),
            "abs_max": self.abs_max,
            "weight_concentration": self.weight_concentration,
            "balance": self.balance,
            "entropy": self.entropy,
            "disagreement": self.disagreement,
            "concentration": self.concentration,
            "s_score": self.s_score,
            "weights_used": dict(self.weights_used or {}),
            "d_unavailable_reason": self.d_unavailable_reason,
            "issues": list(self.issues),
        }


# ---------------------------------------------------------------------------
# Input discovery and loading
# ---------------------------------------------------------------------------
def _is_motif_artifact(name: str) -> bool:
    """True for ``motifs.json`` / ``motifs.<variant>[...].json`` (never ``.data.json``)."""
    if name.endswith(MOTIF_DATA_SUFFIX):
        return False
    return bool(MOTIF_JSON_RE.fullmatch(name))


def _is_motif_data_artifact(name: str) -> bool:
    """True for ``motifs.data.json`` / ``motifs.<variant>[...].data.json``."""
    return bool(MOTIF_DATA_RE.fullmatch(name))


def _accepts(name: str, *, include_data: bool) -> bool:
    return _is_motif_artifact(name) or (include_data and _is_motif_data_artifact(name))


def discover_motif_artifacts(
    path: str | Path, *, include_data: bool = False
) -> list[Path]:
    """Return the motif artifacts under *path* in a deterministic order.

    *path* may be a single artifact (returned as-is when it is a motif file) or a
    directory.  A directory is scanned for ``motifs.json`` and ``motifs.<variant>.json``;
    when it holds none, its subdirectories are searched recursively (so a processed tree
    can be scored in one run).  ``motifs.data.json`` is skipped unless *include_data* --
    it duplicates the canonical payload, so scoring it too would double-count a run.
    """
    candidate = Path(path)
    if candidate.is_file():
        if not _accepts(candidate.name, include_data=include_data):
            raise ScoreValidationError(
                f"[{CODE_SCORE_INPUT}] {candidate.name} is not a motif artifact "
                f"(expected motifs[.<variant>].json)"
            )
        return [candidate]
    if not candidate.is_dir():
        raise ScoreValidationError(f"[{CODE_SCORE_INPUT}] {candidate} is not a file or directory")

    matches = sorted(
        item
        for item in candidate.iterdir()
        if item.is_file() and _accepts(item.name, include_data=include_data)
    )
    if matches:
        return matches
    recursive = sorted(
        item
        for item in candidate.rglob(f"{MOTIF_STEM}*.json")
        if item.is_file() and _accepts(item.name, include_data=include_data)
    )
    return recursive


def load_motif_artifact(path: str | Path) -> dict[str, Any]:
    """Load one motif artifact and validate the blocks the scorer depends on."""
    source = Path(path)
    try:
        payload = read_json(source)
    except Exception as exc:
        raise ScoreValidationError(f"[{CODE_SCORE_INPUT}] could not read {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ScoreValidationError(f"[{CODE_SCORE_INPUT}] {source.name} is not a JSON object")
    pathway = payload.get("pathway")
    if not isinstance(pathway, dict):
        raise ScoreValidationError(
            f"[{CODE_SCORE_INPUT}] {source.name} has no 'pathway' block (is this a Phase 04D "
            f"artifact?)"
        )
    if not isinstance(payload.get("config"), dict):
        raise ScoreValidationError(f"[{CODE_SCORE_INPUT}] {source.name} has no 'config' block")
    edges = pathway.get("edges")
    if not isinstance(edges, list):
        raise ScoreValidationError(f"[{CODE_SCORE_INPUT}] {source.name} pathway.edges is not a list")
    declared = pathway.get("n_edges")
    if isinstance(declared, int) and declared != len(edges):
        raise ScoreValidationError(
            f"[{CODE_SCORE_INPUT}] {source.name} declares n_edges={declared} but lists "
            f"{len(edges)} edges"
        )
    return payload


def matrix_artifact_candidates(motif_path: str | Path) -> list[Path]:
    """Candidate sibling Phase 02 artifacts for a motif artifact, best first.

    ``motifs.json`` -> ``z_matrix.json``; ``motifs.f8652585.json`` ->
    ``z_matrix.f8652585.json``; ``motifs.f8652585.e973fece.json`` ->
    ``z_matrix.f8652585.e973fece.json`` then ``z_matrix.f8652585.json``.

    A trailing ``.<config_hash8>`` segment is the Phase 04 config hash rather than part of
    the Phase 02 variant, but the two spellings are indistinguishable (both are 8 hex
    characters, e.g. the Phase 02 hash ``f8652585``), so the resolution is by existence:
    :func:`matrix_artifact_path` returns the first candidate on disk.
    """
    source = Path(motif_path)
    stem = source.stem
    suffix = stem[len(MOTIF_STEM):] if stem.startswith(MOTIF_STEM) else ""
    segments = [segment for segment in suffix.split(".") if segment]
    without_last = segments[:-1] if segments and HASH8_RE.fullmatch(segments[-1]) else segments
    names = [
        ".".join([MATRIX_STEM, *segments]) if segments else MATRIX_STEM,
        ".".join([MATRIX_STEM, *without_last]) if without_last else MATRIX_STEM,
        MATRIX_STEM,
    ]
    ordered: list[Path] = []
    for name in names:
        candidate = source.parent / f"{name}.json"
        if candidate not in ordered:
            ordered.append(candidate)
    return ordered


def matrix_artifact_path(motif_path: str | Path) -> Path:
    """The resolved sibling Phase 02 artifact (the first candidate that exists, else best)."""
    candidates = matrix_artifact_candidates(motif_path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def load_matrix_contributions(
    motif_path: str | Path,
    payload: dict[str, Any],
) -> dict[tuple[str, str], float] | None:
    """Rebuild ``W_matrix`` for *payload* from the sibling Phase 02 artifact.

    Returns ``None`` when no sibling artifact exists (the caller records why D is then
    unavailable); a mismatched neuron order is a hard :class:`ScoreValidationError`.
    """
    candidate = next(
        (item for item in matrix_artifact_candidates(motif_path) if item.is_file()), None
    )
    if candidate is None:
        return None
    from src.matrices.build_square_matrix import load_z_matrix

    try:
        z_matrix = load_z_matrix(candidate)
    except Exception as exc:
        raise ScoreValidationError(
            f"[{CODE_SCORE_MATRIX}] could not load {candidate}: {exc}"
        ) from exc
    neuron_order = [str(name) for name in payload.get("neuron_order") or ()]
    matrix_order = [str(name) for name in z_matrix.neuron_order]
    if neuron_order and matrix_order != neuron_order:
        raise ScoreValidationError(
            f"[{CODE_SCORE_MATRIX}] {candidate.name} neuron order does not match "
            f"{Path(motif_path).name}"
        )
    groups = payload.get("groups")
    if not isinstance(groups, list) or not groups:
        raise ScoreValidationError(
            f"[{CODE_SCORE_MATRIX}] {Path(motif_path).name} has no groups to aggregate"
        )
    return build_pathway_matrix_contributions(groups, z_matrix.matrix, matrix_order)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def _validate_weights(config: ScoreConfig) -> None:
    """Every S-score weight must be finite and non-negative."""
    for name, value in config.hash_fields().items():
        number = _finite(value, label=f"--{name.replace('_', '-')}")
        if number < 0.0:
            raise ScoreValidationError(
                f"[{CODE_SCORE_CONFIG}] {name} must be >= 0, got {number!r}"
            )


def _polarity_counts(edges: Sequence[dict[str, Any]]) -> tuple[int, int, int]:
    """``(n_positive, n_negative, n_zero)`` over the kept edges."""
    positive = negative = zero = 0
    for edge in edges:
        polarity = edge.get("polarity")
        if polarity is None:
            sign = int(np.sign(_finite(edge.get("weight", 0.0), label="edge weight")))
        else:
            sign = int(np.sign(_finite(polarity, label="edge polarity")))
        if sign > 0:
            positive += 1
        elif sign < 0:
            negative += 1
        else:
            zero += 1
    return positive, negative, zero


def _disagreement_required(rule: str, source: str, alpha: float) -> bool:
    """True when ``D`` is *defined* for this run and therefore needs ``W_matrix``."""
    if rule in _SOURCE_BLEND_RULES:
        return source == "both"
    if rule == "hybrid":
        return source == "hybrid" and float(alpha) > 0.0
    return False



def score_pathway(
    payload: dict[str, Any],
    *,
    matrix_contributions: dict[tuple[str, str], float] | None = None,
    config: ScoreConfig | None = None,
    artifact: str | None = None,
    artifact_path: str | Path | None = None,
    artifact_kind: str = "motifs",
    artifact_sha256: str = "",
    report: ValidationReport | None = None,
) -> PathwayScore:
    """Score one loaded motif payload and return its :class:`PathwayScore` record.

    *matrix_contributions* is the rebuilt ``W_matrix`` map (see
    :func:`build_pathway_matrix_contributions`); passing ``None`` leaves ``D`` undefined with
    a recorded reason.  The payload and the Phase 02 matrix are never mutated.
    """
    settings = config or ScoreConfig()
    _validate_weights(settings)
    pathway = _as_dict(payload.get("pathway"))
    motif_config = _as_dict(payload.get("config"))
    edges: list[dict[str, Any]] = list(pathway.get("edges") or [])
    n_edges = len(edges)
    positive, negative, zero = _polarity_counts(edges)
    name = artifact or (Path(artifact_path).name if artifact_path else "motifs.json")

    declared_positive = pathway.get("n_positive")
    declared_negative = pathway.get("n_negative")
    if isinstance(declared_positive, int) and declared_positive != positive:
        raise ScoreValidationError(
            f"[{CODE_SCORE_INPUT}] {name} declares n_positive={declared_positive} but "
            f"{positive} edge(s) are positive"
        )
    if isinstance(declared_negative, int) and declared_negative != negative:
        raise ScoreValidationError(
            f"[{CODE_SCORE_INPUT}] {name} declares n_negative={declared_negative} but "
            f"{negative} edge(s) are negative"
        )

    rule = str(motif_config.get("pathway_weight_rule") or "signed")
    source = str(pathway.get("source") or motif_config.get("pathway_source") or "mode")
    alpha = float(motif_config.get("pathway_hybrid_alpha") or 0.0)

    issues: list[str] = []
    balance = polarity_balance(positive, negative, n_edges)
    entropy = polarity_entropy(positive, negative, zero)
    concentration = weight_concentration_term(pathway)

    disagreement: float | None = None
    if n_edges == 0:
        unavailable = "no pathway edges"
        issues.append(f"[{CODE_SCORE_EMPTY}] the pathway has no edges; no score is defined")
        if report is not None:
            report.warning(
                CODE_SCORE_EMPTY,
                f"{name}: the pathway has no edges, so S is undefined",
            )
    elif _disagreement_required(rule, source, alpha):
        if matrix_contributions is None:
            missing = matrix_artifact_path(artifact_path or name).name
            unavailable = f"z_matrix artifact not found ({missing})"
            issues.append(f"[{CODE_SCORE_MATRIX}] {unavailable}; D is undefined for this row")
            if report is not None:
                report.warning(
                    CODE_SCORE_MATRIX,
                    f"{name}: {unavailable}, so D is undefined",
                )
        else:
            disagreement, unavailable = mode_matrix_disagreement(
                edges, matrix_contributions, rule=rule, source=source, alpha=alpha
            )
    else:
        _, unavailable = recover_mode_contribution(
            0.0, None, rule=rule, source=source, alpha=alpha
        )

    if rule == "hybrid" and 0.0 < alpha < HYBRID_ALPHA_WARN:
        issues.append(
            f"[{CODE_SCORE_CONFIG}] hybrid alpha {alpha:g} < {HYBRID_ALPHA_WARN:g}: the "
            f"W_mode inversion is numerically ill-conditioned"
        )

    score = None if n_edges == 0 else s_score(
        balance, entropy, disagreement, concentration, settings
    )
    return PathwayScore(
        artifact=name,
        artifact_path=str(artifact_path or ""),
        artifact_kind=artifact_kind,
        artifact_sha256=artifact_sha256,
        config_hash=motif_config.get("config_hash"),
        stage=str(motif_config.get("stage") or payload.get("stage") or "") or None,
        pathway_source=source,
        pathway_weight_rule=rule,
        pathway_polarity_rule=str(motif_config.get("pathway_polarity_rule") or "sign"),
        hybrid_alpha=alpha,
        n_nodes=pathway.get("n_nodes"),
        n_edges=n_edges,
        n_positive=positive,
        n_negative=negative,
        n_zero=zero,
        n_intra=int(pathway.get("n_intra") or 0),
        n_cross=int(pathway.get("n_cross") or 0),
        abs_max=pathway.get("abs_max"),
        weight_concentration=pathway.get("weight_concentration"),
        balance=balance,
        entropy=entropy,
        disagreement=disagreement,
        concentration=concentration,
        s_score=score,
        weights_used=settings.hash_fields(),
        d_unavailable_reason=unavailable,
        issues=tuple(issues),
    )


# ---------------------------------------------------------------------------
# Directory-level scoring, ranking and the summary document
# ---------------------------------------------------------------------------
def _needs_matrix(payload: dict[str, Any]) -> bool:
    """True when this payload's rules make ``D`` defined (so ``W_matrix`` is needed)."""
    pathway = _as_dict(payload.get("pathway"))
    motif_config = _as_dict(payload.get("config"))
    rule = str(motif_config.get("pathway_weight_rule") or "signed")
    source = str(pathway.get("source") or motif_config.get("pathway_source") or "mode")
    alpha = float(motif_config.get("pathway_hybrid_alpha") or 0.0)
    return _disagreement_required(rule, source, alpha)


def rank_records(records: Sequence[PathwayScore]) -> list[PathwayScore]:
    """Return the records ordered by S descending (ties on artifact name), ranks assigned.

    Unscored rows (``S = None``, e.g. an edge-free pathway) sort last with ``rank = None``.
    """
    from dataclasses import replace

    ordered = sorted(
        records,
        key=lambda entry: (
            entry.s_score is None,
            -entry.s_score if entry.s_score is not None else 0.0,
            entry.artifact,
        ),
    )
    ranked: list[PathwayScore] = []
    position = 0
    for item in ordered:
        if item.s_score is None:
            ranked.append(replace(item, rank=None))
        else:
            position += 1
            ranked.append(replace(item, rank=position))
    return ranked


def score_directory(
    directory: str | Path,
    *,
    config: ScoreConfig | None = None,
    include_data: bool = False,
) -> dict[str, Any]:
    """Score every motif artifact under *directory* and return the summary document.

    The summary is JSON-serializable and deterministic; it is the payload written by
    :func:`write_summary`.  Failed artifacts are reported in ``metadata.issues`` (with a
    severity) and never abort the whole directory.
    """
    settings = config or ScoreConfig()
    _validate_weights(settings)
    target = Path(directory)
    artifacts = discover_motif_artifacts(target, include_data=include_data)
    report = ValidationReport()
    records: list[PathwayScore] = []

    for artifact in artifacts:
        try:
            payload = load_motif_artifact(artifact)
            contributions = None
            if _needs_matrix(payload):
                contributions = load_matrix_contributions(artifact, payload)
            records.append(
                score_pathway(
                    payload,
                    matrix_contributions=contributions,
                    config=settings,
                    artifact_path=artifact,
                    artifact_kind="data" if _is_motif_data_artifact(artifact.name) else "motifs",
                    artifact_sha256=sha256_file(artifact),
                    report=report,
                )
            )
        except Exception as exc:  # malformed artifact, mismatched matrix, bad config
            code = CODE_SCORE_CONFIG
            text = str(exc)
            for candidate in (CODE_SCORE_MATRIX, CODE_SCORE_INPUT, CODE_SCORE_CONFIG):
                if f"[{candidate}]" in text:
                    code = candidate
                    break
            report.error(code, f"{artifact.name}: {text}")
            LOGGER.error("%s: [%s] %s", artifact.name, code, text)

    ranked = rank_records(records)
    scores = [record.s_score for record in ranked if record.s_score is not None]
    issues = [
        {"severity": issue.severity, "code": issue.code, "message": issue.message}
        for issue in [*report.errors, *report.warnings]
    ]
    return {
        "provenance": {
            "phase": PHASE,
            "stage": "04B",
            "generator": GENERATOR,
            "scorer_version": SCORE_VERSION,
            "source_directory": str(target),
            "n_artifacts": len(artifacts),
            "n_scored": len(scores),
            "created_utc": utc_timestamp(),
        },
        "scorer_config": settings.to_dict(),
        "summary": {
            "n_artifacts": len(artifacts),
            "n_scored": len(scores),
            "n_skipped": len(artifacts) - len(scores),
            "s_max": max(scores) if scores else None,
            "s_min": min(scores) if scores else None,
            "s_mean": (sum(scores) / len(scores)) if scores else None,
            "best_artifact": ranked[0].artifact if scores else None,
            "worst_artifact": ranked[len(scores) - 1].artifact if scores else None,
        },
        "rankings": [record.to_dict() for record in ranked],
        "metadata": {
            "n_errors": len(report.errors),
            "n_warnings": len(report.warnings),
            "issues": issues,
            "definition": S_SCORE_DEFINITION,
        },
    }



# ---------------------------------------------------------------------------
# Terminal output and artifact writing
# ---------------------------------------------------------------------------
def _score_text(value: float | None, *, digits: int = 6) -> str:
    """Format an optional score component (``n/a`` when undefined)."""
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def render_score_table(records: Sequence[PathwayScore]) -> str:
    """Terminal ranking table for :class:`PathwayScore` records."""
    return render_ranking([record.to_dict() for record in records])


def render_ranking(records: Sequence[dict[str, Any]]) -> str:
    """Terminal ranking table for the summary's ``rankings`` records (sorted by S)."""
    header = (
        f"{'rank':>4}  {'artifact':<34} {'edges':>5}  {'B':>8}  {'H':>8}  "
        f"{'D':>8}  {'C':>8}  {'S':>9}"
    )
    lines = ["Phase 04B -- pathway ranking (S descending)", header, "-" * len(header)]
    for record in records:
        rank = f"{record.get('rank')}" if record.get("rank") is not None else "-"
        lines.append(
            f"{rank:>4}  {str(record.get('artifact')):<34} "
            f"{int(record.get('n_edges') or 0):>5}  "
            f"{_score_text(record.get('balance')):>8}  "
            f"{_score_text(record.get('entropy')):>8}  "
            f"{_score_text(record.get('disagreement')):>8}  "
            f"{_score_text(record.get('concentration')):>8}  "
            f"{_score_text(record.get('s_score')):>9}"
        )
    if not records:
        lines.append("  (no motif artifact matched)")
    return "\n".join(lines)


def render_statistics(summary: dict[str, Any]) -> str:
    """Terminal-only ``--stats`` block for a directory summary."""
    stats = summary.get("summary") or {}
    provenance = summary.get("provenance") or {}
    lines = [
        "Phase 04B -- S-score statistics",
        f"  directory           : {provenance.get('source_directory')}",
        f"  artifacts scored    : {stats.get('n_scored')} of {stats.get('n_artifacts')} "
        f"(skipped {stats.get('n_skipped')})",
        f"  S max / min / mean  : {_score_text(stats.get('s_max'))} / "
        f"{_score_text(stats.get('s_min'))} / {_score_text(stats.get('s_mean'))}",
        f"  best                : {stats.get('best_artifact')}",
        f"  weights             : B {summary['scorer_config']['w_balance']:g} / "
        f"H {summary['scorer_config']['w_entropy']:g} / "
        f"D {summary['scorer_config']['w_disagreement']:g} / "
        f"-C {summary['scorer_config']['w_concentration']:g}  "
        f"(score_hash {summary['scorer_config']['score_hash']})",
        f"  definition          : {S_SCORE_DEFINITION}",
    ]
    for issue in summary.get("metadata", {}).get("issues") or []:
        lines.append(
            f"  {issue['severity']:<7}            : [{issue['code']}] {issue['message']}"
        )
    return "\n".join(lines)


def render_summary_box(
    summary: dict[str, Any],
    paths: dict[str, Any],
    *,
    written: bool = True,
    csv: bool = False,
    stats: bool = False,
) -> str:
    """Post-run completion box (terminal only -- never a GUI popup)."""
    stats_block = summary.get("summary") or {}

    def _skip(reason: str) -> str:
        return f"skipped ({reason})"

    def _name(key: str) -> str:
        path = paths.get(key)
        if written and isinstance(path, (str, Path)) and str(path):
            return Path(path).name
        return _skip("--dry-run")

    if not csv:
        csv_line = "not requested (--csv)"
    else:
        csv_line = _name("csv")
    lines = [
        "Phase 04B complete",
        f"Artifacts scored: {stats_block.get('n_scored')} of {stats_block.get('n_artifacts')}",
        f"S-score range: {_score_text(stats_block.get('s_min'))} .. "
        f"{_score_text(stats_block.get('s_max'))}",
        f"JSON written: {_name('json')}",
        f"CSV written:  {csv_line}",
        "Stats: printed above (--stats)" if stats else "Stats: not requested (--stats)",
    ]
    width = max(max(len(line) for line in lines), 60)
    border = "+" + "-" * (width + 2) + "+"
    body = [f"| {line.ljust(width)} |" for line in lines]
    return "\n".join([border, *body, border])



#: The fixed CSV column order (written only with ``--csv``).
CSV_COLUMNS = (
    "rank",
    "artifact",
    "artifact_kind",
    "stage",
    "config_hash",
    "pathway_source",
    "pathway_weight_rule",
    "pathway_polarity_rule",
    "hybrid_alpha",
    "n_nodes",
    "n_edges",
    "n_positive",
    "n_negative",
    "n_zero",
    "n_intra",
    "n_cross",
    "abs_max",
    "weight_concentration",
    "balance",
    "entropy",
    "disagreement",
    "concentration",
    "s_score",
    "d_unavailable_reason",
)


def _write_text_atomic(path: str | Path, text: str) -> Path:
    """Write *text* atomically (temp file in the destination directory + ``os.replace``)."""
    destination = Path(path)
    ensure_dir(destination.parent)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=str(destination.parent),
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def _csv_text(records: Sequence[dict[str, Any]]) -> str:
    """Render the ranking rows as CSV text (stdout-free, deterministic)."""
    buffer = io.StringIO()
    writer = _csv.writer(buffer, lineterminator="\n")
    writer.writerow(CSV_COLUMNS)
    for record in records:
        row: list[str] = []
        for column in CSV_COLUMNS:
            value = record.get(column)
            if value is None:
                row.append("")
            elif isinstance(value, float):
                row.append(f"{value:.12g}")
            else:
                row.append(str(value))
        writer.writerow(row)
    return buffer.getvalue()


def write_summary(
    directory: str | Path,
    summary: dict[str, Any],
    *,
    csv: bool = False,
) -> dict[str, Path]:
    """Write ``motif_analysis_summary.json`` (always) and the CSV (only when asked).

    Both writes are atomic, so a failure never leaves a partial artifact.  The JSON is
    written first: it is the artifact of record and is self-sufficient without the CSV.
    """
    target = Path(directory)
    paths: dict[str, Path] = {}
    json_path = target / f"{SCORE_STEM}.json"
    paths["json"] = write_json_atomic(json_path, summary)
    if csv:
        paths["csv"] = _write_text_atomic(
            target / f"{SCORE_STEM}.csv", _csv_text(summary.get("rankings") or [])
        )
    return paths



# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the S-score scorer."""
    parser = argparse.ArgumentParser(
        prog="motif_analysis",
        description=(
            "Rank Phase 04D pathway artifacts by the unified S-score "
            "(execution-plans/04_update_a.md)."
        ),
    )
    parser.add_argument(
        "-i", "--input", dest="input", action="append", required=True,
        help="a run directory (or a single motif artifact); repeatable",
    )
    parser.add_argument(
        "--include-data", dest="include_data", action="store_true",
        help="also score motifs.data.json artifacts (skipped by default: they duplicate "
             "the canonical payload)",
    )
    weights = parser.add_argument_group("S-score weights")
    weights.add_argument(
        "--w-balance", dest="w_balance", type=float, default=DEFAULT_W_BALANCE,
        help=f"weight of the polarity balance B (default: {DEFAULT_W_BALANCE})",
    )
    weights.add_argument(
        "--w-entropy", dest="w_entropy", type=float, default=DEFAULT_W_ENTROPY,
        help=f"weight of the polarity entropy H (default: {DEFAULT_W_ENTROPY})",
    )
    weights.add_argument(
        "--w-disagreement", dest="w_disagreement", type=float, default=DEFAULT_W_DISAGREEMENT,
        help=f"weight of the mode-matrix disagreement D (default: {DEFAULT_W_DISAGREEMENT})",
    )
    weights.add_argument(
        "--w-concentration", dest="w_concentration", type=float,
        default=DEFAULT_W_CONCENTRATION,
        help=f"penalty weight of the weight concentration C (default: {DEFAULT_W_CONCENTRATION})",
    )
    outputs = parser.add_argument_group("outputs")
    outputs.add_argument(
        "--csv", dest="csv", action="store_true",
        help="also write motif_analysis_summary.csv (opt-in; no CSV is written otherwise)",
    )
    outputs.add_argument(
        "--stats", action="store_true",
        help="print the S-score statistics block to the terminal",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="fail on any validation issue (warnings are escalated to errors)",
    )
    parser.add_argument(
        "--dry-run", dest="dry_run", action="store_true",
        help="score and print the ranking without writing anything",
    )
    parser.add_argument(
        "--log-level", dest="log_level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="logging verbosity (default: INFO)",
    )
    return parser


def _config_from_args(args: argparse.Namespace) -> ScoreConfig:
    """Build the :class:`ScoreConfig` from the parsed CLI arguments."""
    return ScoreConfig(
        w_balance=float(args.w_balance),
        w_entropy=float(args.w_entropy),
        w_disagreement=float(args.w_disagreement),
        w_concentration=float(args.w_concentration),
    )



def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; returns the process exit code.

    ``0`` success, ``1`` any input failed validation or an artifact could not be written,
    ``2`` no motif artifact matched.
    """
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level), format="%(levelname)s %(message)s"
    )

    try:
        config = _config_from_args(args)
        _validate_weights(config)
    except ScoreValidationError as exc:
        LOGGER.error("%s", exc)
        return 1

    directories: list[Path] = []
    for raw in args.input:
        candidate = Path(raw)
        if candidate.is_dir():
            directories.append(candidate)
        elif candidate.is_file():
            directories.append(candidate.parent)
        else:
            LOGGER.error("%s: no such file or directory", candidate)

    if not directories:
        LOGGER.error("no motif artifact matched the given input")
        return 2

    failures = 0
    matched = 0
    targets = sorted(dict.fromkeys(directories))
    for directory in targets:
        try:
            summary = score_directory(directory, config=config, include_data=args.include_data)
        except ScoreValidationError as exc:
            LOGGER.error("%s: %s", directory, exc)
            failures += 1
            continue

        stats = summary.get("summary") or {}
        metadata = summary.get("metadata") or {}
        n_artifacts = int(stats.get("n_artifacts") or 0)
        matched += n_artifacts
        if n_artifacts == 0:
            LOGGER.error("%s: no motif artifact found (motifs[.<variant>].json)", directory)
            continue

        if args.stats:
            print(render_statistics(summary))
        print(render_ranking(summary.get("rankings") or []))

        n_errors = int(metadata.get("n_errors") or 0)
        n_warnings = int(metadata.get("n_warnings") or 0)
        for issue in metadata.get("issues") or []:
            log = LOGGER.error if issue["severity"] == "error" else LOGGER.warning
            log("%s: [%s] %s", directory, issue["code"], issue["message"])

        paths: dict[str, Path] = {}
        if n_errors or (args.strict and n_warnings):
            LOGGER.error(
                "%s: scoring failed (%d error(s), %d warning(s)); nothing written",
                directory,
                n_errors,
                n_warnings,
            )
            failures += 1
            continue

        if not args.dry_run:
            try:
                paths = write_summary(directory, summary, csv=args.csv)
            except Exception as exc:
                LOGGER.error(
                    "%s: [%s] could not write the summary: %s", directory, CODE_SCORE_WRITE, exc
                )
                failures += 1
                continue

        written = ", ".join(
            f"{key}={Path(path).name}" for key, path in sorted(paths.items())
        ) or "nothing (--dry-run)"
        LOGGER.info("%s: scored %d artifact(s); wrote %s", directory, n_artifacts, written)
        print(
            render_summary_box(
                summary, paths, written=not args.dry_run, csv=args.csv, stats=args.stats
            )
        )

    if not matched:
        LOGGER.error("no motif artifact matched the given input")
        return 2
    if failures:
        LOGGER.error("%d of %d directory(ies) failed", failures, len(targets))
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through subprocess tests
    sys.exit(main())

