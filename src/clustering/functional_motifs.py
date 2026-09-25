"""Phase 04 -- functional motifs (04A motifs, 04B tracking, 04C grouping, 04D pathway).

Sub-phase 04A turns the Phase 03 spectrum (``eigen[.<variant>].json``) into the
**per-mode signed motif structure**: which neurons participate in each mode, how
strongly, and with which polarity.  Sub-phase 04B adds the **cross-mode layer**
(``motif_links`` / ``classify_link`` and the merged families ``merge_families`` /
``family_members``); 04C adds the **functional neuron groups** (``build_groups``); 04D
adds the **pathway graph**: signed group-to-group contributions aggregated from the
per-mode outer products (optionally enriched with the raw Phase 02 matrix) and exported
as a NetworkX GraphML document for interactive exploration in yEd.

Pipeline (fixed and recorded in every artifact)::

    load spectrum -> participation -> thresholds -> membership -> signed members
    -> links -> families -> groups -> pathway -> statistics

Artifacts written per input (``<stem>`` = the source ``.gv`` stem, so Phase 04 lands
next to the Phase 01-03 artifacts)::

    <outdir>/<stem>/motifs.json              # canonical: motifs + frozen schema
    <outdir>/<stem>/motifs.npz               # derived array cache (--no-sidecar opts out)
    <outdir>/<stem>/motifs.data.json         # per-neuron tables (--save-data)
    <outdir>/<stem>/motifs.pathway.graphml   # pathway graph (--no-graphml opts out)

The artifact stem inherits the Phase 03 variant suffix (``eigen.json`` ->
``motifs.json``; ``eigen.f8652585.json`` -> ``motifs.f8652585.json``) and adds
``.<config_hash8>`` for any non-default Phase 04 config, so a variant run can never
clobber the canonical ``motifs.json``.

Participation (the definition that reproduces every verified reference number) is the
root-mean-square of the two Phase 03 loading bases::

    P[i, m] = sqrt((L[i, m]^2 + R[i, m]^2) / 2)

Membership is then ``P[i, m] >= t_m`` with a per-mode threshold.  With the default
``--participation rms --threshold-method relative --relative-threshold 0.25`` the
reference artifact yields 21 motifs, 623 members, 107 unique members, 6 neurons in no
motif, a maximum recurrence of 13 and 31 derived regions -- all asserted by the slow
tests.  The 04D pathway with its defaults yields 22 nodes, 39 edges, ``abs_max =
2.345300``, 0 positive / 39 negative edges, 7 intra / 32 cross edges and
``weight_concentration = 0.055842``.

Everything is deterministic: the participation matrix, thresholds, member ordering,
``rms`` sign selection, mode weights, edge filtering and the GraphML serialisation are
pure functions of the arrays, and ``created_utc`` honours ``SOURCE_DATE_EPOCH``, so two
runs produce byte-identical artifacts.  No CSV files are produced or consumed.

CLI
---
::

    python -m src.clustering.functional_motifs \\
        -i data/processed/<stem>/eigen.json -o data/processed \\
        [--glob 'eigen.json'] [--include-variants] [--anatomy PATH] \\
        [--participation {left,right,max,rms}] \\
        [--threshold-method {relative,absolute,quantile,participation}] \\
        [--relative-threshold 0.25] [--absolute-threshold 0.05] [--quantile 0.8] \\
        [--participation-threshold 0.1] [--min-members 3] [--max-members 0] \\
        [--family-jaccard 0.2] [--family-polarity 0.5] [--link-min-jaccard 0.1] \\
        [--grouping {motif,loadings,hybrid,none}] [--polarity-split] \\
        [--polarity-min-members 3] [--max-group-size 12] [--n-groups N|auto] \\
        [--linkage {average,complete,single,ward}] [--affinity {cosine,euclidean}] \\
        [--merge-threshold 0.9] \\
        [--pathway-source {mode,matrix,both}] \\
        [--pathway-weight {evr,energy,uniform,value,abs_value}] \\
        [--pathway-edge-threshold 0.1] [--pathway-top-edges 5] [--no-intra] \\
        [--graphml] [--no-graphml] \\
        [--config-hash] [--no-sidecar] [--save-data] [--stats] \\
        [--strict] [--dry-run] [--log-level INFO]

Loading (04B-04D and Phase 05)
------------------------------
* :func:`load_motifs` -- the full artifact object (verifies the ``.npz`` cache).
* :func:`load_sidecar_arrays` -- fast array access (``allow_pickle=False``).
* :func:`load_pathway_matrix` -- the optional Phase 02 matrix for pathway enrichment.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import math
import os
import re
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from src.parsing.parse_graphviz import ValidationReport
from src.spectral.spectral_decomposition import (
    load_spectrum,
    validate_spectral_payload_schema,
)
from src.utils.io import (
    dumps_json,
    ensure_dir,
    read_json,
    sha256_file,
    utc_timestamp,
    write_json_atomic,
)

LOGGER = logging.getLogger("functional_motifs")

__all__ = [
    "CANONICAL_STEM",
    "CODE_ANATOMY_MISMATCH",
    "CODE_ANATOMY_MISSING",
    "CODE_ARTIFACT_WRITE",
    "CODE_CONFIG",
    "CODE_DEGENERATE",
    "CODE_EMPTY_MOTIF",
    "CODE_FAMILY_CONFLICT",
    "CODE_INPUT_SCHEMA",
    "CODE_MATRIX_MISMATCH",
    "CODE_MATRIX_MISSING",
    "CODE_MODE_WEIGHTS",
    "CODE_NEURON_ORDER",
    "CODE_NONFINITE",
    "CODE_NO_LINKS",
    "CODE_PARTITION",
    "CODE_BACKGROUND",
    "CODE_PATHWAY_SKIPPED",
    "CODE_SINGLETON",
    "CODE_ZERO_VECTOR",
    "CODE_PAYLOAD_SCHEMA",
    "CODE_SIDECAR_CACHE",
    "CODE_SMALL_GRAPH",
    "CODE_TRIVIAL_MOTIFS",
    "CONFIG_KEYS",
    "DEFAULT_ABSOLUTE_THRESHOLD",
    "DEFAULT_DECISION_FIELDS",
    "DEFAULT_INPUT_PATTERN",
    "DEFAULT_MAX_MEMBERS",
    "DEFAULT_MIN_MEMBERS",
    "DEFAULT_PARTICIPATION",
    "DEFAULT_PARTICIPATION_THRESHOLD",
    "DEFAULT_QUANTILE",
    "DEFAULT_RELATIVE_THRESHOLD",
    "DEFAULT_THRESHOLD_METHOD",
    "EDGE_KEYS",
    "EXCLUDED_INPUT_SUFFIXES",
    "FAMILY_KEYS",
    "GENERATOR",
    "GRAPHML_EDGE_KEYS",
    "GRAPHML_GRAPH_KEYS",
    "GRAPHML_NODE_KEYS",
    "GROUP_KEYS",
    "INPUT_VARIANT_PREFIX",
    "LINK_CLASSES",
    "LINK_KEYS",
    "MATRIX_INPUT_PREFIX",
    "MEMBER_KEYS",
    "METADATA_KEYS",
    "MOTIF_CONFIG_FIELDS",
    "MOTIF_KEYS",
    "NEURON_PARTICIPATION_KEYS",
    "NEUTRAL_PATHWAY",
    "NODE_KEYS",
    "OCCURRENCE_KEYS",
    "PARTICIPATIONS",
    "PATHWAY_KEYS",
    "PATHWAY_THRESHOLD_SWEEP",
    "PATHWAY_TOP_MODES",
    "PATHWAY_WEIGHT_NORMALIZATION",
    "PHASE",
    "PROVENANCE_KEYS",
    "SAVE_DATA_KEYS",
    "STAGE",
    "STAGES",
    "THRESHOLD_METHODS",
    "TOP_LEVEL_KEYS",
    "TOP_MODE_KEYS",
    "XML_DECLARATION",
    "Motif",
    "MotifAnalysis",
    "MotifConfig",
    "MotifFamily",
    "MotifLink",
    "MotifMember",
    "MotifValidationError",
    "NeuronGroup",
    "PathwayDiagram",
    "PathwayEdge",
    "PathwayNode",
    "artifact_paths",
    "build_argument_parser",
    "build_motif_analysis",
    "build_pathway",
    "build_sidecar_arrays",
    "classify_link",
    "compute_participation",
    "derive_region",
    "extract_motif",
    "extract_motifs",
    "family_members",
    "filter_pathway_edges",
    "graphml_edge_metadata",
    "graphml_node_metadata",
    "group_pair_contributions",
    "initial_families",
    "input_variant",
    "load_anatomy",
    "load_motifs",
    "load_node_metadata",
    "load_pathway_matrix",
    "load_sidecar_arrays",
    "main",
    "merge_families",
    "mode_outer_product",
    "mode_weights",
    "motif_links",
    "motif_membership",
    "motif_statistics",
    "motif_thresholds",
    "pathway_matrix_path",
    "recurrence_table",
    "render_statistics",
    "render_summary_box",
    "resolve_inputs",
    "save_data_payload",
    "sidecar_path",
    "validate_motif_payload_schema",
    "variant_stem",
    "write_artifact_set",
    "write_graphml_pathway",
    "write_npz_atomic",
    "write_sidecar",
]

CANONICAL_STEM = "motifs"
GENERATOR = "src.clustering.functional_motifs"
PHASE = "04"
STAGE = "04D"
STAGES = ("04A", "04B", "04C", "04D")
INPUT_VARIANT_PREFIX = "eigen"
EXCLUDED_INPUT_SUFFIXES = (".data.json",)
DEFAULT_INPUT_PATTERN = "eigen.json"
VARIANT_INPUT_PATTERN = "eigen.*.json"
SUPPORTED_SUFFIX = ".json"

#: Participation variants (see :func:`compute_participation`).
PARTICIPATIONS = ("left", "right", "max", "rms")
DEFAULT_PARTICIPATION = "rms"
#: Threshold methods (see :func:`motif_thresholds`).
THRESHOLD_METHODS = ("relative", "absolute", "quantile", "participation")
DEFAULT_THRESHOLD_METHOD = "relative"
DEFAULT_RELATIVE_THRESHOLD = 0.25
DEFAULT_ABSOLUTE_THRESHOLD = 0.05
DEFAULT_QUANTILE = 0.8
DEFAULT_PARTICIPATION_THRESHOLD = 0.1
DEFAULT_MIN_MEMBERS = 3
#: 0 means "no upper clamp".
DEFAULT_MAX_MEMBERS = 0

# ---------------------------------------------------------------------------
# Frozen Phase 04 defaults owned by later sub-phases (declared here so that the
# config block and its hash are complete from 04A onward -- see plan section 11).
# ---------------------------------------------------------------------------
DEFAULT_FAMILY_JACCARD = 0.2
DEFAULT_FAMILY_POLARITY = 0.5
DEFAULT_LINK_MIN_JACCARD = 0.1
DEFAULT_GROUPING = "motif"
DEFAULT_POLARITY_SPLIT = False
DEFAULT_POLARITY_MIN_MEMBERS = 3
DEFAULT_MAX_GROUP_SIZE = 12
DEFAULT_N_GROUPS = "auto"
DEFAULT_LINKAGE = "average"
DEFAULT_AFFINITY = "cosine"
DEFAULT_MERGE_THRESHOLD = 0.9
DEFAULT_PATHWAY_SOURCE = "mode"
DEFAULT_PATHWAY_WEIGHT = "evr"
DEFAULT_PATHWAY_EDGE_THRESHOLD = 0.1
DEFAULT_PATHWAY_TOP_EDGES = 5
DEFAULT_INTRA = True
DEFAULT_MAX_DIAGRAM_NODES = 24
DEFAULT_DIAGRAM_LAYOUT = "circular"

GROUPINGS = ("motif", "loadings", "hybrid", "none")
LINKAGES = ("average", "complete", "single", "ward")
AFFINITIES = ("cosine", "euclidean")
PATHWAY_SOURCES = ("mode", "matrix", "both")
PATHWAY_WEIGHTS = ("evr", "energy", "uniform", "value", "abs_value")
DIAGRAM_LAYOUTS = ("circular", "spring", "kamada_kawai", "shell")
#: Cross-mode link classifications (04B), in canonical report order.
LINK_CLASSES = ("stable", "flipped", "composite", "weak")

# ---------------------------------------------------------------------------
# 04D -- pathway graph constants.
# ---------------------------------------------------------------------------
#: Maximum number of per-mode contributions annotated on a pathway node/edge.
PATHWAY_TOP_MODES = 5
#: ``weight_normalization`` recorded in the GraphML graph metadata.  Stored edge
#: weights are raw; the threshold is applied to the abs-max-normalised scale.
PATHWAY_WEIGHT_NORMALIZATION = "abs-max"
#: Thresholds used for the diagnostics-only threshold sweep (fractions of ``abs_max``).
PATHWAY_THRESHOLD_SWEEP = (0.05, 0.1, 0.2, 0.25)
#: Phase 02 artifact stem searched next to the Phase 03 input for matrix enrichment.
MATRIX_INPUT_PREFIX = "z_matrix"
#: XML declaration prepended to the NetworkX GraphML body.
XML_DECLARATION = '<?xml version="1.0" encoding="UTF-8"?>'
#: The exact GraphML attribute key sets (validated before serialisation).
GRAPHML_NODE_KEYS = (
    "group_id",
    "label",
    "size",
    "dominant_mode",
    "region_composition",
    "centroid",
    "coherence",
    "is_background",
    "is_singleton",
)
GRAPHML_EDGE_KEYS = (
    "source_group",
    "target_group",
    "weight",
    "abs_weight",
    "polarity",
    "contribution_by_mode",
    "threshold_flag",
    "topN_flag",
    "intra_flag",
    "z_contribution",
)
GRAPHML_GRAPH_KEYS = (
    "n_nodes",
    "n_edges",
    "weight_normalization",
    "pathway_source",
    "pathway_weight_rule",
    "threshold",
    "topN",
)

#: The 26 hashed Phase 04 config fields (section 11 of the plan).
MOTIF_CONFIG_FIELDS = (
    "participation",
    "threshold_method",
    "relative_threshold",
    "absolute_threshold",
    "quantile",
    "participation_threshold",
    "min_members",
    "max_members",
    "family_jaccard",
    "family_polarity",
    "link_min_jaccard",
    "grouping",
    "polarity_split",
    "polarity_min_members",
    "max_group_size",
    "n_groups",
    "linkage",
    "affinity",
    "merge_threshold",
    "pathway_source",
    "pathway_weight",
    "pathway_edge_threshold",
    "pathway_top_edges",
    "intra",
    "max_diagram_nodes",
    "diagram_layout",
)
#: Fields compared by :meth:`MotifConfig.is_default` (the 04A-selectable ones).
DEFAULT_DECISION_FIELDS = (
    "participation",
    "threshold_method",
    "relative_threshold",
    "absolute_threshold",
    "quantile",
    "participation_threshold",
    "min_members",
    "max_members",
)

#: The frozen key set of every top-level block.  Declared in full in 04A (section 7
#: of the plan) and validated by :func:`validate_motif_payload_schema` from here on.
TOP_LEVEL_KEYS = frozenset(
    {"provenance", "config", "neuron_order", "motifs", "links", "families", "groups", "pathway", "metadata"}
)
PROVENANCE_KEYS = frozenset(
    {
        "source_artifact",
        "source_artifact_sha256",
        "source_file",
        "source_file_sha256",
        "parsed_created_utc",
        "z_matrix_config_hash",
        "z_matrix_config",
        "spectral_config_hash",
        "spectral_config",
        "phase",
        "stage",
    }
)
CONFIG_KEYS = frozenset(MOTIF_CONFIG_FIELDS) | {"k_resolved", "n_motifs", "stage", "config_hash"}
MEMBER_KEYS = frozenset(
    {
        "index",
        "neuron_id",
        "participation",
        "signed_participation",
        "polarity",
        "left",
        "right",
        "dominant_axis",
        "region",
        "rank",
    }
)
MOTIF_KEYS = frozenset(
    {
        "label",
        "mode",
        "mode_index",
        "value",
        "abs_value",
        "explained_variance_ratio",
        "cumulative_variance_ratio",
        "threshold",
        "threshold_method",
        "n_members",
        "members",
        "sender_members",
        "receiver_members",
        "strength_l1",
        "strength_energy",
        "share",
        "polarity_balance",
        "n_positive",
        "n_negative",
        "n_mixed",
        "region_composition",
        "dominant_region",
    }
)
#: Entry shapes owned by later sub-phases -- declared and validated now, populated later.
LINK_KEYS = frozenset(
    {
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
)
FAMILY_KEYS = frozenset(
    {"family_id", "label", "modes", "n_modes", "size", "n_members", "members", "polarity", "occurrences"}
)
OCCURRENCE_KEYS = frozenset({"mode", "participation", "signed_participation"})
GROUP_KEYS = frozenset(
    {
        "group_id",
        "label",
        "members",
        "n_members",
        "size",
        "dominant_mode",
        "region_composition",
        "cent_mean",
        "coherence",
        "is_background",
        "is_singleton",
    }
)
PATHWAY_KEYS = frozenset(
    {
        "nodes",
        "edges",
        "n_nodes",
        "n_edges",
        "weight",
        "source",
        "edge_threshold",
        "top_edges",
        "intra",
        "abs_max",
        "n_positive",
        "n_negative",
        "n_intra",
        "n_cross",
        "weight_concentration",
        "weights",
    }
)
NODE_KEYS = frozenset(
    {
        "node_id",
        "group_id",
        "label",
        "size",
        "members",
        "cent_mean",
        "region_composition",
        "dominant_mode",
        "top_modes",
    }
)
EDGE_KEYS = frozenset(
    {"source", "target", "weight", "abs_weight", "n_modes", "modes", "is_intra", "polarity", "top_modes"}
)
TOP_MODE_KEYS = frozenset({"mode", "rank", "share", "signed_weight", "abs_weight"})
METADATA_KEYS = frozenset(
    {
        # populated by 04A
        "n_neurons",
        "n_modes",
        "n_motifs",
        "n_members_total",
        "n_members_min",
        "n_members_median",
        "n_members_max",
        "n_members_mean",
        "n_unique_members",
        "n_neurons_in_no_motif",
        "membership_density",
        "n_isolated_neurons",
        "recurrence_max",
        "n_recurrence_ge_1",
        "n_recurrence_ge_2",
        "n_recurrence_ge_5",
        "top_recurrent",
        "n_regions",
        "regions",
        "n_anatomy_matches",
        "n_anatomy_missing",
        "n_anatomy_conflicts",
        "participation",
        "threshold_method",
        "k_resolved",
        "stage",
        "config_hash",
        "numpy_version",
        "generator",
        "created_utc",
        # populated by 04B
        "n_links",
        "link_classes",
        "n_families",
        "n_multi_mode_families",
        # populated by 04C
        "n_groups",
        "group_sizes",
        "n_background_groups",
        "n_singleton_groups",
        "largest_group",
        "smallest_group",
        "mean_group_size",
        "n_polarity_split_groups",
        # populated by 04D
        "n_pathway_nodes",
        "n_pathway_edges",
        "pathway_abs_max",
        "pathway_positive_edges",
        "pathway_negative_edges",
        "pathway_intra_edges",
        "pathway_cross_edges",
        "pathway_weight_concentration",
    }
)
NEURON_PARTICIPATION_KEYS = frozenset(
    {
        "index",
        "neuron_id",
        "region",
        "cent",
        "out_degree",
        "in_degree",
        "recurrence",
        "max_participation",
        "mean_participation",
        "dominant_mode",
        "participation",
    }
)
SAVE_DATA_KEYS = frozenset(
    {
        "provenance",
        "config",
        "stage",
        "neuron_order",
        "neurons",
        "motifs",
        "links",
        "families",
        "groups",
        "pathway",
        "metadata",
        "generator",
        "created_utc",
    }
)
#: The neutral (not-yet-computed) pathway diagram emitted by 04A.
NEUTRAL_PATHWAY: dict[str, Any] = {
    "nodes": [],
    "edges": [],
    "n_nodes": None,
    "n_edges": None,
    "weight": None,
    "source": None,
    "edge_threshold": None,
    "top_edges": None,
    "intra": None,
    "abs_max": None,
    "n_positive": None,
    "n_negative": None,
    "n_intra": None,
    "n_cross": None,
    "weight_concentration": None,
    "weights": [],
}

# Validation issue codes (one per rule).
CODE_INPUT_SCHEMA = "input_schema"
CODE_CONFIG = "config"
CODE_NEURON_ORDER = "neuron_order"
CODE_NONFINITE = "nonfinite"
CODE_EMPTY_MOTIF = "empty_motif"
CODE_SMALL_GRAPH = "small_graph"
CODE_TRIVIAL_MOTIFS = "trivial_motifs"
CODE_DEGENERATE = "degenerate"
CODE_ANATOMY_MISSING = "anatomy_missing"
CODE_ANATOMY_MISMATCH = "anatomy_mismatch"
CODE_PAYLOAD_SCHEMA = "payload_schema"
CODE_SIDECAR_CACHE = "sidecar_cache"
CODE_ARTIFACT_WRITE = "artifact_write"
# 04B -- cross-mode tracking.
CODE_NO_LINKS = "no_links"
CODE_FAMILY_CONFLICT = "family_conflict"
CODE_PARTITION = "partition"
CODE_BACKGROUND = "background_present"
CODE_SINGLETON = "singleton_present"
CODE_ZERO_VECTOR = "zero_vector"
# 04D -- pathway graph.
CODE_MATRIX_MISSING = "matrix_missing"
CODE_MATRIX_MISMATCH = "matrix_mismatch"
CODE_MODE_WEIGHTS = "mode_weights"
CODE_PATHWAY_SKIPPED = "pathway_skipped"

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MotifConfig:
    """The complete, hashable Phase 04 recipe (26 fields, all hashed).

    The fields owned by 04B-04D are declared here at their frozen defaults so that
    ``config_hash`` is identical at every stage and a non-default config gets its
    filename segment from 04A onward.
    """

    participation: str = DEFAULT_PARTICIPATION
    threshold_method: str = DEFAULT_THRESHOLD_METHOD
    relative_threshold: float = DEFAULT_RELATIVE_THRESHOLD
    absolute_threshold: float = DEFAULT_ABSOLUTE_THRESHOLD
    quantile: float = DEFAULT_QUANTILE
    participation_threshold: float = DEFAULT_PARTICIPATION_THRESHOLD
    min_members: int = DEFAULT_MIN_MEMBERS
    max_members: int = DEFAULT_MAX_MEMBERS
    family_jaccard: float = DEFAULT_FAMILY_JACCARD
    family_polarity: float = DEFAULT_FAMILY_POLARITY
    link_min_jaccard: float = DEFAULT_LINK_MIN_JACCARD
    grouping: str = DEFAULT_GROUPING
    polarity_split: bool = DEFAULT_POLARITY_SPLIT
    polarity_min_members: int = DEFAULT_POLARITY_MIN_MEMBERS
    max_group_size: int = DEFAULT_MAX_GROUP_SIZE
    n_groups: str = DEFAULT_N_GROUPS
    linkage: str = DEFAULT_LINKAGE
    affinity: str = DEFAULT_AFFINITY
    merge_threshold: float = DEFAULT_MERGE_THRESHOLD
    pathway_source: str = DEFAULT_PATHWAY_SOURCE
    pathway_weight: str = DEFAULT_PATHWAY_WEIGHT
    pathway_edge_threshold: float = DEFAULT_PATHWAY_EDGE_THRESHOLD
    pathway_top_edges: int = DEFAULT_PATHWAY_TOP_EDGES
    intra: bool = DEFAULT_INTRA
    max_diagram_nodes: int = DEFAULT_MAX_DIAGRAM_NODES
    diagram_layout: str = DEFAULT_DIAGRAM_LAYOUT
    #: Derived/observational fields -- recorded verbatim, never hashed.
    k_resolved: int = 0
    n_motifs: int = 0
    stage: str = STAGE

    def hash_fields(self) -> dict[str, Any]:
        """The 26 config-defining fields hashed into ``config_hash``.

        Built directly (never via :meth:`to_dict`) -- ``to_dict`` adds the derived
        keys, and hashing them would make the hash depend on the data.
        """
        return {
            name: getattr(self, name)
            for name in MOTIF_CONFIG_FIELDS
        }

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON payload written to ``config`` (including the hash)."""
        payload = self.hash_fields()
        payload["k_resolved"] = int(self.k_resolved)
        payload["n_motifs"] = int(self.n_motifs)
        payload["stage"] = str(self.stage)
        payload["config_hash"] = self.config_hash()
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MotifConfig":
        """Rebuild a config from an artifact payload (ignores ``config_hash``)."""
        values = {name: payload.get(name, getattr(_DEFAULT_CONFIG, name)) for name in MOTIF_CONFIG_FIELDS}
        return cls(
            **values,
            k_resolved=int(payload.get("k_resolved", 0)),
            n_motifs=int(payload.get("n_motifs", 0)),
            stage=str(payload.get("stage", STAGE)),
        )

    def config_hash(self) -> str:
        """Deterministic 8-hex-character digest of :meth:`hash_fields`."""
        digest = hashlib.sha256(dumps_json(self.hash_fields()).encode("utf-8")).hexdigest()
        return digest[:8]

    def is_default(self) -> bool:
        """True when every 04A-selectable parameter is at its default."""
        return all(
            getattr(self, name) == getattr(_DEFAULT_CONFIG, name) for name in DEFAULT_DECISION_FIELDS
        )


#: Reference instance used by :meth:`MotifConfig.is_default` comparisons.
_DEFAULT_CONFIG = MotifConfig()


@dataclass(frozen=True)
class MotifMember:
    """One neuron's signed participation in one mode."""

    index: int
    neuron_id: str
    participation: float
    signed_participation: float
    polarity: int
    left: float
    right: float
    dominant_axis: str
    region: str
    rank: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": int(self.index),
            "neuron_id": self.neuron_id,
            "participation": float(self.participation),
            "signed_participation": float(self.signed_participation),
            "polarity": int(self.polarity),
            "left": float(self.left),
            "right": float(self.right),
            "dominant_axis": self.dominant_axis,
            "region": self.region,
            "rank": int(self.rank),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MotifMember":
        return cls(
            index=int(payload.get("index", 0)),
            neuron_id=str(payload.get("neuron_id", "")),
            participation=float(payload.get("participation", 0.0)),
            signed_participation=float(payload.get("signed_participation", 0.0)),
            polarity=int(payload.get("polarity", 0)),
            left=float(payload.get("left", 0.0)),
            right=float(payload.get("right", 0.0)),
            dominant_axis=str(payload.get("dominant_axis", "left")),
            region=str(payload.get("region", "")),
            rank=int(payload.get("rank", 0)),
        )


@dataclass(frozen=True)
class Motif:
    """A mode's signed member set (the 22 ``MOTIF_KEYS``)."""

    label: str
    mode: int
    mode_index: int
    value: float
    abs_value: float
    explained_variance_ratio: float
    cumulative_variance_ratio: float
    threshold: float
    threshold_method: str
    members: tuple[MotifMember, ...]
    sender_members: tuple[str, ...]
    receiver_members: tuple[str, ...]
    strength_l1: float
    strength_energy: float
    share: float
    polarity_balance: float
    n_positive: int
    n_negative: int
    n_mixed: int
    region_composition: dict[str, int]
    dominant_region: str

    @property
    def n_members(self) -> int:
        return len(self.members)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "mode": int(self.mode),
            "mode_index": int(self.mode_index),
            "value": float(self.value),
            "abs_value": float(self.abs_value),
            "explained_variance_ratio": float(self.explained_variance_ratio),
            "cumulative_variance_ratio": float(self.cumulative_variance_ratio),
            "threshold": float(self.threshold),
            "threshold_method": self.threshold_method,
            "n_members": int(self.n_members),
            "members": [member.to_dict() for member in self.members],
            "sender_members": list(self.sender_members),
            "receiver_members": list(self.receiver_members),
            "strength_l1": float(self.strength_l1),
            "strength_energy": float(self.strength_energy),
            "share": float(self.share),
            "polarity_balance": float(self.polarity_balance),
            "n_positive": int(self.n_positive),
            "n_negative": int(self.n_negative),
            "n_mixed": int(self.n_mixed),
            "region_composition": dict(self.region_composition),
            "dominant_region": self.dominant_region,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Motif":
        return cls(
            label=str(payload.get("label", "")),
            mode=int(payload.get("mode", 0)),
            mode_index=int(payload.get("mode_index", 0)),
            value=float(payload.get("value", 0.0)),
            abs_value=float(payload.get("abs_value", 0.0)),
            explained_variance_ratio=float(payload.get("explained_variance_ratio", 0.0)),
            cumulative_variance_ratio=float(payload.get("cumulative_variance_ratio", 0.0)),
            threshold=float(payload.get("threshold", 0.0)),
            threshold_method=str(payload.get("threshold_method", DEFAULT_THRESHOLD_METHOD)),
            members=tuple(MotifMember.from_dict(entry) for entry in payload.get("members") or []),
            sender_members=tuple(str(name) for name in payload.get("sender_members") or []),
            receiver_members=tuple(str(name) for name in payload.get("receiver_members") or []),
            strength_l1=float(payload.get("strength_l1", 0.0)),
            strength_energy=float(payload.get("strength_energy", 0.0)),
            share=float(payload.get("share", 0.0)),
            polarity_balance=float(payload.get("polarity_balance", 0.0)),
            n_positive=int(payload.get("n_positive", 0)),
            n_negative=int(payload.get("n_negative", 0)),
            n_mixed=int(payload.get("n_mixed", 0)),
            region_composition=dict(payload.get("region_composition") or {}),
            dominant_region=str(payload.get("dominant_region", "")),
        )

@dataclass(frozen=True)
class MotifLink:
    """A pairwise link between two modes (populated by 04B)."""

    source_mode: int
    target_mode: int
    source_label: str = ""
    target_label: str = ""
    shared_members: tuple[str, ...] = ()
    jaccard: float = 0.0
    polarity_agreement: float = 0.0
    classification: str = "unclassified"

    @property
    def n_shared(self) -> int:
        return len(self.shared_members)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_mode": int(self.source_mode),
            "target_mode": int(self.target_mode),
            "source_label": self.source_label,
            "target_label": self.target_label,
            "shared_members": list(self.shared_members),
            "n_shared": int(self.n_shared),
            "jaccard": float(self.jaccard),
            "polarity_agreement": float(self.polarity_agreement),
            "classification": self.classification,
        }


@dataclass(frozen=True)
class MotifFamily:
    """A set of modes sharing a motif (04A emits k singletons)."""

    family_id: int
    label: str
    modes: tuple[int, ...]
    members: tuple[str, ...] = ()
    polarity: float = 0.0
    occurrences: tuple[dict[str, Any], ...] = ()

    @property
    def n_modes(self) -> int:
        return len(self.modes)

    @property
    def size(self) -> int:
        return len(self.modes)

    @property
    def n_members(self) -> int:
        return len(self.members)

    def to_dict(self) -> dict[str, Any]:
        return {
            "family_id": int(self.family_id),
            "label": self.label,
            "modes": [int(mode) for mode in self.modes],
            "n_modes": int(self.n_modes),
            "size": int(self.size),
            "n_members": int(self.n_members),
            "members": list(self.members),
            "polarity": float(self.polarity),
            "occurrences": [dict(entry) for entry in self.occurrences],
        }


@dataclass(frozen=True)
class NeuronGroup:
    """A functional neuron population (populated by 04C)."""

    group_id: str
    label: str
    members: tuple[str, ...]
    dominant_mode: int | None = None
    region_composition: dict[str, int] = field(default_factory=dict)
    cent_mean: float | None = None
    coherence: float | None = None
    is_background: bool = False
    is_singleton: bool = False

    @property
    def n_members(self) -> int:
        return len(self.members)

    @property
    def size(self) -> int:
        return len(self.members)

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "label": self.label,
            "members": list(self.members),
            "n_members": int(self.n_members),
            "size": int(self.size),
            "dominant_mode": self.dominant_mode,
            "region_composition": dict(self.region_composition),
            "cent_mean": self.cent_mean,
            "coherence": self.coherence,
            "is_background": bool(self.is_background),
            "is_singleton": bool(self.is_singleton),
        }


@dataclass(frozen=True)
class PathwayNode:
    """A pathway-diagram node (= a functional group; populated by 04D)."""

    node_id: str
    group_id: str
    label: str
    members: tuple[str, ...] = ()
    cent_mean: float | None = None
    region_composition: dict[str, int] = field(default_factory=dict)
    dominant_mode: int | None = None
    top_modes: tuple[dict[str, Any], ...] = ()

    @property
    def size(self) -> int:
        return len(self.members)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "group_id": self.group_id,
            "label": self.label,
            "size": int(self.size),
            "members": list(self.members),
            "cent_mean": self.cent_mean,
            "region_composition": dict(self.region_composition),
            "dominant_mode": self.dominant_mode,
            "top_modes": [dict(entry) for entry in self.top_modes],
        }


@dataclass(frozen=True)
class PathwayEdge:
    """A signed pathway edge between two groups (populated by 04D)."""

    source: str
    target: str
    weight: float
    modes: tuple[int, ...] = ()
    is_intra: bool = False
    polarity: int = 0
    top_modes: tuple[dict[str, Any], ...] = ()

    @property
    def abs_weight(self) -> float:
        return abs(float(self.weight))

    @property
    def n_modes(self) -> int:
        return len(self.modes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "weight": float(self.weight),
            "abs_weight": float(self.abs_weight),
            "n_modes": int(self.n_modes),
            "modes": [int(mode) for mode in self.modes],
            "is_intra": bool(self.is_intra),
            "polarity": int(self.polarity),
            "top_modes": [dict(entry) for entry in self.top_modes],
        }


@dataclass(frozen=True)
class PathwayDiagram:
    """The quantitative pathway graph; 04A emits the neutral value."""

    nodes: tuple[dict[str, Any], ...] = ()
    edges: tuple[dict[str, Any], ...] = ()
    n_nodes: int | None = None
    n_edges: int | None = None
    weight: str | None = None
    source: str | None = None
    edge_threshold: float | None = None
    top_edges: int | None = None
    intra: bool | None = None
    abs_max: float | None = None
    n_positive: int | None = None
    n_negative: int | None = None
    n_intra: int | None = None
    n_cross: int | None = None
    weight_concentration: float | None = None
    weights: tuple[float, ...] = ()

    @classmethod
    def neutral(cls) -> "PathwayDiagram":
        """Return the not-yet-computed diagram written by 04A."""
        return cls()

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [dict(node) for node in self.nodes],
            "edges": [dict(edge) for edge in self.edges],
            "n_nodes": self.n_nodes,
            "n_edges": self.n_edges,
            "weight": self.weight,
            "source": self.source,
            "edge_threshold": self.edge_threshold,
            "top_edges": self.top_edges,
            "intra": self.intra,
            "abs_max": self.abs_max,
            "n_positive": self.n_positive,
            "n_negative": self.n_negative,
            "n_intra": self.n_intra,
            "n_cross": self.n_cross,
            "weight_concentration": self.weight_concentration,
            "weights": [float(value) for value in self.weights],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PathwayDiagram":
        return cls(
            nodes=tuple(dict(node) for node in payload.get("nodes") or []),
            edges=tuple(dict(edge) for edge in payload.get("edges") or []),
            n_nodes=payload.get("n_nodes"),
            n_edges=payload.get("n_edges"),
            weight=payload.get("weight"),
            source=payload.get("source"),
            edge_threshold=payload.get("edge_threshold"),
            top_edges=payload.get("top_edges"),
            intra=payload.get("intra"),
            abs_max=payload.get("abs_max"),
            n_positive=payload.get("n_positive"),
            n_negative=payload.get("n_negative"),
            n_intra=payload.get("n_intra"),
            n_cross=payload.get("n_cross"),
            weight_concentration=payload.get("weight_concentration"),
            weights=tuple(float(value) for value in payload.get("weights") or []),
        )

@dataclass
class MotifAnalysis:
    """In-memory Phase 04 artifact (see :meth:`to_dict` for the JSON schema).

    ``motifs`` carries the retained mode structure while the raw arrays
    (participation, membership, loadings, spectrum) are kept separately in
    ``arrays`` for the ``.npz`` cache and are never serialized into the JSON.
    Fields owned by 04B-04D (``links``/``families``/``groups``/``pathway``) hold
    their neutral values here and are replaced -- never retyped -- later.
    """

    source_artifact: Path
    source_artifact_sha256: str
    source_file: str
    source_file_sha256: str
    parsed_created_utc: str
    z_matrix_config_hash: str
    z_matrix_config: dict[str, Any]
    spectral_config_hash: str
    spectral_config: dict[str, Any]
    config: MotifConfig
    neuron_order: list[str]
    motifs: list[Motif]
    links: list[dict[str, Any]] = field(default_factory=list)
    families: list[dict[str, Any]] = field(default_factory=list)
    groups: list[dict[str, Any]] = field(default_factory=list)
    pathway: dict[str, Any] = field(default_factory=lambda: dict(NEUTRAL_PATHWAY))
    metadata: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    #: Derived arrays for the ``.npz`` cache (never serialized into the JSON).
    arrays: dict[str, np.ndarray] = field(default_factory=dict)
    #: Where this object was read from; never serialized so ``to_dict()``
    #: round-trips byte-for-byte.
    loaded_from: Path | None = None

    @property
    def n_neurons(self) -> int:
        return len(self.neuron_order)

    @property
    def n_modes(self) -> int:
        return len(self.motifs)

    @property
    def k(self) -> int:
        return len(self.motifs)

    @property
    def index(self) -> dict[str, int]:
        """Neuron-id -> row index lookup (the Phase 04 neuron index map)."""
        return {name: position for position, name in enumerate(self.neuron_order)}

    def _array(self, name: str) -> np.ndarray | None:
        return self.arrays.get(name)

    @property
    def participation(self) -> np.ndarray | None:
        return self._array("participation")

    @property
    def membership(self) -> np.ndarray | None:
        return self._array("membership")

    @property
    def loadings_left(self) -> np.ndarray | None:
        return self._array("loadings_left")

    @property
    def loadings_right(self) -> np.ndarray | None:
        return self._array("loadings_right")

    @property
    def values(self) -> np.ndarray | None:
        return self._array("values")

    @property
    def explained_variance_ratio(self) -> np.ndarray | None:
        return self._array("explained_variance_ratio")

    @property
    def thresholds(self) -> np.ndarray | None:
        return self._array("thresholds")

    @property
    def motif_strength_l1(self) -> np.ndarray | None:
        return self._array("motif_strength_l1")

    @property
    def motif_strength_energy(self) -> np.ndarray | None:
        return self._array("motif_strength_energy")

    @property
    def retained_values(self) -> np.ndarray:
        """The retained mode values (derived from ``motifs``, always available)."""
        return np.asarray([motif.value for motif in self.motifs], dtype=np.float64)

    @property
    def abs_retained_values(self) -> np.ndarray:
        return np.asarray([motif.abs_value for motif in self.motifs], dtype=np.float64)

    @property
    def family_of_mode(self) -> np.ndarray:
        """Family index per mode (0-based; the identity when every mode is alone).

        Derived from :attr:`families`, so it is the identity for a 04A payload
        (k singletons) and the merged assignment for a 04B payload.
        """
        return _family_of_mode(self.families, self.k)

    def to_dict(self) -> dict[str, Any]:
        """Return the exact JSON payload written to ``motifs.json``."""
        return {
            "provenance": {
                "source_artifact": str(self.source_artifact),
                "source_artifact_sha256": self.source_artifact_sha256,
                "source_file": self.source_file,
                "source_file_sha256": self.source_file_sha256,
                "parsed_created_utc": self.parsed_created_utc,
                "z_matrix_config_hash": self.z_matrix_config_hash,
                "z_matrix_config": dict(self.z_matrix_config),
                "spectral_config_hash": self.spectral_config_hash,
                "spectral_config": dict(self.spectral_config),
                "phase": PHASE,
                "stage": str(self.config.stage),
            },
            "config": self.config.to_dict(),
            "neuron_order": list(self.neuron_order),
            "motifs": [motif.to_dict() for motif in self.motifs],
            "links": [dict(entry) for entry in self.links],
            "families": [dict(entry) for entry in self.families],
            "groups": [dict(entry) for entry in self.groups],
            "pathway": dict(self.pathway),
            "metadata": dict(self.metadata),
        }


class MotifValidationError(ValueError):
    """Raised when a Phase 04 input/output violates a validation rule.

    The full :class:`src.parsing.parse_graphviz.ValidationReport` is available as
    :attr:`report`, mirroring ``GraphvizValidationError`` / ``ZMatrixValidationError``
    / ``SpectralValidationError``.
    """

    def __init__(self, message: str, report: ValidationReport | None = None) -> None:
        super().__init__(message)
        self.report = report

# ---------------------------------------------------------------------------
# Regions, anatomy and sibling-metadata enrichment
# ---------------------------------------------------------------------------
def derive_region(neuron_id: str) -> str:
    """Derive an anatomical region label from a neuron id.

    Trailing tokens matching ``_[LR]``, ``_C<digits>`` and ``_<digits>`` are stripped
    repeatedly (``hDeltaA_12_C10_1`` -> ``hDeltaA``; ``FB4P_a_R_1`` -> ``FB4P_a``;
    ``MBON09(y3B'1)(AVM17)_L_1`` -> ``MBON09(y3B'1)(AVM17)``).  The id is returned
    unchanged when nothing matches.  On the reference data this yields 31 regions.
    """
    tokens = str(neuron_id).split("_")
    while len(tokens) > 1:
        tail = tokens[-1]
        if tail in ("L", "R") or tail.isdigit() or re.fullmatch(r"C\d+", tail):
            tokens.pop()
            continue
        break
    return "_".join(tokens)


def _anatomy_records(payload: Any) -> dict[str, dict[str, Any]]:
    """Normalize the many accepted anatomy shapes into ``{neuron_id: {...}}``."""
    records: dict[str, dict[str, Any]] = {}

    def _add(neuron_id: Any, entry: Any) -> None:
        name = str(neuron_id)
        if isinstance(entry, dict):
            records[name] = {
                "region": entry.get("region"),
                "cent": entry.get("cent"),
            }
        elif entry is not None:
            records[name] = {"region": str(entry), "cent": None}

    if isinstance(payload, list):
        for entry in payload:
            if isinstance(entry, dict):
                _add(entry.get("neuron_id") or entry.get("id"), entry)
        return records
    if not isinstance(payload, dict):
        return records
    if isinstance(payload.get("neurons"), list):
        for entry in payload["neurons"]:
            if isinstance(entry, dict):
                _add(entry.get("neuron_id") or entry.get("id"), entry)
        return records
    if isinstance(payload.get("regions"), dict):
        for name, region in payload["regions"].items():
            _add(name, region)
        return records
    for name, entry in payload.items():
        _add(name, entry)
    return records


def load_anatomy(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load a tolerant anatomy file: ``{id: region}``, a ``neurons`` record list, a
    nested ``{id: {region, cent}}`` map or a ``parsed_graph.json``-shaped file."""
    candidate = Path(path)
    if not candidate.is_file():
        raise MotifValidationError(f"anatomy file not found: {candidate}")
    return _anatomy_records(read_json(candidate))


def load_node_metadata(path: str | Path) -> dict[str, dict[str, Any]]:
    """Read ``cent``/``out_degree``/``in_degree`` from a Phase 01 ``parsed_graph.json``."""
    payload = read_json(path)
    rows: dict[str, dict[str, Any]] = {}
    for entry in (payload or {}).get("neurons") or []:
        if not isinstance(entry, dict):
            continue
        name = entry.get("neuron_id")
        if name is None:
            continue
        rows[str(name)] = {
            "cent": entry.get("cent"),
            "out_degree": entry.get("out_degree"),
            "in_degree": entry.get("in_degree"),
        }
    return rows


# ---------------------------------------------------------------------------
# Participation, thresholds and membership
# ---------------------------------------------------------------------------
def compute_participation(loadings_left: Any, loadings_right: Any, variant: str = DEFAULT_PARTICIPATION) -> np.ndarray:
    """Return the ``(N, k)`` participation matrix for *variant*.

    ``left``/``right``/``max`` use a single axis; ``rms`` (the default) combines both
    as ``sqrt((L**2 + R**2) / 2)`` -- the definition that reproduces every verified
    reference number.  The result is finite and non-negative by construction.
    """
    left = np.asarray(loadings_left, dtype=np.float64)
    right = np.asarray(loadings_right, dtype=np.float64)
    if left.ndim != 2 or right.ndim != 2 or left.shape != right.shape:
        raise MotifValidationError(
            f"loadings must be matching 2-D arrays, got {tuple(left.shape)} and {tuple(right.shape)}"
        )
    if variant not in PARTICIPATIONS:
        raise MotifValidationError(f"participation {variant!r} is not one of {PARTICIPATIONS}")
    if variant == "left":
        participation = np.abs(left)
    elif variant == "right":
        participation = np.abs(right)
    elif variant == "max":
        participation = np.maximum(np.abs(left), np.abs(right))
    else:  # rms
        participation = np.sqrt((left * left + right * right) / 2.0)
    return np.asarray(participation, dtype=np.float64)


def motif_thresholds(
    participation: Any,
    method: str = DEFAULT_THRESHOLD_METHOD,
    *,
    relative_threshold: float = DEFAULT_RELATIVE_THRESHOLD,
    absolute_threshold: float = DEFAULT_ABSOLUTE_THRESHOLD,
    quantile: float = DEFAULT_QUANTILE,
    participation_threshold: float = DEFAULT_PARTICIPATION_THRESHOLD,
) -> np.ndarray:
    """Return the ``(k,)`` effective per-mode threshold for *method*.

    Each method is a pure function of exactly one knob (section 6 of the plan); an
    all-zero mode column gets ``+inf`` so that it contributes no members instead of
    matching every neuron.
    """
    matrix = np.asarray(participation, dtype=np.float64)
    if matrix.ndim != 2:
        raise MotifValidationError(f"participation must be 2-D, got shape {tuple(matrix.shape)}")
    if method not in THRESHOLD_METHODS:
        raise MotifValidationError(f"threshold_method {method!r} is not one of {THRESHOLD_METHODS}")
    n_modes = int(matrix.shape[1])
    if method == "absolute":
        return np.full(n_modes, float(absolute_threshold), dtype=np.float64)
    if method == "participation":
        return np.full(n_modes, float(participation_threshold), dtype=np.float64)
    if method == "relative":
        maxima = matrix.max(axis=0) if matrix.size else np.zeros(n_modes)
        with np.errstate(invalid="ignore"):
            thresholds = float(relative_threshold) * maxima
        thresholds = np.where(maxima > 0.0, thresholds, np.inf)
        return np.asarray(thresholds, dtype=np.float64)
    # quantile
    thresholds = np.empty(n_modes, dtype=np.float64)
    for mode_index in range(n_modes):
        column = matrix[:, mode_index]
        if column.size == 0 or float(column.max()) <= 0.0:
            thresholds[mode_index] = np.inf
        else:
            thresholds[mode_index] = float(np.quantile(column, float(quantile)))
    return thresholds


def motif_membership(
    participation: Any,
    thresholds: Any,
    *,
    min_members: int = DEFAULT_MIN_MEMBERS,
    max_members: int = DEFAULT_MAX_MEMBERS,
) -> np.ndarray:
    """Return the ``(N, k)`` boolean membership with the size clamps applied.

    ``max_members`` (> 0) truncates a motif to its strongest members and
    ``min_members`` tops a too-small motif up, both on descending participation
    (ties by neuron index), so the result stays deterministic.
    """
    matrix = np.asarray(participation, dtype=np.float64)
    grid = np.asarray(thresholds, dtype=np.float64)
    if matrix.ndim != 2:
        raise MotifValidationError(f"participation must be 2-D, got shape {tuple(matrix.shape)}")
    if grid.shape != (matrix.shape[1],):
        raise MotifValidationError(
            f"thresholds must have length {matrix.shape[1]}, got {grid.shape}"
        )
    n_neurons, n_modes = matrix.shape
    membership = np.zeros((n_neurons, n_modes), dtype=bool)
    upper = int(max_members) if max_members else 0
    lower = max(0, int(min_members))
    for mode_index in range(n_modes):
        column = matrix[:, mode_index]
        order = np.argsort(-column, kind="stable")
        selected = column >= grid[mode_index]
        count = int(selected.sum())
        if upper and count > upper:
            selected = np.zeros(n_neurons, dtype=bool)
            selected[order[:upper]] = True
        elif count < lower:
            selected = np.zeros(n_neurons, dtype=bool)
            selected[order[:lower]] = True
        membership[:, mode_index] = selected
    return membership

# ---------------------------------------------------------------------------
# Motif extraction
# ---------------------------------------------------------------------------
def _sign(value: float) -> int:
    """Return -1, 0 or +1 for *value* (``-0.0`` resolves to 0)."""
    if value > 0.0:
        return 1
    if value < 0.0:
        return -1
    return 0


def extract_motif(
    participation: Any,
    left: Any,
    right: Any,
    mode_index: int,
    *,
    threshold: float,
    threshold_method: str,
    membership: Any,
    neuron_ids: Sequence[str],
    regions: Sequence[str],
    value: float,
    abs_value: float,
    explained_variance_ratio: float,
    cumulative_variance_ratio: float,
) -> Motif:
    """Build one signed motif from the participation and membership columns.

    Member order is descending participation with ties broken by the neuron index;
    the polarity comes from the ``rms`` sign selection (the dominant axis wins, with a
    fallback to the other axis and finally ``+1`` when both components are zero).
    """
    matrix = np.asarray(participation, dtype=np.float64)
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    selected = np.asarray(membership)
    if selected.ndim != 2:
        raise MotifValidationError("membership must be 2-D")
    column = matrix[:, mode_index]
    candidates = np.flatnonzero(selected[:, mode_index])
    order = sorted((int(index) for index in candidates), key=lambda i: (-float(column[i]), i))

    members: list[MotifMember] = []
    n_positive = n_negative = n_mixed = 0
    region_counts: Counter[str] = Counter()
    strength_l1 = 0.0
    strength_energy = 0.0
    for rank, index in enumerate(order, start=1):
        l_value = float(left_array[index, mode_index])
        r_value = float(right_array[index, mode_index])
        participation_value = float(column[index])
        if abs(l_value) >= abs(r_value):
            axis = "left"
            polarity = _sign(l_value)
            if polarity == 0:
                polarity = _sign(r_value)
        else:
            axis = "right"
            polarity = _sign(r_value)
            if polarity == 0:
                polarity = _sign(l_value)
        if polarity == 0:
            polarity = 1
        l_sign, r_sign = _sign(l_value), _sign(r_value)
        if l_sign == 0 or r_sign == 0 or l_sign == r_sign:
            pass
        else:
            n_mixed += 1
        if polarity > 0:
            n_positive += 1
        else:
            n_negative += 1
        region = str(regions[index]) if index < len(regions) else ""
        region_counts[region] += 1
        strength_l1 += participation_value
        strength_energy += participation_value * participation_value
        members.append(
            MotifMember(
                index=index,
                neuron_id=str(neuron_ids[index]) if index < len(neuron_ids) else str(index),
                participation=participation_value,
                signed_participation=polarity * participation_value,
                polarity=polarity,
                left=l_value,
                right=r_value,
                dominant_axis=axis,
                region=region,
                rank=rank,
            )
        )

    n_members = len(members)
    dominant_region = ""
    if region_counts:
        dominant_region = min(region_counts.items(), key=lambda item: (-item[1], item[0]))[0]
    return Motif(
        label=f"M{int(mode_index) + 1:02d}",
        mode=int(mode_index) + 1,
        mode_index=int(mode_index),
        value=float(value),
        abs_value=float(abs_value),
        explained_variance_ratio=float(explained_variance_ratio),
        cumulative_variance_ratio=float(cumulative_variance_ratio),
        threshold=float(threshold),
        threshold_method=str(threshold_method),
        members=tuple(members),
        sender_members=tuple(entry.neuron_id for entry in members if entry.dominant_axis == "left"),
        receiver_members=tuple(entry.neuron_id for entry in members if entry.dominant_axis == "right"),
        strength_l1=float(strength_l1),
        strength_energy=float(strength_energy),
        share=0.0,
        polarity_balance=float(n_positive - n_negative) / n_members if n_members else 0.0,
        n_positive=int(n_positive),
        n_negative=int(n_negative),
        n_mixed=int(n_mixed),
        region_composition={key: int(value_) for key, value_ in sorted(region_counts.items())},
        dominant_region=dominant_region,
    )


def extract_motifs(
    participation: Any,
    left: Any,
    right: Any,
    *,
    thresholds: Any,
    threshold_method: str,
    membership: Any,
    neuron_ids: Sequence[str],
    regions: Sequence[str],
    values: Any,
    abs_values: Any,
    explained_variance_ratio: Any,
    cumulative_variance_ratio: Any,
) -> list[Motif]:
    """Extract every mode's motif and set the energy ``share`` in a second pass."""
    matrix = np.asarray(participation, dtype=np.float64)
    n_modes = int(matrix.shape[1]) if matrix.ndim == 2 else 0
    motifs = [
        extract_motif(
            matrix,
            left,
            right,
            mode_index,
            threshold=float(np.asarray(thresholds, dtype=np.float64)[mode_index]),
            threshold_method=threshold_method,
            membership=membership,
            neuron_ids=neuron_ids,
            regions=regions,
            value=float(np.asarray(values, dtype=np.float64)[mode_index]),
            abs_value=float(np.asarray(abs_values, dtype=np.float64)[mode_index]),
            explained_variance_ratio=float(np.asarray(explained_variance_ratio, dtype=np.float64)[mode_index]),
            cumulative_variance_ratio=float(np.asarray(cumulative_variance_ratio, dtype=np.float64)[mode_index]),
        )
        for mode_index in range(n_modes)
    ]
    total_energy = sum(motif.strength_energy for motif in motifs)
    if total_energy > 0.0:
        motifs = [
            replace(motif, share=float(motif.strength_energy / total_energy)) for motif in motifs
        ]
    return motifs


def initial_families(motifs: Sequence[Motif]) -> list[MotifFamily]:
    """Return one singleton family per mode (the neutral 04A family state)."""
    families: list[MotifFamily] = []
    for position, motif in enumerate(motifs):
        occurrences = tuple(
            {
                "neuron_id": member.neuron_id,
                "occurrences": [
                    {
                        "mode": int(motif.mode),
                        "participation": float(member.participation),
                        "signed_participation": float(member.signed_participation),
                    }
                ],
            }
            for member in motif.members
        )
        families.append(
            MotifFamily(
                family_id=position + 1,
                label=f"F{position + 1:02d}",
                modes=(int(motif.mode),),
                members=tuple(member.neuron_id for member in motif.members),
                polarity=float(motif.polarity_balance),
                occurrences=occurrences,
            )
        )
    return families


# ---------------------------------------------------------------------------
# Cross-mode tracking (04B): links, families and their statistics
# ---------------------------------------------------------------------------
def classify_link(
    jaccard: float,
    polarity_agreement: float,
    *,
    family_jaccard: float = DEFAULT_FAMILY_JACCARD,
    family_polarity: float = DEFAULT_FAMILY_POLARITY,
    link_min_jaccard: float = DEFAULT_LINK_MIN_JACCARD,
) -> str:
    """Classify one mode pair into one of :data:`LINK_CLASSES`.

    * ``weak`` -- ``jaccard < link_min_jaccard`` (barely any overlap);
    * ``stable`` -- ``jaccard >= family_jaccard`` **and**
      ``polarity_agreement >= family_polarity`` (shared members agree in sign);
    * ``flipped`` -- ``jaccard >= family_jaccard`` **and**
      ``polarity_agreement <= -family_polarity`` (shared members disagree in sign);
    * ``composite`` -- everything else (reportable overlap that is neither strong
      nor sign-consistent enough to merge two modes into one family).

    Only ``stable`` and ``flipped`` links may merge families; feeding ``composite``
    links to :func:`merge_families` would collapse most of the mode graph into a
    single giant component.
    """
    overlap = float(jaccard)
    agreement = float(polarity_agreement)
    if overlap < float(link_min_jaccard):
        return "weak"
    if overlap >= float(family_jaccard):
        if agreement >= float(family_polarity):
            return "stable"
        if agreement <= -float(family_polarity):
            return "flipped"
    return "composite"


def _member_index(motifs: Sequence[Motif]) -> dict[str, int]:
    """Neuron-id -> neuron index lookup over a sequence of motifs."""
    lookup: dict[str, int] = {}
    for motif in motifs:
        for member in motif.members:
            lookup.setdefault(member.neuron_id, int(member.index))
    return lookup


def motif_links(
    motifs: Sequence[Motif],
    config: MotifConfig | None = None,
    *,
    family_jaccard: float | None = None,
    family_polarity: float | None = None,
    link_min_jaccard: float | None = None,
) -> list[MotifLink]:
    """Return every mode pair that shares at least one member, as a link table.

    ``shared_members`` is the signed-set intersection in ascending neuron index;
    ``jaccard = |shared| / |union|`` and
    ``polarity_agreement = (n_same - n_diff) / n_shared`` (signed, in ``[-1, 1]``).
    Entries are ordered by ``(source_mode, target_mode)`` so the table is a pure
    function of the motifs and the three thresholds.
    """
    resolved = config or MotifConfig()
    min_jaccard = (
        float(link_min_jaccard) if link_min_jaccard is not None else float(resolved.link_min_jaccard)
    )
    merge_jaccard = (
        float(family_jaccard) if family_jaccard is not None else float(resolved.family_jaccard)
    )
    merge_polarity = (
        float(family_polarity) if family_polarity is not None else float(resolved.family_polarity)
    )
    index_of = _member_index(motifs)
    polarities = [
        {member.neuron_id: int(member.polarity) for member in motif.members} for motif in motifs
    ]
    links: list[MotifLink] = []
    for first in range(len(motifs)):
        for second in range(first + 1, len(motifs)):
            left, right = polarities[first], polarities[second]
            shared = sorted(left.keys() & right.keys(), key=lambda name: index_of[name])
            if not shared:
                continue
            union = len(left.keys() | right.keys())
            jaccard = len(shared) / union if union else 0.0
            same = sum(1 for name in shared if left[name] == right[name])
            agreement = (2 * same - len(shared)) / len(shared)
            links.append(
                MotifLink(
                    source_mode=int(motifs[first].mode),
                    target_mode=int(motifs[second].mode),
                    source_label=motifs[first].label,
                    target_label=motifs[second].label,
                    shared_members=tuple(shared),
                    jaccard=float(jaccard),
                    polarity_agreement=float(agreement),
                    classification=classify_link(
                        jaccard,
                        agreement,
                        family_jaccard=merge_jaccard,
                        family_polarity=merge_polarity,
                        link_min_jaccard=min_jaccard,
                    ),
                )
            )
    return links


def family_members(
    modes: Sequence[int],
    by_mode: dict[int, Motif] | Sequence[Motif],
    signs: dict[int, int] | None = None,
) -> tuple[tuple[str, ...], tuple[dict[str, Any], ...]]:
    """Union the members of *modes* and collect their per-mode occurrences.

    Members are ordered by ``(mode asc, member rank asc)``; occurrences carry the
    ``mode``, the unsigned ``participation`` and the ``signed_participation`` after
    applying the family's polarity ``signs`` (``+1`` keeps, ``-1`` flips), so a
    flipped mode contributes sign-aligned values.
    """
    lookup = by_mode if isinstance(by_mode, dict) else {motif.mode: motif for motif in by_mode}
    ordered: list[str] = []
    entries: dict[str, list[dict[str, Any]]] = {}
    for mode in modes:
        motif = lookup[int(mode)]
        sign = 1 if signs is None else int(signs.get(int(mode), 1))
        for member in motif.members:
            if member.neuron_id not in entries:
                entries[member.neuron_id] = []
                ordered.append(member.neuron_id)
            entries[member.neuron_id].append(
                {
                    "mode": int(motif.mode),
                    "participation": float(member.participation),
                    "signed_participation": float(sign * member.signed_participation),
                }
            )
    occurrences = tuple({"neuron_id": name, "occurrences": entries[name]} for name in ordered)
    return tuple(ordered), occurrences


def merge_families(
    motifs: Sequence[Motif],
    links: Sequence[MotifLink],
) -> list[MotifFamily]:
    """Merge modes into families using **only** the ``stable``/``flipped`` links.

    ``composite`` and ``weak`` links never merge: union-find over every reported
    link would collapse the whole mode graph into one giant component.  Within a
    component the lowest mode is the representative (sign ``+1``); a ``stable`` edge
    keeps the sign and a ``flipped`` edge negates it, so a contradiction is reported
    through :data:`CODE_FAMILY_CONFLICT`.  Families are ordered by their lowest mode
    and labelled ``F01``, ``F02``, ... (``family_id`` is 1-based).  ``families``
    stays a partition of all modes.
    """
    total = len(motifs)
    position = {int(motif.mode): index for index, motif in enumerate(motifs)}
    parent = list(range(total))

    def _find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    adjacency: dict[int, list[tuple[int, int]]] = {}
    for link in links:
        if link.classification not in ("stable", "flipped"):
            continue
        if int(link.source_mode) not in position or int(link.target_mode) not in position:
            continue
        first = position[int(link.source_mode)]
        second = position[int(link.target_mode)]
        if first == second:
            continue
        root_first, root_second = _find(first), _find(second)
        if root_first != root_second:
            parent[max(root_first, root_second)] = min(root_first, root_second)
        sign = 1 if link.classification == "stable" else -1
        adjacency.setdefault(first, []).append((second, sign))
        adjacency.setdefault(second, []).append((first, sign))

    components: dict[int, list[int]] = {}
    for node in range(total):
        components.setdefault(_find(node), []).append(node)

    families: list[MotifFamily] = []
    roots = sorted(components, key=lambda key: min(components[key]))
    for family_id, root in enumerate(roots, start=1):
        group = sorted(components[root])
        representative = group[0]
        signs: dict[int, int] = {representative: 1}
        stack = [representative]
        while stack:
            node = stack.pop()
            for neighbour, sign in adjacency.get(node, ()):
                wanted = signs[node] * sign
                if neighbour in signs:
                    if signs[neighbour] != wanted:
                        LOGGER.warning(
                            "[%s] modes %d and %d disagree on the family polarity",
                            CODE_FAMILY_CONFLICT,
                            motifs[neighbour].mode,
                            motifs[node].mode,
                        )
                else:
                    signs[neighbour] = wanted
                    stack.append(neighbour)
        for node in group:
            signs.setdefault(node, 1)
        modes = tuple(motifs[node].mode for node in group)
        signed = {motifs[node].mode: signs[node] for node in group}
        members, occurrences = family_members(modes, motifs, signed)
        polarity = sum(signs[node] * motifs[node].polarity_balance for node in group) / len(group)
        families.append(
            MotifFamily(
                family_id=family_id,
                label=f"F{family_id:02d}",
                modes=modes,
                members=members,
                polarity=float(polarity),
                occurrences=occurrences,
            )
        )
    return families


def recurrence_table(
    motifs: Sequence[Motif],
    families: Sequence[MotifFamily] | Sequence[dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Per-neuron recurrence across modes (and families when supplied).

    Returns ``{neuron_id: {n_modes, modes, n_families, families,
    max_participation, max_signed_participation}}`` with the mode/family lists
    sorted.  It complements the frozen ``motifs.data.json`` picture and is the
    natural input for 04C's membership-based grouping.
    """
    table: dict[str, dict[str, Any]] = {}

    def _entry(neuron_id: str) -> dict[str, Any]:
        return table.setdefault(
            neuron_id,
            {
                "neuron_id": neuron_id,
                "n_modes": 0,
                "modes": [],
                "n_families": 0,
                "families": [],
                "max_participation": 0.0,
                "max_signed_participation": 0.0,
            },
        )

    for motif in motifs:
        for member in motif.members:
            entry = _entry(member.neuron_id)
            entry["n_modes"] += 1
            entry["modes"].append(int(motif.mode))
            entry["max_participation"] = max(
                float(entry["max_participation"]), float(member.participation)
            )
            entry["max_signed_participation"] = max(
                float(entry["max_signed_participation"]), abs(float(member.signed_participation))
            )
    if families:
        for family in families:
            if isinstance(family, dict):
                family_id = int(family.get("family_id", 0))
                members = [str(name) for name in family.get("members") or []]
            else:
                family_id = int(family.family_id)
                members = [str(name) for name in family.members]
            for name in members:
                entry = _entry(name)
                entry["n_families"] += 1
                entry["families"].append(family_id)
    for entry in table.values():
        entry["modes"] = sorted(entry["modes"])
        entry["families"] = sorted(entry["families"])
    return table


# ---------------------------------------------------------------------------
# Functional grouping (04C)
# ---------------------------------------------------------------------------
def assign_groups(participation: Any) -> np.ndarray:
    """Assign each neuron to its strongest participation mode; zero rows get ``G00``."""
    matrix = np.asarray(participation, dtype=np.float64)
    if matrix.ndim != 2:
        raise MotifValidationError("participation must be a 2-D matrix")
    labels = np.zeros(matrix.shape[0], dtype="<i8")
    nonzero = np.any(matrix > 0.0, axis=1)
    if np.any(nonzero):
        labels[nonzero] = np.argmax(matrix[nonzero], axis=1).astype("<i8") + 1
    return labels


def initial_group_labels(participation: Any) -> np.ndarray:
    """Return the deterministic, mode-aligned default motif grouping labels."""
    return assign_groups(participation)


def group_from_labels(labels: Any, *, mode_aligned: bool = False) -> np.ndarray:
    """Canonicalize raw labels, preserving background label zero.

    ``mode_aligned`` is used by the motif path: its labels already mean mode numbers and
    must remain ``G01 == M01``. Loading-space cluster labels use the size/min-label ordering.
    """
    raw = np.asarray(labels, dtype="<i8").reshape(-1)
    if raw.size == 0:
        return raw.copy()
    if mode_aligned:
        return raw.copy()
    result = np.zeros_like(raw)
    groups = [int(value) for value in np.unique(raw) if int(value) != 0]
    groups.sort(key=lambda value: (-int(np.count_nonzero(raw == value)), value))
    for new_id, old_id in enumerate(groups, start=1):
        result[raw == old_id] = new_id
    return result


def _dominant_signs(left: Any, right: Any) -> np.ndarray:
    """Return the 04A dominant-axis polarity for every neuron/mode pair."""
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    signs = np.sign(np.where(np.abs(left_array) >= np.abs(right_array), left_array, right_array))
    fallback = signs == 0
    signs[fallback] = np.sign(np.where(np.abs(left_array[fallback]) >= np.abs(right_array[fallback]),
                                       right_array[fallback], left_array[fallback]))
    signs[signs == 0] = 1.0
    return signs.astype("<i8")


def group_centroid(members: Sequence[int], participation: Any) -> np.ndarray:
    """Return the mean unsigned participation vector for a group."""
    matrix = np.asarray(participation, dtype=np.float64)
    indices = np.asarray(list(members), dtype=np.int64)
    if indices.size == 0:
        return np.zeros(matrix.shape[1], dtype="<f8")
    return np.asarray(matrix[indices].mean(axis=0), dtype="<f8")


def _cosine_matrix(rows: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(rows, axis=1)
    normalized = np.divide(rows, norms[:, None], out=np.zeros_like(rows), where=norms[:, None] > 0)
    return normalized @ normalized.T


def group_coherence(vectors: Any, labels: Any) -> dict[int, float | None]:
    """Return mean pairwise cosine coherence for each labeled group."""
    rows = np.asarray(vectors, dtype=np.float64)
    group_labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if rows.ndim != 2 or rows.shape[0] != group_labels.size:
        raise MotifValidationError("vectors and labels have incompatible shapes")
    result: dict[int, float | None] = {}
    for group_id in sorted(int(value) for value in np.unique(group_labels)):
        indices = np.flatnonzero(group_labels == group_id)
        if indices.size < 2:
            result[group_id] = None
            continue
        matrix = _cosine_matrix(rows[indices])
        upper = np.triu_indices(indices.size, 1)
        result[group_id] = float(matrix[upper].mean())
    return result


def region_composition(members: Sequence[int], regions: Sequence[str]) -> dict[str, int]:
    """Return a deterministic region-count mapping for neuron indices."""
    counts = Counter(str(regions[index]) for index in sorted(int(value) for value in members))
    return {name: int(counts[name]) for name in sorted(counts)}


def split_large_groups(
    labels: Any,
    signs: Any,
    *,
    polarity_min_members: int = DEFAULT_POLARITY_MIN_MEMBERS,
) -> tuple[np.ndarray, int]:
    """Split sufficiently large minority-polarity sides and return labels/count."""
    result = np.asarray(labels, dtype="<i8").copy()
    polarity = np.asarray(signs, dtype=np.int64)
    next_id = int(result.max(initial=0)) + 1
    created = 0
    for group_id in sorted(int(value) for value in np.unique(result) if int(value) != 0):
        members = np.flatnonzero(result == group_id)
        if members.size == 0 or polarity.ndim != 2:
            continue
        mode = min(group_id - 1, polarity.shape[1] - 1)
        positive = members[polarity[members, mode] > 0]
        negative = members[polarity[members, mode] < 0]
        if not positive.size or not negative.size:
            continue
        minority = negative if negative.size <= positive.size else positive
        if minority.size >= int(polarity_min_members):
            result[minority] = next_id
            next_id += 1
            created += 1
    return result, created


def _centroid_similarity(first: np.ndarray, second: np.ndarray) -> float:
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm == 0.0 or second_norm == 0.0:
        return 0.0
    return float(np.dot(first, second) / (first_norm * second_norm))


def merge_similar_groups(
    labels: Any,
    features: Any,
    *,
    max_group_size: int = DEFAULT_MAX_GROUP_SIZE,
    merge_threshold: float = DEFAULT_MERGE_THRESHOLD,
) -> np.ndarray:
    """Merge the most similar pair while an oversized group can be reduced."""
    result = np.asarray(labels, dtype="<i8").copy()
    matrix = np.asarray(features, dtype=np.float64)
    changed = False
    while True:
        ids = [int(value) for value in np.unique(result) if int(value) != 0]
        oversized = [value for value in ids if np.count_nonzero(result == value) > int(max_group_size)]
        if not oversized or len(ids) < 2:
            break
        candidates: list[tuple[float, int, int]] = []
        for position, first in enumerate(ids):
            first_members = np.flatnonzero(result == first)
            first_centroid = matrix[first_members].mean(axis=0)
            for second in ids[position + 1:]:
                second_members = np.flatnonzero(result == second)
                similarity = _centroid_similarity(first_centroid, matrix[second_members].mean(axis=0))
                candidates.append((similarity, min(first, second), max(first, second)))
        if not candidates:
            break
        similarity, first, second = max(candidates, key=lambda item: (item[0], -item[1], -item[2]))
        if similarity < float(merge_threshold):
            break
        result[result == second] = first
        changed = True
    return group_from_labels(result) if changed else result


def loadings_clustering(
    features: Any,
    *,
    n_clusters: int,
    linkage: str = DEFAULT_LINKAGE,
    affinity: str = DEFAULT_AFFINITY,
) -> np.ndarray:
    """Run deterministic agglomerative clustering on nonzero loading rows."""
    from sklearn.cluster import AgglomerativeClustering

    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < int(n_clusters):
        raise MotifValidationError("features cannot support the requested number of groups")
    model = AgglomerativeClustering(
        n_clusters=int(n_clusters), metric=str(affinity), linkage=str(linkage)
    )
    return np.asarray(model.fit_predict(matrix), dtype="<i8") + 1


def _group_statistics(groups: Sequence[NeuronGroup | dict[str, Any]], n_polarity_split: int) -> dict[str, Any]:
    sizes = [int(group.get("size", 0) if isinstance(group, dict) else group.size) for group in groups]
    background = [group for group in groups if bool(group.get("is_background") if isinstance(group, dict) else group.is_background)]
    non_background_singletons = [group for group in groups if (int(group.get("size", 0)) if isinstance(group, dict) else group.size) == 1 and group not in background]
    expected = {
        "n_groups": int(len(groups)),
        "group_sizes": sizes,
        "n_background_groups": int(len(background)),
        "n_singleton_groups": int(len(non_background_singletons)),
        "largest_group": int(max(sizes)) if sizes else 0,
        "smallest_group": int(min(sizes)) if sizes else 0,
        "mean_group_size": float(np.mean(np.asarray(sizes, dtype=np.float64))) if sizes else 0.0,
        "n_polarity_split_groups": int(n_polarity_split),
    }
    return expected


def build_groups(
    motifs: Sequence[Motif],
    config: MotifConfig,
    *,
    participation: Any,
    loadings_left: Any,
    loadings_right: Any,
    neuron_ids: Sequence[str],
    regions: Sequence[str],
    node_metadata: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[NeuronGroup], np.ndarray | None, np.ndarray | None, int]:
    """Build 04C groups and return groups, labels, centroids, and split count."""
    matrix = np.asarray(participation, dtype=np.float64)
    left = np.asarray(loadings_left, dtype=np.float64)
    right = np.asarray(loadings_right, dtype=np.float64)
    n_neurons, n_modes = matrix.shape
    if config.grouping == "none":
        return [], None, None, 0
    zero_rows = ~np.any(matrix > 0.0, axis=1)
    if np.any(zero_rows):
        LOGGER.info("[%s] removed and re-attached %d zero-vector neuron(s) as G00", CODE_ZERO_VECTOR, int(zero_rows.sum()))
    features = np.hstack([left, right])
    if config.grouping == "motif" or config.grouping == "hybrid":
        labels = initial_group_labels(matrix)
        active_features = matrix if config.grouping == "motif" else features
    else:
        active = np.flatnonzero(~zero_rows)
        requested = n_modes if str(config.n_groups) == "auto" else int(config.n_groups)
        clustered = loadings_clustering(features[active], n_clusters=requested, linkage=config.linkage, affinity=config.affinity)
        labels = np.zeros(n_neurons, dtype="<i8")
        labels[active] = clustered
        active_features = features
        labels = group_from_labels(labels)
    signs = _dominant_signs(left, right)
    split_count = 0
    if config.polarity_split:
        labels, split_count = split_large_groups(labels, signs, polarity_min_members=config.polarity_min_members)
    labels = merge_similar_groups(labels, active_features, max_group_size=config.max_group_size, merge_threshold=config.merge_threshold)
    if config.grouping == "motif" and not config.polarity_split:
        labels = initial_group_labels(matrix)
    labels = np.asarray(labels, dtype="<i8")
    signed = matrix * signs
    coherence = group_coherence(signed, labels)
    groups: list[NeuronGroup] = []
    metadata = node_metadata or {}
    for group_id in range(int(labels.max(initial=0)) + 1):
        members = tuple(int(index) for index in np.flatnonzero(labels == group_id))
        if group_id == 0:
            dominant_mode = None
        else:
            centroid = matrix[list(members)].mean(axis=0) if members else np.zeros(n_modes)
            dominant_mode = int(np.argmax(centroid)) + 1 if members else group_id
        cents = [metadata.get(neuron_ids[index], {}).get("cent") for index in members]
        cents = [float(value) for value in cents if value is not None]
        groups.append(NeuronGroup(
            group_id=f"G{group_id:02d}", label=f"G{group_id:02d}", members=tuple(neuron_ids[index] for index in members),
            dominant_mode=dominant_mode, region_composition=region_composition(members, regions),
            cent_mean=float(np.mean(cents)) if cents else None, coherence=coherence.get(group_id),
            is_background=group_id == 0, is_singleton=len(members) == 1,
        ))
    sizes = np.asarray([group.size for group in groups], dtype="<i8")
    centroids = np.vstack([group_centroid(np.flatnonzero(labels == group_id), matrix) for group_id in range(len(groups))]).astype("<f8")
    if int(sizes.sum()) != n_neurons or len(np.unique(np.concatenate([np.flatnonzero(labels == group_id) for group_id in range(len(groups))]))) != n_neurons:
        raise MotifValidationError("group partition invariant violated")
    if np.any(zero_rows & (labels != 0)):
        raise MotifValidationError("zero-vector neuron was not re-attached to G00")
    return groups, labels, centroids, split_count


def _family_of_mode(
    families: Sequence[MotifFamily] | Sequence[dict[str, Any]], n_modes: int
) -> np.ndarray:
    """``(k,) <i8`` family index per mode (0-based, identity for singletons)."""
    array = np.full(int(n_modes), -1, dtype="<i8")
    for index, family in enumerate(families):
        modes = family.get("modes") if isinstance(family, dict) else family.modes
        for mode in modes or ():
            if 1 <= int(mode) <= int(n_modes):
                array[int(mode) - 1] = index
    return array


def _link_matrices(
    links: Sequence[MotifLink] | Sequence[dict[str, Any]], n_modes: int
) -> tuple[np.ndarray, np.ndarray]:
    """``(k, k)`` symmetric Jaccard and polarity-agreement matrices (unit diagonal)."""
    jaccard = np.zeros((int(n_modes), int(n_modes)), dtype="<f8")
    polarity = np.zeros((int(n_modes), int(n_modes)), dtype="<f8")
    np.fill_diagonal(jaccard, 1.0)
    np.fill_diagonal(polarity, 1.0)
    for link in links:
        if isinstance(link, dict):
            source = int(link.get("source_mode", 0)) - 1
            target = int(link.get("target_mode", 0)) - 1
            jaccard_value = float(link.get("jaccard", 0.0))
            polarity_value = float(link.get("polarity_agreement", 0.0))
        else:
            source = int(link.source_mode) - 1
            target = int(link.target_mode) - 1
            jaccard_value = float(link.jaccard)
            polarity_value = float(link.polarity_agreement)
        if 0 <= source < n_modes and 0 <= target < n_modes:
            jaccard[source, target] = jaccard[target, source] = jaccard_value
            polarity[source, target] = polarity[target, source] = polarity_value
    return jaccard, polarity


def _tracking_statistics(
    links: Sequence[MotifLink] | Sequence[dict[str, Any]],
    families: Sequence[MotifFamily] | Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """The 04B ``metadata`` block (``n_links``, ``link_classes``, family counts)."""
    classes = {name: 0 for name in LINK_CLASSES}
    for link in links:
        classification = (
            link.get("classification") if isinstance(link, dict) else link.classification
        )
        if classification in classes:
            classes[classification] += 1
    multi = 0
    for family in families:
        n_modes = family.get("n_modes") if isinstance(family, dict) else family.n_modes
        if int(n_modes) > 1:
            multi += 1
    return {
        "n_links": len(links),
        "link_classes": classes,
        "n_families": len(families),
        "n_multi_mode_families": multi,
    }


# ---------------------------------------------------------------------------
# 04D -- pathway graph
# ---------------------------------------------------------------------------
def mode_weights(values: Any, rule: str = DEFAULT_PATHWAY_WEIGHT) -> np.ndarray:
    """Return the ``(k,)`` mode weights for *rule* (always ``Sigma |w_j| == 1``).

    ``evr`` and ``energy`` are numerically equivalent (``energy = value**2``,
    ``evr = value**2 / Sigma value**2``) because the weights are renormalised over the
    retained modes; both enum values are kept for readability.  ``value`` keeps the
    sign of the mode value and can therefore produce negative weights; ``abs_value``
    uses the magnitude only.
    """
    array = np.asarray(values, dtype=np.float64).ravel()
    if rule in ("evr", "energy"):
        raw = array**2
    elif rule == "abs_value":
        raw = np.abs(array)
    elif rule == "value":
        raw = array.copy()
    elif rule == "uniform":
        raw = np.ones_like(array)
    else:
        raise MotifValidationError(
            f"unknown pathway weight rule {rule!r}; expected one of {PATHWAY_WEIGHTS}"
        )
    scale = float(np.abs(raw).sum())
    if not math.isfinite(scale) or scale <= 0.0:
        raise MotifValidationError(
            f"[{CODE_MODE_WEIGHTS}] the retained mode values are all zero: no weight rule applies"
        )
    weights = np.asarray(raw / scale, dtype="<f8")
    if rule == "value" and float(weights.sum()) <= 0.0:
        raise MotifValidationError(
            f"[{CODE_MODE_WEIGHTS}] --pathway-weight value needs a non-negative retained "
            f"spectrum (Sigma value = {float(array.sum())}); use abs_value instead"
        )
    return weights


def mode_outer_product(loadings_left: Any, loadings_right: Any, values: Any, mode: int) -> np.ndarray:
    """Return the ``(N, N)`` rank-1 outer product ``value_m * u_m v_m^T`` (*mode* 1-based)."""
    left = np.asarray(loadings_left, dtype=np.float64)
    right = np.asarray(loadings_right, dtype=np.float64)
    array = np.asarray(values, dtype=np.float64).ravel()
    if left.ndim != 2 or right.ndim != 2 or left.shape != right.shape:
        raise MotifValidationError("the loading bases must be two arrays of the same shape")
    index = int(mode) - 1
    if not 0 <= index < array.size:
        raise MotifValidationError(f"mode {mode} is outside the retained range 1..{array.size}")
    return np.asarray(array[index] * np.outer(left[:, index], right[:, index]), dtype="<f8")


def _mode_outer_product_stack(loadings_left: Any, loadings_right: Any, values: Any) -> np.ndarray:
    """Return the ``(k, N, N)`` stack of ``value_m * u_m v_m^T`` (mode-major)."""
    array = np.asarray(values, dtype=np.float64).ravel()
    return np.asarray(
        [
            mode_outer_product(loadings_left, loadings_right, array, index + 1)
            for index in range(array.size)
        ],
        dtype="<f8",
    )


def _group_pair_sum(matrix: Any, group_labels: Any, n_groups: int) -> np.ndarray:
    """Sum a square ``(N, N)`` matrix over group member pairs -> ``(G, G)``."""
    values = np.asarray(matrix, dtype=np.float64)
    labels = np.asarray(group_labels, dtype="<i8").ravel()
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise MotifValidationError("pair contributions must come from a square matrix")
    if labels.size != values.shape[0]:
        raise MotifValidationError("group_labels length does not match the matrix")
    out = np.zeros((int(n_groups), int(n_groups)), dtype="<f8")
    if labels.size == 0 or int(n_groups) <= 0:
        return out
    indicator = np.zeros((labels.size, int(n_groups)), dtype=np.float64)
    indicator[np.arange(labels.size), labels] = 1.0
    return np.asarray(indicator.T @ values @ indicator, dtype="<f8")


def group_pair_contributions(outer_products: Any, group_labels: Any, weights: Any) -> np.ndarray:
    """Aggregate per-mode outer products into signed ``(G, G)`` group contributions.

    ``W[A, B] = sum_m weights[m] * sum_{i in A, j in B} outer_products[m, i, j]``.
    """
    stack = np.asarray(outer_products, dtype=np.float64)
    labels = np.asarray(group_labels, dtype="<i8").ravel()
    weight_array = np.asarray(weights, dtype=np.float64).ravel()
    if stack.ndim != 3 or stack.shape[1] != stack.shape[2]:
        raise MotifValidationError("outer_products must have shape (k, N, N)")
    if stack.shape[0] != weight_array.size:
        raise MotifValidationError(
            "the number of mode weights does not match the outer-product stack"
        )
    if labels.size != stack.shape[1]:
        raise MotifValidationError("group_labels length does not match the outer products")
    if labels.size and int(labels.min()) < 0:
        raise MotifValidationError("group_labels must be non-negative")
    n_groups = int(labels.max()) + 1 if labels.size else 0
    out = np.zeros((n_groups, n_groups), dtype="<f8")
    for index in range(stack.shape[0]):
        if weight_array[index] == 0.0:
            continue
        out += weight_array[index] * _group_pair_sum(stack[index], labels, n_groups)
    return np.asarray(out, dtype="<f8")


def filter_pathway_edges(
    contributions: Any,
    *,
    edge_threshold: float = DEFAULT_PATHWAY_EDGE_THRESHOLD,
    top_edges: int = DEFAULT_PATHWAY_TOP_EDGES,
    intra: bool = DEFAULT_INTRA,
) -> tuple[list[tuple[int, int, float]], dict[str, Any]]:
    """Threshold and top-N filter a ``(G, G)`` contribution matrix.

    The threshold is a **fraction of ``abs_max``** (the largest absolute contribution over
    the whole matrix), so the stored weights stay raw and the scale never shifts with
    ``intra``/top-N.  Returns the kept ``(source, target, weight)`` triples ordered by
    ``(source, target)`` plus the pathway statistics (including the diagnostics-only
    threshold sweep).
    """
    matrix = np.asarray(contributions, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise MotifValidationError("pathway contributions must be a square (G, G) matrix")
    n_groups = int(matrix.shape[0])
    magnitude = np.abs(matrix)
    abs_max = float(magnitude.max()) if magnitude.size else 0.0
    total = float(magnitude.sum())
    sweep = {
        f"{value:g}": (int((magnitude >= value * abs_max).sum()) if abs_max > 0.0 else 0)
        for value in PATHWAY_THRESHOLD_SWEEP
    }
    cutoff = abs_max * float(edge_threshold)
    candidates: list[tuple[int, int, float]] = []
    if abs_max > 0.0:
        for source in range(n_groups):
            for target in range(n_groups):
                if source == target and not intra:
                    continue
                weight = float(matrix[source, target])
                if abs(weight) >= cutoff:
                    candidates.append((source, target, weight))
    candidates.sort(key=lambda item: (-abs(item[2]), item[0], item[1]))
    kept: list[tuple[int, int, float]] = []
    per_source: dict[int, int] = {}
    for source, target, weight in candidates:
        if per_source.get(source, 0) >= int(top_edges):
            continue
        per_source[source] = per_source.get(source, 0) + 1
        kept.append((source, target, weight))
    kept.sort(key=lambda item: (item[0], item[1]))
    weights = [weight for _, _, weight in kept]
    stats: dict[str, Any] = {
        "abs_max": abs_max,
        "cutoff": cutoff,
        "n_candidates": len(candidates),
        "n_edges": len(kept),
        "n_positive": sum(1 for weight in weights if weight > 0.0),
        "n_negative": sum(1 for weight in weights if weight < 0.0),
        "n_zeros": sum(1 for weight in weights if weight == 0.0),
        "n_intra": sum(1 for source, target, _ in kept if source == target),
        "n_cross": sum(1 for source, target, _ in kept if source != target),
        "weight_concentration": (abs_max / total) if total > 0.0 else 0.0,
        "threshold_sweep": sweep,
        "per_source_counts": dict(sorted(per_source.items())),
        "weights": weights,
    }
    return kept, stats


def _top_mode_records(values: Any, limit: int) -> list[dict[str, Any]]:
    """Return the top-*limit* ``TOP_MODE_KEYS`` records of a mode profile.

    ``share`` is relative to the profile's total absolute magnitude, so the shares of the
    retained records always sum to at most 1.
    """
    array = np.asarray(values, dtype=np.float64).ravel()
    total = float(np.abs(array).sum())
    if array.size == 0 or total <= 0.0:
        return []
    order = sorted(range(array.size), key=lambda index: (-abs(array[index]), index))
    records: list[dict[str, Any]] = []
    for rank, index in enumerate(order[: int(limit)], start=1):
        value = float(array[index])
        if value == 0.0:
            break
        records.append(
            {
                "mode": index + 1,
                "rank": rank,
                "share": abs(value) / total,
                "signed_weight": value,
                "abs_weight": abs(value),
            }
        )
    return records


def build_pathway(
    groups: Sequence[NeuronGroup | dict[str, Any]],
    *,
    participation: Any,
    loadings_left: Any,
    loadings_right: Any,
    values: Any,
    neuron_ids: Sequence[str],
    config: MotifConfig,
    matrix: Any = None,
) -> tuple[PathwayDiagram, dict[str, Any]]:
    """Build the 04D pathway diagram and return ``(diagram, diagnostics)``.

    ``matrix`` is the optional Phase 02 effective matrix used by
    ``--pathway-source matrix``/``both``; when it is ``None`` the contributions come from
    the per-mode outer products only.  The Phase 03/04 objects are never mutated.
    """
    if not groups:
        return PathwayDiagram.neutral(), {
            "abs_max": 0.0,
            "cutoff": 0.0,
            "n_candidates": 0,
            "threshold_sweep": {},
            "per_source_counts": {},
            "mode_weights": [],
            "z_contributions": {},
        }
    profile_matrix = np.asarray(participation, dtype=np.float64)
    left = np.asarray(loadings_left, dtype=np.float64)
    right = np.asarray(loadings_right, dtype=np.float64)
    retained = np.asarray(values, dtype=np.float64).ravel()
    position = {name: index for index, name in enumerate(neuron_ids)}
    labels = np.full(len(neuron_ids), -1, dtype="<i8")
    group_members: list[list[int]] = []
    group_records: list[dict[str, Any]] = []
    for index, group in enumerate(groups):
        record = group if isinstance(group, dict) else group.to_dict()
        members = [
            position[name] for name in record.get("members") or () if name in position
        ]
        group_members.append(members)
        group_records.append(record)
        for member in members:
            labels[member] = index
    if np.any(labels < 0):
        raise MotifValidationError("the groups do not partition the neuron order")
    weights = mode_weights(retained, config.pathway_weight)
    if not math.isclose(float(np.abs(weights).sum()), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise MotifValidationError(f"[{CODE_MODE_WEIGHTS}] the mode weights do not sum to 1")
    stack = _mode_outer_product_stack(left, right, retained)
    mode_matrix = group_pair_contributions(stack, labels, weights)
    requested = str(config.pathway_source)
    matrix_matrix: np.ndarray | None = None
    if matrix is not None and requested in ("matrix", "both"):
        z_values = np.asarray(matrix, dtype=np.float64)
        if z_values.shape != (len(neuron_ids), len(neuron_ids)):
            raise MotifValidationError(
                "[matrix_mismatch] the Phase 02 matrix shape does not match the neuron order"
            )
        matrix_matrix = _group_pair_sum(z_values, labels, len(groups))
    if matrix_matrix is None:
        resolved = "mode"
        contributions = mode_matrix
    elif requested == "matrix":
        resolved = "matrix"
        contributions = matrix_matrix
    else:
        resolved = "both"
        contributions = mode_matrix + matrix_matrix
    kept, stats = filter_pathway_edges(
        contributions,
        edge_threshold=config.pathway_edge_threshold,
        top_edges=config.pathway_top_edges,
        intra=bool(config.intra),
    )
    signs = _dominant_signs(left, right)
    signed = profile_matrix * signs
    nodes: list[dict[str, Any]] = []
    for index, record in enumerate(group_records):
        members = group_members[index]
        profile = signed[members].mean(axis=0) if members else np.zeros(retained.size)
        nodes.append(
            {
                "node_id": str(record.get("group_id")),
                "group_id": str(record.get("group_id")),
                "label": str(record.get("label") or record.get("group_id")),
                "size": int(record.get("size", len(members))),
                "members": list(record.get("members") or ()),
                "cent_mean": record.get("cent_mean"),
                "region_composition": dict(record.get("region_composition") or {}),
                "dominant_mode": record.get("dominant_mode"),
                "top_modes": _top_mode_records(profile, PATHWAY_TOP_MODES),
            }
        )
    node_ids = [node["node_id"] for node in nodes]
    if len(set(node_ids)) != len(node_ids) or any(not name for name in node_ids):
        raise MotifValidationError("pathway node ids must be unique and non-empty")
    edges: list[dict[str, Any]] = []
    z_contributions: dict[tuple[str, str], float] = {}
    seen: set[tuple[str, str]] = set()
    threshold_value = float(config.pathway_edge_threshold)
    for source_index, target_index, weight in kept:
        source_id = node_ids[source_index]
        target_id = node_ids[target_index]
        by_mode = np.zeros(retained.size, dtype=np.float64)
        if resolved != "matrix":
            rows = group_members[source_index]
            columns = group_members[target_index]
            if rows and columns:
                for mode_index in range(retained.size):
                    by_mode[mode_index] = weights[mode_index] * float(
                        stack[mode_index][np.ix_(rows, columns)].sum()
                    )
        modes = [
            mode_index + 1
            for mode_index in range(retained.size)
            if by_mode[mode_index] != 0.0
        ]
        key = (source_id, target_id)
        if key in seen:
            raise MotifValidationError(f"duplicate pathway edge {key}")
        seen.add(key)
        if abs(weight) < threshold_value * float(stats["abs_max"]) - 1e-12:
            raise MotifValidationError(f"the pathway edge {key} violates the edge threshold")
        edges.append(
            {
                "source": source_id,
                "target": target_id,
                "weight": float(weight),
                "abs_weight": abs(float(weight)),
                "n_modes": len(modes),
                "modes": modes,
                "is_intra": bool(source_index == target_index),
                "polarity": int(np.sign(weight)),
                "top_modes": _top_mode_records(by_mode, PATHWAY_TOP_MODES),
            }
        )
        if matrix_matrix is not None:
            z_contributions[key] = float(matrix_matrix[source_index, target_index])
    diagram = PathwayDiagram(
        nodes=tuple(nodes),
        edges=tuple(edges),
        n_nodes=len(nodes),
        n_edges=len(edges),
        weight=str(config.pathway_weight),
        source=resolved,
        edge_threshold=threshold_value,
        top_edges=int(config.pathway_top_edges),
        intra=bool(config.intra),
        abs_max=float(stats["abs_max"]),
        n_positive=int(stats["n_positive"]),
        n_negative=int(stats["n_negative"]),
        n_intra=int(stats["n_intra"]),
        n_cross=int(stats["n_cross"]),
        weight_concentration=float(stats["weight_concentration"]),
        weights=tuple(float(value) for value in stats["weights"]),
    )
    diagnostics: dict[str, Any] = {
        "abs_max": float(stats["abs_max"]),
        "cutoff": float(stats["cutoff"]),
        "n_candidates": int(stats["n_candidates"]),
        "threshold_sweep": dict(stats["threshold_sweep"]),
        "per_source_counts": dict(stats["per_source_counts"]),
        "mode_weights": [float(value) for value in weights],
        "z_contributions": dict(z_contributions),
    }
    return diagram, diagnostics


def _pathway_adjacency(edges: Sequence[dict[str, Any]], node_ids: Sequence[str]) -> np.ndarray:
    """Return the ``(G, G)`` signed adjacency of the *filtered* pathway edges.

    Purely derivable from the payload, so it is part of the sidecar verification table.
    """
    index = {str(name): position for position, name in enumerate(node_ids)}
    matrix = np.zeros((len(index), len(index)), dtype="<f8")
    for edge in edges:
        source = index.get(str(edge.get("source")))
        target = index.get(str(edge.get("target")))
        if source is None or target is None:
            continue
        matrix[source, target] = float(edge.get("weight", 0.0))
    return matrix


def _pathway_statistics(diagram: PathwayDiagram | dict[str, Any]) -> dict[str, Any]:
    """The 04D ``metadata`` block (the eight pre-declared ``pathway_*`` keys)."""
    payload = diagram if isinstance(diagram, dict) else diagram.to_dict()
    n_nodes = payload.get("n_nodes")
    n_edges = payload.get("n_edges")
    return {
        "n_pathway_nodes": int(n_nodes) if n_nodes is not None else None,
        "n_pathway_edges": int(n_edges) if n_edges is not None else None,
        "pathway_abs_max": payload.get("abs_max"),
        "pathway_positive_edges": payload.get("n_positive"),
        "pathway_negative_edges": payload.get("n_negative"),
        "pathway_intra_edges": payload.get("n_intra"),
        "pathway_cross_edges": payload.get("n_cross"),
        "pathway_weight_concentration": payload.get("weight_concentration"),
    }


def pathway_matrix_path(source_artifact: str | Path | None) -> Path | None:
    """The sibling Phase 02 artifact for *source_artifact* (``z_matrix[.variant].json``)."""
    if source_artifact is None:
        return None
    source = Path(source_artifact)
    if source.stem == "<in-memory>":
        return None
    return source.parent / f"{MATRIX_INPUT_PREFIX}{input_variant(source)}.json"


def load_pathway_matrix(source_artifact: str | Path | None) -> tuple[np.ndarray, list[str]]:
    """Load the Phase 02 effective matrix for pathway enrichment.

    Raises :class:`MotifValidationError` when the sibling artifact is missing or cannot be
    loaded -- ``--pathway-source matrix``/``both`` require it.
    """
    candidate = pathway_matrix_path(source_artifact)
    if candidate is None or not candidate.is_file():
        raise MotifValidationError(
            f"[{CODE_MATRIX_MISSING}] --pathway-source matrix/both needs the Phase 02 "
            f"artifact {candidate}; rebuild Phase 02 or use --pathway-source mode"
        )
    from src.matrices.build_square_matrix import load_z_matrix

    try:
        z_matrix = load_z_matrix(candidate)
    except Exception as exc:  # malformed/stale Phase 02 artifact
        raise MotifValidationError(
            f"[{CODE_MATRIX_MISSING}] could not load {candidate}: {exc}"
        ) from exc
    return np.asarray(z_matrix.matrix, dtype=np.float64), list(z_matrix.neuron_order)


# ---------------------------------------------------------------------------
# Statistics and the top-level builder
# ---------------------------------------------------------------------------
def motif_statistics(
    participation: Any,
    membership: Any,
    motifs: Sequence[Motif],
    config: MotifConfig,
    *,
    neuron_ids: Sequence[str],
    regions: Sequence[str],
    n_anatomy_matches: int = 0,
    n_anatomy_missing: int = 0,
    n_anatomy_conflicts: int = 0,
    now: Any = None,
) -> dict[str, Any]:
    """Build the 04A ``metadata`` block (the later sub-phase blocks stay ``null``)."""
    matrix = np.asarray(participation, dtype=np.float64)
    mask = np.asarray(membership, dtype=bool)
    n_neurons, n_modes = matrix.shape
    counts = [motif.n_members for motif in motifs]
    total = int(sum(counts))
    recurrence = mask.sum(axis=1) if mask.size else np.zeros(n_neurons, dtype=np.int64)
    order = sorted(range(n_neurons), key=lambda index: (-int(recurrence[index]), index))
    top_recurrent = [
        {"neuron_id": str(neuron_ids[index]), "recurrence": int(recurrence[index])}
        for index in order[:5]
        if int(recurrence[index]) > 0
    ]
    region_set = sorted({str(region) for region in regions})
    return {
        "n_neurons": int(n_neurons),
        "n_modes": int(n_modes),
        "n_motifs": int(len(motifs)),
        "n_members_total": total,
        "n_members_min": int(min(counts)) if counts else 0,
        "n_members_median": float(np.median(np.asarray(counts, dtype=np.float64))) if counts else 0.0,
        "n_members_max": int(max(counts)) if counts else 0,
        "n_members_mean": float(np.mean(np.asarray(counts, dtype=np.float64))) if counts else 0.0,
        "n_unique_members": int(np.count_nonzero(recurrence >= 1)),
        "n_neurons_in_no_motif": int(np.count_nonzero(recurrence == 0)),
        "membership_density": float(total / (n_neurons * n_modes)) if n_neurons * n_modes else 0.0,
        "n_isolated_neurons": int(np.count_nonzero(matrix.max(axis=1) <= 0.0)) if matrix.size else 0,
        "recurrence_max": int(recurrence.max()) if recurrence.size else 0,
        "n_recurrence_ge_1": int(np.count_nonzero(recurrence >= 1)),
        "n_recurrence_ge_2": int(np.count_nonzero(recurrence >= 2)),
        "n_recurrence_ge_5": int(np.count_nonzero(recurrence >= 5)),
        "top_recurrent": top_recurrent,
        "n_regions": int(len(region_set)),
        "regions": region_set,
        "n_anatomy_matches": int(n_anatomy_matches),
        "n_anatomy_missing": int(n_anatomy_missing),
        "n_anatomy_conflicts": int(n_anatomy_conflicts),
        "participation": config.participation,
        "threshold_method": config.threshold_method,
        "k_resolved": int(len(motifs)),
        "stage": str(config.stage),
        "config_hash": config.config_hash(),
        "numpy_version": np.__version__,
        "generator": GENERATOR,
        "created_utc": utc_timestamp(now),
        # populated by 04B
        "n_links": None,
        "link_classes": None,
        "n_families": None,
        "n_multi_mode_families": None,
        # populated by 04C
        "n_groups": None,
        "group_sizes": None,
        "n_background_groups": None,
        "n_singleton_groups": None,
        "largest_group": None,
        "smallest_group": None,
        "mean_group_size": None,
        "n_polarity_split_groups": None,
        # populated by 04D
        "n_pathway_nodes": None,
        "n_pathway_edges": None,
        "pathway_abs_max": None,
        "pathway_positive_edges": None,
        "pathway_negative_edges": None,
        "pathway_intra_edges": None,
        "pathway_cross_edges": None,
        "pathway_weight_concentration": None,
    }


def _resolve_regions(
    neuron_order: Sequence[str],
    anatomy: dict[str, dict[str, Any]] | None,
    report: ValidationReport,
) -> tuple[list[str], dict[str, int]]:
    """Return per-neuron region labels plus the anatomy match/missing/conflict counts."""
    derived = [derive_region(name) for name in neuron_order]
    if not anatomy:
        return derived, {"matches": 0, "missing": 0, "conflicts": 0}
    regions: list[str] = []
    matches = missing = conflicts = 0
    for name, fallback in zip(neuron_order, derived):
        entry = anatomy.get(name)
        if entry is None:
            missing += 1
            regions.append(fallback)
            continue
        region = entry.get("region")
        if region in (None, ""):
            regions.append(fallback)
            continue
        region = str(region)
        regions.append(region)
        if region == fallback:
            matches += 1
        else:
            conflicts += 1
    if missing:
        LOGGER.info(
            "[%s] %d neuron(s) are absent from the anatomy file; the derived region is used",
            CODE_ANATOMY_MISSING,
            missing,
        )
    if conflicts:
        report.warning(
            CODE_ANATOMY_MISMATCH,
            f"{conflicts} neuron(s) have an anatomy region that differs from the derived one",
        )
    return regions, {"matches": matches, "missing": missing, "conflicts": conflicts}


def _validate_config(config: MotifConfig, report: ValidationReport) -> None:
    """Validate the CLI-controlled Phase 04 parameters (04A fields plus the frozen enums)."""
    if config.participation not in PARTICIPATIONS:
        report.error(CODE_CONFIG, f"participation {config.participation!r} is not one of {PARTICIPATIONS}")
    if config.threshold_method not in THRESHOLD_METHODS:
        report.error(
            CODE_CONFIG,
            f"threshold_method {config.threshold_method!r} is not one of {THRESHOLD_METHODS}",
        )
    if not math.isfinite(config.relative_threshold) or not 0.0 <= config.relative_threshold <= 1.0:
        report.error(
            CODE_CONFIG, f"relative_threshold must be in [0, 1], got {config.relative_threshold}"
        )
    if not math.isfinite(config.absolute_threshold) or config.absolute_threshold < 0.0:
        report.error(
            CODE_CONFIG, f"absolute_threshold must be finite and >= 0, got {config.absolute_threshold}"
        )
    if not math.isfinite(config.quantile) or not 0.0 <= config.quantile <= 1.0:
        report.error(CODE_CONFIG, f"quantile must be in [0, 1], got {config.quantile}")
    if not math.isfinite(config.participation_threshold) or config.participation_threshold < 0.0:
        report.error(
            CODE_CONFIG,
            f"participation_threshold must be finite and >= 0, got {config.participation_threshold}",
        )
    if int(config.min_members) < 0:
        report.error(CODE_CONFIG, f"min_members must be >= 0, got {config.min_members}")
    if int(config.max_members) < 0:
        report.error(CODE_CONFIG, f"max_members must be >= 0 (0 = uncapped), got {config.max_members}")
    if int(config.max_members) and int(config.min_members) > int(config.max_members):
        report.error(
            CODE_CONFIG,
            f"min_members ({config.min_members}) exceeds max_members ({config.max_members})",
        )
    # 04B -- cross-mode tracking thresholds.
    if not math.isfinite(config.family_jaccard) or not 0.0 <= config.family_jaccard <= 1.0:
        report.error(
            CODE_CONFIG, f"family_jaccard must be in [0, 1], got {config.family_jaccard}"
        )
    if not math.isfinite(config.family_polarity) or not 0.0 <= config.family_polarity <= 1.0:
        report.error(
            CODE_CONFIG, f"family_polarity must be in [0, 1], got {config.family_polarity}"
        )
    if not math.isfinite(config.link_min_jaccard) or not 0.0 <= config.link_min_jaccard <= 1.0:
        report.error(
            CODE_CONFIG, f"link_min_jaccard must be in [0, 1], got {config.link_min_jaccard}"
        )
    if float(config.link_min_jaccard) > float(config.family_jaccard):
        report.error(
            CODE_CONFIG,
            f"link_min_jaccard ({config.link_min_jaccard}) must not exceed "
            f"family_jaccard ({config.family_jaccard})",
        )
    if config.grouping not in GROUPINGS:
        report.error(CODE_CONFIG, f"grouping {config.grouping!r} is not one of {GROUPINGS}")
    if config.linkage not in LINKAGES:
        report.error(CODE_CONFIG, f"linkage {config.linkage!r} is not one of {LINKAGES}")
    if config.affinity not in AFFINITIES:
        report.error(CODE_CONFIG, f"affinity {config.affinity!r} is not one of {AFFINITIES}")
    if int(config.polarity_min_members) < 1:
        report.error(CODE_CONFIG, f"polarity_min_members must be >= 1, got {config.polarity_min_members}")
    if int(config.max_group_size) < 2:
        report.error(CODE_CONFIG, f"max_group_size must be >= 2, got {config.max_group_size}")
    if not math.isfinite(float(config.merge_threshold)) or not 0.0 <= float(config.merge_threshold) <= 1.0:
        report.error(CODE_CONFIG, f"merge_threshold must be in [0, 1], got {config.merge_threshold}")
    if str(config.n_groups) != "auto":
        try:
            if int(config.n_groups) < 2:
                raise ValueError
        except (TypeError, ValueError):
            report.error(CODE_CONFIG, f"n_groups must be 'auto' or an integer >= 2, got {config.n_groups!r}")
    if config.linkage == "ward" and config.affinity != "euclidean":
        report.error(CODE_CONFIG, "linkage='ward' requires affinity='euclidean'")
    if config.pathway_source not in PATHWAY_SOURCES:
        report.error(CODE_CONFIG, f"pathway_source {config.pathway_source!r} is not one of {PATHWAY_SOURCES}")
    if config.pathway_weight not in PATHWAY_WEIGHTS:
        report.error(CODE_CONFIG, f"pathway_weight {config.pathway_weight!r} is not one of {PATHWAY_WEIGHTS}")
    if config.diagram_layout not in DIAGRAM_LAYOUTS:
        report.error(CODE_CONFIG, f"diagram_layout {config.diagram_layout!r} is not one of {DIAGRAM_LAYOUTS}")
    # 04D -- pathway graph ranges (the enums above are already validated).
    if not math.isfinite(float(config.pathway_edge_threshold)) or not 0.0 <= float(config.pathway_edge_threshold) <= 1.0:
        report.error(
            CODE_CONFIG,
            f"pathway_edge_threshold must be in [0, 1] (a fraction of abs_max), "
            f"got {config.pathway_edge_threshold}",
        )
    if int(config.pathway_top_edges) < 1:
        report.error(CODE_CONFIG, f"pathway_top_edges must be >= 1, got {config.pathway_top_edges}")
    if int(config.max_diagram_nodes) < 1:
        report.error(CODE_CONFIG, f"max_diagram_nodes must be >= 1, got {config.max_diagram_nodes}")


def _build_arrays(
    participation: np.ndarray,
    membership: np.ndarray,
    spectrum: Any,
    motifs: Sequence[Motif],
    thresholds: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return the ordered ``motifs.npz`` array bundle (fixed dtypes)."""
    order_length = max((len(name) for name in spectrum.neuron_order), default=1)
    values = np.asarray(spectrum.spectrum, dtype="<f8")
    return {
        "stage": np.asarray([STAGE], dtype="<U8"),
        "participation": np.asarray(participation, dtype="<f8"),
        "membership": np.asarray(membership, dtype="<i8"),
        "loadings_left": np.asarray(spectrum.loadings_left, dtype="<f8"),
        "loadings_right": np.asarray(spectrum.loadings_right, dtype="<f8"),
        "values": values,
        "retained_values": np.asarray([motif.value for motif in motifs], dtype="<f8"),
        "abs_retained_values": np.asarray([motif.abs_value for motif in motifs], dtype="<f8"),
        "explained_variance_ratio": np.asarray(spectrum.explained_variance_ratio, dtype="<f8"),
        "thresholds": np.asarray(thresholds, dtype="<f8"),
        "motif_strength_l1": np.asarray([motif.strength_l1 for motif in motifs], dtype="<f8"),
        "motif_strength_energy": np.asarray([motif.strength_energy for motif in motifs], dtype="<f8"),
        "mode_outer_products": _mode_outer_product_stack(
            spectrum.loadings_left,
            spectrum.loadings_right,
            np.asarray([motif.value for motif in motifs], dtype=np.float64),
        ),
        "family_of_mode": np.arange(len(motifs), dtype="<i8"),
        "neuron_order": np.asarray(list(spectrum.neuron_order), dtype=f"<U{order_length}"),
    }


def build_motif_analysis(
    spectrum: Any,
    *,
    source_artifact: str | Path | None = None,
    config: MotifConfig | None = None,
    anatomy: dict[str, dict[str, Any]] | None = None,
    node_metadata: dict[str, dict[str, Any]] | None = None,
    now: Any = None,
) -> tuple[MotifAnalysis, ValidationReport]:
    """Build the Phase 04A analysis from a Phase 03 spectrum.

    Returns the analysis plus the accumulated :class:`ValidationReport`; the caller
    decides whether the errors are fatal (the CLI fails on any error, and ``--strict``
    escalates warnings).  The Phase 03 object is never mutated.
    """
    report = ValidationReport()
    config = config or MotifConfig()
    _validate_config(config, report)

    problems = validate_spectral_payload_schema(spectrum.to_dict())
    for problem in problems:
        report.error(CODE_INPUT_SCHEMA, problem)

    neuron_order = list(spectrum.neuron_order)
    if not neuron_order:
        report.error(CODE_NEURON_ORDER, "the spectrum has no neurons")
    elif len(set(neuron_order)) != len(neuron_order):
        report.error(CODE_NEURON_ORDER, "neuron_order contains duplicate ids")

    left = np.asarray(spectrum.loadings_left, dtype=np.float64)
    right = np.asarray(spectrum.loadings_right, dtype=np.float64)
    n_neurons = len(neuron_order)
    n_modes = int(spectrum.n_modes)
    if left.shape != (n_neurons, n_modes) or right.shape != (n_neurons, n_modes):
        report.error(
            CODE_NEURON_ORDER,
            f"loading shapes {tuple(left.shape)} / {tuple(right.shape)} do not match "
            f"({n_neurons}, {n_modes})",
        )

    participation = compute_participation(left, right, config.participation)
    if not bool(np.all(np.isfinite(participation))):
        report.error(CODE_NONFINITE, "the participation matrix contains non-finite entries")
    if bool(np.any(participation < 0.0)):
        report.error(CODE_NONFINITE, "the participation matrix contains negative entries")

    thresholds = motif_thresholds(
        participation,
        config.threshold_method,
        relative_threshold=config.relative_threshold,
        absolute_threshold=config.absolute_threshold,
        quantile=config.quantile,
        participation_threshold=config.participation_threshold,
    )
    membership = motif_membership(
        participation,
        thresholds,
        min_members=config.min_members,
        max_members=config.max_members,
    )

    regions, anatomy_counts = _resolve_regions(neuron_order, anatomy, report)
    values = np.asarray(spectrum.spectrum, dtype=np.float64)
    abs_values = np.asarray(spectrum.abs_spectrum, dtype=np.float64)
    evr = np.asarray(spectrum.explained_variance_ratio, dtype=np.float64)
    cumulative = np.asarray(spectrum.cumulative_variance_ratio, dtype=np.float64)
    if values.size < n_modes or evr.size < n_modes:
        report.error(CODE_INPUT_SCHEMA, "the spectrum is shorter than its retained modes")

    motifs = extract_motifs(
        participation,
        left,
        right,
        thresholds=thresholds,
        threshold_method=config.threshold_method,
        membership=membership,
        neuron_ids=neuron_order,
        regions=regions,
        values=values,
        abs_values=abs_values,
        explained_variance_ratio=evr,
        cumulative_variance_ratio=cumulative,
    )
    links = motif_links(motifs, config)
    if not links:
        LOGGER.info(
            "[%s] no pair of modes shares a member; every mode is its own family",
            CODE_NO_LINKS,
        )
    merged_families = merge_families(motifs, links)
    families = [family.to_dict() for family in merged_families]
    resolved_config = replace(
        config, k_resolved=len(motifs), n_motifs=len(motifs), stage=STAGE
    )
    metadata = motif_statistics(
        participation,
        membership,
        motifs,
        resolved_config,
        neuron_ids=neuron_order,
        regions=regions,
        n_anatomy_matches=anatomy_counts["matches"],
        n_anatomy_missing=anatomy_counts["missing"],
        n_anatomy_conflicts=anatomy_counts["conflicts"],
        now=now,
    )
    metadata.update(_tracking_statistics(links, merged_families))

    if n_neurons < 2:
        report.warning(CODE_SMALL_GRAPH, f"only {n_neurons} neuron(s): density is degenerate")
    if participation.size == 0 or float(participation.max()) == 0.0:
        report.warning(CODE_TRIVIAL_MOTIFS, "every participation value is zero: no motif exists")
    empty = [motif.label for motif in motifs if motif.n_members == 0]
    if empty:
        report.warning(CODE_EMPTY_MOTIF, f"{len(empty)} motif(s) have no member: {empty[:5]}")
    for mode_index in range(n_modes):
        if participation.size == 0 or float(participation[:, mode_index].max()) <= 0.0:
            LOGGER.info("[%s] mode %d has an all-zero participation column", CODE_DEGENERATE, mode_index + 1)

    source_path = Path(source_artifact) if source_artifact is not None else spectrum.loaded_from
    digest = ""
    if source_path is not None and Path(source_path).is_file():
        digest = sha256_file(source_path)
    analysis = MotifAnalysis(
        source_artifact=Path(source_path) if source_path is not None else Path("<in-memory>"),
        source_artifact_sha256=digest,
        source_file=spectrum.source_file,
        source_file_sha256=spectrum.source_file_sha256,
        parsed_created_utc=spectrum.parsed_created_utc,
        z_matrix_config_hash=spectrum.z_matrix_config_hash,
        z_matrix_config=dict(spectrum.z_matrix_config),
        spectral_config_hash=spectrum.config.config_hash(),
        spectral_config=spectrum.config.to_dict(),
        config=resolved_config,
        neuron_order=neuron_order,
        motifs=motifs,
        links=[link.to_dict() for link in links],
        families=families,
        groups=[],
        pathway=dict(NEUTRAL_PATHWAY),
        metadata=metadata,
        diagnostics=dict(metadata),
        arrays={},
    )
    analysis.arrays = _build_arrays(participation, membership, spectrum, motifs, thresholds)
    analysis.arrays["family_of_mode"] = _family_of_mode(merged_families, len(motifs))
    link_jaccard, link_polarity = _link_matrices(links, len(motifs))
    analysis.arrays["link_jaccard"] = link_jaccard
    analysis.arrays["link_polarity"] = link_polarity
    pathway_diagnostics: dict[str, Any] = {}
    if config.grouping != "none":
        groups, group_labels, group_centroids, n_polarity_split = build_groups(
            motifs,
            resolved_config,
            participation=participation,
            loadings_left=left,
            loadings_right=right,
            neuron_ids=neuron_order,
            regions=regions,
            node_metadata=node_metadata,
        )
        analysis.groups = [group.to_dict() for group in groups]
        analysis.metadata.update(_group_statistics(groups, n_polarity_split))
        analysis.arrays["group_labels"] = np.asarray(group_labels, dtype="<i8")
        analysis.arrays["group_centroids"] = np.asarray(group_centroids, dtype="<f8")
        analysis.arrays["group_sizes"] = np.asarray([group.size for group in groups], dtype="<i8")
        if analysis.groups and analysis.groups[0]["is_background"]:
            LOGGER.info("[%s] background group %s has %d member(s)", CODE_BACKGROUND, analysis.groups[0]["group_id"], analysis.groups[0]["size"])
        singleton_count = int(analysis.metadata["n_singleton_groups"])
        if singleton_count:
            LOGGER.info("[%s] %d non-background singleton group(s)", CODE_SINGLETON, singleton_count)
        pathway_matrix = None
        if str(config.pathway_source) in ("matrix", "both"):
            matrix_candidate = pathway_matrix_path(source_path)
            try:
                matrix_values, matrix_order = load_pathway_matrix(source_path)
            except MotifValidationError as exc:
                report.error(CODE_MATRIX_MISSING, str(exc))
            else:
                if list(matrix_order) != list(neuron_order):
                    report.error(
                        CODE_MATRIX_MISMATCH,
                        f"the Phase 02 matrix {matrix_candidate} covers a different neuron "
                        f"order than the spectrum",
                    )
                else:
                    pathway_matrix = matrix_values
        pathway, pathway_diagnostics = build_pathway(
            groups,
            participation=participation,
            loadings_left=left,
            loadings_right=right,
            values=values[: len(motifs)],
            neuron_ids=neuron_order,
            config=resolved_config,
            matrix=pathway_matrix,
        )
        analysis.pathway = pathway.to_dict()
        analysis.metadata.update(_pathway_statistics(pathway))
        node_ids = [node["node_id"] for node in analysis.pathway["nodes"]]
        adjacency = _pathway_adjacency(analysis.pathway["edges"], node_ids)
        analysis.arrays["pathway_adjacency"] = adjacency
        analysis.arrays["pathway_adjacency_abs"] = np.abs(adjacency)
    else:
        LOGGER.info(
            "[%s] grouping is disabled: the pathway layer is skipped", CODE_PATHWAY_SKIPPED
        )
    analysis.arrays["source_json_sha256"] = np.asarray([""], dtype="<U64")
    analysis.arrays["numpy_version"] = np.asarray([np.__version__], dtype="<U64")
    analysis.diagnostics = {
        **metadata,
        "node_metadata": dict(node_metadata or {}),
        "neuron_regions": list(regions),
        "recurrence_table": recurrence_table(motifs, merged_families),
        "pathway": pathway_diagnostics,
    }

    for problem in validate_motif_payload_schema(analysis.to_dict()):
        report.error(CODE_PAYLOAD_SCHEMA, problem)
    return analysis, report


# ---------------------------------------------------------------------------
# Payload validation (the frozen Phase 04 schema)
# ---------------------------------------------------------------------------
def validate_motif_payload_schema(payload: dict[str, Any]) -> list[str]:
    """Hand-rolled schema check for the Phase 04 JSON artifact.

    Validates **every** frozen key set -- including the link/family/group/pathway
    entry shapes owned by 04B-04D -- so later sub-phases never touch this function.
    Returns a list of human-readable problems (empty list == conforming).
    """
    problems: list[str] = []

    def _require_keys(actual: Iterable[str], expected: frozenset[str], where: str) -> None:
        missing = sorted(expected - set(actual))
        extra = sorted(set(actual) - expected)
        if missing:
            problems.append(f"{where}: missing key(s) {missing}")
        if extra:
            problems.append(f"{where}: unexpected key(s) {extra}")

    if not isinstance(payload, dict):
        return ["payload is not an object"]

    _require_keys(payload.keys(), TOP_LEVEL_KEYS, "top level")

    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        problems.append("provenance: not an object")
    else:
        _require_keys(provenance.keys(), PROVENANCE_KEYS, "provenance")
        if provenance.get("phase") != PHASE:
            problems.append(f"provenance.phase: expected {PHASE!r}")
        if provenance.get("stage") not in STAGES:
            problems.append(f"provenance.stage: {provenance.get('stage')!r} is not one of {STAGES}")

    config = payload.get("config")
    if not isinstance(config, dict):
        problems.append("config: not an object")
    else:
        _require_keys(config.keys(), CONFIG_KEYS, "config")
        if not isinstance(config.get("config_hash"), str) or len(str(config["config_hash"])) != 8:
            problems.append("config.config_hash: expected 8 hex characters")
        if config.get("participation") not in PARTICIPATIONS:
            problems.append(f"config.participation: {config.get('participation')!r} is not one of {PARTICIPATIONS}")
        if config.get("threshold_method") not in THRESHOLD_METHODS:
            problems.append(
                f"config.threshold_method: {config.get('threshold_method')!r} is not one of {THRESHOLD_METHODS}"
            )

    neuron_order = payload.get("neuron_order")
    if not isinstance(neuron_order, list) or not all(isinstance(name, str) for name in neuron_order):
        problems.append(
            "neuron_order: expected a list of strings (a dict would lose the index order)"
        )
        neuron_order = []
    elif len(set(neuron_order)) != len(neuron_order):
        problems.append("neuron_order: contains duplicate ids")
    n_neurons = len(neuron_order)

    motifs = payload.get("motifs")
    if not isinstance(motifs, list) or not motifs:
        problems.append("motifs: expected a non-empty list")
    else:
        for position, entry in enumerate(motifs):
            if not isinstance(entry, dict):
                problems.append(f"motifs[{position}]: not an object")
                continue
            _require_keys(entry.keys(), MOTIF_KEYS, f"motifs[{position}]")
            members = entry.get("members")
            if not isinstance(members, list):
                problems.append(f"motifs[{position}].members: expected a list")
                continue
            if entry.get("n_members") != len(members):
                problems.append(f"motifs[{position}].n_members: does not match len(members)")
            for slot, member in enumerate(members):
                if not isinstance(member, dict):
                    problems.append(f"motifs[{position}].members[{slot}]: not an object")
                    continue
                _require_keys(member.keys(), MEMBER_KEYS, f"motifs[{position}].members[{slot}]")
                if member.get("polarity") not in (-1, 1):
                    problems.append(f"motifs[{position}].members[{slot}].polarity: expected -1 or 1")
                if member.get("dominant_axis") not in ("left", "right"):
                    problems.append(
                        f"motifs[{position}].members[{slot}].dominant_axis: expected 'left' or 'right'"
                    )
            for side in ("sender_members", "receiver_members"):
                if not isinstance(entry.get(side), list):
                    problems.append(f"motifs[{position}].{side}: expected a list")

    links = payload.get("links")
    if not isinstance(links, list):
        problems.append("links: expected a list (neutral value is [])")
    else:
        for position, entry in enumerate(links):
            if not isinstance(entry, dict):
                problems.append(f"links[{position}]: not an object")
                continue
            _require_keys(entry.keys(), LINK_KEYS, f"links[{position}]")

    families = payload.get("families")
    if not isinstance(families, list):
        problems.append("families: expected a list")
    else:
        for position, entry in enumerate(families):
            if not isinstance(entry, dict):
                problems.append(f"families[{position}]: not an object")
                continue
            _require_keys(entry.keys(), FAMILY_KEYS, f"families[{position}]")
            modes = entry.get("modes")
            if not isinstance(modes, list) or not all(isinstance(mode, int) for mode in modes):
                problems.append(f"families[{position}].modes: expected a list of mode integers")
            for slot, occurrence in enumerate(entry.get("occurrences") or []):
                if not isinstance(occurrence, dict) or "neuron_id" not in occurrence:
                    problems.append(
                        f"families[{position}].occurrences[{slot}]: expected a record with neuron_id"
                    )
                    continue
                for item in occurrence.get("occurrences") or []:
                    if not isinstance(item, dict):
                        problems.append(f"families[{position}].occurrences[{slot}]: entry not an object")
                        continue
                    _require_keys(
                        item.keys(),
                        OCCURRENCE_KEYS,
                        f"families[{position}].occurrences[{slot}].occurrences",
                    )

    groups = payload.get("groups")
    if not isinstance(groups, list):
        problems.append("groups: expected a list (neutral value is [])")
    else:
        for position, entry in enumerate(groups):
            if not isinstance(entry, dict):
                problems.append(f"groups[{position}]: not an object")
                continue
            _require_keys(entry.keys(), GROUP_KEYS, f"groups[{position}]")

    pathway = payload.get("pathway")
    if not isinstance(pathway, dict):
        problems.append("pathway: not an object")
    else:
        _require_keys(pathway.keys(), PATHWAY_KEYS, "pathway")
        if not isinstance(pathway.get("nodes"), list) or not isinstance(pathway.get("edges"), list):
            problems.append("pathway.nodes/edges: expected lists")
        else:
            for position, node in enumerate(pathway["nodes"]):
                if not isinstance(node, dict):
                    problems.append(f"pathway.nodes[{position}]: not an object")
                    continue
                _require_keys(node.keys(), NODE_KEYS, f"pathway.nodes[{position}]")
                for slot, top in enumerate(node.get("top_modes") or []):
                    if not isinstance(top, dict):
                        problems.append(f"pathway.nodes[{position}].top_modes[{slot}]: not an object")
                        continue
                    _require_keys(top.keys(), TOP_MODE_KEYS, f"pathway.nodes[{position}].top_modes[{slot}]")
            for position, edge in enumerate(pathway["edges"]):
                if not isinstance(edge, dict):
                    problems.append(f"pathway.edges[{position}]: not an object")
                    continue
                _require_keys(edge.keys(), EDGE_KEYS, f"pathway.edges[{position}]")
                for slot, top in enumerate(edge.get("top_modes") or []):
                    if not isinstance(top, dict):
                        problems.append(f"pathway.edges[{position}].top_modes[{slot}]: not an object")
                        continue
                    _require_keys(top.keys(), TOP_MODE_KEYS, f"pathway.edges[{position}].top_modes[{slot}]")

    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        problems.append("metadata: not an object")
    else:
        _require_keys(metadata.keys(), METADATA_KEYS, "metadata")
        if not isinstance(metadata.get("n_neurons"), int):
            problems.append("metadata.n_neurons: expected an integer")
        if metadata.get("n_neurons") != n_neurons:
            problems.append("metadata.n_neurons: does not match len(neuron_order)")

    return problems


# ---------------------------------------------------------------------------
# Artifact paths
# ---------------------------------------------------------------------------
def input_variant(source_artifact: str | Path | None) -> str:
    """Variant suffix of a Phase 03 artifact (``eigen.f8652585`` -> ``.f8652585``).

    04A inherits it so that a motif run over a variant spectrum can never overwrite
    the canonical ``motifs.json``.
    """
    if source_artifact is None:
        return ""
    stem = Path(source_artifact).stem
    if stem == INPUT_VARIANT_PREFIX:
        return ""
    if stem.startswith(f"{INPUT_VARIANT_PREFIX}."):
        return stem[len(INPUT_VARIANT_PREFIX) :]
    return ""


def variant_stem(
    config: MotifConfig,
    *,
    source_artifact: str | Path | None = None,
    force_config_hash: bool = False,
) -> str:
    """Return ``motifs[<input variant>][.<config_hash8>]`` for *config*."""
    stem = CANONICAL_STEM + input_variant(source_artifact)
    if force_config_hash or not config.is_default():
        stem += f".{config.config_hash()}"
    return stem


def artifact_paths(
    analysis: MotifAnalysis,
    outdir: str | Path,
    *,
    force_config_hash: bool = False,
) -> dict[str, Path]:
    """Return the four Phase 04 artifact paths for *analysis*.

    ``{"json", "npz", "data", "graphml"}`` under ``<outdir>/<gv-stem>/`` -- the same
    directory Phase 01-03 write to.
    """
    stem = variant_stem(
        analysis.config,
        source_artifact=analysis.source_artifact,
        force_config_hash=force_config_hash,
    )
    directory = Path(outdir) / Path(analysis.source_file).stem
    return {
        "json": directory / f"{stem}.json",
        "npz": directory / f"{stem}.npz",
        "data": directory / f"{stem}.data.json",
        "graphml": directory / f"{stem}.pathway.graphml",
    }


def sidecar_path(json_path: str | Path) -> Path:
    """``motifs[.variant].json`` -> ``motifs[.variant].npz``."""
    return Path(json_path).with_suffix(".npz")


# ---------------------------------------------------------------------------
# Sidecar (derived npz array cache)
# ---------------------------------------------------------------------------
def build_sidecar_arrays(
    analysis: MotifAnalysis, *, source_json_sha256: str
) -> dict[str, np.ndarray]:
    """Return the ordered array bundle stored in the ``.npz`` cache.

    Ordering and dtypes are fixed (``<f8`` / ``<i8`` / ``<U*``) so the archive is
    byte-reproducible: numpy writes every zip entry with a constant ``date_time``
    and the ``.npy`` header carries no timestamp.
    """
    if not analysis.arrays:
        raise MotifValidationError(
            "the analysis carries no arrays (it was loaded from JSON); rebuild it from "
            "the spectrum before writing a sidecar"
        )
    arrays = {name: value for name, value in analysis.arrays.items() if name != "source_json_sha256"}
    arrays["numpy_version"] = np.asarray([np.__version__], dtype="<U64")
    arrays["source_json_sha256"] = np.asarray([source_json_sha256], dtype="<U64")
    return arrays


def write_npz_atomic(path: str | Path, arrays: dict[str, np.ndarray]) -> Path:
    """Write *arrays* to *path* (``.npz``) atomically and byte-reproducibly.

    The archive is built in memory first, then written to a temporary file in the
    destination directory, flushed and ``fsync``-ed before ``os.replace`` -- so a
    reader sees either the previous cache or the complete new one.
    """
    destination = Path(path)
    ensure_dir(destination.parent)

    buffer = io.BytesIO()
    np.savez(buffer, **arrays)  # deterministic: numpy pins the zip date_time
    payload = buffer.getvalue()

    handle = tempfile.NamedTemporaryFile(
        mode="wb",
        dir=str(destination.parent),
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def _write_text_atomic(path: str | Path, text: str) -> Path:
    """Write *text* to *path* atomically (temp file -> ``fsync`` -> ``os.replace``).

    ``src/utils/io.py`` deliberately exposes only JSON writers and the GraphML artifact is
    not JSON, so the Phase 04 module owns this small raw-text variant.
    """
    destination = Path(path)
    ensure_dir(destination.parent)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
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


def graphml_node_metadata(
    node: dict[str, Any],
    group: dict[str, Any] | None = None,
    centroid: Any = None,
) -> dict[str, Any]:
    """Return the yEd-facing GraphML attribute record for one pathway node.

    Unavailable optional attributes (``dominant_mode`` for the background group,
    ``coherence`` for singletons, ``centroid`` without in-memory arrays) are **omitted**:
    NetworkX refuses ``None`` GraphML values.
    """
    record = dict(group or {})
    metadata: dict[str, Any] = {
        "group_id": str(node.get("group_id")),
        "label": str(node.get("label") or node.get("group_id")),
        "size": int(node.get("size", 0)),
        "region_composition": json.dumps(node.get("region_composition") or {}, sort_keys=True),
        "is_background": bool(record.get("is_background", False)),
        "is_singleton": bool(record.get("is_singleton", False)),
    }
    dominant_mode = node.get("dominant_mode")
    if dominant_mode is not None:
        metadata["dominant_mode"] = int(dominant_mode)
    if centroid is not None:
        metadata["centroid"] = json.dumps([float(value) for value in np.asarray(centroid).ravel()])
    coherence = record.get("coherence")
    if coherence is not None:
        metadata["coherence"] = float(coherence)
    return metadata


def graphml_edge_metadata(
    edge: dict[str, Any], *, z_contribution: float | None = None
) -> dict[str, Any]:
    """Return the yEd-facing GraphML attribute record for one pathway edge."""
    metadata: dict[str, Any] = {
        "source_group": str(edge.get("source")),
        "target_group": str(edge.get("target")),
        "weight": float(edge.get("weight", 0.0)),
        "abs_weight": float(edge.get("abs_weight", 0.0)),
        "polarity": int(edge.get("polarity", 0)),
        "contribution_by_mode": json.dumps(
            [dict(entry) for entry in edge.get("top_modes") or ()], sort_keys=True
        ),
        "threshold_flag": True,
        "topN_flag": True,
        "intra_flag": bool(edge.get("is_intra", False)),
    }
    if z_contribution is not None:
        metadata["z_contribution"] = float(z_contribution)
    return metadata


def write_graphml_pathway(
    path: str | Path,
    diagram: PathwayDiagram | dict[str, Any],
    *,
    groups: Sequence[dict[str, Any]] | None = None,
    config: MotifConfig | None = None,
    source: str | None = None,
    centroids: Any = None,
    z_by_edge: dict[tuple[str, str], float] | None = None,
) -> Path:
    """Write the NetworkX-backed GraphML pathway artifact and return its path.

    NetworkX is imported lazily (the Matplotlib precedent of Phases 02/03) and stays the
    mandated GraphML backend; the output is deterministic (fixed node/edge/attribute order,
    primitives or sorted-key JSON only) and is written atomically.
    """
    try:
        import networkx as nx
    except ImportError as exc:  # pragma: no cover - the environment pins networkx
        raise MotifValidationError(
            "networkx is required for the GraphML pathway artifact; conda install -c "
            "conda-forge networkx (or use --no-graphml)"
        ) from exc

    payload = diagram if isinstance(diagram, dict) else diagram.to_dict()
    nodes = [dict(node) for node in payload.get("nodes") or ()]
    edges = [dict(edge) for edge in payload.get("edges") or ()]
    groups_by_id = {str(group.get("group_id")): dict(group) for group in groups or ()}
    centroid_by_id: dict[str, Any] = {}
    if centroids is not None:
        column = np.asarray(centroids, dtype=np.float64)
        for position, node in enumerate(nodes):
            if position < column.shape[0]:
                centroid_by_id[str(node.get("group_id"))] = column[position]

    graph = nx.DiGraph()
    for node in nodes:
        node_id = str(node.get("node_id"))
        graph.add_node(
            node_id,
            **graphml_node_metadata(
                node, groups_by_id.get(str(node.get("group_id"))), centroid_by_id.get(node_id)
            ),
        )
    for edge in edges:
        metadata = graphml_edge_metadata(
            edge,
            z_contribution=(z_by_edge or {}).get(
                (str(edge.get("source")), str(edge.get("target")))
            ),
        )
        graph.add_edge(str(edge.get("source")), str(edge.get("target")), **metadata)
    edge_threshold = payload.get("edge_threshold")
    top_edges = payload.get("top_edges")
    graph.graph.update(
        {
            "n_nodes": int(len(nodes)),
            "n_edges": int(len(edges)),
            "weight_normalization": PATHWAY_WEIGHT_NORMALIZATION,
            "pathway_source": str(
                source if source is not None else payload.get("source") or DEFAULT_PATHWAY_SOURCE
            ),
            "pathway_weight_rule": str(
                config.pathway_weight
                if config is not None
                else payload.get("weight") or DEFAULT_PATHWAY_WEIGHT
            ),
            "threshold": float(
                config.pathway_edge_threshold
                if config is not None
                else (edge_threshold if edge_threshold is not None else DEFAULT_PATHWAY_EDGE_THRESHOLD)
            ),
            "topN": int(
                config.pathway_top_edges
                if config is not None
                else (top_edges if top_edges is not None else DEFAULT_PATHWAY_TOP_EDGES)
            ),
        }
    )

    for node_id, attributes in graph.nodes(data=True):
        unknown = sorted(set(attributes) - set(GRAPHML_NODE_KEYS))
        if unknown:
            raise MotifValidationError(f"GraphML node {node_id} carries unknown keys {unknown}")
    for source_id, target_id, attributes in graph.edges(data=True):
        unknown = sorted(set(attributes) - set(GRAPHML_EDGE_KEYS))
        if unknown:
            raise MotifValidationError(
                f"GraphML edge {source_id}->{target_id} carries unknown keys {unknown}"
            )
    unknown_graph = sorted(set(graph.graph) - set(GRAPHML_GRAPH_KEYS))
    if unknown_graph:
        raise MotifValidationError(
            f"the GraphML graph metadata carries unknown keys {unknown_graph}"
        )

    try:
        body = "\n".join(nx.generate_graphml(graph))
    except Exception as exc:
        raise MotifValidationError(f"could not serialise the GraphML pathway: {exc}") from exc
    return _write_text_atomic(path, f"{XML_DECLARATION}\n{body}\n")


def write_sidecar(analysis: MotifAnalysis, json_path: str | Path) -> Path:
    """Write the ``.npz`` cache for *analysis*, keyed by the JSON's SHA-256."""
    json_path = Path(json_path)
    digest = sha256_file(json_path)
    arrays = build_sidecar_arrays(analysis, source_json_sha256=digest)
    return write_npz_atomic(sidecar_path(json_path), arrays)


def load_sidecar_arrays(path: str | Path) -> dict[str, np.ndarray]:
    """Fast array access to a ``.npz`` cache (``allow_pickle=False``).

    Accepts either the ``.npz`` path or the ``.json`` path it belongs to.
    """
    candidate = Path(path)
    if candidate.suffix != ".npz":
        candidate = sidecar_path(candidate)
    if not candidate.is_file():
        raise FileNotFoundError(f"no npz sidecar at {candidate}")
    with np.load(candidate, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _expected_sidecar_arrays(analysis: MotifAnalysis) -> dict[str, np.ndarray]:
    """The arrays a correct cache must reproduce for *analysis* (JSON-derivable)."""
    n_modes = len(analysis.motifs)
    link_jaccard, link_polarity = _link_matrices(analysis.links, n_modes)
    return {
        "retained_values": np.asarray([motif.value for motif in analysis.motifs], dtype="<f8"),
        "abs_retained_values": np.asarray([motif.abs_value for motif in analysis.motifs], dtype="<f8"),
        "thresholds": np.asarray([motif.threshold for motif in analysis.motifs], dtype="<f8"),
        "motif_strength_l1": np.asarray([motif.strength_l1 for motif in analysis.motifs], dtype="<f8"),
        "motif_strength_energy": np.asarray([motif.strength_energy for motif in analysis.motifs], dtype="<f8"),
        "family_of_mode": _family_of_mode(analysis.families, n_modes),
        "link_jaccard": link_jaccard,
        "link_polarity": link_polarity,
    }
    if analysis.groups:
        labels = np.zeros(len(analysis.neuron_order), dtype="<i8")
        centroids = []
        sizes = []
        for index, group in enumerate(analysis.groups):
            members = [analysis.neuron_order.index(name) for name in group.get("members") or []]
            labels[members] = index
            sizes.append(len(members))
            if analysis.participation is not None:
                centroids.append(group_centroid(members, analysis.participation))
        expected.update({
            "group_labels": labels,
            "group_sizes": np.asarray(sizes, dtype="<i8"),
        })
        if centroids:
            expected["group_centroids"] = np.asarray(centroids, dtype="<f8")
        node_ids = [node.get("node_id") for node in analysis.pathway.get("nodes") or ()]
        if node_ids:
            adjacency = _pathway_adjacency(analysis.pathway.get("edges") or (), node_ids)
            expected["pathway_adjacency"] = adjacency
            expected["pathway_adjacency_abs"] = np.abs(adjacency)
    return expected


def _verify_sidecar(
    arrays: dict[str, np.ndarray],
    analysis: MotifAnalysis,
    json_path: Path,
) -> None:
    """Raise :class:`MotifValidationError` when the cache disagrees with the JSON."""
    expected_sha = sha256_file(json_path)
    recorded = arrays.get("source_json_sha256")
    if recorded is None or str(recorded[0]) != expected_sha:
        raise MotifValidationError(
            f"sidecar {sidecar_path(json_path).name} was built from a different JSON "
            f"(recorded {None if recorded is None else str(recorded[0])[:12]}, "
            f"actual {expected_sha[:12]})"
        )
    for name, expected in _expected_sidecar_arrays(analysis).items():
        actual = arrays.get(name)
        if actual is None or not np.array_equal(np.asarray(actual), expected):
            raise MotifValidationError(f"sidecar `{name}` differs from the JSON motifs")
    if "neuron_order" in arrays and list(arrays["neuron_order"]) != list(analysis.neuron_order):
        raise MotifValidationError("sidecar `neuron_order` differs from the JSON order")
    if "stage" in arrays and str(arrays["stage"][0]) != str(analysis.config.stage):
        raise MotifValidationError("sidecar `stage` differs from the JSON provenance stage")


def _analysis_from_payload(payload: dict[str, Any], *, loaded_from: Path | None) -> MotifAnalysis:
    """Rebuild a :class:`MotifAnalysis` from an artifact payload."""
    provenance = dict(payload.get("provenance") or {})
    config = MotifConfig.from_dict(dict(payload.get("config") or {}))
    neuron_order = [str(name) for name in payload.get("neuron_order") or []]
    motifs = [Motif.from_dict(entry) for entry in payload.get("motifs") or []]
    families = [dict(entry) for entry in payload.get("families") or []]
    return MotifAnalysis(
        source_artifact=Path(str(provenance.get("source_artifact") or "<in-memory>")),
        source_artifact_sha256=str(provenance.get("source_artifact_sha256") or ""),
        source_file=str(provenance.get("source_file") or ""),
        source_file_sha256=str(provenance.get("source_file_sha256") or ""),
        parsed_created_utc=str(provenance.get("parsed_created_utc") or ""),
        z_matrix_config_hash=str(provenance.get("z_matrix_config_hash") or ""),
        z_matrix_config=dict(provenance.get("z_matrix_config") or {}),
        spectral_config_hash=str(provenance.get("spectral_config_hash") or ""),
        spectral_config=dict(provenance.get("spectral_config") or {}),
        config=config,
        neuron_order=neuron_order,
        motifs=motifs,
        links=[dict(entry) for entry in payload.get("links") or []],
        families=families,
        groups=[dict(entry) for entry in payload.get("groups") or []],
        pathway=dict(payload.get("pathway") or NEUTRAL_PATHWAY),
        metadata=dict(payload.get("metadata") or {}),
        diagnostics=dict(payload.get("metadata") or {}),
        arrays={},
        loaded_from=loaded_from,
    )


def load_motifs(path: str | Path, *, prefer_sidecar: bool = True) -> MotifAnalysis:
    """Load a Phase 04 artifact (the entry point for 04B-04D and Phase 05).

    The canonical JSON is always the source of truth.  When the sibling ``.npz`` cache
    exists it is *verified* against the JSON's SHA-256 and the JSON-derivable arrays; a
    stale or tampered cache is ignored with a warning instead of raising.  ``path`` may
    also be the ``.npz`` file, in which case the sibling JSON is required.
    """
    candidate = Path(path)
    if candidate.suffix == ".npz":
        json_path = candidate.with_suffix(".json")
        if not json_path.is_file():
            raise FileNotFoundError(
                f"the npz sidecar {candidate} has no sibling JSON artifact at {json_path}"
            )
        analysis = _analysis_from_payload(read_json(json_path), loaded_from=json_path)
        _verify_sidecar(load_sidecar_arrays(candidate), analysis, json_path)
        return analysis

    payload = read_json(candidate)
    analysis = _analysis_from_payload(payload, loaded_from=candidate)
    if prefer_sidecar:
        cache = sidecar_path(candidate)
        if cache.is_file():
            try:
                _verify_sidecar(load_sidecar_arrays(cache), analysis, candidate)
            except Exception as exc:  # stale/tampered/legacy cache: fall back to JSON
                LOGGER.warning("ignoring sidecar %s: %s", cache.name, exc)
    return analysis


# ---------------------------------------------------------------------------
# Diagnostics output: --save-data, --stats, the completion box
# ---------------------------------------------------------------------------
def save_data_payload(analysis: MotifAnalysis, *, now: Any = None) -> dict[str, Any]:
    """The ``motifs.data.json`` payload (per-neuron participation tables)."""
    neurons: list[dict[str, Any]] = []
    participation = analysis.participation
    membership = analysis.membership
    node_metadata = dict(analysis.diagnostics.get("node_metadata") or {})
    neuron_regions = list(analysis.diagnostics.get("neuron_regions") or [])
    if participation is not None and membership is not None:
        recurrence = membership.sum(axis=1)
        for index, name in enumerate(analysis.neuron_order):
            row = np.asarray(participation[index], dtype=np.float64)
            meta = dict(node_metadata.get(name) or {})
            dominant = int(np.argmax(row)) + 1 if row.size and float(row.max()) > 0.0 else 0
            neurons.append(
                {
                    "index": int(index),
                    "neuron_id": name,
                    "region": str(neuron_regions[index]) if index < len(neuron_regions) else "",
                    "cent": meta.get("cent"),
                    "out_degree": meta.get("out_degree"),
                    "in_degree": meta.get("in_degree"),
                    "recurrence": int(recurrence[index]),
                    "max_participation": float(row.max()) if row.size else 0.0,
                    "mean_participation": float(row.mean()) if row.size else 0.0,
                    "dominant_mode": dominant,
                    "group_id": int(analysis.arrays.get("group_labels", np.zeros(len(analysis.neuron_order), dtype="<i8"))[index]) if analysis.groups else None,
                    "participation": [float(value) for value in row],
                }
            )
    return {
        "provenance": analysis.to_dict()["provenance"],
        "config": analysis.config.to_dict(),
        "stage": str(analysis.config.stage),
        "neuron_order": list(analysis.neuron_order),
        "neurons": neurons,
        "motifs": [motif.to_dict() for motif in analysis.motifs],
        "links": [dict(entry) for entry in analysis.links],
        "families": [dict(entry) for entry in analysis.families],
        "groups": [dict(entry) for entry in analysis.groups],
        "pathway": dict(analysis.pathway),
        "metadata": dict(analysis.metadata),
        "generator": GENERATOR,
        "created_utc": utc_timestamp(now),
    }


def render_statistics(analysis: MotifAnalysis) -> str:
    """Terminal-only statistics block printed by ``--stats``."""
    metadata = analysis.metadata
    top = metadata.get("top_recurrent") or []
    top_text = ", ".join(
        f"{entry['neuron_id']} ({entry['recurrence']})" for entry in top[:2]
    ) or "n/a"
    n_families = len(analysis.families)
    classes = metadata.get("link_classes") or {}
    n_links = metadata.get("n_links") or 0
    n_multi = metadata.get("n_multi_mode_families") or 0
    if metadata.get("n_pathway_nodes") is None:
        pathway_lines = ["  pathway             : not computed yet (grouping none)"]
    else:
        pathway = analysis.pathway
        pathway_lines = [
            f"  pathway             : {metadata.get('n_pathway_nodes')} nodes / "
            f"{metadata.get('n_pathway_edges')} edges "
            f"(source {pathway.get('source')}, weight {pathway.get('weight')})",
            f"  pathway threshold   : {pathway.get('edge_threshold')} x abs_max "
            f"(abs_max {float(metadata.get('pathway_abs_max') or 0.0):.6f}, "
            f"top {pathway.get('top_edges')} per source, intra {pathway.get('intra')})",
            f"  pathway polarity    : {metadata.get('pathway_positive_edges')} positive / "
            f"{metadata.get('pathway_negative_edges')} negative   "
            f"intra {metadata.get('pathway_intra_edges')} / cross {metadata.get('pathway_cross_edges')}",
            f"  pathway concentration: {float(metadata.get('pathway_weight_concentration') or 0.0):.6f}",
        ]
    lines = [
        "Phase 04D -- motif statistics",
        f"  participation       : {metadata.get('participation')}   "
        f"threshold method: {metadata.get('threshold_method')}",
        f"  shape               : {metadata.get('n_neurons')} neurons x {metadata.get('n_modes')} modes",
        f"  motifs              : {metadata.get('n_motifs')} (k = {metadata.get('k_resolved')})",
        f"  members total       : {metadata.get('n_members_total')}  (min/median/max/mean: "
        f"{metadata.get('n_members_min')} / {metadata.get('n_members_median')} / "
        f"{metadata.get('n_members_max')} / {float(metadata.get('n_members_mean') or 0.0):.3f})",
        f"  unique members      : {metadata.get('n_unique_members')}   "
        f"in no motif: {metadata.get('n_neurons_in_no_motif')}",
        f"  membership density  : {float(metadata.get('membership_density') or 0.0):.6f}",
        f"  recurrence >=1/2/5  : {metadata.get('n_recurrence_ge_1')} / "
        f"{metadata.get('n_recurrence_ge_2')} / {metadata.get('n_recurrence_ge_5')}  "
        f"(max {metadata.get('recurrence_max')})",
        f"  top recurrent       : {top_text}",
        f"  isolated neurons    : {metadata.get('n_isolated_neurons')}",
        f"  regions             : {metadata.get('n_regions')} derived",
        f"  links               : {n_links} (stable {classes.get('stable', 0)} / "
        f"flipped {classes.get('flipped', 0)} / composite {classes.get('composite', 0)} / "
        f"weak {classes.get('weak', 0)})",
        f"  families            : {n_families} ({n_multi} multi-mode)",
        f"  groups              : {metadata.get('n_groups')} (background {metadata.get('n_background_groups')} / singletons {metadata.get('n_singleton_groups')})",
        *pathway_lines,
    ]
    return "\n".join(lines)


def render_summary_box(
    analysis: MotifAnalysis,
    paths: dict[str, Path],
    *,
    written: bool = True,
    sidecar: bool = True,
    save_data: bool = False,
    graphml: bool = True,
    stats: bool = False,
) -> str:
    """The boxed terminal-only Phase 04D completion summary."""
    metadata = analysis.metadata

    def _name(key: str) -> str:
        return paths[key].name

    def _skip(reason: str) -> str:
        return f"skipped ({reason})"

    if written and sidecar:
        npz_line = _name("npz")
    elif sidecar:
        npz_line = _skip("--dry-run")
    else:
        npz_line = _skip("--no-sidecar")
    if not graphml:
        graphml_line = "not requested (--no-graphml)"
    elif written:
        graphml_line = _name("graphml")
    else:
        graphml_line = _skip("--dry-run")

    lines = [
        "Phase 04D complete" if written else "Phase 04D dry-run (nothing written)",
        f"Stage: {STAGE} (participation: {metadata.get('participation')})",
        f"Threshold: {metadata.get('threshold_method')}",
        f"Shape: {metadata.get('n_neurons')} neurons x {metadata.get('n_modes')} modes",
        f"Motifs: {metadata.get('n_motifs')}  members: {metadata.get('n_members_total')}",
        f"Families: {len(analysis.families)}  links: {metadata.get('n_links') or 0}",
        f"Groups: {metadata.get('n_groups')}",
        (
            f"Pathway: {metadata.get('n_pathway_nodes')} nodes / "
            f"{metadata.get('n_pathway_edges')} edges"
            if metadata.get("n_pathway_nodes") is not None
            else "Pathway: not computed (grouping none)"
        ),
        f"JSON written: {_name('json') if written else _skip('--dry-run')}",
        f"NPZ written:  {npz_line}",
        (
            "Save-data:   "
            + (
                _name("data")
                if (written and save_data)
                else ("not requested (--save-data)" if not save_data else _skip("--dry-run"))
            )
        ),
        f"GraphML:     {graphml_line}",
        "Stats: printed above (--stats)" if stats else "Stats: not requested (--stats)",
    ]
    width = max(max(len(line) for line in lines), 60)
    border = "+" + "-" * (width + 2) + "+"
    body = [f"| {line.ljust(width)} |" for line in lines]
    return "\n".join([border, *body, border])


def write_artifact_set(
    analysis: MotifAnalysis,
    outdir: str | Path,
    *,
    sidecar: bool = True,
    save_data: bool = False,
    graphml: bool = True,
    force_config_hash: bool = False,
    report: ValidationReport | None = None,
    log: logging.Logger | None = None,
) -> dict[str, Path]:
    """Write the requested artifacts and return the paths that were written.

    The canonical JSON goes first (it is self-sufficient); the ``.npz`` cache, the
    save-data file and the GraphML pathway follow.  A failed derived write is recorded as an
    error in *report*, never a partial file: every writer builds its bytes first and then
    swaps them in.
    """
    logger = log or LOGGER
    report = report if report is not None else ValidationReport()
    paths = artifact_paths(analysis, outdir, force_config_hash=force_config_hash)
    written: dict[str, Path] = {}

    written["json"] = write_json_atomic(paths["json"], analysis.to_dict())

    if sidecar:
        try:
            written["npz"] = write_sidecar(analysis, written["json"])
        except Exception as exc:
            report.error(CODE_ARTIFACT_WRITE, f"could not write the npz sidecar {paths['npz']}: {exc}")

    if save_data:
        try:
            written["data"] = write_json_atomic(paths["data"], save_data_payload(analysis))
        except Exception as exc:
            report.error(CODE_ARTIFACT_WRITE, f"could not write the save-data file {paths['data']}: {exc}")

    if graphml:
        try:
            written["graphml"] = write_graphml_pathway(
                paths["graphml"],
                analysis.pathway,
                groups=analysis.groups,
                config=analysis.config,
                source=analysis.pathway.get("source"),
                centroids=analysis.arrays.get("group_centroids"),
                z_by_edge=(analysis.diagnostics.get("pathway") or {}).get("z_contributions"),
            )
        except Exception as exc:
            report.error(
                CODE_ARTIFACT_WRITE, f"could not write the GraphML pathway {paths['graphml']}: {exc}"
            )

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("wrote %s", {key: str(value) for key, value in written.items()})
    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _is_excluded_name(name: str) -> bool:
    """True for Phase 03 save-data artifacts (``eigen.data.json``)."""
    return name.endswith(EXCLUDED_INPUT_SUFFIXES)


def _looks_like_phase03(name: str) -> bool:
    """True for a Phase 03 artifact name (``eigen[.variant].json``)."""
    return name.startswith(INPUT_VARIANT_PREFIX) and not _is_excluded_name(name)


def resolve_inputs(
    paths: Sequence[str | Path],
    pattern: str = DEFAULT_INPUT_PATTERN,
    *,
    include_variants: bool = False,
) -> list[Path]:
    """Expand CLI inputs into a deterministic list of Phase 03 JSON artifacts.

    Directory inputs are searched **recursively** and only pick up Phase 03 artifacts,
    so a Phase 04 output (``motifs.json``) or a save-data file is never re-consumed.
    ``--include-variants`` also resolves ``eigen.<variant>.json``.
    """
    patterns = [pattern]
    if include_variants and VARIANT_INPUT_PATTERN not in patterns:
        patterns.append(VARIANT_INPUT_PATTERN)

    resolved: list[Path] = []
    for raw in paths:
        candidate = Path(raw)
        if candidate.is_dir():
            matches = {
                path
                for pat in patterns
                for path in candidate.rglob(pat)
                if path.is_file() and _looks_like_phase03(path.name)
            }
            resolved.extend(sorted(matches))
        elif candidate.is_file():
            if candidate.suffix.lower() != SUPPORTED_SUFFIX:
                LOGGER.error("%s: unsupported suffix (expected %s)", candidate, SUPPORTED_SUFFIX)
                continue
            if _is_excluded_name(candidate.name):
                LOGGER.error("%s: this is a save-data artifact, not a spectrum", candidate)
                continue
            resolved.append(candidate)
        else:
            LOGGER.error("%s: no such file or directory", candidate)

    return sorted(dict.fromkeys(resolved))


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.clustering.functional_motifs",
        description="Extract per-mode signed motifs from a Phase 03 eigen.json (motifs.json)",
    )
    parser.add_argument(
        "--input", "-i", nargs="+", required=True, metavar="PATH",
        help="one or more eigen.json files or directories (searched recursively)",
    )
    parser.add_argument(
        "--outdir", "-o", default="data/processed",
        help="output root; artifacts land in <outdir>/<stem>/ (default: data/processed)",
    )
    parser.add_argument(
        "--glob", dest="glob_pattern", default=DEFAULT_INPUT_PATTERN,
        help=f"glob used for directory inputs (default: {DEFAULT_INPUT_PATTERN})",
    )
    parser.add_argument(
        "--include-variants", dest="include_variants", action="store_true",
        help="also process Phase 03 variant artifacts (eigen.<config_hash8>.json)",
    )
    parser.add_argument(
        "--anatomy", metavar="PATH", default=None,
        help="optional JSON region map that overrides the region derived from the neuron id",
    )

    participation = parser.add_argument_group("participation and motifs")
    participation.add_argument(
        "--participation", choices=PARTICIPATIONS, default=DEFAULT_PARTICIPATION,
        help=f"how the two loading axes are combined (default: {DEFAULT_PARTICIPATION})",
    )
    participation.add_argument(
        "--threshold-method", dest="threshold_method", choices=THRESHOLD_METHODS,
        default=DEFAULT_THRESHOLD_METHOD,
        help=f"membership threshold rule (default: {DEFAULT_THRESHOLD_METHOD})",
    )
    participation.add_argument(
        "--relative-threshold", dest="relative_threshold", type=float,
        default=DEFAULT_RELATIVE_THRESHOLD,
        help=f"fraction of each mode's maximum participation (default: {DEFAULT_RELATIVE_THRESHOLD})",
    )
    participation.add_argument(
        "--absolute-threshold", dest="absolute_threshold", type=float,
        default=DEFAULT_ABSOLUTE_THRESHOLD,
        help=f"constant participation threshold (default: {DEFAULT_ABSOLUTE_THRESHOLD})",
    )
    participation.add_argument(
        "--quantile", type=float, default=DEFAULT_QUANTILE,
        help=f"per-mode participation quantile (default: {DEFAULT_QUANTILE})",
    )
    participation.add_argument(
        "--participation-threshold", dest="participation_threshold", type=float,
        default=DEFAULT_PARTICIPATION_THRESHOLD,
        help=f"constant threshold used by --threshold-method participation "
             f"(default: {DEFAULT_PARTICIPATION_THRESHOLD})",
    )
    participation.add_argument(
        "--min-members", dest="min_members", type=int, default=DEFAULT_MIN_MEMBERS,
        help=f"top a motif up to this many members (default: {DEFAULT_MIN_MEMBERS})",
    )
    participation.add_argument(
        "--max-members", dest="max_members", type=int, default=DEFAULT_MAX_MEMBERS,
        help=f"truncate a motif to this many members, 0 = uncapped (default: {DEFAULT_MAX_MEMBERS})",
    )

    tracking = parser.add_argument_group("cross-mode tracking")
    tracking.add_argument(
        "--family-jaccard", dest="family_jaccard", type=float, default=DEFAULT_FAMILY_JACCARD,
        help=f"minimum member Jaccard for a stable/flipped link to merge two modes "
             f"(default: {DEFAULT_FAMILY_JACCARD})",
    )
    tracking.add_argument(
        "--family-polarity", dest="family_polarity", type=float, default=DEFAULT_FAMILY_POLARITY,
        help=f"signed polarity agreement needed for a stable (resp. flipped) family link "
             f"(default: {DEFAULT_FAMILY_POLARITY})",
    )
    tracking.add_argument(
        "--link-min-jaccard", dest="link_min_jaccard", type=float, default=DEFAULT_LINK_MIN_JACCARD,
        help=f"below this Jaccard a link is weak; at or above it the link is reported "
             f"(default: {DEFAULT_LINK_MIN_JACCARD})",
    )

    grouping = parser.add_argument_group("functional neuron groups")
    grouping.add_argument("--grouping", choices=GROUPINGS, default=DEFAULT_GROUPING)
    grouping.add_argument("--polarity-split", action="store_true", default=DEFAULT_POLARITY_SPLIT)
    grouping.add_argument("--polarity-min-members", type=int, default=DEFAULT_POLARITY_MIN_MEMBERS)
    grouping.add_argument("--max-group-size", type=int, default=DEFAULT_MAX_GROUP_SIZE)
    grouping.add_argument("--n-groups", default=DEFAULT_N_GROUPS, metavar="N|auto")
    grouping.add_argument("--linkage", choices=LINKAGES, default=DEFAULT_LINKAGE)
    grouping.add_argument("--affinity", choices=AFFINITIES, default=DEFAULT_AFFINITY)
    grouping.add_argument("--merge-threshold", type=float, default=DEFAULT_MERGE_THRESHOLD)

    pathway = parser.add_argument_group("pathway graph")
    pathway.add_argument(
        "--pathway-source", dest="pathway_source", choices=PATHWAY_SOURCES,
        default=DEFAULT_PATHWAY_SOURCE,
        help=f"contribution source: per-mode outer products, the raw Phase 02 matrix, or "
             f"both (default: {DEFAULT_PATHWAY_SOURCE})",
    )
    pathway.add_argument(
        "--pathway-weight", dest="pathway_weight", choices=PATHWAY_WEIGHTS,
        default=DEFAULT_PATHWAY_WEIGHT,
        help=f"mode weighting rule for the aggregated contributions "
             f"(default: {DEFAULT_PATHWAY_WEIGHT}; evr and energy are equivalent)",
    )
    pathway.add_argument(
        "--pathway-edge-threshold", dest="pathway_edge_threshold", type=float,
        default=DEFAULT_PATHWAY_EDGE_THRESHOLD,
        help=f"minimum |edge weight| as a fraction of the strongest contribution "
             f"(default: {DEFAULT_PATHWAY_EDGE_THRESHOLD})",
    )
    pathway.add_argument(
        "--pathway-top-edges", dest="pathway_top_edges", type=int,
        default=DEFAULT_PATHWAY_TOP_EDGES,
        help=f"strongest edges kept per source group (default: {DEFAULT_PATHWAY_TOP_EDGES})",
    )
    pathway.add_argument(
        "--no-intra", dest="intra", action="store_false", default=DEFAULT_INTRA,
        help="exclude intra-group (self) edges from the pathway graph",
    )
    pathway.add_argument(
        "--graphml", dest="graphml", action="store_true", default=True,
        help="write the GraphML pathway artifact (default: on)",
    )
    pathway.add_argument(
        "--no-graphml", dest="graphml", action="store_false",
        help="do not write motifs.pathway.graphml",
    )

    outputs = parser.add_argument_group("outputs")
    outputs.add_argument("--config-hash", dest="config_hash", action="store_true",
                         help="always include <config_hash8> in the artifact filenames")
    outputs.add_argument("--no-sidecar", dest="no_sidecar", action="store_true",
                         help="do not write motifs.npz (the JSON stays canonical)")
    outputs.add_argument("--save-data", dest="save_data", action="store_true",
                         help="write motifs.data.json (per-neuron participation tables)")
    outputs.add_argument("--stats", action="store_true",
                         help="print the motif statistics to the terminal")

    parser.add_argument("--strict", action="store_true",
                        help="fail on any validation issue (JSON, NPZ and save-data writing)")
    parser.add_argument("--dry-run", dest="dry_run", action="store_true",
                        help="validate, print statistics and the summary box without writing anything")
    parser.add_argument(
        "--log-level", dest="log_level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="logging verbosity (default: INFO)",
    )
    return parser


def _sibling_node_metadata(source: Path) -> dict[str, dict[str, Any]] | None:
    """Read ``cent``/degrees from the sibling ``parsed_graph.json`` when it exists."""
    for candidate in (source.parent / "parsed_graph.json", source.parent / "parsed_graph.filtered.json"):
        if candidate.is_file():
            try:
                return load_node_metadata(candidate)
            except Exception as exc:  # pragma: no cover - defensive
                LOGGER.warning("could not read %s: %s", candidate.name, exc)
                return None
    return None


def _config_from_args(args: argparse.Namespace) -> MotifConfig:
    """Build the Phase 04 config from the CLI surface (later sub-phase fields stay frozen)."""
    return MotifConfig(
        participation=args.participation,
        threshold_method=args.threshold_method,
        relative_threshold=float(args.relative_threshold),
        absolute_threshold=float(args.absolute_threshold),
        quantile=float(args.quantile),
        participation_threshold=float(args.participation_threshold),
        min_members=int(args.min_members),
        max_members=int(args.max_members),
        family_jaccard=float(args.family_jaccard),
        family_polarity=float(args.family_polarity),
        link_min_jaccard=float(args.link_min_jaccard),
        grouping=args.grouping,
        polarity_split=bool(args.polarity_split),
        polarity_min_members=int(args.polarity_min_members),
        max_group_size=int(args.max_group_size),
        n_groups=str(args.n_groups),
        linkage=args.linkage,
        affinity=args.affinity,
        merge_threshold=float(args.merge_threshold),
        pathway_source=str(args.pathway_source),
        pathway_weight=str(args.pathway_weight),
        pathway_edge_threshold=float(args.pathway_edge_threshold),
        pathway_top_edges=int(args.pathway_top_edges),
        intra=bool(args.intra),
        stage=STAGE,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; returns the process exit code.

    ``0`` success, ``1`` any input failed validation or an artifact could not be
    written, ``2`` no input matched.
    """
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(message)s")

    inputs = resolve_inputs(args.input, args.glob_pattern, include_variants=args.include_variants)
    if not inputs:
        LOGGER.error("no Phase 03 artifact matched the given input")
        return 2

    config = _config_from_args(args)
    anatomy = None
    if args.anatomy:
        try:
            anatomy = load_anatomy(args.anatomy)
        except Exception as exc:
            LOGGER.error("could not load the anatomy file: %s", exc)
            return 1

    failures = 0
    for source in inputs:
        try:
            spectrum = load_spectrum(source)
        except Exception as exc:
            LOGGER.error("%s: %s", source.name, exc)
            failures += 1
            continue

        try:
            analysis, report = build_motif_analysis(
                spectrum,
                source_artifact=source,
                config=config,
                anatomy=anatomy,
                node_metadata=_sibling_node_metadata(source),
            )
        except Exception as exc:  # malformed artifact or invalid config
            LOGGER.error("%s: %s", source.name, exc)
            failures += 1
            continue

        if report.has_errors:
            for issue in report.errors:
                LOGGER.error("%s: [%s] %s", source.name, issue.code, issue.message)
            LOGGER.error("%s: validation failed (%s)", source.name, report.summary())
            failures += 1
            continue

        metadata = analysis.metadata
        LOGGER.info(
            "%s: participation=%s threshold=%s neurons=%d modes=%d motifs=%d members=%d "
            "unique=%d no-motif=%d config_hash=%s",
            source.name,
            metadata["participation"],
            metadata["threshold_method"],
            metadata["n_neurons"],
            metadata["n_modes"],
            metadata["n_motifs"],
            metadata["n_members_total"],
            metadata["n_unique_members"],
            metadata["n_neurons_in_no_motif"],
            metadata["config_hash"],
        )

        if args.stats:
            print(render_statistics(analysis))

        paths = artifact_paths(analysis, args.outdir, force_config_hash=args.config_hash)
        if not args.dry_run:
            write_artifact_set(
                analysis,
                args.outdir,
                sidecar=not args.no_sidecar,
                save_data=args.save_data,
                graphml=args.graphml,
                force_config_hash=args.config_hash,
                report=report,
                log=LOGGER,
            )

        for issue in report.warnings:
            LOGGER.warning("%s: [%s] %s", source.name, issue.code, issue.message)

        if not args.dry_run and (report.has_errors or (args.strict and report.warnings)):
            for issue in report.errors:
                LOGGER.error("%s: [%s] %s", source.name, issue.code, issue.message)
            if args.strict and report.warnings:
                LOGGER.error(
                    "%s: --strict escalates %d warning(s) to errors",
                    source.name,
                    len(report.warnings),
                )
            LOGGER.error("%s: artifact writing failed (%s)", source.name, report.summary())
            failures += 1
            continue

        if not args.dry_run:
            for key, path in paths.items():
                if path.is_file():
                    LOGGER.info("%s: wrote %s [%s]", source.name, path, key)

        print(
            render_summary_box(
                analysis,
                paths,
                written=not args.dry_run,
                sidecar=not args.no_sidecar,
                save_data=args.save_data,
                graphml=args.graphml,
                stats=args.stats,
            )
        )

    if failures:
        LOGGER.error("%d of %d input file(s) failed", failures, len(inputs))
    return 1 if failures else 0


if __name__ == "__main__":  # pragma: no cover - exercised through subprocess tests
    sys.exit(main())
