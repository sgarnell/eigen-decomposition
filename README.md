# Functional Motifs Pipeline

The functional motifs pipeline turns a spectral decomposition into an interpretable
description of circuit structure. It is designed for directed functional-connectivity
data such as the Drosophila fan-shaped body, but the artifacts and algorithms are general
to any compatible spectral output.

## Overview

The workflow is a single command with several analysis layers:

1. **Motif extraction** computes each neuron's participation in every retained mode,
   applies per-mode thresholds, and records ordered signed members.
2. **Cross-mode tracking** compares motif membership between modes. It produces pairwise
   links and merges strongly related modes into motif families.
3. **Functional neuron grouping** assigns every neuron to a population group using motif
   dominance, loading-space clustering, or a hybrid strategy. Zero-participation neurons
   are retained in an explicit background group.
4. **Pathway construction** builds a directed graph whose nodes are the functional groups
   and whose edges are signed group-to-group mode contributions, exported as a NetworkX
   GraphML document for interactive exploration in yEd.

Every completed run writes structured JSON, an optional NumPy sidecar, an optional expanded
per-neuron report, and the GraphML pathway graph, plus terminal statistics. The JSON is the
source of truth; the sidecar is a verified cache for fast array access.

## Installation and environment

The supported environment is conda-forge with Python 3.11:

```bash
conda env create -f environment.yml
conda activate eigen-decomposition
```

To update an existing environment:

```bash
conda env update -f environment.yml --prune
```

The environment includes NumPy and SciPy for decomposition, scikit-learn for optional
loading-space grouping, NetworkX and Matplotlib for graph/visualization work, and the
project's parsing and test dependencies.

## Input requirements

The motif command consumes an `eigen.json` file produced by the spectral decomposition
command. It must contain:

- `neuron_order`: the ordered neuron IDs used as matrix rows;
- `values`: the complete ranked spectrum;
- `modes`: retained mode records with `left` and `right` loading vectors;
- decomposition configuration and metadata.

The loading vectors must have one row per entry in `neuron_order`, and all numeric values
used by the analysis must be finite. A sibling `parsed_graph.json` is used automatically
when present to add `cent`, `out_degree`, and `in_degree`. An optional anatomy JSON file
can override the region inferred from neuron IDs. `eigen.data.json`, PNG files, and CSV
files are not valid motif inputs.

## Quick Start

For the canonical reference input:

```bash
python -m src.clustering.functional_motifs \
  -i eigen.json -o data/processed --stats --save-data
```

The command writes `motifs.json`, `motifs.npz`, the optional `motifs.data.json` report and
the `motifs.pathway.graphml` pathway graph, and prints the layer statistics.

For an input at `data/processed/sample/eigen.json`, outputs are placed in
`data/processed/sample/`.

## Analysis layers

### Motif extraction

Participation is computed from left (`L`) and right (`R`) loadings:

| Mode | Participation |
|---|---|
| `left` | `abs(L)` |
| `right` | `abs(R)` |
| `max` | `max(abs(L), abs(R))` |
| `rms` (default) | `sqrt((L² + R²) / 2)` |

Membership is `P[i,m] >= t_m`. The default threshold is relative to each mode's maximum:
`t_m = 0.25 * max_i(P[i,m])`. Each member also records signed participation, polarity,
dominant axis, region, and rank. A completely zero participation row is isolated and is
never assigned to a motif.

### Cross-mode tracking

For every pair of motifs with shared members, the pipeline records shared member IDs and
count, Jaccard overlap, signed polarity agreement, and a `stable`, `flipped`, `composite`,
or `weak` classification. Only stable and flipped links merge modes into families.
Composite and weak links remain available for downstream analysis but do not collapse
unrelated modes.

### Functional neuron grouping

Grouping creates a total partition of the neurons, including `G00` for background rows.
The default `motif` mode assigns each nonzero row to its strongest motif. `loadings` uses
agglomerative clustering on concatenated left/right loadings, `hybrid` starts with motif
assignments and merges similar groups when needed, and `none` leaves the group block
neutral. Group records include members, size, dominant mode, region composition, centroid,
coherence, and background/singleton flags.

### Pathway graph

The pathway layer reduces the network to a directed graph whose nodes are the functional
groups and whose edges are signed group-to-group contributions aggregated from the
per-mode outer products:

```text
w          = mode weights (rule: evr | energy | uniform | value | abs_value, Σ|w| = 1)
OP_m[i,j]  = value_m · U[i,m] · V[j,m]
W[A,B]     = Σ_m w_m · Σ_{i∈A} Σ_{j∈B} OP_m[i,j]        # --pathway-source mode
           = Σ_{i∈A} Σ_{j∈B} Z[i,j]                      # --pathway-source matrix
           = mode + matrix                               # --pathway-source both
```

Edges are kept when `|W| >= --pathway-edge-threshold × abs_max` (the threshold is a
**fraction of the strongest contribution**, so the stored weights stay on the Phase 02
scale), then at most `--pathway-top-edges` are kept per source group. `--no-intra` drops
self-edges. Each edge records its weight, magnitude, polarity, intra flag, contributing
modes and the top five per-mode contributions; each node records its group metadata and its
strongest signed-participation modes.

The graph is exported with **NetworkX** to `motifs.pathway.graphml` for interactive
exploration in yEd (custom layouts, recoloring, metadata inspection, SVG/PDF export). Node
and edge attributes carry the full metadata, and the graph element records
`n_nodes`, `n_edges`, `weight_normalization`, `pathway_source`, `pathway_weight_rule`,
`threshold` and `topN`. Use `--no-graphml` to skip the artifact.

`--pathway-source matrix` and `both` enrich the contributions with the raw Phase 02
`z_matrix[.<variant>].json` sibling (`z_contribution` on every GraphML edge); a missing or
mismatched matrix artifact is an error, because the requested source cannot be honoured.

With the defaults on the reference dataset the pathway has 22 nodes and 39 edges,
`abs_max = 2.345300`, 0 positive / 39 negative edges, 7 intra / 32 cross edges and
`weight_concentration = 0.055842`. Every kept edge is negative at the defaults — a direct
consequence of Phase 02's documented negative-weight bias (77 % of unified weights are
negative).

## Full CLI reference

```text
python -m src.clustering.functional_motifs \
  -i PATH [PATH ...] -o OUTDIR [options]
```

### Motif extraction options

| Option | Default | Description |
|---|---:|---|
| `-i, --input PATH ...` | required | One or more `eigen.json` files or directories. Directories are searched recursively. |
| `--glob PATTERN` | `eigen.json` | Pattern used for directory inputs. |
| `--include-variants` | off | Also process `eigen.<config_hash8>.json` inputs. |
| `--anatomy PATH` | none | JSON region map or neuron-record file; overrides inferred regions. |
| `--participation {left,right,max,rms}` | `rms` | Combines the two loading axes. |
| `--threshold-method {relative,absolute,quantile,participation}` | `relative` | Selects the per-mode threshold rule. |
| `--relative-threshold FLOAT` | `0.25` | Fraction of each mode's maximum. |
| `--absolute-threshold FLOAT` | `0.05` | Constant threshold for `absolute`. |
| `--quantile FLOAT` | `0.8` | Per-mode quantile for `quantile`. |
| `--participation-threshold FLOAT` | `0.1` | Constant threshold for `participation`. |
| `--min-members INT` | `3` | Minimum motif size; tops up from highest participation. |
| `--max-members INT` | `0` | Maximum motif size; `0` means uncapped. |

### Tracking options

| Option | Default | Description |
|---|---:|---|
| `--family-jaccard FLOAT` | `0.2` | Minimum Jaccard for a family-eligible link. |
| `--family-polarity FLOAT` | `0.5` | Signed agreement required for stable/flipped classification. |
| `--link-min-jaccard FLOAT` | `0.1` | Links below this overlap are classified as weak. |

### Grouping options

| Option | Default | Description |
|---|---:|---|
| `--grouping {motif,loadings,hybrid,none}` | `motif` | Grouping algorithm. |
| `--polarity-split` | off | Split a group by seed-mode polarity when the minority is large enough. |
| `--polarity-min-members INT` | `3` | Minimum minority size for a polarity split. |
| `--max-group-size INT` | `12` | Size cap that can trigger deterministic merging. |
| `--n-groups N\|auto` | `auto` | Cluster count for loading grouping; `auto` resolves to retained mode count. |
| `--linkage {ward,complete,average,single}` | `average` | Agglomerative linkage. |
| `--affinity {euclidean,cosine,manhattan}` | `cosine` | Loading-space distance metric. |
| `--merge-threshold FLOAT` | `0.9` | Minimum centroid cosine for hybrid merging. |

### Pathway graph options

| Option | Default | Description |
|---|---:|---|
| `--pathway-source {mode,matrix,both}` | `mode` | Per-mode outer products, the raw Phase 02 matrix, or both. |
| `--pathway-weight {evr,energy,uniform,value,abs_value}` | `evr` | Mode weighting rule (`evr` and `energy` are equivalent). |
| `--pathway-edge-threshold FLOAT` | `0.1` | Minimum edge weight as a fraction of `abs_max`. |
| `--pathway-top-edges INT` | `5` | Strongest edges kept per source group. |
| `--no-intra` | off | Exclude intra-group (self) edges. |
| `--graphml` / `--no-graphml` | on | Write `motifs.pathway.graphml` (NetworkX). |

### Global and output options

| Option | Description |
|---|---|
| `-o, --outdir PATH` | Output root; default `data/processed`. |
| `--config-hash` | Always include the eight-character configuration hash in filenames. |
| `--no-sidecar` | Do not write `motifs.npz`; JSON remains complete and canonical. |
| `--save-data` | Write the expanded per-neuron `motifs.data.json` report. |
| `--stats` | Print motif, tracking, and grouping statistics. |
| `--log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}` | Set logging verbosity; default `INFO`. |

### Batch options

Pass multiple paths after `-i`, or pass a directory:

```bash
python -m src.clustering.functional_motifs \
  -i data/processed --glob eigen.json --include-variants \
  -o data/processed --no-sidecar --strict
```

Each resolved input is processed independently and deterministically. Variant inputs are
excluded unless `--include-variants` is supplied; motif outputs inherit the input variant
suffix.

### Dry-run and validation options

| Option | Description |
|---|---|
| `--dry-run` | Resolve, analyze, validate, and report without writing artifacts. |
| `--strict` | Treat validation warnings as failures. |

Exit status is `0` for success, `1` for validation or write failure, and `2` when no input
matches.

## Full pipeline examples

### Default pipeline

```bash
python -m src.clustering.functional_motifs \
  -i data/processed/FB4Yaffect_FB45_999prePost_001_all/eigen.json \
  -o data/processed --stats --save-data
```

### Custom motif thresholds

```bash
python -m src.clustering.functional_motifs \
  -i eigen.json -o data/processed \
  --participation rms --threshold-method absolute \
  --absolute-threshold 0.10 --min-members 5 --max-members 40 \
  --config-hash --stats
```

### Custom tracking thresholds

```bash
python -m src.clustering.functional_motifs \
  -i eigen.json -o data/processed \
  --family-jaccard 0.35 --family-polarity 0.75 \
  --link-min-jaccard 0.15 --config-hash --stats
```

Use `--config-hash` whenever changing tracking or grouping settings to make the output
filename unambiguous and prevent accidental replacement of the canonical artifact.

### Loading clustering

```bash
python -m src.clustering.functional_motifs \
  -i eigen.json -o data/processed \
  --grouping loadings --n-groups 12 \
  --linkage average --affinity cosine --config-hash --save-data
```

The reference data has low within-group loading coherence, so loading clustering should
normally use an explicit `--n-groups` rather than relying on `auto`.

### Hybrid grouping and polarity splitting

```bash
python -m src.clustering.functional_motifs \
  -i eigen.json -o data/processed \
  --grouping hybrid --merge-threshold 0.85 --max-group-size 10 \
  --polarity-split --polarity-min-members 3 \
  --config-hash --stats
```

### Pathway graph

```bash
# default pathway: mode contributions, evr weights, threshold 0.1 x abs_max, top 5 per source
python -m src.clustering.functional_motifs \
  -i data/processed/FB4Yaffect_FB45_999prePost_001_all/eigen.json \
  -o data/processed --stats

# cross-group edges only, stricter threshold, JSON + GraphML
python -m src.clustering.functional_motifs \
  -i eigen.json -o data/processed \
  --no-intra --pathway-edge-threshold 0.25 --pathway-top-edges 3 --stats

# enrich the contributions with the raw Phase 02 matrix (needs z_matrix.json next to eigen.json)
python -m src.clustering.functional_motifs \
  -i data/processed/<stem>/eigen.json -o data/processed \
  --pathway-source both --config-hash --stats

# JSON and NPZ only (no GraphML)
python -m src.clustering.functional_motifs \
  -i eigen.json -o data/processed --no-graphml --no-sidecar
```

### Batch processing

```bash
python -m src.clustering.functional_motifs \
  -i data/processed -o data/processed \
  --glob eigen.json --include-variants --strict --save-data
```

### Dry-run validation

```bash
python -m src.clustering.functional_motifs \
  -i eigen.json -o data/processed --dry-run --stats --strict
```

## Artifact description

Artifacts are written under `<outdir>/<input-stem>/`. For a variant or hashed run, the
variant/hash segment is retained in the filename.

### `motifs.json`

The canonical, self-contained artifact contains provenance, the complete hashed config,
`neuron_order`, motif records, links, families, groups, pathway placeholders, and metadata.
Motif records include mode/value fields, threshold, membership, signed participation,
sender/receiver members, strength, variance share, polarity counts, and region composition.

Link records contain `source_mode`, `target_mode`, labels, shared members, `n_shared`,
`jaccard`, `polarity_agreement`, and `classification`. Family records contain family ID,
modes, merged members, polarity, and per-mode occurrences.

Group records contain:

```text
group_id, label, members, n_members, size, dominant_mode,
region_composition, cent_mean, coherence, is_background, is_singleton
```

The group list is a total partition when grouping is active. `G00` is the background group;
singleton groups are flagged explicitly. The `pathway` object holds the 04D group-to-group
graph: `nodes` (every group), `edges` (the filtered signed contributions), `n_nodes`,
`n_edges`, `weight`, `source`, `edge_threshold`, `top_edges`, `intra`, `abs_max`,
`n_positive`, `n_negative`, `n_intra`, `n_cross`, `weight_concentration` and `weights`.

### `motifs.npz`

Unless `--no-sidecar` is used, the verified sidecar contains **24 arrays** for an active
grouping run:

| Array | Shape | Meaning |
|---|---:|---|
| `stage` | `(1,)` | Writer stage marker. |
| `participation` | `(N,k)` | Non-negative participation matrix. |
| `membership` | `(N,k)` | Binary motif membership. |
| `loadings_left`, `loadings_right` | `(N,k)` | Original loading bases. |
| `values` | `(N,)` | Complete ranked spectrum. |
| `retained_values`, `abs_retained_values` | `(k,)` | Retained values and magnitudes. |
| `explained_variance_ratio` | `(N,)` | Ranked explained variance. |
| `thresholds` | `(k,)` | Effective motif thresholds. |
| `motif_strength_l1`, `motif_strength_energy` | `(k,)` | Motif strength summaries. |
| `family_of_mode` | `(k,)` | Family index for each mode. |
| `link_jaccard`, `link_polarity` | `(k,k)` | Symmetric link matrices with unit diagonal. |
| `neuron_order` | `(N,)` | Row order. |
| `source_json_sha256` | `(1,)` | Digest of the canonical JSON. |
| `numpy_version` | `(1,)` | NumPy writer version. |
| `group_labels` | `(N,)` | Group index for each neuron. |
| `group_centroids` | `(G,k)` | Mean unsigned participation per group. |
| `group_sizes` | `(G,)` | Group member counts. |
| `mode_outer_products` | `(k,N,N)` | Rank-1 `value_m · u_m v_mᵀ` per mode. |
| `pathway_adjacency`, `pathway_adjacency_abs` | `(G,G)` | Filtered signed pathway adjacency and its magnitude. |

The three 04D arrays: `mode_outer_products` is always written; the two adjacency arrays are
written only when the pathway layer runs (i.e. grouping is active). The sidecar is derived
from the JSON digest, read with `allow_pickle=False`, and ignored when stale or tampered;
the JSON remains authoritative.

### `motifs.data.json`

Written with `--save-data`, this expanded report contains the same motifs, links, families,
groups, pathway block, provenance, config, metadata, and a `neurons` table. Each neuron row
contains ID, index, region, centrality/degrees when available, recurrence, maximum and mean
participation, dominant mode, group ID, and the full participation row.

### `motifs.pathway.graphml`

The pathway graph is exported with NetworkX for interactive exploration in yEd. Node
attributes: `group_id`, `label`, `size`, `dominant_mode`, `region_composition` (JSON),
`centroid` (JSON), `coherence`, `is_background`, `is_singleton`. Edge attributes:
`source_group`, `target_group`, `weight`, `abs_weight`, `polarity`, `contribution_by_mode`
(JSON), `threshold_flag`, `topN_flag`, `intra_flag` and, with `--pathway-source
matrix|both`, `z_contribution`. The graph element records `n_nodes`, `n_edges`,
`weight_normalization`, `pathway_source`, `pathway_weight_rule`, `threshold` and `topN`.
Attributes that do not apply (a background group's dominant mode, a singleton's coherence,
`z_contribution` for mode-only contributions) are omitted rather than emitted as null.

## Reference numbers

With default settings on the canonical reference dataset (`rms`, relative threshold
`0.25`, motif grouping), the expected output is:

| Quantity | Value |
|---|---:|
| Neurons / retained modes / motifs | `113 / 21 / 21` |
| Total motif memberships | `623` |
| Minimum / median / maximum / mean motif size | `11 / 23 / 60 / 29.666667` |
| Unique members / neurons in no motif | `107 / 6` |
| Membership density | `0.26253687315634217` |
| Recurrence ≥1 / ≥2 / ≥5 / maximum | `107 / 99 / 70 / 13` |
| Derived regions | `31` |
| Cross-mode links | `187` |
| Stable / flipped / composite / weak links | `1 / 2 / 133 / 51` |
| Families / multi-mode families | `18 / 3` |
| Functional groups | `22` |
| Background groups | `1` |
| Non-background singleton groups | `3` |
| Largest / smallest / mean group size | `10 / 1 / 5.136364` |
| Group coherence min / max / mean | `-0.155531 / 0.876450 / 0.147827` |
| Pathway nodes / edges | `22 / 39` |
| Pathway `abs_max` / smallest kept \|edge\| | `2.345300 / 0.235975` |
| Pathway positive / negative edges | `0 / 39` |
| Pathway intra / cross edges | `7 / 32` |
| Pathway `weight_concentration` | `0.055842` |
| Pathway edges at threshold only (0.05/0.1/0.2/0.25 × `abs_max`) | `91 / 46 / 17 / 14` |

The reference non-background group sizes are:

```text
10, 10, 9, 7, 7, 7, 7, 7, 6, 6, 5, 5, 5, 4, 4, 4, 4, 2, 1, 1, 1
```

The isolated background neuron is `DNa03_R_1`. Representative group compositions include
`G01 ≈ FB4K/hDeltaA`, `G02 ≈ FB4E/hDeltaI`, and `G03 ≈ FB5O/R/S/U/hDeltaH`.
Group coherence is the mean pairwise cosine of signed participation rows; singleton and
background coherence are null.

Set `SOURCE_DATE_EPOCH` to make timestamps deterministic. With the same input, options,
software versions, and epoch, the JSON, NPZ and GraphML outputs are byte-identical.

## Advanced usage

### Threshold tuning

Relative thresholds preserve a comparable fraction of each mode's strongest participants.
Absolute and participation thresholds are useful when amplitudes are on a calibrated scale;
quantiles control the fraction retained per mode. `--min-members` and `--max-members` apply
after thresholding and use deterministic participation ordering.

### Jaccard and polarity math

For motif member sets `A` and `B`:

```text
jaccard(A,B) = |A ∩ B| / |A ∪ B|
polarity_agreement = (same_polarity - opposite_polarity) / |A ∩ B|
```

The thresholds are inclusive for stable/flipped family links. A link below
`--link-min-jaccard` is weak; strong overlap with positive agreement is stable, strong
overlap with negative agreement is flipped, and all other reported links are composite.

### Group coherence

For signed participation rows `S_i = polarity_i * P_i`, coherence is the mean cosine over
the strict upper triangle of all member pairs. It is undefined for groups with fewer than
two members. This measures signed functional agreement, not anatomical similarity.

### Clustering modes and zero vectors

`motif` is deterministic and NumPy-only. `loadings` uses scikit-learn agglomerative
clustering on `[left | right]`; `hybrid` performs deterministic centroid-based merging;
`none` emits no group arrays. All-zero vectors are removed before scikit-learn runs and
reattached to `G00`, preventing undefined cosine distances and preserving the total
partition invariant.

### Determinism and config hashing

Member ties are broken by neuron index, family components by their lowest mode, and group
labels by size and raw cluster label. `--config-hash` adds the first eight characters of
the complete analysis configuration to filenames. Presentation choices such as statistics,
save-data, and sidecar suppression do not alter the analysis hash.

## Status

| Capability | Status | Main output |
|---|---|---|
| Motif extraction | ✔ Implemented | `motifs.json`, `motifs.npz`, optional `motifs.data.json` |
| Cross-mode tracking | ✔ Implemented | links, families, link matrices |
| Functional neuron grouping | ✔ Implemented | groups, group arrays, coherence statistics |
| Pathway graph (GraphML) | ✔ Implemented | `motifs.pathway.graphml`, pathway block, adjacency arrays |

The source entry point is `src/clustering/functional_motifs.py`. The upstream parsing,
matrix construction, and spectral commands produce the `eigen.json` input consumed here.

## License

MIT License.