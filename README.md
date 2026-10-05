# Functional Motifs Pipeline

The functional motifs pipeline turns a spectral decomposition into an interpretable
description of circuit structure. It is designed for directed functional-connectivity
data such as the Drosophila fan-shaped body, but the artifacts and algorithms are general
to any compatible spectral output.

## Overview

The workflow is four commands run in order (see [Pipeline overview](#pipeline-overview)); the
motif command itself is a single command with several analysis layers:

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

## Pipeline overview

The motif pipeline is the **last** of four stages. Each stage is a separate command, each
reads the previous stage's JSON artifact, and each writes into the same
`data/processed/<gv-stem>/` directory:

| # | Stage | Command | Consumes | Produces |
|---|---|---|---|---|
| 01 | Parse Graphviz | `python -m src.parsing.parse_graphviz` | `data/raw_dot/*.gv` | `parsed_graph.json` |
| 02 | Unified-weight matrix | `python -m src.matrices.build_square_matrix` | `parsed_graph.json` | `z_matrix.json` |
| 03 | Spectral decomposition | `python -m src.spectral.spectral_decomposition` | `z_matrix.json` | `eigen.json` |
| 04 | Functional motifs | `python -m src.clustering.functional_motifs` | `eigen.json` | `motifs.json`, `motifs.pathway.graphml` |

**Every stage is required and nothing is chained automatically.** In particular *Phase 02 is
not wrapped by Phase 03*: `spectral_decomposition` only *loads* a `z_matrix.json` (via
`load_z_matrix`) and re-reads it for the `--source`/`--hybrid`/`--delta` options; it never
builds the matrix. Pointing Phase 03 at a Phase 01 artifact fails with
`could not load the Z matrix: 'weight'`, and pointing it at a raw `.gv` fails with
`no input files matched`.

Phase 02 is also needed *after* Phase 03: `functional_motifs` reads the sibling
`z_matrix[.<variant>].json` for `--pathway-source matrix|both`,
`--pathway-weight-rule matrix-only|hybrid` and `--pathway-polarity-rule delta`, and the
04B S-score scorer rebuilds `W_matrix` from it. So keep the Phase 02 artifact on disk.

The reference dataset used throughout this file is
`data/raw_dot/FB4Yaffect_FB45_999prePost_001_all.gv`; the stem
`FB4Yaffect_FB45_999prePost_001_all` comes from that file name.

## Phase 01 — Graphviz parsing

Phase 01 parses a Graphviz `.gv` file into a validated `parsed_graph.json` holding the
neuron list, the collapsed pairs with their `pre_z` / `post_z` scores, and the raw edges.

```bash
python -m src.parsing.parse_graphviz \
  -i data/raw_dot/FB4Yaffect_FB45_999prePost_001_all.gv \
  -o data/processed
```

On the reference file this reports `neurons=113 hubs=1355 raw_edges=2710 pairs=1355
density=0.107064 reciprocity=574` and writes
`data/processed/FB4Yaffect_FB45_999prePost_001_all/parsed_graph.json`.

### Options

| Option | Default | Description |
|---|---:|---|
| `-i, --input PATH ...` | required | One or more `.gv`/`.dot` files or directories. Directories are globbed (non-recursive). |
| `-o, --outdir PATH` | `data/processed` | Output root; the artifact lands in `<outdir>/<stem>/`. |
| `--glob PATTERN` | `*.gv` | Pattern for directory inputs. `.dot` files are included too when the default is used. |
| `--min-pre FLOAT` | none | Drop pairs whose `pre_z` is below this value. |
| `--min-post FLOAT` | none | Drop pairs whose `post_z` is below this value. |
| `--top-k INT` | none | Keep only the top-k pairs ranked by `pre_z + post_z`. |
| `--allow-self-loops` | off | Allow hubs of the form `"A--A"`; self-loops become warnings instead of errors and are kept in the collapsed pair list. |
| `--strict` | off | Escalate validation warnings to errors (non-zero exit). |
| `--dry-run` | off | Validate and report without writing any artifact. |
| `--log-level LEVEL` | `INFO` | Logging verbosity. |

### Filtered inputs

Any of `--min-pre`, `--min-post` or `--top-k` makes the run *filtered*, and the artifact is
then named `parsed_graph.filtered.json` so it can never overwrite the canonical
`parsed_graph.json`. Phase 02 consumes a filtered artifact only with `--include-filtered`
(and names its own output `z_matrix.filtered.json`).

Artifacts: `parsed_graph.json` (blocks `provenance`, `source_file`, `file_sha256`,
`neurons`, `pairs`, `raw_edges`, `metadata`). No sidecar, no CSV.

### Self-loops

A hub named `"A--A"` encodes the pair `(A, A)`. Self-loops violate the hub semantics assumed
by the rest of the pipeline, so they are a validation **error** by default:

```text
A -> "A--A" [label="Zscore = 0.5"]
"A--A" -> A [label="Zscore = 0.5"]
```

Real neural datasets may encode auto-connections, so `--allow-self-loops` downgrades the
`self_loop` finding to a **warning** and keeps the pair in `pairs` (degrees, strengths,
density and `reciprocity` are then derived from it like any other pair):

```bash
python -m src.parsing.parse_graphviz \
  -i data/raw_dot/with_autapses.gv \
  -o data/processed --allow-self-loops
```

`metadata.allow_self_loops` records the flag (so the artifact stays self-describing), and
combining `--allow-self-loops` with `--strict` still fails the run, because `--strict`
escalates every warning back to an error. A kept self-loop counts once towards
`metadata.reciprocity` (it is its own reverse) and contributes its `pre_z` / `post_z` to that
neuron's out/in degree and pre/post strength. Phase 02 consumes such an artifact only when it
is also given `--allow-self-loops` (see [Phase 02 — Self-loops](#self-loops-1)).

## Phase 02 — unified-weight Z matrix

Phase 02 is the only owner of the pre/post → unified-weight computation and of the square
matrix assembly:

```text
s = (|pre_z| + |post_z|) / 2
g = tanh(alpha * log(post_z / (pre_z + eps)))
w = s * g
```

with the default `zero-in-zero-out` policy mapping any pair whose `pre_z` or `post_z` is
exactly `0` to a weight of `0` (a zero z-score means "no measurable relationship"). The
fixed pipeline order is `fuse → assemble Z → normalize → symmetrize`.

```bash
python -m src.matrices.build_square_matrix \
  -i data/processed/FB4Yaffect_FB45_999prePost_001_all/parsed_graph.json \
  -o data/processed --stats
```

On the reference data this yields a `113 x 113` matrix with 1350 stored non-zero entries
(sparsity `0.894275`), 302 positive / 1048 negative / 5 exact-zero weights,
`density = 0.1070638432364096`, `reciprocity = 574`, `frobenius_norm = 8.005821952321897`,
`largest_singular_value = 4.441772842851828` and `config_hash = c1623ccb`. Stored weights are
predominantly negative (1048 of 1350), which is why the Phase 04 reference pathway is
entirely negative.

### Unified-weight options

| Option | Default | Description |
|---|---:|---|
| `--eps FLOAT` | `0.1` | Ratio stabilizer in `log(post_z / (pre_z + eps))`. |
| `--alpha FLOAT` | `1.0` | Gain sensitivity of the `tanh` term. |
| `--zero-policy {zero-in-zero-out,formula}` | `zero-in-zero-out` | How a zero `pre_z`/`post_z` is treated. |

### Matrix structure options

| Option | Default | Description |
|---|---:|---|
| `--symmetric` | off | Make the effective matrix `Z_sym = (Z + Z.T) / 2` (real eigenvalues, orthogonal eigenvectors). |
| `--symmetrize {mean,sum,max-abs,min-abs}` | `mean` | Rule used by `--symmetric` and for the always-present `matrix_symmetric` key. |
| `--normalize {none,rows,cols,unit,spectral,zscore-nonzero}` | `none` | Row/col L1, max-abs, largest-singular-value, or in-place standardisation of stored weights. |
| `--allow-self-loops` | off | Allow pairs of the form `(A, A)`: self-loops become warnings instead of errors and their unified weight is written to the matrix diagonal. |

`--symmetric` is what makes Phase 03's **eigen** path valid, and `matrix_symmetric` is always
written (as `(Z + Z.T)/2` at the default `mean`), so Phase 03's `--source symmetric` always
has a symmetric matrix available.

### Self-loops

A Phase 01 artifact built from a graph with auto-connections (hub `"A--A"`, produced with
Phase 01's own `--allow-self-loops`) carries self-loop pairs `(A, A)`. Phase 02 forbids these
by default — the pair is reported as a `self_loop` **error** and the run aborts. To include
them, pass `--allow-self-loops`:

```bash
python -m src.matrices.build_square_matrix \
  -i data/processed/with_autapses/parsed_graph.json \
  -o data/processed --allow-self-loops
```

With the flag set, each self-loop becomes a `self_loop` **warning**, its unified weight is
computed by the usual rule and written to the diagonal `Z[i, i]` (the diagonal is no longer
forced to zero), and the pair appears in `pairs`. `metadata.n_self_loops` counts the populated
diagonal entries and `config.allow_self_loops` records the flag, so the artifact stays
self-describing; the flag is part of the configuration hash, so the run writes
`z_matrix.<config_hash8>.json` and never clobbers the canonical `z_matrix.json`. Symmetrization
preserves the diagonal for every `--symmetrize` rule (the diagonal is its own transpose), and
the diagonal entry participates in `matrix_stats`, the row/column norms and every `--normalize`
mode — so the spectral radius Phase 03 sees may change. Combining `--allow-self-loops` with
`--strict` still fails the run, because `--strict` escalates every warning back to an error
(nothing is written).

### Input and output options

| Option | Description |
|---|---|
| `-i, --input PATH ...` | One or more `parsed_graph.json` files or directories. Directories are searched recursively. |
| `-o, --outdir PATH` | Output root; default `data/processed`. |
| `--glob PATTERN` | Pattern for directory inputs; default `parsed_graph.json`. |
| `--include-filtered` | Also process Phase 01 `parsed_graph.filtered.json` artifacts. |
| `--plot` | Write `z_matrix.png` and show the heatmap popup. |
| `--interactive` | Open a window of the effective matrix with per-cell hover tooltips (source, target, weight, `(i, j)`); needs mplcursors and a GUI backend; writes no artifact. |
| `--no-popup` | With `--plot`: save the PNG without opening the GUI window. |
| `--cmap NAME` | Heatmap colormap; default `viridis`. |
| `--save-data` | Write `z_matrix.data.json` (per-pair weights, histogram, row/col norms, statistics). |
| `--stats` | Print the Z statistics to the terminal. |
| `--hist-bins INT` | Histogram bin count for `--save-data`; default `10`. |
| `--config-hash` | Always include the eight-character configuration hash in filenames. |
| `--no-sidecar` | Do not write `z_matrix.npz`; the JSON stays canonical and `load_z_matrix` still works. |
| `--strict` | Fail on any validation issue (JSON and NPZ writing). |
| `--dry-run` | Validate, print statistics and the summary box without writing anything. |
| `--log-level LEVEL` | Logging verbosity; default `INFO`. |

Exit status is `0` for success, `1` for a failed input or write, and `2` when no input
matches.

### Artifacts

| File | Contents |
|---|---|
| `z_matrix.json` | Canonical, self-sufficient: `provenance`, `config`, `neuron_order`, `pairs`, `matrix`, `matrix_symmetric`, `metadata`. |
| `z_matrix.npz` | Verified array cache: `matrix`, `matrix_symmetric`, `neuron_order`, `edge_rows`, `edge_cols`, `edge_weights`, `source_json_sha256`, `numpy_version`. |
| `z_matrix.png` | Heatmap (only with `--plot`). |
| `z_matrix.data.json` | Per-pair weights, histogram, norms and statistics (only with `--save-data`). |

A non-default matrix config inserts `.<config_hash8>` (`z_matrix.<config_hash8>.json`) and a
filtered input adds `.filtered`, so a variant run can never clobber the canonical artifact;
`--config-hash` forces the hash segment even for a default config.

## Phase 03 — spectral decomposition

Phase 03 extracts the latent functional modes of the Phase 02 `Z` matrix. It owns exactly two
decompositions and nothing else:

- **eigen** — a true eigen-decomposition of a *symmetric* matrix (`scipy.linalg.eigh`): real
  eigenvalues, orthonormal eigenvectors. Valid for `Z_sym = (Z + Z.T) / 2`, because a general
  directed `Z` has a complex spectrum.
- **svd** — a singular value decomposition of the *directed* matrix (`scipy.linalg.svd`,
  `gesdd`): non-negative singular values plus the two orthonormal loading bases `U` (source /
  sender side) and `V` (target / receiver side).

`--method auto` (the default) picks eigen when the selected source matrix is numerically
symmetric and SVD otherwise, so a default Phase 02 artifact decomposes with SVD and a
`--symmetric` one with eigen, with no extra flags. `--method eigen` on an asymmetric source is
a **hard error**.

```bash
python -m src.spectral.spectral_decomposition \
  -i data/processed/FB4Yaffect_FB45_999prePost_001_all/z_matrix.json \
  -o data/processed --stats --save-data
```

On the reference data this reports `SVD neurons=113 values=113 retained=21
sparsity-free top|v|=4.441773 cum@k=0.896063 config_hash=c5983254`, and logs that the directed
matrix has 4 numerical-null singular values (effective rank 109 of 113) plus one degenerate
spectral cluster of size 4.

Explained variance is defined **identically for both paths** as `value**2 / sum(value**2)` —
for SVD that is the classical `sigma**2 / sum(sigma**2)`, for a symmetric matrix
`lambda**2 / sum(lambda**2)`. That is what makes the two paths comparable and gives the exact
invariant `sum(sigma**2) == sum(lambda**2) == ||A||_F**2`.

### Input options

| Option | Default | Description |
|---|---:|---|
| `-i, --input PATH ...` | required | One or more `z_matrix.json` files or directories. Directories are searched recursively. |
| `-o, --outdir PATH` | `data/processed` | Output root; artifacts land in `<outdir>/<stem>/`. |
| `--glob PATTERN` | `z_matrix.json` | Glob used for directory inputs. |
| `--include-variants` | off | Also process `z_matrix.<config_hash8>.json` inputs. |

### Decomposition options

| Option | Default | Description |
|---|---:|---|
| `--method {auto,eigen,svd}` | `auto` | `auto` picks eigen for a symmetric source and SVD for a directed one. |
| `--source {effective,symmetric}` | `effective` | `effective` uses Phase 02's matrix (`Z_sym` in symmetric mode, else the directed `Z`); `symmetric` always uses `matrix_symmetric`. |
| `--k auto\|0\|N` | `auto` | Modes to retain: `auto` (heuristic median), `0`/`all` (every mode), or an integer. |
| `--rank-by {magnitude,value}` | `magnitude` | Rank modes by \|value\| or by the signed value. |
| `--sign-convention {max-abs-positive,none}` | `max-abs-positive` | Fixes the arbitrary eigenvector sign for byte-reproducible artifacts. |
| `--driver {evr,evd,ev,evx}` | `evr` | LAPACK driver for `scipy.linalg.eigh`. |
| `--residual-tol FLOAT` | `1e-09` | Relative reconstruction tolerance for a full (`k == N`) decomposition. |

Artifacts are byte-identical across runs: eigenvector signs are pinned by
`--sign-convention max-abs-positive`, the heuristics are a pure function of the spectrum, and
`created_utc` honours `SOURCE_DATE_EPOCH`.

### Dimensionality heuristics

All three estimators are advisory: they never drop a mode by themselves. Only `--k auto`
follows the recommendation, which is the **median of the enabled heuristic k values, clipped
to `[1, N]`** (the `heuristics.rule` string in the artifact says exactly that).

| Option | Default | Description |
|---|---:|---|
| `--variance-threshold FLOAT` | `0.9` | Cumulative explained-variance target. |
| `--elbow` / `--no-elbow` | on | Estimate k from the elbow of the variance curve. |
| `--elbow-curve {cumulative,scree}` | `cumulative` | Curve the elbow is detected on. |
| `--elbow-method {l-method,second-difference}` | `l-method` | Knee detector (self-contained, no `kneed` dependency). |
| `--spectral-gap` / `--no-spectral-gap` | on | Estimate k from the largest leading spectral gap. |
| `--gap-window INT` | `25` | Leading components searched for the gap. |
| `--gap-metric {ratio,gap}` | `ratio` | Spectral-gap criterion. |

On the reference spectrum the estimates are 22 (variance), 21 (elbow) and 1 (spectral gap), so
`recommended_k = 21`. The gap search is deliberately restricted to the leading `gap_window`
components: an unrestricted `argmax` lands on the numerical-null tail (it returns `k = 109`
for the reference spectrum, where four singular values are ~1e-17 apart), which is a
measurement artefact rather than a mode boundary.

### Output options

| Option | Description |
|---|---|
| `--config-hash` | Always include `<config_hash8>` in the artifact filenames. |
| `--no-sidecar` | Do not write `eigen.npz`; the JSON stays canonical and `load_spectrum` still works. |
| `--plot` | Write `eigen.png` (scree / cumulative variance / spectrum) and show its popup. |
| `--no-popup` | With `--plot`: save the PNG without opening the GUI window. |
| `--cmap NAME` | Plot colormap; default `viridis`. |
| `--plot-modes INT` | Single-mode heatmap panels in `eigen.modes.png`, `0` disables; default `4`. |
| `--save-data` | Write `eigen.data.json` (per-mode loading stats, residuals, energy shares). |
| `--stats` | Print the spectral statistics to the terminal. |

Global options: `--strict`, `--dry-run`, `--log-level LEVEL` (default `INFO`). Exit status is
`0` for success, `1` when any input failed validation/decomposition or an artifact could not be
written, and `2` when no input file matched.

### Artifacts

| File | Contents |
|---|---|
| `eigen.json` | Canonical: `provenance`, `config`, `neuron_order`, `values` (all N ranked entries), `modes` (retained loading vectors), `heuristics`, `metadata`. |
| `eigen.npz` | Verified array cache: `values`, `loadings_left`, `loadings_right`, `explained_variance_ratio`, `rank_indices`, `neuron_order`, `source_json_sha256`, `numpy_version`. |
| `eigen.png` | Scree + cumulative variance + spectrum figure (only with `--plot`). |
| `eigen.modes.png` | Loading-matrix and per-mode contribution heatmaps (only with `--plot --plot-modes > 0`). |
| `eigen.data.json` | Per-mode loading statistics, residuals and energy shares (only with `--save-data`). |

The artifact stem **inherits the Phase 02 variant suffix** (`z_matrix.json` → `eigen.json`;
`z_matrix.f8652585.json` → `eigen.f8652585.json`) and adds `.<config_hash8>` for any non-default
Phase 03 config, so a variant run can never clobber the canonical `eigen.json`. `values` always
carries **every** mode (all `N`), while `modes` carries the retained loading vectors (`k`), so
Phase 04 always has a ready `(N x k)` loading matrix.

### Reference numbers

With the defaults on the canonical reference dataset (SVD of the directed `Z`), the expected
output is:

| Quantity | Value |
|---|---:|
| Neurons / spectrum values / modes retained | `113 / 113 / 21` |
| Resolved method / source | `svd / effective` |
| Heuristic k: variance / elbow / spectral gap | `22 / 21 / 1` (`recommended_k = 21`) |
| Explained variance top-1 | `0.30782283556009055` |
| Cumulative variance at k | `0.8960630436858558` |
| Frobenius norm / `sum(value²)` | `8.005821952321897 / 64.09318513227922` |
| Participation ratio | `35.6323231094758` |
| Reconstruction error (relative, retained modes) | `0.3223925500289118` |
| Orthogonality error | `1.7763568394002505e-15` |
| Max mode residual | `1.887379141862766e-15` |
| Numerical rank (SVD only) / condition number | `109 / 409126.97892414214` |
| Degenerate clusters | `1` cluster of size `4` |
| `config_hash` | `c5983254` |

The `sum(value²) == frobenius_norm**2` identity holds exactly here
(`8.005821952321897² = 64.09318513227922`), and the orthogonality and per-mode residuals are at
machine precision — the checks Phase 03 reports as `orthogonality_error` and `max_mode_residual`.

## Input requirements

The motif command consumes an `eigen.json` file produced by the Phase 03 spectral
decomposition command (see [Phase 03 — spectral decomposition](#phase-03--spectral-decomposition)),
which in turn consumes the Phase 02 `z_matrix.json`. It must contain:

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

This assumes `eigen.json` already exists. To produce the full chain from the raw Graphviz
file, run the four commands in [End-to-end (Phases 01 → 04)](#end-to-end-phases-01--04).

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

`--pathway-weight-rule` then transforms `W` into a filter weight (used for the threshold and the
top-N ranking) and a visualisation weight (the stored edge `weight`):

| Rule | Filter weight | Visualisation weight | Needs Phase 02 `Z`? |
|---|---|---|---|
| `signed` (default) | `W` | `W` | inherits `--pathway-source` |
| `abs` | `\|W\|` | `W` | inherits |
| `positive` | `max(W, 0)` | `W` | inherits |
| `negative` | `min(W, 0)` | `W` | yes (via the source) |
| `mode-only` | `W_mode` | `W_mode` | no |
| `matrix-only` | `W_matrix` | `W_matrix` | yes |
| `hybrid` | `α·W_mode + (1−α)·W_matrix` | same | yes |

Any rule other than `signed` **overrides** `--pathway-source` (an informational log line records
it), and the effective choice is written to `pathway.source` and to the GraphML graph element.
**`abs` is mathematically a no-op** on the kept edge set, because the threshold already uses
`|W|`; it is provided for explicitness and for the ranking magnitude. `positive`/`negative`
threshold against the strongest *surviving* contribution, so they are not subsets of `signed`.

`--pathway-polarity-rule` sets the edge `polarity` (and therefore the GraphML `edge_color`):

| Rule | Polarity |
|---|---|
| `sign` (default) | `sign(W_viz)` |
| `pre_z` | `sign(Σ_{i∈A} Σ_m P̃[i,m])` — the source group's signed participation |
| `post_z` | `sign(Σ_{j∈B} Σ_m P̃[j,m])` — the target group's signed participation |
| `delta` | `sign(W_mode − W_matrix)` (needs the Phase 02 matrix) |

`P̃` is the **signed** participation (the participation matrix carries no sign), so `pre_z`/`post_z`
are group-level rules: an edge inherits its source's (resp. target's) polarity, and a whole
row/column of the graph shares one colour.

Edges are kept when `|filter weight| >= --pathway-edge-threshold × abs_max` (the threshold is a
**fraction of the strongest contribution**, so the stored weights stay on the Phase 02
scale), then at most `--pathway-top-edges` are kept per source group. `--no-intra` drops
self-edges. Each edge records its weight, magnitude, polarity, intra flag, contributing
modes and the top five per-mode contributions; each node records its group metadata and its
strongest signed-participation modes.

The graph is exported with **NetworkX** to `motifs.pathway.graphml` for interactive
exploration in yEd (custom layouts, recoloring, metadata inspection, SVG/PDF export). Node
and edge attributes carry the full metadata; every edge also carries a human-readable
`label` (for example `G04 → G01: -2.3453`) and a yFiles `edgegraphics` style block that
maps `abs_weight` to line width and `polarity` to colour, and every node carries a yFiles
`nodegraphics` block whose multi-line label lists the group ID and its region
composition (each `cell_type (count)`), with a role-based fill, border and shape, so both
weights and node identities are visible on import.
The graph element records `n_nodes`, `n_edges`, `abs_max`, `weight_normalization`, `edge_style`,
`pathway_source`, `pathway_weight_rule` (the *mode* rule), `pathway_unified_weight_rule`,
`pathway_polarity_rule`, `pathway_hybrid_alpha` (hybrid only), `threshold` and `topN`. Use
`--no-graphml` to skip the artifact.

`--pathway-source matrix` and `both` enrich the contributions with the raw Phase 02
`z_matrix[.<variant>].json` sibling (`z_contribution` on every GraphML edge); a missing or
mismatched matrix artifact is an error, because the requested source cannot be honoured.

With the defaults on the reference dataset the pathway has 22 nodes and 39 edges,
`abs_max = 2.345300`, 0 positive / 39 negative edges, 7 intra / 32 cross edges and
`weight_concentration = 0.055842`. Every kept edge is negative at the defaults — a direct
consequence of Phase 02's documented negative-weight bias (77 % of unified weights are
negative).

### Pathway ranking (S-score)

Comparing many runs by eye does not scale, so `src/clustering/motif_analysis.py` reduces every
pathway artifact of a run directory to a single scalar S-score:

```text
S = w_B · B + w_H · H + w_D · D − w_C · C          # all weights default to 1
```

| Component | Definition | Meaning |
|---|---|---|
| `B` polarity balance | `1 − \|n_pos − n_neg\| / n_edges` | 0 = one sign only, 1 = perfectly balanced |
| `H` polarity entropy | `−Σ_s p_s · ln p_s`, `s ∈ {−1, 0, +1}` | 0 = one polarity class, max `ln 3 ≈ 1.0986` |
| `D` mode–matrix disagreement | `#(sign(W_mode) ≠ sign(W_matrix)) / n_edges` | 0 = mode and matrix agree everywhere |
| `C` concentration | Phase 04D's `pathway.weight_concentration` | lower = more distributed influence |

A component that is not defined for a run is reported as `null`, its term is **dropped** from
`S` (never replaced by zero), and the row records why in `d_unavailable_reason`:

| `pathway_weight_rule` | `pathway.source` | `D` |
|---|---|---|
| `signed`/`abs`/`positive`/`negative` | `mode` | `null` (`mode-only: no matrix contribution`) |
| `signed`/`abs`/`positive`/`negative` | `both` | `sign(W − W_matrix)` vs `sign(W_matrix)` |
| `mode-only` | `mode` | `null` |
| `matrix-only` | `matrix` | `null` (`matrix-only: no mode contribution stored`) |
| `hybrid`, `α > 0` | `hybrid` | `sign((W − (1−α)·W_matrix)/α)` vs `sign(W_matrix)` |
| `hybrid`, `α = 0` | `matrix` | `null` |

`W_matrix` is rebuilt exactly as 04D builds it (`Σ_{i∈A} Σ_{j∈B} Z[i,j]`) from the sibling
Phase 02 `z_matrix[.<variant>].json`; a missing sibling matrix is a **warning** that leaves `D`
undefined (so the other three components still score the row), while a matrix that exists but
does not match the artifact's neuron order is an **error**. An edge-free pathway (for example
`--grouping none`) receives no score at all.

`S` is not normalised: it is unbounded above and **negative on the canonical reference run**,
because that pathway is entirely negative (`B = H = 0`, so `S = −C = −0.055842`). That is the
expected consequence of the definition, not a defect — compare scores only at equal weights.

The scorer is read-only: it never rewrites `motifs.json`, `motifs.npz`, `motifs.data.json` or
`motifs.pathway.graphml`, it never touches a Phase 01–03 artifact, and it does not change the
Phase 04 `config_hash`. It writes `motif_analysis_summary.json` into the same run directory
(plus `motif_analysis_summary.csv` **only** with `--csv` — this is the single, opt-in exception
to the project's no-CSV rule).

```bash
# rank every motif artifact of one run directory
python -m src.clustering.motif_analysis -i data/processed/FB4Yaffect_FB45_999prePost_001_all --stats

# a processed tree (recursive) with an explicit weight profile, writing the CSV too
python -m src.clustering.motif_analysis -i data/processed --w-balance 2 --w-concentration 0.5 --csv

# also score the motifs.data.json duplicates (skipped by default) and write nothing
python -m src.clustering.motif_analysis -i outdir/<stem> --include-data --dry-run
```

Options: `-i/--input` (repeatable, a directory or a single artifact), `--include-data`,
`--w-balance`, `--w-entropy`, `--w-disagreement`, `--w-concentration`, `--csv`, `--stats`,
`--dry-run`, `--strict`, `--log-level`. Exit codes are `0` success, `1` a failed artifact
(including a strict-mode warning), `2` no motif artifact matched.

The terminal output is a ranking table (always) plus, with `--stats`, the S-score statistics
block:

```text
Phase 04B -- S-score statistics
  directory           : data/processed/FB4Yaffect_FB45_999prePost_001_all
  artifacts scored    : 1 of 1 (skipped 0)
  S max / min / mean  : -0.055842 / -0.055842 / -0.055842
  best                : motifs.json
  weights             : B 1 / H 1 / D 1 / -C 1  (score_hash 7835a30e)

Phase 04B -- pathway ranking (S descending)
rank  artifact                           edges         B         H         D         C          S
-------------------------------------------------------------------------------------------------
   1  motifs.json                           39  0.000000  0.000000       n/a  0.055842  -0.055842
```

The summary JSON contains `provenance` (phase, scorer version, source directory, artifact
counts, `created_utc`), `scorer_config` (the four weights, the definition, `score_hash`),
`summary` (counts, `s_max`/`s_min`/`s_mean`, best/worst artifact), `rankings` (one record per
artifact, sorted by `S` descending with ties broken on the artifact name) and `metadata`
(issues with severities). It honours `SOURCE_DATE_EPOCH`, so repeated runs are byte-identical.

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
| `--pathway-weight {evr,energy,uniform,value,abs_value}` | `evr` | Mode weighting rule (`evr` and `energy` are equivalent). Alias: `--pathway-mode-weights`. |
| `--pathway-weight-rule {signed,abs,positive,negative,mode-only,matrix-only,hybrid}` | `signed` | Transform of the aggregated contribution. Any rule other than `signed` overrides `--pathway-source`. |
| `--pathway-polarity-rule {sign,post_z,pre_z,delta}` | `sign` | Edge polarity (and GraphML colour). `pre_z`/`post_z` are group-level; `delta` needs the Phase 02 matrix. |
| `--pathway-hybrid-alpha FLOAT` | `0.5` | Blend weight of `W_mode` for `--pathway-weight-rule hybrid`. |
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

### End-to-end (Phases 01 → 04)

The complete chain from the raw Graphviz file to the pathway graph. Every stage is a separate
command; each reads the previous stage's artifact from the same
`data/processed/FB4Yaffect_FB45_999prePost_001_all/` directory.

```bash
# Phase 01 -- parse the raw Graphviz file into parsed_graph.json
python -m src.parsing.parse_graphviz \
  -i data/raw_dot/FB4Yaffect_FB45_999prePost_001_all.gv \
  -o data/processed

# Phase 02 -- build the unified-weight square matrix Z (required: Phase 03 only loads it)
python -m src.matrices.build_square_matrix \
  -i data/processed/FB4Yaffect_FB45_999prePost_001_all/parsed_graph.json \
  -o data/processed --stats

# Phase 03 -- eigen/SVD decomposition of Z into per-mode loadings
python -m src.spectral.spectral_decomposition \
  -i data/processed/FB4Yaffect_FB45_999prePost_001_all/z_matrix.json \
  -o data/processed --stats

# Phase 04 -- motifs, families, groups and the GraphML pathway graph
python -m src.clustering.functional_motifs \
  -i data/processed/FB4Yaffect_FB45_999prePost_001_all/eigen.json \
  -o data/processed --stats --save-data
```

`--plot` and `--interactive` are the only flags that open a GUI window; on a headless machine
add `--no-popup` (Phase 02/03) or simply omit them. Phases 01–03 write no CSV.

The whole tree in one pass — Phase 01/02 glob directories (non-recursively and recursively
respectively) while Phase 03/04 search recursively for their own artifact names, so passing
the processed root never re-consumes a downstream output:

```bash
python -m src.parsing.parse_graphviz -i data/raw_dot -o data/processed
python -m src.matrices.build_square_matrix -i data/processed -o data/processed
python -m src.spectral.spectral_decomposition -i data/processed -o data/processed
python -m src.clustering.functional_motifs -i data/processed -o data/processed
```

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

# configurable unified weight: 60/40 mode-matrix blend with a delta polarity (needs z_matrix.json)
python -m src.clustering.functional_motifs \
  -i eigen.json -o data/processed \
  --pathway-weight-rule hybrid --pathway-hybrid-alpha 0.6 \
  --pathway-polarity-rule delta --stats

# only the negative contributions, coloured by the receiver group's polarity
python -m src.clustering.functional_motifs \
  -i eigen.json -o data/processed \
  --pathway-weight-rule negative --pathway-polarity-rule post_z --stats

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

### The Whole Enchilada!

```bash
python -m src.clustering.functional_motifs \
  -i data/processed/FB4Yaffect_FB45_999prePost_001_all/eigen.json  \
  -o outdir \
  \
  --participation rms \
  --threshold-method relative \
  --relative-threshold 0.25 \
  --absolute-threshold 0.05 \
  --quantile 0.8 \
  --participation-threshold 0.1 \
  --min-members 3 \
  --max-members 40 \
  \
  --family-jaccard 0.2 \
  --family-polarity 0.5 \
  --link-min-jaccard 0.1 \
  \
  --grouping motif \
  --polarity-split \
  --polarity-min-members 3 \
  --max-group-size 12 \
  --n-groups auto \
  --linkage average \
  --affinity cosine \
  --merge-threshold 0.9 \
  \
  --pathway-source mode \
  --pathway-weight evr \
  --pathway-weight-rule signed \
  --pathway-polarity-rule sign \
  --pathway-hybrid-alpha 0.5 \
  --pathway-edge-threshold 0.1 \
  --pathway-top-edges 5 \
  --no-intra \
  --graphml \
  \
  --save-data \
  --stats \
  --config-hash
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
`source_group`, `target_group`, `label`, `weight`, `abs_weight`, `polarity`,
`contribution_by_mode` (JSON), `threshold_flag`, `topN_flag`, `intra_flag`, `weight_ratio`,
`edge_width`, `edge_color` and, with `--pathway-source matrix|both`, `z_contribution`.

Each edge also carries a yFiles `edgegraphics` visual-properties block:

- the `label` text (for example `G04 → G01: -2.3453`) is rendered as the edge label;
- the line width ranges from `1.0` to `5.0` as `1.0 + 4.0 × abs_weight / abs_max`;
- negative edges are red (`#C62828`), positive edges blue (`#1565C0`), zero-polarity edges
  gray (`#757575`), with matching arrows.

Each node also carries a yFiles `nodegraphics` block (`<y:ShapeNode>`). Its
`<y:NodeLabel>` is **multi-line**: the first line is the node's `label` (the group ID), and
each following line is one `cell_type (count)` entry of the node's `region_composition`,
ordered by descending count then cell type, for example:

```text
G14
hDeltaA (3)
FB4Z (1)
```

A group with an empty composition keeps a single-line label. The fill, border and shape are
chosen by role:

| Role | Fill | Border | Shape |
|---|---|---|---|
| background (`is_background`) | `#EEEEEE` | `#9E9E9E` | `rectangle` |
| singleton (`is_singleton`) | `#FFF3E0` | `#EF6C00` | `ellipse` |
| default | `#E8EEF7` | `#37474F` | `roundrectangle` |

The node realizer's `<y:Geometry>` is **auto-sized from the label text** and placed on a
deterministic grid (a placeholder that yEd's own layouts replace on import). The box is

```text
width  = max(80, longest_line_chars × 7  + 2 × 10)
height = max(30, line_count × 12 × 1.2   + 2 × 7)
```

so a single-line label keeps the legacy `80 × 30` box while multi-line or long labels grow
instead of overflowing. The grid step tracks the largest box in the diagram, so nodes cannot
overlap. The sizing is a pure function of the label text (no font metrics), keeping the export
byte-identical under a fixed `SOURCE_DATE_EPOCH`.

The graph element records `n_nodes`, `n_edges`, `abs_max`, `weight_normalization`,
`edge_style` (the width formula and colour palette), `pathway_source`, `pathway_weight_rule`
(the *mode* weighting rule), `pathway_unified_weight_rule`, `pathway_polarity_rule`,
`pathway_hybrid_alpha` (only for `hybrid`), `threshold` and `topN`. Attributes that do not apply
(a background group's dominant mode, a
singleton's coherence, `z_contribution` for mode-only contributions) are omitted rather than
emitted as null. These GraphML-only fields never enter the JSON payload, the NPZ arrays or
the configuration hash.

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
| Graphviz parsing (Phase 01) | ✔ Implemented | `parsed_graph.json` |
| Unified-weight matrix (Phase 02) | ✔ Implemented | `z_matrix.json`, `z_matrix.npz` |
| Spectral decomposition (Phase 03) | ✔ Implemented | `eigen.json`, `eigen.npz` |
| Motif extraction | ✔ Implemented | `motifs.json`, `motifs.npz`, optional `motifs.data.json` |
| Cross-mode tracking | ✔ Implemented | links, families, link matrices |
| Functional neuron grouping | ✔ Implemented | groups, group arrays, coherence statistics |
| Pathway graph (GraphML) | ✔ Implemented | `motifs.pathway.graphml`, pathway block, adjacency arrays |
| Pathway ranking (S-score) | ✔ Implemented | `motif_analysis_summary.json` (+ `.csv` with `--csv`) |

The pipeline is four separate commands, run in order (each reads the previous stage's JSON
artifact and nothing is chained automatically):

| Stage | Module | Output |
|---|---|---|
| 01 | `src/parsing/parse_graphviz.py` (`python -m src.parsing.parse_graphviz`) | `parsed_graph.json` |
| 02 | `src/matrices/build_square_matrix.py` (`python -m src.matrices.build_square_matrix`) | `z_matrix.json` |
| 03 | `src/spectral/spectral_decomposition.py` (`python -m src.spectral.spectral_decomposition`) | `eigen.json` |
| 04 | `src/clustering/functional_motifs.py` (`python -m src.clustering.functional_motifs`) | `motifs.json`, `motifs.pathway.graphml` |

The read-only scorer is `src/clustering/motif_analysis.py`
(`python -m src.clustering.motif_analysis`).

**No CSV files are produced by the pipeline.** The only CSV in the project is the optional
`motif_analysis_summary.csv`, written exclusively when `--csv` is passed to the scorer.

## License

MIT License.