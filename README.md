# Eigen-Decomposition: Functional Mode Analysis of the Drosophila Fan-Shaped Body

## Overview
**Eigen-Decomposition** is a computational neuroscience project designed to extract
functional structure, latent modes, and dynamical motifs from the **Drosophila
fan-shaped body (FB)** using data-driven spectral methods.

The FB contains rich cross–cell-type interactions that do not map cleanly onto
anatomical wedges or layers. This project provides a principled workflow for
discovering **functional subsystems**, **population variables**, and **reduced
dynamical models** directly from functional connectivity data.

The pipeline operates on Graphviz `.dot` files containing:
- Neuron IDs  
- Directed edges  
- Z-score weights representing functional relationships  

From these, the project constructs a full **Z-matrix**, extracts **functional
modes**, clusters neurons into **functional groups**, and analyzes the **dynamical
behavior** of these reduced subsystems.

---

## Goals
- Build a reproducible computational pipeline for FB functional analysis  
- Identify latent functional axes via spectral decomposition  
- Cluster neurons into population-level functional groups  
- Construct reduced dynamical models for each subsystem  
- Analyze fixed points, stability, attractors, and motif structure  
- Provide a modular codebase for future FB circuit research  

---

## Pipeline Architecture

### 1. Graphviz Parsing
Convert `.dot` files into structured triples:
(source_neuron, target_neuron, z_score_weight)

Code
This step extracts all nodes, edges, and functional weights.

### 2. Z-Matrix Construction
Build a square functional-connectivity matrix **Z** from the parsed pre/post
z-scores:
- Rows = source neurons, Columns = target neurons
- Missing edges = 0, diagonal = 0 (there are no self-loops)
- Optional normalization (`--normalize`) and symmetrization (`--symmetric`)

Every edge also gets a single **unified weight** fused from its pre- and
post-synaptic z-scores.  See **Phase 02** below for the rule, the artifacts and
the full command-line interface.

### 3. Spectral Decomposition
Apply:
- Eigen-decomposition (for symmetric Z)  
- Singular Value Decomposition (SVD) for directed Z  

Extract:
- Leading functional modes  
- Mode loadings for each neuron  
- Cross-type coupling axes  

These modes reveal latent structure not visible anatomically.

### 4. Mode-Based Clustering
Cluster neurons using their mode loading vectors:
- k-means  
- spectral clustering  
- HDBSCAN  
- UMAP embeddings (optional)

Clusters represent **functional groups** or **population variables**.

### 5. Reduced Dynamical Models
For each functional group:
- Define population activity variables  
- Construct reduced rate/Wilson–Cowan equations  
- Parameterize interactions using mode projections  

This yields compact dynamical systems describing subsystem behavior.

### 6. Dynamical Analysis
Compute:
- Fixed points  
- Jacobians  
- Stability  
- Attractor classes  
- Bifurcation structure  

Determine whether each subsystem forms:
- A single dynamical motif  
- A composite motif  
- A multi-attractor functional unit  

---

## Phase 02 — Build Square Z-Matrix (`src/matrices/`)

Phase 02 turns the Phase 01 artifact into the square weight matrix **Z** that
Phases 03–06 consume.  It is the only stage that performs the pre/post →
unified-weight computation; it does no spectral analysis, no clustering and no
pair filtering.

Pipeline order (fixed and recorded in every artifact):

```
fuse (unified weight)  ->  assemble Z  ->  normalize  ->  symmetrize
```

### Inputs

`data/processed/<stem>/parsed_graph.json` — the Phase 01 artifact
(`<stem>` = the source `.gv` stem, e.g. `FB4Yaffect_FB45_999prePost_001_all`).

Relevant fields:

| Field | Role |
|---|---|
| `neurons[].neuron_id` | row/column order (already sorted by `neuron_id`) |
| `pairs[] = {source, target, pre_z, post_z}` | the raw triples Phase 02 fuses — **no `weight` key**, Phase 02 owns it |
| `file_sha256`, `source_file`, `metadata.created_utc` | provenance |
| `metadata.density`, `metadata.reciprocity` | copied verbatim into the Phase 02 artifacts |
| `metadata.filters` | non-empty ⇒ filtered input (output name gets `.filtered`) |

The input is re-validated with Phase 01's own schema checker; `.gv` files are
never re-parsed.

### The unified weight (pre/post → one scalar)

Each edge carries two z-scores: `pre_z` (how strongly the source neuron
participates) and `post_z` (how strongly the target does).  Spectral work needs a
*single* number per edge that keeps directionality, strength and boundedness, so
Phase 02 applies the project's unification rule:

```
s = (|pre_z| + |post_z|) / 2                      # strength
g = tanh(alpha * log(post_z / (pre_z + eps)))     # bounded gain, g in (-1, 1)
w = s * g                                         # the value placed in Z
```

- `g > 0` ⇒ **gain** (the target amplifies the relationship), `g < 0` ⇒
  **attenuation**.
- `g` is bounded, so a tiny `pre_z` can never explode the weight; `eps`
  (`0.1` by default) stabilizes the ratio and `alpha` (`1.0` by default) sets the
  sensitivity.  Both are exposed as `--eps` / `--alpha`.
- Defaults reproduce every worked example in `execution-plans/zscore_context.md`:
  `(0.2, 0.9) → 0.44`, `(0.6, 0.7) → 0.0`, `(0.9, 0.2) → −0.51`,
  `(0.1, 0.5) → 0.22`, `(0.05, 0.08) → −0.036`.
- A zero z-score means "no measurable relationship": with the default
  `--zero-policy zero-in-zero-out`, either z-score being `0` yields exactly
  `w = 0` (the raw formula would instead give the extreme attenuation limit).
  `--zero-policy formula` restores the raw behaviour.

On the reference dataset this yields 1355 weights (302 positive, 1048 negative,
5 exactly zero) with `min = −0.563605`, `max = 0.507692`, `mean = −0.096312` over
the stored entries.  Note the documented `eps` shifts the neutral point to
`post_z = pre_z + eps`, so equal pre/post weights are mildly attenuating.

### Outputs — the JSON + NPZ dual-artifact design

Four artifacts land in `data/processed/<stem>/`:

| Artifact | Written | Purpose |
|---|---|---|
| `z_matrix.json` | always | **canonical, self-sufficient** record: matrix, `neuron_order`, per-pair weights, provenance, config, statistics |
| `z_matrix.npz` | unless `--no-sidecar` | derived NumPy array cache for fast loading by Phases 03+ |
| `z_matrix.png` | `--plot` | heatmap of the effective matrix |
| `z_matrix.data.json` | `--save-data` | per-pair weights, histogram, row/column norms, statistics |

**Why two artifacts.**  `z_matrix.json` is the human-inspectable, byte-reproducible
artifact of record: it holds the provenance, the exact construction recipe
(`config`, including a `config_hash`) and the statistics, and it is the only thing
`load_z_matrix()` treats as authoritative.  `z_matrix.npz` is a **derived cache** —
arrays only, no metadata — whose content is a pure function of the JSON bytes; it is
keyed by that JSON's SHA-256 (`source_json_sha256`) and carries `numpy_version`, so a
stale or hand-edited cache is detected and ignored (with a warning) rather than
trusted.  Nothing in the JSON points at the cache, so the canonical artifact never
depends on it, and `--no-sidecar` runs still work end to end.  Both files are written
atomically (temp file + `os.replace`) and both are byte-identical across runs when
`SOURCE_DATE_EPOCH` is fixed.

The `.npz` bundle (fixed order and little-endian dtypes):

| Key | Meaning |
|---|---|
| `matrix` | the effective matrix — `Z`, or `Z_sym` in symmetric mode |
| `matrix_symmetric` | the symmetrized matrix (always present) |
| `neuron_order` | row/column index order |
| `edge_rows`, `edge_cols`, `edge_weights` | the stored edges, aligned 1:1 with `pairs` |
| `source_json_sha256` | digest of the JSON this cache was built from |
| `numpy_version` | the numpy version that wrote it |

**No CSV files are produced or consumed anywhere in this project.**

### CLI

```bash
python -m src.matrices.build_square_matrix \
  -i data/processed/<stem>/parsed_graph.json \
  -o data/processed \
  [--glob 'parsed_graph.json'] [--include-filtered] \
  [--eps 0.1] [--alpha 1.0] [--zero-policy {zero-in-zero-out,formula}] \
  [--symmetric] [--symmetrize {mean,sum,max-abs,min-abs}] \
  [--normalize {none,rows,cols,unit,spectral,zscore-nonzero}] \
  [--config-hash] [--no-sidecar] \
  [--plot] [--no-popup] [--cmap viridis] [--interactive] \
  [--save-data] [--stats] [--hist-bins 10] \
  [--strict] [--dry-run] [--log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}]
```

| Option | Effect |
|---|---|
| `--plot` | write `z_matrix.png` (heatmap) **and** open a GUI popup window (skipped on a headless backend, or with `--no-popup`) |
| `--interactive` | open a matplotlib window of the effective matrix with per-cell hover tooltips (source neuron, target neuron, unified weight, `(i, j)`); writes **no** artifact; needs `mplcursors` and a GUI backend |
| `--no-popup` | with `--plot`: save the PNG without opening the window (does not affect `--interactive`) |
| `--symmetric` | use `Z_sym = (Z + Zᵀ)/2` instead of `Z` in the JSON, the NPZ and the plot |
| `--normalize` | `rows` (each row ÷ its L1 norm), `cols` (each column ÷ its L1 norm), `unit` (÷ max abs weight), plus `spectral`, `zscore-nonzero`, `none` |
| `--config-hash` | force `<config_hash8>` into the filenames (non-default configs get it automatically) |
| `--no-sidecar` | skip `z_matrix.npz`; the JSON stays canonical and loading still works |
| `--strict` | fail on any validation issue (warnings escalate to errors) |
| `--save-data` | write `z_matrix.data.json` (per-pair weights + statistics) |
| `--stats` | print the Z statistics to the terminal, then the boxed summary |

Exit codes: `0` success, `1` an input failed validation/writing, `2` nothing matched.

### Example commands

```bash
# canonical artifacts: z_matrix.json + z_matrix.npz
python -m src.matrices.build_square_matrix \
  -i data/processed/FB4Yaffect_FB45_999prePost_001_all/parsed_graph.json \
  -o data/processed --stats

# symmetric, row-normalized variant with plot, save-data and hashed filenames
#   -> z_matrix.<config_hash8>.{json,npz,png,data.json}
python -m src.matrices.build_square_matrix \
  -i data/processed/FB4Yaffect_FB45_999prePost_001_all/parsed_graph.json \
  -o data/processed --symmetric --normalize rows --config-hash \
  --plot --save-data --stats

# JSON-only run over a whole processed tree (recursive), strict
python -m src.matrices.build_square_matrix \
  -i data/processed -o data/processed --no-sidecar --strict --include-filtered

# CI-safe: save the heatmap without opening a window, and validate without writing
python -m src.matrices.build_square_matrix -i <artifact> -o data/processed --plot --no-popup
python -m src.matrices.build_square_matrix -i <artifact> -o data/processed --dry-run --stats

# interactive inspection: hover any cell for its neurons, weight and (i, j)
python -m src.matrices.build_square_matrix -i <artifact> -o data/processed --interactive

# PNG on disk *and* the hover-tooltip window (only one window opens)
python -m src.matrices.build_square_matrix -i <artifact> -o data/processed --plot --interactive
```

### The heatmap (`--plot`)

`--plot` renders the effective matrix as a heatmap with a perceptually uniform
colormap (`viridis` by default, `--cmap plasma` for the alternative): x axis =
target neuron index, y axis = source neuron index, with a colorbar labelled
`unified weight w`.  Positive (gain) edges appear warm, negative (attenuation)
edges cold, and the block structure of the functional connectivity stands out
against the zero (absent-edge) background.  For the reference dataset the figure is
saved as a 1093×923 PNG (`z_matrix.png`).  The PNG is always written; the GUI popup
is only shown when the matplotlib backend is interactive (see below).

### Interactive mode (`--interactive`)

`--interactive` opens a real matplotlib window (`plt.show()`) of the **same**
effective matrix and attaches an [`mplcursors`](https://mplcursors.readthedocs.io/)
hover cursor to the heatmap: move the mouse over any cell and a popup appears —
no click required (right-click removes it, `Shift` + arrow keys step one cell at
a time).  The tooltip reports exactly four things:

```
source: ExR7(ring)_L_1
target: ExR7(ring)_L_2
unified weight: -0.102144
(i, j) = (17, 42)
```

Two extra lines keep the tooltip honest:

* a cell with no stored edge adds `stored edge: none (absent)` (and marks the
  weight line `(absent edge)`) — a stored edge whose unified weight is exactly
  `0.0` (`--zero-policy zero-in-zero-out`) is **not** called absent;
* when `--normalize` or `--symmetric` makes the displayed value differ from the
  unified weight of the originating pair, the raw value is added as
  `raw unified weight: …` (or `raw unified weight (reverse direction): …`).

**`--plot` vs `--interactive`.**  They are independent, presentation-only flags:

| | `--plot` | `--interactive` |
|---|---|---|
| writes an artifact | `z_matrix.png` | none |
| opens a window | yes (plain `plt.show()`, skipped when headless or with `--no-popup`) | yes (hover tooltips; skipped only when headless) |
| CI / headless | safe — PNG is written either way | degrades to a warning, nothing is written |
| needs `mplcursors` | no | yes |
| good for | sharing/committing a figure | exploring cell by cell |

Because the tooltip window is the richer one, `--plot --interactive` writes the
PNG **and** suppresses the plain `--plot` popup, so exactly one window opens.
`--interactive` never changes the matrix, the `config_hash`, the artifact names
or any JSON/NPZ content: it is presentation only, and under `--dry-run` it opens
nothing.  With several inputs (`-i data/processed`) one window is opened per
artifact, sequentially.

`mplcursors` is a conda-forge dependency of this project
(`conda install -c conda-forge mplcursors`, already listed in
`environment.yml`); it is imported lazily, so every non-interactive run works
even without it.

### `--stats` / the completion box (terminal) vs the GUI popup

These two outputs are deliberately different things:

* **Terminal only** — `--stats` prints min/max/mean/std, sparsity, positive/negative
  counts, row and column norm summaries, the spectral radius (only in symmetric
  mode) and density/reciprocity; every successful run then prints a boxed summary:

  ```
  +--------------------------------------------------------------+
  | Phase 02 complete                                            |
  | Z matrix shape: 113x113                                      |
  | Unified weights: OK                                          |
  | Symmetric mode: no                                           |
  | Normalization: none                                          |
  | JSON written: z_matrix.json                                  |
  | NPZ written:  z_matrix.npz                                   |
  | Plot:        not requested (--plot)                          |
  | Save-data:   not requested (--save-data)                     |
  | Stats: min/max/mean/std printed above (--stats)              |
  +--------------------------------------------------------------+
  ```

* **GUI windows** — Phase 02 opens exactly two kinds of window, both optional:
  * the `--plot` heatmap popup, shown after the PNG is saved (forced off with
    `--no-popup`, skipped on a non-interactive backend);
  * the `--interactive` hover-tooltip window, which *replaces* the plain popup
    when both flags are given.

  Both are skipped automatically on a non-interactive backend (e.g. headless CI —
  a file-only backend such as `Agg` is recognised as headless, while the desktop
  backends `qtagg`/`tkagg`/`gtk3agg`/`wxagg` are not).  Artifacts are written
  either way, so a batch run never blocks on a window.  A run with
  `--interactive` adds one line to the summary box:

  ```
  | Interactive: window shown (--interactive)         |
  | Interactive: skipped (non-interactive backend)    |
  | Interactive: skipped (--dry-run)                  |
  ```

### How Phase 03 loads the artifacts

```python
from src.matrices.build_square_matrix import load_sidecar_arrays, load_z_matrix

# fast path: arrays only (no JSON parsing, no pickle)
arrays = load_sidecar_arrays("data/processed/<stem>/z_matrix.npz")  # or the .json path
z_directed  = arrays["matrix"]             # effective matrix (Z_sym with --symmetric)
z_symmetric = arrays["matrix_symmetric"]   # always symmetric
order       = arrays["neuron_order"]       # row/column order

# full object: verified against the cache, with config/metadata/pairs
z = load_z_matrix("data/processed/<stem>/z_matrix.json")
z.config                       # symmetric / normalize / eps / alpha / config_hash
z.index                        # neuron_id -> row/column
z.metadata["spectral_radius"]  # sanity checks Phase 03 can cross-validate
```

`load_z_matrix()` verifies the `.npz` against the JSON's SHA-256 and arrays, and
silently falls back to the JSON when the cache is missing or stale, so Phases 03+
can always load *something* correct.

### Symmetric mode and eigen-decomposition

Phase 03 extracts latent functional modes with either an eigen-decomposition or an
SVD.  A symmetric matrix has **real eigenvalues and orthogonal eigenvectors**,
which is exactly what a true eigen-decomposition requires and what makes the modes
interpretable as orthogonal axes of functional variation.  The directed `Z` is not
symmetric (574 of 1355 edges are reciprocal), so its spectrum would be complex —
that path uses SVD instead.

`--symmetric` therefore computes `Z_sym = (Z + Zᵀ)/2` (or `sum` / `max-abs` /
`min-abs` via `--symmetrize`) and puts it in the JSON, the `.npz` and the plot, with
`config.symmetric` and the save-data `symmetric` flag recording the choice.
`matrix_symmetric` is written in *every* run, so Phase 03 can always choose the eigen
path without re-running Phase 02.  On the reference data the symmetric variant has a
spectral radius of 2.433314 (directed `Z` largest singular value 4.441773).

---

## Repository Structure
eigen-decomposition/
│
├── data/
│   ├── raw_dot/          # Original Graphviz files
│   └── processed/        # Parsed triples, Z-matrices, metadata
│
├── src/
│   ├── parsing/          # Graphviz → triples
│   ├── matrices/         # Z-matrix construction
│   ├── spectral/         # SVD, eigen, mode extraction
│   ├── clustering/       # Mode-based clustering
│   ├── dynamics/         # Reduced models, Jacobians, stability
│   └── utils/            # Shared helpers
│
├── tests/                # Unit tests for each module
│
├── environment.yml       # Conda-forge environment specification
└── README.md             # Project overview

Code

---

## Environment
This project uses a **conda-forge only** environment for numerical stability and
dependency consistency.

Key packages include:
- numpy, scipy, numba  
- pandas, networkx, pydot, graphviz  
- scikit-learn, hdbscan, umap-learn  
- sympy, jax, pyDSTool  
- matplotlib, seaborn, mplcursors  

See `environment.yml` for full details.

---

## Status

| Phase | Module | State |
|---|---|---|
| 01 — Parse Graphviz `.gv` files | `src/parsing/parse_graphviz.py` | implemented → `parsed_graph.json` |
| 02 — Build square Z-matrix | `src/matrices/build_square_matrix.py` | implemented → `z_matrix.json` + `z_matrix.npz` (`--plot`, `--interactive`, `--save-data`) |
| 03 — Eigen / SVD decomposition | `src/spectral/` | planned |
| 04 — Mode loadings & clustering | `src/clustering/` | planned |
| 05 — Reduced dynamical models | `src/dynamics/` | planned |
| 06 — Dynamical structure analysis | `src/dynamics/` | planned |

Each phase has an execution plan in `execution-plans/`.  Run the test suite with
`python -m pytest -m "not slow"` (fast) or `python -m pytest` (adds the integration
tests against the real dataset).

---

## License
MIT License (recommended for open scientific work).