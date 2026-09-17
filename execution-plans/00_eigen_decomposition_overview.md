# Eigen-Decomposition Pipeline: Full Phased Execution Plan

This document defines the complete phased architecture for the
Eigen-Decomposition project. Each phase is designed to be implemented
independently, with clear inputs, outputs, and responsibilities.  
Sub‑plans (`01_parse_graphviz.md`, `02_build_square_matrix.md`, etc.)
should be generated directly from the specifications below.

---

## Phase 01 — Parse Graphviz `.gv` Files

### Objective
Convert raw Graphviz `.gv` / `.dot` files into structured data suitable
for matrix construction and spectral analysis.

### Inputs
- Raw `.gv` files containing:
  - Neuron IDs
  - Directed edges
  - Z-score weights

### Outputs
- Parsed triples:
(source_neuron, target_neuron, z_score_weight)

Code
- Node list
- Edge list
- Metadata (counts, connectivity stats)

### Requirements
- Robust parsing using `pydot` or `networkx.drawing.nx_pydot`
- Validation of node/edge consistency
- Handling of missing weights (default = 0)
- Optional filtering (e.g., thresholding low Z-scores)

### Deliverables
- `parse_graphviz.py` module
- `01_parse_graphviz.md` execution plan

---

## Phase 02 — Build Square Z‑Matrix

### Objective
Construct a square functional connectivity matrix **Z** from parsed triples.
This phase performs the **pre/post → unified-weight computation** that converts
raw z-score metadata into the final scalar weight used in the Z-matrix.

For the mathematical definition of Z and the unification rule, see:
`execution-plans/zscore_context.md`.

---

### Inputs
Phase 01 emits raw functional triples:

(source_neuron, target_neuron, pre_z, post_z)

Code

These represent the upstream statistical relationships extracted from the `.gv`
file. Phase 02 consumes these raw values and computes the unified weight.

---

### Unified Weight Computation
Each edge’s final scalar weight is computed using the project’s unification rule:

w = s * tanh( a * log( post_z / (pre_z + ε) ) )

Code


## Phase 03 — Eigen / SVD Decomposition

### Objective
Extract latent functional modes from the Z-matrix using spectral methods.

### Inputs
- Z-matrix from Phase 02

### Outputs
- Eigenvalues / singular values
- Eigenvectors / singular vectors
- Mode ranking (by magnitude or explained variance)
- Diagnostic plots:
- Scree plot
- Eigenvalue spectrum
- Mode heatmaps

### Requirements
- Support for:
- Eigen-decomposition (symmetric Z)
- SVD (directed Z)
- Numerical stability (SciPy)
- Optional dimensionality selection heuristics:
- Elbow method
- Variance threshold
- Spectral gap

### Deliverables
- `spectral_decomposition.py` module
- `03_eigen_decomposition.md` execution plan

---

## Phase 04 — Mode Loadings & Functional Clustering

### Objective
Cluster neurons into functional groups based on their mode loading vectors.

### Inputs
- Mode vectors from Phase 03
- Neuron index map

### Outputs
- Loading matrix `L` of shape `(N × k)`
- Cluster assignments for each neuron
- Cluster centroids
- Visualizations:
- UMAP / PCA embeddings
- Cluster heatmaps
- Mode contribution plots

### Requirements
- Support for multiple clustering algorithms:
- k-means
- spectral clustering
- HDBSCAN
- Optional nonlinear embeddings (UMAP)
- Export cluster metadata for dynamical modeling

### Deliverables
- `mode_clustering.py` module
- `04_mode_loadings.md` execution plan

---

## Phase 05 — Reduced Dynamical Models

### Objective
Construct population-level dynamical models using functional groups
identified in Phase 04.

### Inputs
- Functional clusters
- Mode loadings
- Z-matrix projections

### Outputs
- Reduced system variables:
x_1, x_2, ..., x_m

Code
- Interaction matrix between population variables
- Rate/Wilson–Cowan equations
- Symbolic Jacobians (SymPy)
- Parameter sets for simulation

### Requirements
- Define population activity variables
- Project Z-matrix into reduced space
- Construct differential equations:
dx_i/dt = -x_i + f( Σ_j W_ij x_j + I_i )

Code
- Support for nonlinear activation functions

### Deliverables
- `reduced_models.py` module
- `05_reduced_dynamical_models.md` execution plan

---

## Phase 06 — Dynamical Structure Analysis

### Objective
Analyze the dynamical behavior of reduced subsystems to identify motifs,
attractors, and stability properties.

### Inputs
- Reduced dynamical models from Phase 05

### Outputs
- Fixed points
- Jacobians
- Stability classification
- Attractor types:
- point attractor
- limit cycle
- multi-stable
- saddle-node
- Bifurcation diagrams
- Motif classification:
- single motif
- composite motif
- hierarchical motif

### Requirements
- Symbolic and numeric Jacobian computation
- Stability via eigenvalues of Jacobian
- Phase portrait visualization
- Optional bifurcation analysis (pyDSTool)

### Deliverables
- `dynamics_analysis.py` module
- `06_analyze_dynamical_structures.md` execution plan

---

## Phase Dependencies

Phase 01 → Phase 02 → Phase 03 → Phase 04 → Phase 05 → Phase 06

Code

Each phase produces artifacts consumed by the next.

---

## Repository Integration

All phases map directly to modules in:

src/
parsing/
matrices/
spectral/
clustering/
dynamics/

Code

Execution plans live in:

execution-plans/
00_eigen_decomp_overview.md
01_parse_graphviz.md
02_build_square_matrix.md
03_eigen_decomposition.md
04_mode_loadings.md
05_reduced_dynamical_models.md
06_analyze_dynamical_structures.md

Code

---

## Next Steps

You can now generate each sub-plan using this master blueprint.

01_parse_graphviz.md
02_build_square_matrix.md
03_eigen_decomposition.md
04_mode_loadings.md
05_reduced_dynamical_models.md
06_analyze_dynamical_structures.md

Code

Each sub-plan should expand its phase into:
- goals  
- inputs/outputs  
- algorithms  
- data structures  
- module layout  
- testing strategy  