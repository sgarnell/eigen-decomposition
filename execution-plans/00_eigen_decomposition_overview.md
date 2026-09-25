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
# Phase 04 — Functional Motifs, Neuron Grouping & Pathway Diagrams

### Objective
Extract functional motifs from spectral mode structure, group neurons by shared participation patterns, and synthesize quantitative pathway diagrams that reveal the core computational architecture of the network.

### Inputs
- Mode loadings matrix (N × k)
- Mode outer‑product heatmaps (k × N × N)
- Eigen statistics JSON (`eigen.json`, `eigen.<variant>.json`, `eigen.<config>.json`)
- Eigenvalue / singular value spectrum
- Explained variance ratios (EVR)
- Recommended k and heuristic metadata
- Anatomical metadata (neuron labels, regions)

### Outputs
- Mode‑specific motifs (subcircuits with polarity and strength)
- Cross‑mode motif tracking (recurrence, polarity flips, strength changes)
- Functional neuron groups (motif‑based + loading‑space clustering)
- Quantitative pathway diagrams (nodes = groups, edges = contributions)
- Motif JSON artifacts (structure, polarity, strength, recurrence)

### Tasks
- Identify strongest motifs per mode using loadings, heatmaps, and EVR
- Track motifs across modes to find stable, polarity‑sensitive, or composite motifs
- Group neurons by motif membership and loading‑vector similarity
- Build pathway diagrams with nodes (functional groups), edges (signed/weighted contributions), and annotations (mode index, eigenvalue, EVR, anatomical region)
- Produce diagram‑ready graph structures for Phase 05

---

# Phase 05 — Reduced Dynamical Models on Motif-Based Diagrams

### Objective
Construct reduced dynamical models using the motif‑based pathway diagrams from Phase 04 as the structural substrate. Each motif becomes a population variable; each diagram edge becomes an effective coupling.

### Inputs
- Pathway diagrams from Phase 04
- Functional groups and motif definitions
- Eigen statistics JSON (mode strengths, polarity, EVR, residuals)
- Mode recurrence and motif metadata

### Outputs
- Reduced ODE models (rate models, Wilson–Cowan, linearized systems)
- Effective coupling matrices derived from motif structure
- Population variables linked to neuron groups
- Model JSON artifacts (equations, parameters, coupling structure)

### Tasks
- Define population activity variables for each motif/group
- Derive effective couplings from motif polarity and strength
- Construct reduced dynamical equations (rate models, WC equations, low‑dimensional ODEs)
- Prepare models for stability and attractor analysis in Phase 06
- Validate models against motif structure and spectral signatures

---

# Phase 06 — Dynamical Structure Analysis of Reduced Models

### Objective
Analyze the dynamical behavior of the reduced models from Phase 05 to classify motifs by their dynamical roles: attractors, oscillators, switches, gain‑control loops, etc.

### Inputs
- Reduced dynamical models from Phase 05
- Effective coupling matrices
- Population variables and motif diagrams
- Eigen statistics JSON (for linking dynamical behavior back to spectral structure)

### Outputs
- Fixed points and equilibrium states
- Jacobians and linear stability analysis
- Stability classification (stable, unstable, saddle, bistable)
- Attractor types (point, limit cycle, multi‑stable, saddle‑node)
- Bifurcation diagrams
- Dynamical motif taxonomy (single, composite, hierarchical)
- Mapping from dynamical behavior → anatomical motif

### Tasks
- Compute fixed points and evaluate Jacobians
- Classify stability and attractor structure
- Identify bifurcations under parameter variation
- Relate dynamical behavior back to motif diagrams and spectral signatures
- Produce dynamical‑motif JSON artifacts for Phase 07 (if needed)
