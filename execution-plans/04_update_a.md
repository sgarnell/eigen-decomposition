
```md
# Phase 04 Update B — Unified Pathway Scoring (S-score)

## Overview

Phase 04D produces multiple motif and pathway artifacts for each run, often with
different configuration hashes. Visual inspection of the `.gv` pathway graphs is
no longer scalable. This update introduces a **single scalar score (S-score)**
that ranks pathway graphs by structural richness, allowing automated comparison
across many runs.

The S-score is computed from established metrics in signed graph theory,
information theory, and multimodal connectomics. Each component is standard; the
composite score provides a simple, objective measure of “interestingness.”

Higher S indicates a more informative, diverse, and scientifically meaningful
pathway graph.

---

# S-score Definition

The S-score combines four metrics:

- **B** — polarity balance  
- **H** — polarity entropy  
- **D** — mode–matrix disagreement  
- **C** — weight concentration (penalty)

The unified score is:

\[
S = w_B B + w_H H + w_D D - w_C C
\]

Default weights:

- `w_B = 1`  
- `w_H = 1`  
- `w_D = 1`  
- `w_C = 1`

Weights are user-configurable via CLI.

---

# S-score Components

## 1. Polarity Balance (B)

Measures how evenly positive and negative edges are represented.

\[
B = 1 - \left|\frac{n_{pos} - n_{neg}}{n_{edges}}\right|
\]

- `B = 0` → all edges same sign (uniform)  
- `B = 1` → perfect balance (maximally diverse)

---

## 2. Polarity Entropy (H)

Measures diversity of polarity categories.

\[
H = -\sum_{s \in \{-1,0,+1\}} p_s \log p_s
\]

- `H = 0` → only one polarity type present  
- `H > 0` → mixed polarity (structurally richer)

Entropy is computed even if `p_zero = 0`.

---

## 3. Mode–Matrix Disagreement (D)

Defined only when both contributions exist (`source=both`, `hybrid`,
`matrix-only`).

For each edge `(A,B)`:

\[
d_{AB} = \text{sign}(W_{\text{mode}}[A,B]) \neq \text{sign}(W_{\text{matrix}}[A,B])
\]

Then:

\[
D = \frac{\#\{d_{AB} = \text{True}\}}{n_{edges}}
\]

- `D = 0` → mode and matrix agree everywhere  
- `D = 1` → complete disagreement  
- `D = None` → mode-only runs (ignored in S)

---

## 4. Concentration (C)

Provided by Phase 04D:

\[
C = \text{weight\_concentration}
\]

Lower concentration indicates a more distributed influence pattern.

---

# Interpretation

- **High S** → mixed polarity, high entropy, meaningful mode–matrix tension,
  low concentration.
- **Low S** → uniform polarity, low entropy, low disagreement, high
  concentration.

The S-score correlates strongly with visually rich pathway graphs while
remaining fully objective.

---

# Input Directory

The scoring script (`motif_analysis.py`) operates on the **Phase 04D output
directory**, typically:

```
outdir/<stem>/
```

or:

```
data/processed/<stem>/
```

Example:

```
outdir/FB4Yaffect_FB45_999prePost_001_all/
```

The script must locate all motif artifacts inside this directory:

- `motifs.json`
- `motifs.<hash>.json`
- `motifs.data.json`
- `motifs.<hash>.data.json`

Hashed filenames are derived from `config_hash8` and must be treated as valid
inputs.

The script does **not** inspect Phase 02 or Phase 03 artifacts.

---

# Output Location

All scoring results are written **into the same directory** as the motif
artifacts, preserving the one-directory-per-run invariant and keeping analysis
artifacts colocated with the `.gv` pathway graph.

Outputs:

- `motif_analysis_summary.json`
- `motif_analysis_summary.csv` (optional)
- terminal table sorted by S-score

Example:

```
outdir/FB4Yaffect_FB45_999prePost_001_all/motif_analysis_summary.json
```

This mirrors Phase 04D behavior and keeps all run artifacts together.

```
