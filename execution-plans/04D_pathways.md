# Sub‑phase 04D — Pathway Graph Construction & Phase 05 Handoff  
*(GraphML‑based visualization; PNG removed)*

## **Objective**  
Construct the **functional pathway graph** between neuron groups and export it as a richly annotated **GraphML document** suitable for interactive exploration in yEd.  
This replaces the earlier PNG‑based visualization layer.

## **Input / Output**  
**Input:**  
- Completed grouping output from 04C  
- Optional `z_matrix.json` for raw matrix contribution enrichment  

**Output:**  
- `pathway` block populated in `motifs.json` and `motifs.data.json`  
- `mode_outer_products` `(k,N,N)` added to `motifs.npz`  
- `pathway_adjacency` `(G,G)` and `pathway_adjacency_abs` `(G,G)` added to `motifs.npz`  
- **New artifact:**  
  - `motifs.pathway.graphml` — full interactive pathway graph  
- **PNG deliverables removed**  

## **New Symbols**  
- `mode_weights`  
- `mode_outer_product`  
- `group_pair_contributions`  
- `filter_pathway_edges`  
- `build_pathway`  
- `load_pathway_matrix` (Phase 02 enrichment)  
- `write_graphml_pathway` *(replaces plot functions)*  
- `graphml_node_metadata`  
- `graphml_edge_metadata`  

## **CLI Additions**  
| Option | Default | Description |
| --- | --- | --- |
| ``--pathway-source ``{mode,matrix,both}`` | ``mode`` | Source of pathway contributions: per-mode outer products, raw matrix, or both. |
| ``--pathway-weight ``{evr,energy,uniform,value,abs_value}`` | ``evr`` | Mode weighting rule for aggregating contributions. |
| ``--pathway-edge-threshold ``FLOAT`` | ``0.1`` | Minimum absolute edge weight to include in the pathway graph. |
| ``--pathway-top-edges ``INT`` | ``5`` | Maximum number of strongest edges kept per source group. |
| ``--no-intra`` | off | Exclude intra-group (self) edges from the pathway graph. |
| ``--graphml`` | on | Write the GraphML pathway artifact (``motifs.pathway.graphml``). |

**Retroactive Removal needed from previous phases:**  
- `--plot`  
- `--plot-pathways`  
- `--no-popup`  
- `--cmap`  
- `--diagram-layout`  
- PNG generation  

## **Validation Rules**  
- pathway config enums/ranges  
- `Σ w_j == 1` for mode weights  
- valid node/edge IDs  
- threshold and top‑N filtering correctness  
- matrix-source enrichment:  
  - missing Phase 02 artifact → warning, `z_contribution: null`  
- GraphML write failure → error  
- GraphML schema validation (node/edge keys)  

## **Reference Numbers (owned by 04D)**  
Using defaults (`mode` / `evr` / threshold 0.1 / top‑5):

| Quantity | Value |
|---|---:|
| `n_pathway_nodes` / `n_pathway_edges` | 22 / **39** |
| `pathway_abs_max` | 2.345300 |
| smallest kept \|edge\| | 0.235975 |
| positive edges / negative edges | **0 / 39** |
| intra-group edges / cross-group edges | 7 / 32 |
| weight concentration | 0.0558 |
| edges @ threshold-only (0.05 / 0.1 / 0.2 / 0.25) | 91 / 46 / 17 / 14 |
| documented finding | all edges negative at defaults (consistent with Phase 02 bias) |

These numbers must be reproduced exactly.

## **GraphML Specification**

## Required GraphML Backend

Pathway graph construction in 04D **must use NetworkX** as the GraphML backend.  
NetworkX provides a stable, well‑tested, schema‑compliant GraphML writer that supports:

- directed graphs,
- arbitrary node and edge metadata,
- deterministic output,
- yEd compatibility,
- safe XML generation without manual serialization.

This requirement ensures that pathway artifacts are generated using a standard, widely
supported library rather than custom XML writers. NetworkX is already included in the
`eigen-decomposition` environment and should be used for all GraphML output unless a
specific technical need arises that NetworkX cannot satisfy.

If future requirements demand features beyond NetworkX’s GraphML capabilities (e.g.,
custom schema extensions, specialized XML structures, or advanced metadata encoding),
the backend may be replaced with a more specialized library such as `lxml`. Any such
change must be justified in the execution plan and validated against yEd compatibility.

Until such a need is demonstrated, **NetworkX remains the mandated GraphML generator**
for 04D.



### **Nodes**  
Each group becomes a GraphML node with keys:

- `group_id`  
- `label`  
- `size`  
- `dominant_mode`  
- `region_composition`  
- `centroid`  
- `coherence`  
- `is_background`  
- `is_singleton`  

### **Edges**  
Each pathway edge becomes a GraphML edge with keys:

- `source_group`  
- `target_group`  
- `weight`  
- `abs_weight`  
- `polarity`  
- `contribution_by_mode` (JSON‑encoded)  
- `threshold_flag`  
- `topN_flag`  
- `intra_flag`  
- `z_contribution` (optional)  

### **Graph-level metadata**  
- `n_nodes`  
- `n_edges`  
- `weight_normalization`  
- `pathway_source`  
- `pathway_weight_rule`  
- `threshold`  
- `topN`  

## **Tests (≈32 + 2 slow)**  
- contribution formula correctness  
- weight normalization  
- threshold + top‑N filtering  
- intra-group edge handling  
- matrix/both enrichment  
- GraphML node/edge validity  
- deterministic ordering  
- `motifs.json` pathway block correctness  
- `motifs.npz` pathway arrays correctness  
- slow tests for reference numbers  
- round-trip GraphML load  

## **Acceptance**  
- All master-plan acceptance criteria satisfied  
- GraphML replaces PNG 
- Phase 05 can consume `pathway_adjacency` + GraphML directly  
- `git diff --stat` touches only clustering module, tests, README, and plan docs  

## **Non-goals**  
- PNG diagrams  
- SVG/Graphviz export  
- interactive windows  
- Phase 05/06 dynamics  

## **Deliverables**  
- `execution-plans/04D_pathways.md`  
- updated module + `__init__.py`  
- new tests  
- updated README (GraphML pathway section)  
- `motifs.pathway.graphml` artifact  
