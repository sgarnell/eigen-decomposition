# Sub-phase 04D — Pathway graph construction & Phase 05 handoff: detailed design

*Implementation: `src/clustering/functional_motifs.py` (04D scope) with `src/clustering/__init__.py`.*
*Artifacts: populated `pathway` block in `motifs.json` and `motifs.data.json`; `motifs.npz`
grows from **21 to 24 arrays**; **new artifact** `motifs.pathway.graphml`.*
*Tests: `tests/test_functional_motifs.py`, `tests/test_motif_schema.py`.*

> This document is the detailed design for the spec in `execution-plans/04D_pathways.md`.
> The spec file is left intact (mirroring the `04C_grouping.md` /
> `04C_grouping_detailed.md` split). It contains **mandatory corrections** to the spec,
> discovered by reverse-verifying every reference number against the real artifacts.

---

## 1. Corrections to the spec (each one verified numerically)

| # | Spec statement | Verified reality | Resolution adopted |
|---|---|---|---|
| C1 | `--pathway-edge-threshold` is the *"minimum **absolute** edge weight"* | Absolute filtering gives **86** edges at `0.1`, not the required **46**. The required sweep `91 / 46 / 17 / 14` is reproduced **only** when the threshold is applied to the **abs-max-normalised** weight (equivalently `threshold × abs_max` on raw weights). | The threshold is a **fraction of `abs_max`**; the CLI help text is corrected. |
| C2 | `weight_concentration = 0.0558` | `max|w| / Σ|w|` over the **full `G×G` matrix (intra diagonal included)** = **0.055842135916593295**. Herfindahl (`0.0137`) and the off-diagonal-only share (`0.0619`) do not match. | `weight_concentration = abs_max / Σ_{all G×G} |w|`. |
| C3 | `--pathway-weight {evr,energy,...}` implies two distinct rules | `energy = λ²` and `evr = λ²/Σλ²`; after the mandatory `Σ=1` renormalisation they are **numerically identical**. | Both enum values are kept (frozen config enum) and documented as equivalent; a test asserts equality. |
| C4 | `smallest kept |edge| = 0.235975`, `abs_max = 2.345300` | These are **raw** (pre-normalisation) values. | Stored weights are **raw**; `abs_max` is reported separately; the normalisation is used **only** for thresholding. |
| C5 | *"Retroactive removal: `--plot`, `--plot-pathways`, `--no-popup`, `--cmap`, `--diagram-layout`, PNG generation"* | None of these were ever implemented in 04A–04C (`grep` over `src/clustering/functional_motifs.py` → 0 hits); there is nothing to remove. | **No removal.** The frozen config fields `max_diagram_nodes` / `diagram_layout` stay declared + hashed but unused (no CLI flag), so `config_hash` never changes. |
| C6 | *"missing Phase 02 artifact → warning, `z_contribution: null`"* | Superseded by the approved decision: a missing matrix artifact with `--pathway-source matrix\|both` is a **hard error**. | `CODE_MATRIX_MISSING` error, exit code `1`, actionable message; `z_contribution` is absent only for `--pathway-source mode`. |
| C7 | GraphML edge `z_contribution` / node `dominant_mode`, `coherence` are "optional" | NetworkX 3.6.1 raises `NetworkXError: GraphML writer does not support <class 'NoneType'> as data values`. | Unavailable attributes are **omitted**, never `None`. |
| C8 | `modes` / `n_modes` describe the *dominant* modes | On the reference data **all 21 modes contribute a non-zero weight to every kept edge**. | `modes` = every mode with a non-zero contribution (documented finding: `n_modes == k == 21` here); the *top* modes live in `top_modes`. |
| C9 | Master decomposition doc: 04D adds `motifs.png`, `motifs.pathways.png` | Superseded by the GraphML-only decision in the spec. | The 04D row of `execution-plans/04_functional_motifs_diagrams.md` is corrected to `motifs.pathway.graphml`. |

## 2. Scope and contract

04D populates the **already frozen** `pathway` payload block as a directed group-to-group
graph, adds three arrays to the sidecar, and writes the yEd-facing GraphML artifact. It adds
**no** schema key, no config field, and no new dependency (NetworkX and Matplotlib are already
declared; NetworkX is the mandated GraphML backend, Matplotlib is unused).

* **Purely additive.** The frozen key sets, `validate_motif_payload_schema`, the 26 config
  fields, `config_hash`, and every 04A–04C behaviour are untouched.
* **Only permitted existing-test edits:** assertions that pinned a field at its *pre-04D
  neutral value* (§13). No 04A–04C behaviour changes.
* **No plotting.** `--plot`, `--plot-pathways`, `--no-popup`, `--cmap`, PNG generation and
  `--diagram-layout` are **not** implemented (C5).

## 3. The pathway formula (reverse-verified against the real artifacts)

```
w               = mode_weights(retained_values, rule)          # Σ|w| = 1, (k,)
OP_m[i, j]      = value_m · U[i, m] · V[j, m]                  # (k, N, N)
W_mode[A, B]    = Σ_m w_m · Σ_{i∈A} Σ_{j∈B} OP_m[i, j]         # (G, G)
                = Σ_m w_m · value_m · (Σ_{i∈A} U[i,m]) · (Σ_{j∈B} V[j,m])
W_matrix[A, B]  = Σ_{i∈A} Σ_{j∈B} Z[i,j]                       # Phase 02 effective matrix
W_total         = W_mode | W_matrix | W_mode + W_matrix        # source mode | matrix | both
```

**Verification on the reference artifact** (`mode` / `evr`, `intra` on, `top_edges` 5):

| Quantity | Required | Computed |
|---|---:|---:|
| `max|W_mode|` (`abs_max`) | 2.345300 | **2.345299756658418** |
| threshold-only counts @ 0.05/0.1/0.2/0.25 (`× abs_max`) | 91 / 46 / 17 / 14 | **91 / 46 / 17 / 14** |
| kept edges | 39 | **39** |
| smallest kept `|w|` | 0.235975 | **0.23597480790052852** |
| positive / negative | 0 / 39 | **0 / 39** |
| intra / cross | 7 / 32 | **7 / 32** |
| `weight_concentration` | 0.0558 | **0.055842135916593295** |

**Mode-weight rules** (`raw` normalised by `Σ|raw|`, so `Σ|w_j| == 1` always):

| rule | `raw_m` | `Σ w_j == 1`? |
|---|---|---|
| `evr` (default) | `λ_m²/Σ_all λ²` | ✔ |
| `energy` | `λ_m²` (≡ `evr`, C3) | ✔ |
| `uniform` | `1` | ✔ |
| `abs_value` | `|λ_m|` | ✔ |
| `value` | `λ_m` (signed ⇒ `w` may be negative) | ✔ iff `Σλ > 0` (always for SVD); otherwise `CODE_MODE_WEIGHTS` error |

For the reference run `w = [0.343528, 0.122317, 0.097866, 0.094065, 0.058372, …]`
(`Σ = 1`, `Σ|λ| = 29.163366343542684`).

## 4. Thresholding, normalisation and top-N (fixed order)

1. `abs_max = max|W_total|` over the **full `G×G` matrix** (pre-filter, independent of
   `--no-intra` / top-N) — so the threshold scale never shifts when the intra switch changes.
2. `cutoff = edge_threshold × abs_max`; keep candidates with `|W_total| ≥ cutoff`, skipping
   `source == target` when `--no-intra`. A threshold-only sweep is recorded for
   `0.05 / 0.1 / 0.2 / 0.25` (diagnostics only, never serialized).
3. Keep at most `top_edges` per **source** group, ranked by `(-|w|, source, target)`
   (deterministic tie-break).
4. Emit edges ordered by `(source_group_id, target_group_id)`; zero-padded ids make the
   lexicographic order equal the numeric order.
5. `polarity = sign(weight) ∈ {-1, 0, 1}`; `is_intra = (source == target)`.
6. An all-zero contribution matrix yields `abs_max = 0`, no edges, and no division by zero.

**Reference leftovers** (used by the slow tests): kept-per-source counts
`G01:2 G02:2 G03:2 G04:5 G06:2 G07:5 G08:5 G09:5 G11:1 G12:1 G17:1 G18:5 G20:2 G21:1`;
intra edges `G01, G02, G03, G04, G06, G09, G18`; isolated nodes
`G00, G10, G13, G15, G16, G19`; strongest edge `G04 → G01 = −2.345300` (mode 1 carries
`0.991164` of its absolute mode contribution).

## 5. Payload changes (frozen keys only)

`pathway` (all 16 `PATHWAY_KEYS`), reference values:

```
nodes  = 22 (all groups G00…G21, group order)   edges = 39
n_nodes = 22   n_edges = 39
weight = "evr"   source = "mode"   edge_threshold = 0.1   top_edges = 5   intra = true
abs_max = 2.345299756658418   n_positive = 0   n_negative = 39
n_intra = 7                   n_cross = 32
weight_concentration = 0.055842135916593295
weights = [39 raw weights, in edges order]
```

`nodes[]` (`NODE_KEYS`): `node_id = group_id = label` (`G00…G21`), `size`, `members`,
`cent_mean`, `region_composition`, `dominant_mode`, and `top_modes` = the top **5** modes of
the group's **mean signed participation** profile (`signed = participation ·
_dominant_signs(left, right)`, the 04A rule), ranked by `(-|signed|, mode)`, with
`share = |signed_m| / Σ_m' |signed_m'|`. `G00` has a zero profile ⇒ `top_modes = []`.

`edges[]` (`EDGE_KEYS`): `source`, `target`, `weight` (raw signed), `abs_weight`, `n_modes`,
`modes` (ascending, every non-zero mode contribution), `is_intra`, `polarity`, and
`top_modes` (top 5 by `|signed_weight|`, `share = |signed_weight| / Σ_m |contribution_m|`).
`--pathway-source matrix` has no per-mode breakdown (`modes = []`, `top_modes = []`) because
the weight comes from the raw matrix, not from mode outer products; for `both` the
breakdown covers the mode part only and the matrix part is recorded as `z_contribution`.

`metadata` (8 keys declared as `null` since 04A, now filled): `n_pathway_nodes`,
`n_pathway_edges`, `pathway_abs_max`, `pathway_positive_edges`, `pathway_negative_edges`,
`pathway_intra_edges`, `pathway_cross_edges`, `pathway_weight_concentration`.

`provenance.stage` / `config.stage` / the npz `stage` array → `"04D"`.
`motifs.data.json` gains the filled block automatically (`save_data_payload` already
forwards `analysis.pathway`).

**Neutral fallback.** With `--grouping none` there are no groups: the pathway stays neutral
(`NEUTRAL_PATHWAY`), the 8 metadata keys stay `null`, the two adjacency arrays are not
written (the sidecar then holds 19 arrays: the 18 group-free 04A–04C arrays plus
`mode_outer_products`), and `[pathway_skipped]` is logged. `--graphml` still writes a valid
metadata-only GraphML document (0 nodes, 0 edges) so the artifact set stays uniform.

## 6. `motifs.npz` additions (21 → 24 arrays)

| Key | Shape/dtype | Meaning | JSON-verified? |
|---|---|---|---|
| `mode_outer_products` | `(k, N, N) <f8` | `value_m · U[:,m] ⊗ V[:,m]`, mode-major | no (needs the loadings, like `loadings_left/right`) |
| `pathway_adjacency` | `(G, G) <f8` | the **filtered** signed adjacency (0 where no kept edge), group order | **yes**, derived from `pathway.edges` |
| `pathway_adjacency_abs` | `(G, G) <f8` | `abs(pathway_adjacency)` | **yes** |

* `mode_outer_products` is group-independent and is always written by 04D; the two adjacency
  arrays are written only when the pathway is built (groups exist), so an active-grouping
  sidecar holds 24 arrays and a `--grouping none` sidecar holds 19. A stale 04C cache lacks
  them and is therefore ignored with a warning, with the JSON winning — the designed
  staleness path. Nothing else about the cache changes (fixed dtypes, numpy's frozen zip
  `date_time`, in-memory → temp file → `fsync` → `os.replace`, `allow_pickle=False`).

## 7. GraphML artifact — `motifs.pathway.graphml` (NetworkX-mandated)

* `networkx.DiGraph`, built by :func:`write_graphml_pathway`; NetworkX is imported **lazily**
  inside that function (the Matplotlib precedent of Phases 02/03), and a missing NetworkX
  raises an actionable `MotifValidationError`.
* Nodes are added in `pathway.nodes` order, edges in `pathway.edges` order, attributes in a
  fixed insertion order ⇒ **byte-deterministic** output (verified: two `generate_graphml`
  calls are identical and `nx.parse_graphml` round-trips nodes, edges and graph metadata).
* Serialisation: `<?xml version="1.0" encoding="UTF-8"?>` + `nx.generate_graphml(graph)`,
  written through a local atomic text writer (`_write_text_atomic`, mirroring
  `write_npz_atomic`; `src/utils/io.py` stays untouched).

| Level | keys (exactly) |
|---|---|
| node | `group_id`, `label`, `size`, `dominant_mode`, `region_composition` (JSON), `centroid` (JSON of the group's `(k,)` participation centroid), `coherence`, `is_background`, `is_singleton` |
| edge | `source_group`, `target_group`, `weight`, `abs_weight`, `polarity`, `contribution_by_mode` (JSON of `top_modes`), `threshold_flag`, `topN_flag`, `intra_flag`, `z_contribution` |
| graph | `n_nodes`, `n_edges`, `weight_normalization` (`"abs-max"`), `pathway_source`, `pathway_weight_rule`, `threshold`, `topN` |

* `threshold_flag` / `topN_flag` are `True` for every emitted edge **by construction** (the
  artifact *is* the filtered graph); they exist for yEd filtering and future exports.
* Optional attributes are **omitted** when unavailable (C7): `dominant_mode` for `G00`,
  `coherence` for singleton/background groups, `centroid` when the in-memory centroids are
  unavailable (an analysis loaded from JSON), `z_contribution` for `--pathway-source mode`.
* `GRAPHML_NODE_KEYS` / `GRAPHML_EDGE_KEYS` / `GRAPHML_GRAPH_KEYS` are validated against the
  built graph *before* serialisation; a write failure is `CODE_ARTIFACT_WRITE` (error, never a
  partial file). `--no-graphml` suppresses the artifact; `--dry-run` writes nothing.

## 8. Matrix-source enrichment (`--pathway-source {mode,matrix,both}`)

* `pathway_matrix_path(source_artifact)` returns the sibling Phase-02 artifact
  `z_matrix<inherited variant>.json` (so `eigen.f8652585.json` → `z_matrix.f8652585.json`).
* `load_pathway_matrix(source_artifact)` loads it through Phase 02's `load_z_matrix` and
  returns `(effective_matrix, neuron_order)`. No other auto-discovery is attempted.
* `matrix`: `W_total = W_matrix`; `both`: `W_total = W_mode + W_matrix`; per edge the matrix
  part is recorded as `z_contribution` in the GraphML.
* **Missing artifact** ⇒ `CODE_MATRIX_MISSING` **error** (exit code `1`, no artifacts
  written); **neuron-order mismatch** ⇒ `CODE_MATRIX_MISMATCH` **error** (approved §1 C6).
* `pathway.source` records the resolved source (`mode` default; `matrix`/`both` when used).
  For `mode` the matrix is never loaded.

## 9. New symbols (exactly the spec names)

Public: `mode_weights`, `mode_outer_product`, `group_pair_contributions`,
`filter_pathway_edges`, `build_pathway`, `load_pathway_matrix`, `write_graphml_pathway`,
`graphml_node_metadata`, `graphml_edge_metadata`, plus `pathway_matrix_path`,
`PATHWAY_TOP_MODES`, `GRAPHML_NODE_KEYS`, `GRAPHML_EDGE_KEYS`, `GRAPHML_GRAPH_KEYS`,
`PATHWAY_WEIGHT_NORMALIZATION`, `PATHWAY_THRESHOLD_SWEEP`, `XML_DECLARATION` and the four new
issue codes.
Private: `_mode_outer_product_stack`, `_group_pair_sum`, `_pathway_adjacency`,
`_pathway_statistics`, `_write_text_atomic`.

```python
def mode_weights(values, rule=DEFAULT_PATHWAY_WEIGHT) -> np.ndarray           # (k,), Σ|w| = 1
def mode_outer_product(loadings_left, loadings_right, values, mode) -> np.ndarray  # (N, N)
def group_pair_contributions(outer_products, group_labels, weights) -> np.ndarray  # (G, G)
def filter_pathway_edges(contributions, *, edge_threshold=..., top_edges=..., intra=True)
        -> tuple[list[tuple[int, int, float]], dict[str, Any]]
def build_pathway(groups, *, participation, loadings_left, loadings_right, values,
                  neuron_ids, config, matrix=None)
        -> tuple[PathwayDiagram, dict[str, Any]]          # (diagram, diagnostics)
def pathway_matrix_path(source_artifact) -> Path | None
def load_pathway_matrix(source_artifact) -> tuple[np.ndarray, list[str]]
def graphml_node_metadata(node, group=None, centroid=None) -> dict[str, Any]
def graphml_edge_metadata(edge, *, z_contribution=None) -> dict[str, Any]
def write_graphml_pathway(path, diagram, *, groups=None, config=None, source=None,
                          centroids=None, z_by_edge=None) -> Path
```

`build_pathway` returns `(diagram, diagnostics)` — the 04C `build_groups` tuple-return
convention. `diagnostics` = `{abs_max, cutoff, n_candidates, threshold_sweep,
per_source_counts, mode_weights, z_contributions}`; it is stored in
`analysis.diagnostics["pathway"]` (never serialized) so `write_artifact_set` can attach
`z_contribution` to the GraphML edges.

New constants and issue codes:

```python
PATHWAY_TOP_MODES = 5
PATHWAY_WEIGHT_NORMALIZATION = "abs-max"
PATHWAY_THRESHOLD_SWEEP = (0.05, 0.1, 0.2, 0.25)
MATRIX_INPUT_PREFIX = "z_matrix"
XML_DECLARATION = '<?xml version="1.0" encoding="UTF-8"?>'
CODE_MATRIX_MISSING = "matrix_missing"
CODE_MATRIX_MISMATCH = "matrix_mismatch"
CODE_MODE_WEIGHTS = "mode_weights"
CODE_PATHWAY_SKIPPED = "pathway_skipped"
```

## 10. Config and CLI

The five pathway config fields already exist in `MOTIF_CONFIG_FIELDS` (hashed) and in
`MotifConfig`, so **`config_hash` is unchanged**. A new `pathway graph` argument group adds:

| Option | Default | Behaviour |
|---|---|---|
| `--pathway-source {mode,matrix,both}` | `mode` | contribution source (§8) |
| `--pathway-weight {evr,energy,uniform,value,abs_value}` | `evr` | mode-weight rule (§3) |
| `--pathway-edge-threshold FLOAT` | `0.1` | minimum `|w|` as a **fraction of `abs_max`** (C1) |
| `--pathway-top-edges INT` | `5` | strongest edges kept per source group |
| `--no-intra` | off | drop intra-group (self) edges |
| `--graphml` / `--no-graphml` | **on** | write `motifs.pathway.graphml` |

`_config_from_args` passes the five config fields through (`pathway_source`,
`pathway_weight`, `pathway_edge_threshold`, `pathway_top_edges`, `intra`);
`max_diagram_nodes` / `diagram_layout` keep their frozen defaults (C5).
`_validate_config` gains: `pathway_edge_threshold` finite ∈ `[0, 1]`, `pathway_top_edges ≥ 1`,
`max_diagram_nodes ≥ 1` (the enums were already validated).

## 11. Pipeline integration (`build_motif_analysis`)

1. `_build_arrays` gains `mode_outer_products` (group-independent, always present).
2. After the 04C group block and **only when groups exist**:
   * resolve/load the Phase-02 matrix when `config.pathway_source in {"matrix","both"}`
     (errors → `CODE_MATRIX_MISSING` / `CODE_MATRIX_MISMATCH`, analysis still constructed so
     the CLI skips the write);
   * `diagram, diagnostics = build_pathway(...)`;
   * `analysis.pathway = diagram.to_dict()`; `analysis.metadata.update(_pathway_statistics(diagram))`;
   * `analysis.arrays["pathway_adjacency"]` / `["pathway_adjacency_abs"]` = `_pathway_adjacency(diagram.edges, node_ids)`
     and its absolute value;
   * `analysis.diagnostics["pathway"] = diagnostics`.
3. When groups do **not** exist (`--grouping none`): pathway stays neutral, `[pathway_skipped]`
   is logged, no adjacency arrays.
4. `artifact_paths` gains `"graphml"` → `<stem>.pathway.graphml`;
   `write_artifact_set(..., graphml: bool = True)` writes it **last** (JSON → npz → data →
   GraphML) and records any failure as `CODE_ARTIFACT_WRITE` (never partial).
5. `render_statistics` replaces `pathway : not computed yet (04D)` with four real lines
   (`pathway`, `pathway threshold`, `pathway polarity`, `pathway concentration`), or the
   "not computed yet (grouping none)" line.
6. `render_summary_box` gains a `graphml: bool = True` keyword, the heading
   `Phase 04D complete`, a `Pathway: 22 nodes / 39 edges` line and a
   `GraphML written: …` line (`skipped (--dry-run)` / `not requested (--no-graphml)`).

## 12. Validation rules, codes and invariants

| Check | Severity |
|---|---|
| `pathway_source` / `pathway_weight` / `diagram_layout` enums (already present) | Error |
| `pathway_edge_threshold` finite and in `[0, 1]`; `pathway_top_edges ≥ 1`; `max_diagram_nodes ≥ 1` | Error |
| `Σ|w_j| == 1` for the mode weights; `Σ w_j > 0` for `--pathway-weight value` | Error (`CODE_MODE_WEIGHTS`) |
| Phase-02 matrix missing for `matrix`/`both` | Error (`CODE_MATRIX_MISSING`) |
| Phase-02 matrix neuron order ≠ spectrum order | Error (`CODE_MATRIX_MISMATCH`) |
| Contribution matrix square; node ids unique/non-empty; edge endpoints ∈ node ids; no duplicate `(source, target)` | Error |
| Every emitted edge satisfies `|w| ≥ threshold × abs_max` and `≤ top_edges` per source | Error |
| GraphML attribute keys ⊆ the three declared key sets | Error |
| Emitted payload conforms to the frozen schema | Error (`CODE_PAYLOAD_SCHEMA`) |
| GraphML / npz / save-data write failure (never a partial file) | Error (`CODE_ARTIFACT_WRITE`) |
| Grouping disabled (pathway skipped) | Info (`CODE_PATHWAY_SKIPPED`) |
| Pathway with zero kept edges (e.g. a trivial matrix) | Info |

**Invariants (asserted in tests):** `n_pathway_nodes == len(groups)`;
`n_pathway_edges == len(edges) == len(weights)`; `n_positive + n_negative + n_zeros == n_edges`;
`n_intra + n_cross == n_edges`; `is_intra == (source == target)`; `polarity == sign(weight)`;
`abs_weight == |weight|`; `n_modes == len(modes)`; `Σ top_modes.share ≤ 1` per edge and node;
`mode_outer_products[m] == value_m · outer(U[:,m], V[:,m])` (rank-1);
`pathway_adjacency[i,j]` equals the corresponding edge weight;
`Σ|w_j| == 1`; `motifs.json`, `motifs.npz` and `motifs.pathway.graphml` are byte-identical
under a fixed `SOURCE_DATE_EPOCH`; the Phase 03 object is never mutated; **no `*.csv`**.

## 13. Tests (~35 fast + 2 slow) and permitted 04A–04C assertion updates

New `# 04D — pathway graph` sections:

* *Rules / formula*: `mode_weights` for all five rules (incl. `evr ≡ energy`, `Σ|w| = 1`,
  `value` degeneracy error); `mode_outer_product` rank-1 identity vs `np.outer`;
  `group_pair_contributions` vs a brute-force member-pair sum on a synthetic `(8, 3)`
  spectrum; zero labels / zero weights guards.
* *Filtering*: threshold relative to `abs_max`; threshold-only sweep counts; top-N per source
  with deterministic tie-break; `--no-intra`; empty/all-zero matrix ⇒ no edges; polarity and
  concentration definitions.
* *Build*: node set = all groups; edge ordering; `top_modes` share/rank identities; `G00`
  empty profile; determinism (two builds identical); `--grouping none` ⇒ neutral pathway and
  no adjacency arrays; metadata block cross-checks.
* *Matrix source*: synthetic sibling `z_matrix.json` ⇒ `both` = mode + matrix and
  `z_contribution` correct; `matrix` alone; **missing sibling ⇒ `--pathway-source matrix`
  exits 1 with `[matrix_missing]` and writes nothing**; neuron-order mismatch ⇒ error.
* *Artifacts*: npz 24 arrays; `pathway_adjacency` matches the JSON edges;
  `mode_outer_products` shape/identity; GraphML key sets, node/edge/graph attributes,
  round-trip through `nx.read_graphml`, determinism, `--no-graphml`, write failure →
  `CODE_ARTIFACT_WRITE`.
* *Terminal/CLI*: `--stats` pathway lines; box heading / `Pathway:` / `GraphML written:` lines;
  `python -m … --pathway-source mode --stats` smoke test.
* *Slow*: the §3 reference block (22/39, `abs_max`, smallest kept, 0/39, 7/32, concentration,
  the sweep, per-source counts, isolated nodes); the reference GraphML round-trip.

**Permitted updates to 04A–04C assertions (neutral-value exceptions only):**

| File / line | Assertion | New form |
|---|---|---|
| `test_functional_motifs.py:684` | `len(arrays) == 21` | `== 24` |
| `test_functional_motifs.py:662` | `set(written) == {"json","npz","data"}` | `+ "graphml"` |
| `test_functional_motifs.py:746–751` | `write_artifact_set(sidecar=False)` → `{"json"}` + one file | `{"json","graphml"}` + both files; a new test covers `graphml=False` |
| `test_functional_motifs.py:770, 860, 871` | `"Phase 04C …"` | `"Phase 04D …"` |
| `test_functional_motifs.py:875` | `"not computed yet (04D)"` | real pathway lines |
| `test_functional_motifs.py:1253` | `metadata["n_pathway_nodes"] is None` | `== 22` |
| `test_motif_schema.py:108–147` | pathway neutral / pathway keys `None` | populated pathway + typed metadata |

## 14. Deliverables

* `execution-plans/04D_pathways_detailed.md` (this document)
* `src/clustering/functional_motifs.py`, `src/clustering/__init__.py`
* `tests/test_functional_motifs.py`, `tests/test_motif_schema.py`
* `README.md` (GraphML pathway section, options, examples, npz table 21 → 24, reference
  numbers, status row) and the 04D row of `execution-plans/04_functional_motifs_diagrams.md`
* Artifacts: `motifs.json` / `motifs.npz` / `motifs.data.json` (richer) and
  `motifs.pathway.graphml`

**Untouched:** `src/utils/io.py`, `src/parsing/*`, `src/matrices/*`, `src/spectral/*`,
`pytest.ini`, `environment.yml`, every Phase 01–03 test/artifact, all frozen key sets, the
schema validator and `config_hash`.

## 15. Risks and mitigations

| Risk | Mitigation |
|---|---|
| The spec's "absolute threshold" wording contradicts its numbers (C1) | documented correction + the threshold is a fraction of `abs_max`; both the sweep and the kept set are reference-tested |
| `evr` ≡ `energy` (C3) | documented; a test pins the equivalence instead of inventing a difference |
| `None` attributes crash NetworkX (C7) | optional attributes are omitted; the key sets are validated before serialisation |
| A stale 04C cache breaking loads | the new arrays are JSON-verified where possible; a missing/old cache is ignored with a warning and the JSON wins |
| GraphML growth / non-determinism | primitive or sorted-key-JSON attributes, fixed insertion order, a determinism test |
| A silent empty pathway for a missing matrix | hard error with an actionable message (approved), plus an explicit CLI test |
| A non-square / all-zero contribution matrix | shape validation + the `abs_max == 0` guard |
| `--grouping none` producing a half-populated pathway | explicit neutral fallback with an info log and no adjacency arrays |

## 16. Non-goals

PNG/SVG/Graphviz export, interactive windows, diagram layout / `--max-diagram-nodes` (C5),
GraphML-side edge filtering, Phase 05/06 dynamics, any new schema key, config field or
dependency.

## 17. Verification protocol

```bash
PY=/home/mr-miracle/miniconda3/envs/eigen-decomposition/bin/python
$PY -m pytest -m "not slow" -q                       # 531 + ~35 fast, green
$PY -m pytest -q                                     # + 2 slow, green
$PY -m src.clustering.functional_motifs \
    -i data/processed/FB4Yaffect_FB45_999prePost_001_all/eigen.json \
    -o data/processed --save-data --stats            # 22 / 39 / 2.345300 / 0.055842
SOURCE_DATE_EPOCH=... double run                     # byte-identical json/npz/graphml
git --no-pager diff --stat && git --no-pager status --short
```