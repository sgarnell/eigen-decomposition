# Phase 04D Update A — Unified Weight Specification (Configurable Weight, Polarity & Hybrid Rules)

*Implementation: `src/clustering/functional_motifs.py` (stage `04D`, engine `build_pathway`).*

*Artifacts touched: `motifs.json` (config block only), `motifs.data.json` (config block only),
`motifs.npz` (bytes only via the recorded `config_hash`), `motifs.pathway.graphml`
(3 new `<graph>` keys).*

*Tests: `tests/test_functional_motifs.py`, `tests/test_motif_schema.py`.*

*Status: approved design — decision (A) locked (new fields are hashed; the default
`config_hash` changes `3fcaf50e → e973fece`); `pre_z`/`post_z` use the group-level
signed-participation reduction; `--pathway-mode-weights` is an alias of the shipped
`--pathway-weight`.*

---

## 1. Why this update exists

Phase 04D reduced the network to a directed group-to-group pathway graph whose edge weight was a
fixed recipe: mode weights → per-mode outer products → group aggregation → `sign(W)` polarity.
Every stage of that recipe except the mode-weight rule was hard-coded.

This update makes the **transform of the unified weight**, the **edge polarity rule** and the
**hybrid blending ratio** scientifically configurable, without changing any other behaviour:

1. `--pathway-weight-rule {signed,abs,positive,negative,mode-only,matrix-only,hybrid}`
2. `--pathway-polarity-rule {sign,post_z,pre_z,delta}`
3. `--pathway-hybrid-alpha <float>` (default `0.5`)

The three options become three `MotifConfig` fields, three CLI flags, three records in the
`config` JSON block and three `<graph>` keys in the GraphML artifact.

The update is **purely additive** with respect to behaviour: the defaults
(`signed` / `sign` / `0.5`) reproduce the current pathway bit-for-bit in every *scientific*
field (22 nodes, 39 edges, `abs_max = 2.345299756658418`, 0 positive / 39 negative, 7 intra /
32 cross). The only artifact diff is the `config` block (3 new keys) and the derived
`config_hash` (decision **A**, §12).

---

## 2. Ground truth — the current implementation map

All references are to `src/clustering/functional_motifs.py` unless stated otherwise.

| Concern | Symbol / line | Current behaviour |
|---|---|---|
| Mode weights | `mode_weights(values, rule)` @ L2322 | `Σ|w| = 1`; rules `evr`,`energy`,`uniform`,`value`,`abs_value` |
| Weight enum | `PATHWAY_WEIGHTS` @ L298 | `("evr","energy","uniform","value","abs_value")` |
| Default mode rule | `DEFAULT_PATHWAY_WEIGHT` @ L287 | `"evr"` |
| Outer product | `mode_outer_product(...)` @ L2358 | `value_m · u_m v_mᵀ`, `(N,N)` |
| Stack | `_mode_outer_product_stack(...)` @ L2371 | `(k,N,N)`, mode-major |
| Group sum | `_group_pair_sum(matrix, labels, n_groups)` @ L2383 | `Iᵀ · M · I`, `(G,G)` |
| Group aggregation | `group_pair_contributions(...)` @ L2399 | `W[A,B] = Σ_m w_m · C_m[A,B]` |
| Contribution source | `build_pathway(...)` @ L2518, L2569–2586 | `mode` / `matrix` / `both = mode + matrix` |
| Threshold + top-N | `filter_pathway_edges(...)` @ L2426 | `cutoff = abs_max · edge_threshold`; `abs_max = max|W|` @ L2446 |
| Edge polarity | `build_pathway` @ L2651 | **hard-coded** `int(np.sign(weight))` |
| Signed participation | `build_pathway` @ L2593–2594 | `signs = _dominant_signs(left,right); signed = P · signs` |
| Participation matrix | `compute_participation(...)` @ L1405 | **non-negative** (`left`/`right`/`max`/`rms`) |
| Config | `MotifConfig` @ L719 | 26 hashed fields; `MOTIF_CONFIG_FIELDS` @ L408 |
| Config hash | `MotifConfig.config_hash()` @ L790 | `sha256(dumps_json(hash_fields()))[:8]` |
| Default detection | `is_default()` @ L795 | only the 8 `DEFAULT_DECISION_FIELDS` (04A) |
| Config schema | `CONFIG_KEYS` @ L468 | `MOTIF_CONFIG_FIELDS ∪ {k_resolved,n_motifs,stage,config_hash}` |
| Config validation | `_validate_config(...)` @ L2881 | enum + range checks, code `CODE_CONFIG` |
| Payload validation | `validate_motif_payload_schema(...)` @ L3230 | frozen key sets |
| GraphML graph keys | `GRAPHML_GRAPH_KEYS` @ L395 | 9 keys, **includes `pathway_weight_rule`** (the *mode* rule) |
| GraphML writer | `write_graphml_pathway(...)` @ L3984 | graph metadata assembled @ L4047–4084 |
| CLI | `build_argument_parser()` @ L4540 | `pathway` argument group @ L4632 |
| Statistics | `render_statistics(...)` @ L4308 | terminal `--stats` block |
| Summary box | `render_summary_box(...)` @ L4363 | terminal completion box |
| Artifact stem | `variant_stem(...)` @ L3425 | `motifs[.variant][.<hash8>]` |

**Reference artifact** (`data/processed/FB4Yaffect_FB45_999prePost_001_all/motifs.json`):
`n_nodes = 22`, `n_edges = 39`, `abs_max = 2.345299756658418`, `weight = "evr"`,
`source = "mode"`, `config_hash = 3fcaf50e`, `n_positive = 0`, `n_negative = 39`,
`n_intra = 7`, `n_cross = 32`, `weight_concentration = 0.05584213591659331`.

---

## 3. Scope boundary

| Owned by this update | Explicitly **not** this update |
|---|---|
| `W_filter` / `W_viz` transform rules; hybrid α; edge polarity rules | grouping logic (04C), participation matrix, motifs/links/families (04A/04B) |
| 2 new engine functions, 3 config fields, 3 CLI flags, 3 GraphML graph keys | NPZ array set, `PATHWAY_KEYS`, `EDGE_KEYS`, `NODE_KEYS` |
| `_validate_config` rules, informational log lines | nodegraphics / edgegraphics formatting, yEd styles, node geometry |
| config-hash change (decision A) and its artifact regeneration | Phase 01/02/03 modules, artifacts, tests, `pytest.ini`, `environment.yml` |

**Purely additive** to the code path: `filter_pathway_edges` gains one optional keyword whose
default is `None` (⇒ byte-identical behaviour); `build_pathway` keeps its signature and its
`--pathway-source` semantics; `PathwayDiagram` gains no fields.

---

## 4. Definitions

Let:
- `A`, `B` be functional groups (04C groups, index space `0..G-1`).
- `i ∈ A`, `j ∈ B` be neurons (index space `0..N-1`).
- `m = 1..k` be the retained functional modes (Phase 03).
- `U[i,m]` = `loadings_left` (source/sender axis), `V[j,m]` = `loadings_right` (target/receiver).
- `value_m` = the mode scalar (Phase 03 `modes[m].value`).
- `P[i,m]` = the non-negative participation matrix (`compute_participation`, L1405).
- `P̃[i,m]` = the **signed** participation `P[i,m] · s[i,m]` with `s = _dominant_signs(U,V)` (L2030).
- `Z[i,j]` = the Phase 02 effective matrix (`z_matrix[.<variant>].json`).
- `w_m` = the mode weight coefficient (`mode_weights`, `Σ|w_m| = 1`).

The **per-mode outer product** is

```
OP_m[i,j] = value_m · U[i,m] · V[j,m]
```

The **per-mode group contribution** is

```
C_m[A,B] = Σ_{i∈A} Σ_{j∈B} OP_m[i,j]
```

The **mode contribution** and the **matrix contribution** are

```
W_mode[A,B]   = Σ_{m=1..k} w_m · C_m[A,B]                 # (G,G)
W_matrix[A,B] = Σ_{i∈A} Σ_{j∈B} Z[i,j]                    # (G,G), requires Phase 02
```

The **raw unified weight** is

```
W[A,B] = W_mode[A,B]                          # --pathway-source mode   (default)
       = W_matrix[A,B]                        # --pathway-source matrix
       = W_mode[A,B] + W_matrix[A,B]          # --pathway-source both
```

The **transformed unified weights** are `W_filter` (thresholding and top-N) and `W_viz` (the
stored edge `weight`), defined per rule in §6.

---

## 5. Mode weight coefficients (existing — unchanged)

Mode weights satisfy `Σ_m |w_m| = 1` and are selected by the shipped
`--pathway-weight {evr,energy,uniform,value,abs_value}` (alias `--pathway-mode-weights`, §11).

| Rule | Definition | Reference |
|---|---|---|
| `evr` (default) | `w_m = value_m² / Σ value²` | `w = [0.343528, 0.122317, 0.097866, …]` |
| `energy` | numerically identical to `evr` (both renormalise `value²`) | equal to `evr` |
| `uniform` | `w_m = 1/k` | `1/21 ≈ 0.047619` |
| `value` | `w_m = value_m / Σ|value|` (signed; needs `Σ value > 0`) | `CODE_MODE_WEIGHTS` otherwise |
| `abs_value` | `w_m = |value_m| / Σ|value|` | `Σ|w| = 1` |

`mode_weights` (L2322) is **not modified**. The new `pathway_weight_rule` is a *separate*,
downstream transform (§6) — the two never interact except that both consume `values`.

**Naming note.** The spec's §2 spells the mode-weight flag `--pathway-mode-weights`; the shipped
flag is `--pathway-weight`. Both are accepted (same `argparse` `dest`), so existing command lines
keep working and the spec spelling is honoured. `--pathway-weight-rule` (§6) is a different,
new option.

---

## 6. Unified weight rule (NEW)

`--pathway-weight-rule <rule>` determines how the unified weight is transformed into
`W_filter` (used for thresholding and top-N ranking) and `W_viz` (the stored edge weight and
therefore the label, width and default polarity).

### 6.1 Rule table

| Rule | `W_filter[A,B]` | `W_viz[A,B]` | Needs Phase 02 `Z`? |
|---|---|---|---|
| `signed` (default) | `W[A,B]` | `W[A,B]` | inherits `--pathway-source` |
| `abs` | `max(W,0) − min(W,0) = |W|` | `W[A,B]` | inherits `--pathway-source` |
| `positive` | `max(W[A,B], 0)` | `W[A,B]` | inherits `--pathway-source` |
| `negative` | `min(W[A,B], 0)` | `W[A,B]` | inherits `--pathway-source` |
| `mode-only` | `W_mode[A,B]` | `W_mode[A,B]` | no |
| `matrix-only` | `W_matrix[A,B]` | `W_matrix[A,B]` | **yes** |
| `hybrid` | `α·W_mode + (1−α)·W_matrix` | same as `W_filter` | **yes** |

`α = --pathway-hybrid-alpha` (default `0.5`). `hybrid(α=1) ≡ mode-only` and
`hybrid(α=0) ≡ matrix-only`, element-wise, and these are asserted identities (§17 item 3).

**Consequence for the one-sided rules.** `positive`/`negative` threshold against
`abs_max = max|W_filter|` of their own truncated matrix, so their scale is the strongest
*surviving* contribution — smaller than the `signed` scale. They therefore keep *different*
edges from `signed` (not a subset); the exact behaviour is pinned by the §17 item 5 tests and
the reference numbers in §25.

### 6.2 `abs_max` definition

```
abs_max = max |W_filter[A,B]|          # computed AFTER the transform
```

This is self-consistent with the threshold rule `|W_filter| ≥ t · abs_max` and is
**bit-identical to today** under `signed` + `source=mode` (`abs_max = 2.345299756658418`).
For every rule, `abs_max > 0 ⇒ weight_ratio = |W_filter| / abs_max ∈ [0, 1]` holds, so the
GraphML edge-width formula stays valid.

The diagnostics additionally record `abs_max_mode` (`max|W_mode|`), `abs_max_matrix`
(`max|W_matrix|`, `0.0` when the artifact is not loaded) and `abs_max_raw`
(`max|W_viz|`, i.e. the **pre-transform** scale, so a one-sided rule shows how far it
truncated: `positive` reports `abs_max = 0.120921` against `abs_max_raw = 2.345300` on the
reference data). None of the three is serialized.

### 6.3 Documented property of `abs` — a deliberate no-op

Because §9 already thresholds on `|W_filter|` and `abs_max = max|W_filter|`, the `abs` rule
produces **exactly the same kept edge set as `signed`**; it changes only the intermediate
magnitude that drives ranking. This is a true property of the specification as written, **not**
an implementation artefact. It is implemented faithfully, documented here, and pinned by an
explicit test (`abs` and `signed` yield identical edge sets and identical `abs_max`) so the
behaviour can never drift silently.
---

## 7. Interaction with the existing `--pathway-source`

`--pathway-source {mode,matrix,both}` already expresses `mode-only` / `matrix-only` / an
unweighted `mode + matrix`. To avoid two competing selectors the precedence is:

```
needs_matrix = (weight_rule in {"matrix-only", "hybrid"})
            or (weight_rule in {"signed", "abs", "positive", "negative"}
                and pathway_source in {"matrix", "both"})
            or (polarity_rule == "delta")

if weight_rule == "signed":
    resolved_source = pathway_source            # full backward compatibility
    W = W_mode | W_matrix | W_mode + W_matrix   # per --pathway-source
else:
    resolved_source = {"mode-only":   "mode",
                       "matrix-only": "matrix",
                       "hybrid":      "hybrid",
                       "abs":         pathway_source,
                       "positive":    pathway_source,
                       "negative":    pathway_source}[weight_rule]
    W = per the §6.1 table                      # weight_rule OVERRIDES --pathway-source
```

Rules:

1. `weight_rule == "signed"` (default) leaves `--pathway-source` in complete control — the
   existing behaviour, unchanged.
2. A non-default `weight_rule` **overrides** `--pathway-source`; the effective choice is recorded
   in the existing `pathway.source` field and in the new GraphML key, so the artifact is
   self-describing. `abs`/`positive`/`negative` do not change the source selection.
3. An **informational** log line (`CODE_WEIGHT_RULE`) fires whenever a non-default `weight_rule`
   overrides a non-default `--pathway-source`, so the override is never silent.
4. `W_matrix` is computed whenever `needs_matrix` is true, and the `CODE_MATRIX_MISSING`
   hard-error path (L3177–3191) is extended to cover `matrix-only`, `hybrid` and
   `polarity_rule == "delta"` — previously only `pathway_source ∈ {matrix, both}` triggered the
   Phase 02 load.
5. `--pathway-source both` (the unweighted sum) remains exactly as-is under `signed`; `hybrid`
   is the new, explicitly weighted alternative.

---

## 8. Polarity rule (NEW)

`--pathway-polarity-rule <rule>` determines the integer `polarity ∈ {-1, 0, +1}` stored on every
edge. This drives the GraphML `edge_color`, and the `n_positive` / `n_negative` statistics.

| Rule | Definition | Needs Phase 02 `Z`? |
|---|---|---|
| `sign` (default) | `polarity = sign(W_viz[A,B])` — **current behaviour, unchanged** | no |
| `post_z` | `polarity = sign( Σ_{j∈B} Σ_m P̃[j,m] )` | no |
| `pre_z` | `polarity = sign( Σ_{i∈A} Σ_m P̃[i,m] )` | no |
| `delta` | `polarity = sign( W_mode[A,B] − W_matrix[A,B] )` | **yes** |

### 8.1 Why `P̃` and not `P` (spec correction)

The specification writes `polarity = sign(Σ_{j∈B} P_{j,·})`. That expression is **not
implementable as written**:

* `P_{j,·}` is a **k-vector** (one participation per retained mode), and `sign()` of a vector is
  undefined — the expression has no scalar value.
* The project's participation matrix `P` (`compute_participation`, L1405) is **non-negative by
  construction** (`abs` / `max` / `rms` of the loadings), so the sign of any sum of it would be
  the constant `+1` — scientifically vacuous.

**Resolution.** Use the **signed** participation

```
P̃[i,m] = P[i,m] · s[i,m]        s = _dominant_signs(U, V)
```

which `build_pathway` already builds at L2593–2594 (`signs = _dominant_signs(left, right);
signed = profile_matrix * signs`), and reduce over **members and modes** to the required scalar:

```
post_z[A,B] = sign( Σ_{j∈B} Σ_{m=1..k} P̃[j,m] )
pre_z [A,B] = sign( Σ_{i∈A} Σ_{m=1..k} P̃[i,m] )
```

**Documented consequence.** `pre_z` / `post_z` are **group-level** rules: an edge's colour
depends only on its source (`pre_z`) or only on its target (`post_z`). A whole row (resp. column)
of the graph therefore shares one colour. This is meaningful ("this receiver group is net
negative") but yields far fewer distinct colours than `sign`. An **informational** log line
(`CODE_POLARITY_RULE`) announces this at run time.

### 8.2 `delta`

`delta` is a purely edge-local rule: it compares the two candidate contributions directly. It
therefore requires the Phase 02 artifact even when `--pathway-source mode`, and a missing matrix
is a hard `CODE_MATRIX_MISSING` error.

---

## 9. Thresholding (existing — unchanged semantics)

Edges are kept when

```
|W_filter[A,B]| ≥ edge_threshold · abs_max
```

with `abs_max = max|W_filter|` (§6.2) and
`--pathway-edge-threshold <float>` (default `0.1`). Each kept edge stores
`weight = W_viz[A,B]` and `abs_weight = |W_filter[A,B]|`; the existing self-check in
`build_pathway` (L2640) is retargeted from `|weight|` to `|W_filter|`.

---

## 10. Top-N selection (existing — unchanged)

For each source group `A`, at most `--pathway-top-edges <int>` edges (default `5`) are kept,
ordered by the deterministic `(-|W_filter|, source, target)` key so ties break on the group
indices. `--no-intra` continues to drop `source == target` before ranking.

---

## 11. CLI reference (additions)

Added to the existing `pathway` argument group (`build_argument_parser`, L4632), after
`--pathway-top-edges`:

```
--pathway-weight-rule   {signed,abs,positive,negative,mode-only,matrix-only,hybrid}   default signed
--pathway-polarity-rule {sign,post_z,pre_z,delta}                                    default sign
--pathway-hybrid-alpha  FLOAT                                                        default 0.5
```

Plus an **alias** for the mode rule so the specification's spelling works:

```
--pathway-mode-weights  {evr,energy,uniform,value,abs_value}   dest=pathway_weight, default evr
```

All four keep the project's `dest` naming (`pathway_weight_rule`, `pathway_polarity_rule`,
`pathway_hybrid_alpha`, `pathway_weight`) and are passed straight into `MotifConfig` in
`main()` (L4725).

Example commands:

```bash
# default: identical science to the shipped run (new config block only)
python -m src.clustering.functional_motifs \
  -i data/processed/<stem>/eigen.json -o data/processed --stats

# absolute filter scale, group-level receiver polarity
python -m src.clustering.functional_motifs \
  -i data/processed/<stem>/eigen.json -o data/processed \
  --pathway-weight-rule abs --pathway-polarity-rule post_z --stats

# explicit 60/40 mode/matrix blend (requires z_matrix.json next to eigen.json)
python -m src.clustering.functional_motifs \
  -i data/processed/<stem>/eigen.json -o data/processed \
  --pathway-weight-rule hybrid --pathway-hybrid-alpha 0.6 --stats

# matrix-only edges with a delta polarity (requires z_matrix.json)
python -m src.clustering.functional_motifs \
  -i data/processed/<stem>/eigen.json -o data/processed \
  --pathway-weight-rule matrix-only --pathway-polarity-rule delta
```

---

## 12. Config and config hash (decision A)

Three fields are inserted into `MotifConfig` after `pathway_top_edges` (L750):

```python
pathway_weight_rule:   str   = DEFAULT_PATHWAY_WEIGHT_RULE     # "signed"
pathway_polarity_rule: str   = DEFAULT_PATHWAY_POLARITY_RULE   # "sign"
pathway_hybrid_alpha:  float = DEFAULT_PATHWAY_HYBRID_ALPHA    # 0.5
```

`MOTIF_CONFIG_FIELDS` (L408) 26 → **29**; `CONFIG_KEYS` (L468) 30 → **33**.

**Decision (A): the three fields are hashed.** Measured consequences (verified with the pinned
interpreter against the real artifacts):

| Item | Before | After |
|---|---|---|
| Hashed fields | 26 | 29 |
| **Default `config_hash`** | **`3fcaf50e`** | **`e973fece`** |
| `--pathway-weight-rule abs` | — | `73521b46` |
| `--pathway-weight-rule hybrid` (α 0.5) | — | `2cc6ff8c` |
| Reconstructed `outdir` config | `e9acfb54` | `7a609e43` |
| Canonical stem `variant_stem(...)` | `motifs` | `motifs` — **unchanged** |

**Why the canonical filename does not change:** `variant_stem` (L3425) appends `.<hash8>` only
when `force_config_hash` is set or `config.is_default()` is false, and `is_default()` (L795)
inspects **only** the 8 `DEFAULT_DECISION_FIELDS` from 04A. The pathway fields are not in that
list, so a canonical run still writes `motifs.json` / `motifs.npz` / `motifs.data.json` /
`motifs.pathway.graphml`. Only the **recorded hash value** inside them changes.

### 12.1 Required artifact regeneration

Because the hash is hashed-in, the four tracked files under
`data/processed/FB4Yaffect_FB45_999prePost_001_all/` must be regenerated once:

`motifs.json`, `motifs.npz`, `motifs.data.json`, `motifs.pathway.graphml`.

Their **scientific content is unchanged** (22 nodes, 39 edges, `abs_max`, polarity counts,
concentration, NPZ arrays); the diff is confined to the `config` block (3 new keys + the new
`config_hash`). The untracked `outdir/…/motifs.e9acfb54.*` scratch files become
`motifs.7a609e43.*` and can be discarded or regenerated.

### 12.2 Known follow-up (not fixed here)

`is_default()` ignoring the 04D fields is a pre-existing behaviour: a non-default
`--pathway-weight` (or any other 04D flag) run overwrites the canonical artifact instead of
getting its own `.<hash8>` segment. This update does **not** change that (it would alter
long-standing filename behaviour outside this update's scope); it is recorded here as a
follow-up candidate. Decision (A) nonetheless makes the *hash* distinguish the runs, so the
collision is detectable from the file contents.

---

## 13. GraphML metadata

`GRAPHML_GRAPH_KEYS` (L395) 9 → 12.

| Key | Meaning | Change |
|---|---|---|
| `n_nodes`, `n_edges`, `abs_max`, `weight_normalization`, `edge_style` | existing | unchanged |
| `pathway_source` | resolved source | unchanged; now may read `"hybrid"` |
| `pathway_weight_rule` | **mode-weight rule** (`evr`/`energy`/…) | **unchanged meaning** — do not repurpose |
| `pathway_unified_weight_rule` | **new** — `signed`/`abs`/`positive`/`negative`/`mode-only`/`matrix-only`/`hybrid` | added |
| `pathway_polarity_rule` | **new** — `sign`/`post_z`/`pre_z`/`delta` | added |
| `pathway_hybrid_alpha` | **new** — float | added **only when `weight_rule == "hybrid"`** |
| `threshold`, `topN` | existing | unchanged |

**Name-collision note.** The specification's §8 asks the `<graph>` element to record
`pathway_weight_rule`; that key **already exists** and holds the *mode* rule. Renaming or
repurposing it would break every existing GraphML consumer (and the README contract), so the new
unified rule gets the distinct `pathway_unified_weight_rule` name. Both rules are therefore
recorded, unambiguously.

**Injection.** `write_graphml_pathway` (L4047–4084) reads the three new values from `config`
with the same payload/`DEFAULT_*` fallback chain used by the existing five keys;
`pathway_hybrid_alpha` is omitted (not set) for every non-hybrid rule — NetworkX rejects `None`
GraphML values, so omission (not a null) is required. The unknown-key guard (L4086–4100) and the
yFiles `edgegraphics` block are **not touched**: `edge_color` still derives from the `polarity`
field, which now simply carries the rule-derived value.

---

## 14. Validation rules

All new checks live in `_validate_config` (L2881–2969), after the existing 04D range checks
(L2959–2969).

| Check | Severity | Code |
|---|---|---|
| `pathway_weight_rule ∈ PATHWAY_WEIGHT_RULES` | Error | `CODE_CONFIG` |
| `pathway_polarity_rule ∈ PATHWAY_POLARITY_RULES` | Error | `CODE_CONFIG` |
| `pathway_hybrid_alpha` finite and in `[0, 1]` | Error | `CODE_CONFIG` |
| `weight_rule ∈ {matrix-only, hybrid}` or `polarity_rule = delta` without the Phase 02 artifact | Error | `CODE_MATRIX_MISSING` |
| `weight_rule != signed` overriding a non-default `--pathway-source` | Informational | `CODE_WEIGHT_RULE` |
| `post_z` / `pre_z` selected (group-level, row/column-monochromatic colouring) | Informational | `CODE_POLARITY_RULE` |
| the resulting edge set is entirely one-signed under `positive`/`negative` | Informational | `CODE_POLARITY_RULE` |
| `matrix-only` ⇒ per-edge `modes` list is empty (no mode contributes) | Informational | `CODE_WEIGHT_RULE` |
| `grouping = none` ⇒ pathway skipped | Informational | `CODE_PATHWAY_SKIPPED` (existing) |

`CODE_WEIGHT_RULE = "weight_rule"` and `CODE_POLARITY_RULE = "polarity_rule"` are added to the
code block (L710–714), alongside the existing `CODE_MATRIX_MISSING`, `CODE_MATRIX_MISMATCH`,
`CODE_MODE_WEIGHTS`, `CODE_PATHWAY_SKIPPED`.

The frozen payload schema is validated by the **unchanged** `validate_motif_payload_schema`
(L3230): `PATHWAY_KEYS`, `EDGE_KEYS`, `NODE_KEYS` and `TOP_MODE_KEYS` are all **unchanged**, so the
JSON `pathway` block gains no keys. Only `CONFIG_KEYS` grows (30 → 33).

---

## 15. Statistics and terminal output

`render_statistics` (L4308) prints the pathway block from `analysis.pathway` /
`analysis.metadata`; the existing lines are unchanged, and one line is added under the pathway
block:

```
  pathway rules       : weight signed / polarity sign   (hybrid alpha 0.5)
```

with the alpha segment printed only for `hybrid`. `render_summary_box` (L4363) adds the same
information to the `Pathway:` line:

```
| Pathway: 22 nodes / 39 edges (source mode, weight evr, unified signed, polarity sign) |
```

Neither function changes any of the existing lines, so the existing terminal-output tests keep
passing; the `--stats` block remains terminal-only and separate from any GUI window (Phase 03
precedent).

---

## 16. Determinism

* `resolve_unified_weight` and `resolve_edge_polarity` are **pure functions** of
  `(W_mode, W_matrix, config)` — no RNG, no clock, no dict-order dependence.
* Group reductions reuse the existing `indicatorᵀ · M · indicator` formulation (L2394–2396), whose
  summation order is fixed by numpy for a given shape ⇒ bit-reproducible.
* The hybrid blend is a single fixed-order `α·a + (1−α)·b`; no reordering, no `np.average`.
* `pre_z`/`post_z` sums are `numpy` reductions over a fixed member order (ascending neuron index)
  and a fixed mode order (rank order).
* `created_utc` continues to honour `SOURCE_DATE_EPOCH`; `motifs.json`, `motifs.npz` and
  `motifs.pathway.graphml` are byte-identical across repeated runs at a fixed epoch, for every
  rule combination.
* The GraphML writer keeps the fixed node/edge/attribute order and the sorted-key JSON in
  `edge_style`, so the new keys cannot perturb it.

---

## 17. Invariants (asserted in tests)

1. **Backward compatibility (the anchor):** the defaults reproduce the reference run exactly —
   22 nodes, **39 edges**, `abs_max = 2.345299756658418`, 0 positive / 39 negative, 7 intra /
   32 cross, `weight_concentration = 0.05584213591659331`. The only diffs are the 3 new `config`
   keys and `config_hash 3fcaf50e → e973fece`.
2. `Σ_m |w_m| == 1` for every `--pathway-weight` rule (unchanged).
3. `hybrid(α=1) ≡ mode-only` and `hybrid(α=0) ≡ matrix-only`, element-wise.
4. `abs` and `signed` yield the **same** kept edge set and the same `abs_max` (the no-op is
   pinned).
5. Each one-sided rule keeps only its own sign (`positive` ⇒ `weight >= 0`, `negative` ⇒
   `weight <= 0`) and the two are **disjoint**; both rescale `abs_max` to the strongest
   *surviving* contribution, so neither is a subset of `signed` in general (verified:
   `negative` keeps 2 synthetic / 39 reference edges that `signed` rejects because `signed`'s
   larger `abs_max` puts them below its cutoff).
6. `polarity == sign(weight)` when `pathway_polarity_rule == "sign"` (the existing 04D invariant).
7. `polarity ∈ {-1, 0, +1}` for every rule; `delta` equals `sign(W_mode − W_matrix)` element-wise.
8. Every kept edge satisfies `|W_filter| ≥ edge_threshold · abs_max` (with the existing `1e-12`
   slack).
9. Per-source top-N never exceeds `--pathway-top-edges`; ties break on `(source, target)`.
10. `matrix-only` / `hybrid` / `polarity_rule = delta` without the Phase 02 artifact ⇒
    `CODE_MATRIX_MISSING` error, exit code `1`, **no output directory created** (matching the
    existing `test_04d_matrix_source_requires_the_phase02_artifact`).
11. GraphML is byte-deterministic per rule combination; the three new `<graph>` keys are present
    with the right values, and `pathway_hybrid_alpha` appears **only** for `hybrid`.
12. `config.config_hash()` is unchanged by GraphML writing (the existing 04D invariant).
13. `PATHWAY_KEYS`, `EDGE_KEYS`, `NODE_KEYS`, `TOP_MODE_KEYS` are unchanged; `CONFIG_KEYS` is 33.
14. No `*.csv` is produced.
15. The Phase 03 `SpectralDecomposition` object and the Phase 02 `ZMatrix` object are never
    mutated by any rule.

---

## 18. Implementation plan (ordered)

**Step 1 — constants & codes.** Add `DEFAULT_PATHWAY_WEIGHT_RULE`, `DEFAULT_PATHWAY_POLARITY_RULE`,
`DEFAULT_PATHWAY_HYBRID_ALPHA`, `PATHWAY_WEIGHT_RULES`, `PATHWAY_POLARITY_RULES` (L286–299);
`CODE_WEIGHT_RULE`, `CODE_POLARITY_RULE` (L710–714).

**Step 2 — config.** Add the 3 fields to `MotifConfig` (L750); extend `MOTIF_CONFIG_FIELDS`
(L408). `hash_fields`/`to_dict`/`from_dict`/`config_hash` need no code change (they iterate the
tuple) but their values change (§12).

**Step 3 — engine.** Add `resolve_unified_weight(...)` and `resolve_edge_polarity(...)` in the
`04D -- pathway graph` block (~L2320), immediately after `group_pair_contributions`. Both raise
`MotifValidationError` on shape/rule violations, mirroring the existing helpers.

**Step 4 — filtering.** Extend `filter_pathway_edges` with the optional `matrix=None` keyword and
let it report `abs_max_mode` / `abs_max_matrix` / `abs_max_raw` in `stats`.

**Step 5 — `build_pathway`.** Compute `mode_matrix` (always) and `matrix_matrix` (when
`needs_matrix`); call `resolve_unified_weight`; pass `W_filter` to `filter_pathway_edges`; store
`weight = W_viz`, `abs_weight = |W_filter|`; derive `polarity` via `resolve_edge_polarity`;
retarget the L2640 self-check; extend `diagnostics` with the new keys (set to `None` on the
neutral `grouping=none` path).

**Step 6 — matrix load condition.** In `build_motif_analysis` (L3176–3191) replace
`str(config.pathway_source) in ("matrix", "both")` with the `needs_matrix` expression (§7).

**Step 7 — CLI.** Add the three flags (plus the `--pathway-mode-weights` alias) to the `pathway`
group (L4632) and thread them into `MotifConfig(...)` in `main()` (L4725).

**Step 8 — validation.** Add the new checks to `_validate_config` (L2959–2969) and the
informational log lines in `build_motif_analysis`.

**Step 9 — GraphML.** Extend `GRAPHML_GRAPH_KEYS` (L395) and the graph-metadata assembly in
`write_graphml_pathway` (L4047–4084).

**Step 10 — terminal output.** Extend `render_statistics` (L4308) and `render_summary_box`
(L4363) with the new rule line (existing lines untouched).

**Step 11 — re-exports.** Add the new public symbols to `src/clustering/__init__.py` (the lazy
`__getattr__`/`__all__` pattern already used for `PATHWAY_WEIGHTS`, `GRAPHML_GRAPH_KEYS`, …).

**Step 12 — tests & README.** Add ≈60 tests (fast + slow) and the README rows.

**Step 13 — regenerate artifacts.** Re-run the canonical Phase 04D command once to refresh the
four tracked artifacts (§12.1).

---

## 19. Testing strategy

**Framework:** the existing `pytest.ini` (`pythonpath = .`, `--strict-markers`, the registered
`slow` marker). No new markers, no new dependencies. The Phase 04 test modules already import
matplotlib lazily / pin a headless backend as required.

```
fast   /home/mr-miracle/miniconda3/envs/eigen-decomposition/bin/python -m pytest -m "not slow" -q
full   /home/mr-miracle/miniconda3/envs/eigen-decomposition/bin/python -m pytest -q
```

**Baseline:** 598 tests collected before this update. This update adds **105** (**96** fast +
**9** slow) for a total of **703** collected; the full suite passes in ≈117 s. Every existing test
must still pass; there are exactly **three** sanctioned edits to existing tests:

1. `tests/test_motif_schema.py::test_key_sets_are_exactly_the_frozen_ones` —
   `len(CONFIG_KEYS) == 30` → `== 33  # 29 hashed + …`.
2. `tests/test_functional_motifs.py::test_config_has_26_hashed_fields` — renamed to
   `test_config_has_29_hashed_fields`, `26 → 29`, plus the three new field names.
3. `tests/test_functional_motifs.py::test_04d_build_pathway_structure` — the `diagnostics`
   key-set assertion gains the 7 new keys (this test uses an **exact** set comparison).

`GRAPHML_GRAPH_KEYS` is only ever asserted as a subset (`<=`) by the existing tests, so the three
new GraphML graph keys need no edit.

### 19.1 New test surface

* **`resolve_unified_weight` unit** — each of the 7 rules against hand-computed 2×2 / 3×3
  fixtures; the `needs_matrix` flag per rule; `abs_max`; shape/rule validation errors; the α
  boundary values 0, 1, and the range error.
* **`resolve_edge_polarity` unit** — all 4 rules including the `0 → 0` case; `delta` requiring the
  matrix; the group-level `pre_z`/`post_z` reduction checked against a brute-force sum over
  members and modes.
* **Rule × source precedence** — `signed`/`abs`/`positive`/`negative` × `{mode,matrix,both}`;
  the `CODE_WEIGHT_RULE` override log line fires exactly when expected; `hybrid` at
  α ∈ {0, 0.25, 0.5, 0.75, 1}.
* **Filtering invariants** — `abs ≡ signed` edge set; `positive`/`negative` ⊆ `signed`;
  threshold respected; per-source top-N with ties; `--no-intra`; the all-zero matrix.
* **CLI** — every new flag reaches the config; defaults are `signed` / `sign` / `0.5`; every enum
  value parses; `--pathway-mode-weights uniform` sets `pathway_weight`; `--pathway-hybrid-alpha 1.5`
  and `-0.1` ⇒ `CODE_CONFIG`.
* **Errors** — `matrix-only` / `hybrid` / `delta` without `z_matrix.json` ⇒ `CODE_MATRIX_MISSING`,
  exit `1`, no `outdir`; a mismatched neuron order ⇒ `CODE_MATRIX_MISMATCH`.
* **Schema** — `CONFIG_KEYS == 33`; `PATHWAY_KEYS` / `EDGE_KEYS` / `TOP_MODE_KEYS` unchanged;
  negative cases for each new config key (missing/extra/wrong-type).
* **GraphML** — the three new keys carry the right values; `pathway_hybrid_alpha` present **only**
  for `hybrid`; the existing `pathway_weight_rule` still holds the *mode* rule; the unknown-key
  guard still trips; byte-determinism across repeated writes; `config_hash` unaffected by the
  GraphML write.
* **Backward compatibility (the anchor)** — a full default run reproduces §17 item 1 exactly and
  records `config_hash == "e973fece"`.
* **Integration (`@pytest.mark.slow`, ≈5)** — the real reference `eigen.json` under `signed`,
  `abs`, `positive`, `negative`, and `hybrid(0.5)`; one end-to-end `main()` CLI run per rule with
  `--stats`; the summary box showing the new rules.

### 19.2 Test discipline

No new test may assert a *changed* scientific number. The reference numbers in §17 item 1 are
frozen; only the two config-hash values (§12) and the `CONFIG_KEYS` count change.

---

## 20. Risks & mitigations

| Risk | Mitigation |
|---|---|
| **`pathway_weight_rule` name collision** in GraphML (existing key = mode rule) | Keep the existing key bound to the mode rule; the new rule uses `pathway_unified_weight_rule`; a test asserts the old key still carries `evr`/… |
| **Config-hash churn** invalidates the four tracked artifacts | Decision (A) accepted; the change is content-only (no filename change, §12), the scientific numbers are unchanged, and the four files are regenerated (§12.1) |
| `abs` is a **no-op** — a rule that appears to do nothing | Implemented faithfully, documented in §6.3, pinned by an explicit test |
| `pre_z`/`post_z` give **monochromatic rows/columns** | Documented in §8.1; an informational log line fires; the per-mode alternative is recorded as a future option |
| `delta` needs `Z` even with `--pathway-source mode` | `needs_matrix` includes `polarity_rule == delta`; a missing artifact is a hard, actionable error |
| `hybrid` rescales stored weights (vs the unweighted `both`) ⇒ `abs_max`/`weight_ratio` shift | `abs_max` is recomputed from `W_filter`, so `weight_ratio ∈ [0, 1]` always; the GraphML label shows the true value |
| Per-edge `top_modes` are meaningless under `matrix-only`/`hybrid` | `top_modes`/`modes` are still emitted from the **mode** decomposition; `modes` is empty under `matrix-only` and an informational log line notes it |
| Two overlapping selectors (`--pathway-source` vs `--pathway-weight-rule`) confuse users | §7 precedence table + an always-logged override when both are non-default, and the resolved choice recorded in `pathway.source` and the GraphML key |
| Pre-existing latent bug: non-default 04D fields do not affect `is_default()`, so they overwrite the canonical artifact | Explicitly **out of scope** (§12.2); recorded as a follow-up; the hash still distinguishes the runs |
| Growing `motifs.json` (`config` gains 3 keys) | Negligible; `PATHWAY_KEYS`/`EDGE_KEYS` stay frozen, so edge-level size is unchanged |

---

## 21. Acceptance criteria

1. The three new options exist, parse, validate and reach `MotifConfig`; defaults are
   `signed` / `sign` / `0.5`; `--pathway-mode-weights` is accepted as an alias.
2. The default run reproduces the reference science exactly (22 nodes, 39 edges,
   `abs_max = 2.345299756658418`, 0 pos / 39 neg, 7 intra / 32 cross,
   `weight_concentration = 0.05584213591659331`) and records `config_hash = "e973fece"`.
3. Each of the 7 weight rules produces the specified `W_filter` / `W_viz`; each of the 4 polarity
   rules produces `polarity ∈ {-1, 0, +1}` per §8.
4. `matrix-only` / `hybrid` / `delta` require the Phase 02 artifact and fail hard with
   `CODE_MATRIX_MISSING` (exit `1`, no output written) without it.
5. Threshold, top-N, intra, tie-breaks, concentration and the neutral (`grouping=none`) path
   behave exactly as before.
6. The GraphML `<graph>` element records the two new rules (+ α for `hybrid`) without disturbing
   the existing keys; `edge_color` follows the rule-derived `polarity`.
7. All 598 existing tests pass (plus the three sanctioned edits) and **105** new tests pass
   (**703** total); `--strict` passes on the reference dataset; exit codes `0/1/2`.
8. Byte-determinism under a fixed `SOURCE_DATE_EPOCH` for every rule combination; no CSV; no
   Phase 01/02/03 file, test or artifact changes (`git diff --stat` touches only
   `src/clustering/*`, the two Phase 04 test modules, `README.md`, this plan, and the four
   regenerated `motifs.*` artifacts).

---

## 22. Deliverables

- `execution-plans/03_update_a.md` — this document.
- `src/clustering/functional_motifs.py` — new constants/codes, `resolve_unified_weight`,
  `resolve_edge_polarity`, `filter_pathway_edges` + `build_pathway` extensions, the
  `needs_matrix` load condition, 3 config fields, 3 CLI flags + 1 alias, `_validate_config`
  rules, the 3 GraphML graph keys, the terminal-output lines.
- `src/clustering/__init__.py` — lazy re-exports of the new public symbols.
- `tests/test_functional_motifs.py`, `tests/test_motif_schema.py` — 105 new tests plus the three
  sanctioned existing-test edits (§19).
- `README.md` — the three new CLI rows, the weight-rule and polarity-rule tables, the GraphML key
  list (`pathway_unified_weight_rule`, `pathway_polarity_rule`, `pathway_hybrid_alpha`), the
  `abs` no-op note, the `pre_z`/`post_z` caveat, and the updated `config_hash`.
- Regenerated `data/processed/FB4Yaffect_FB45_999prePost_001_all/motifs.{json,npz,data.json,pathway.graphml}`.
- **No changes** to `src/utils/io.py`, `src/parsing/*`, `src/matrices/*`, `src/spectral/*`,
  `pytest.ini`, `environment.yml`, or any Phase 01/02/03 artifact.

---

## 23. Out of scope (explicit non-goals)

Grouping logic (04C); the participation matrix; motifs/links/families (04A/04B); the NPZ array
set; `nodegraphics` / `edgegraphics` formatting and the yEd styles / node geometry; the JSON
`pathway` block key set; the Phase 01/02/03 modules, artifacts and tests; `pytest.ini`;
`environment.yml`; multi-file union; an `--interactive` window; and the `is_default()` follow-up
(§12.2).

---

## 24. Resolved decisions

1. **Config-hash strategy — (A): the three new fields are hashed.** The default `config_hash`
   changes `3fcaf50e → e973fece`; the canonical *filenames* do **not** change (§12); the four
   tracked Phase 04D artifacts are regenerated once.
2. **`pre_z`/`post_z` reduction — group-level** over signed participation,
   `sign(Σ_{members} Σ_m P̃[·,m])` (§8.1), with the row/column-monochromatic consequence
   documented and logged.
3. **CLI naming — `--pathway-mode-weights` is added as an alias** of the shipped
   `--pathway-weight` (same `dest`), so both spellings work and no existing command line breaks.

---

## 25. Verified reference numbers (update A)

Measured with the pinned interpreter on
`data/processed/FB4Yaffect_FB45_999prePost_001_all/eigen.json` (113 neurons, 21 modes, 22
groups), and pinned by the `@pytest.mark.slow` tests in `tests/test_functional_motifs.py`.

### 25.1 Weight rules (`--pathway-source mode` unless noted)

| Rule | source | edges | `abs_max` | pos / neg | intra / cross | concentration |
|---|---|---:|---:|---|---|---:|
| `signed` (default) | `mode` | 39 | 2.345299756658418 | 0 / 39 | 7 / 32 | 0.05584213591659331 |
| `abs` | `mode` | **39** (identical set) | 2.345299756658418 | 0 / 39 | 7 / 32 | 0.05584213591659331 |
| `positive` | `mode` | 13 | **0.12092130881131712** | 13 / 0 | 0 / 13 | 0.14397661726515523 |
| `negative` | `mode` | 39 | 2.345299756658418 | 0 / 39 | 7 / 32 | **0.05698162285236808** |
| `hybrid` (α 0.5) | `hybrid` | 53 | **4.7958655723736365** | 5 / 48 | 7 / 46 | 0.046509305936680886 |

Diagnostics for the hybrid run: `abs_max_mode = 2.345299756658418`,
`abs_max_matrix = 7.246431388088855`, `abs_max_raw = 4.7958655723736365`. For `positive`:
`abs_max = 0.12092130881131712` against `abs_max_raw = 2.345299756658418`.

**`abs` is an exact no-op** (same 39 edges, same `abs_max`); **`negative` matches `signed`'s edge
set but not its concentration** (its filter zeroes the positive entries, which are not kept
anyway but do contribute to `total`); **`positive`/`negative` are not subsets of `signed`**
because each rescales `abs_max` to its own strongest surviving contribution (§6.1).

### 25.2 Polarity rules (default weight rule)

| `--pathway-polarity-rule` | edges | pos / neg |
|---|---:|---|
| `sign` (default) | 39 | 0 / 39 |
| `pre_z` | 39 | **17 / 22** |
| `post_z` | 39 | **23 / 16** |
| `delta` | 39 | **39 / 0** |

### 25.3 Default artifact diff (the backward-compatibility anchor)

Regenerating the canonical artifact at a fixed `SOURCE_DATE_EPOCH` changes **only**:

* `config`: 3 new keys (`pathway_weight_rule = "signed"`, `pathway_polarity_rule = "sign"`,
  `pathway_hybrid_alpha = 0.5`) and `config_hash 3fcaf50e → e973fece`;
* `metadata.config_hash` (the recorded duplicate) — same value change;
* `motifs.npz`: only `source_json_sha256` (the digest of the changed JSON bytes);
* `motifs.pathway.graphml`: the 3 new `<graph>` keys **plus** the pre-existing yFiles styling,
  which the previously committed `data/processed/…/motifs.pathway.graphml` lacked (it was a
  stale pre-styling draft with zero `ygraphml` blocks; the current writer emits 76). This is a
  one-time catch-up, not caused by update A.

Everything else is byte-identical: the `pathway` block, `motifs.*`, `links`, `families`,
`groups` and every other NPZ array.

