# Phase 02 — Build Square Z‑Matrix

*Implementation: `src/matrices/build_square_matrix.py` (with `src/matrices/__init__.py`).*
*Artifacts: `data/processed/<stem>/z_matrix.json`, `z_matrix.npz`, `z_matrix.png`, `z_matrix.data.json`.*
*Tests: `tests/test_build_square_matrix.py`, `tests/test_z_matrix_schema.py` (+ `tests/fixtures/zero_z.gv`).*

---

## 1. Goals

Turn the Phase 01 artifact into the **square functional connectivity matrix `Z`** that
Phases 03–06 consume. Phase 02 is the **only** owner of the pre/post → unified-weight
computation, of the neuron row/column index order, and of the directed / symmetrized
matrix variants.

Pipeline order is fixed and recorded in every artifact:

```
fuse (unified weight)  ->  assemble Z  ->  normalize  ->  symmetrize
```

Phase 02 performs **no spectral analysis, no clustering, no dimensionality reduction and
no pair filtering** (Phase 01 owns filtering).

**Purely additive.** No existing file is modified or deleted: `src/utils/io.py`,
`src/parsing/*`, `pytest.ini`, `environment.yml`, the existing tests and the existing
`parsed_graph.json` stay byte-identical. No new dependencies (numpy + matplotlib, both
already pinned).

## 2. Scope boundary

| Owned by Phase 02 | Explicitly **not** Phase 02 |
|---|---|
| pre/post unification (`unified_weight`) | `.gv` parsing / hub collapse (Phase 01) |
| dense matrix assembly + neuron index order | pair filtering / thresholding (Phase 01 CLI) |
| diagonal-zero policy, duplicate-coordinate checks | eigendecomposition, SVD, scree plots (Phase 03) |
| `--normalize` variants, `--symmetric` + `--symmetrize` | clustering, UMAP, loadings (Phase 04) |
| the four artifacts, `--plot`, `--stats`, `--save-data` | dynamical models, Jacobians (Phases 05–06) |
| `load_z_matrix()` / `load_sidecar_arrays()` for Phase 03+ | multi-file union (deliberately unsupported, §24) |

## 3. Inputs — the Phase 01 → Phase 02 contract

`data/processed/<stem>/parsed_graph.json` (or `parsed_graph.filtered.json`), with:

- `neurons: [{neuron_id, cent, out_degree, in_degree, pre_strength, post_strength}]` —
  the row/column order source, already sorted by `neuron_id`;
- `pairs: [{source, target, pre_z, post_z}]` — 1355 entries for the reference dataset,
  sorted by `(source, target)`; **no `weight` key** (Phase 02 owns it);
- `file_sha256` / `source_file` / `metadata.created_utc` — provenance;
- `metadata.density` / `metadata.reciprocity` — **copied verbatim** into the Phase 02
  artifacts (recomputed only when absent);
- `metadata.filters` — non-empty ⇒ filtered input (affects artifact naming, §10).

The input is re-validated with Phase 01's own `validate_payload_schema` (imported, not
modified). Phase 02 never re-parses `.gv` files.

## 4. Outputs — the artifact set

Four artifacts per input, all under `<outdir>/<stem>/` where `<stem>` is the source
`.gv` stem (so Phase 02 lands next to its input):

| Artifact | Written | Purpose |
|---|---|---|
| `z_matrix.json` | always | **canonical, self-sufficient** matrix + provenance + config + metadata |
| `z_matrix.npz` | unless `--no-sidecar` | derived array cache (strictly a function of the JSON bytes) |
| `z_matrix.png` | `--plot` | heatmap of the effective matrix (**plus** a GUI popup) |
| `z_matrix.data.json` | `--save-data` | per-pair weights + histogram + norms + statistics |

`--interactive` (and `--no-popup`) write **no** artifact of their own: the tooltip window
renders the in-memory effective matrix and only changes what appears on screen (§16.1).

**Naming policy** (`variant_stem`): the canonical config on a canonical input produces the
names above; a Phase 01 *filtered* input inserts `.filtered`, and any **non-default
matrix config** inserts `.<config_hash8>` — so a variant run can never clobber the
canonical artifact:

```
z_matrix.json                                   default config, canonical input
z_matrix.filtered.json                          default config, filtered input
z_matrix.<config_hash8>.json                    non-default config
z_matrix.filtered.<config_hash8>.json           non-default config + filtered input
```

`--config-hash` forces the hash segment even for a default config. All four artifacts
share the same variant stem (`z_matrix.<hash8>.png`, `z_matrix.<hash8>.data.json`, …).

**No CSV files are produced or consumed anywhere in this project.**

## 5. `z_matrix.json` schema

```json
{
  "provenance": {
    "source_artifact": "data/processed/<stem>/parsed_graph.json",
    "source_artifact_sha256": "<sha256 of the Phase 01 JSON bytes>",
    "source_file": "<stem>.gv",
    "source_file_sha256": "c44c24f85756ed07acc1184565b19407df6fa4e0df8493c761f7d84957fa6b57",
    "parsed_created_utc": "2026-09-16T00:44:24Z",
    "phase": "02"
  },
  "config": {
    "unification": "w = s * tanh(alpha * log(post_z / (pre_z + eps))), s = (|pre_z| + |post_z|) / 2",
    "eps": 0.1, "alpha": 1.0, "zero_policy": "zero-in-zero-out",
    "symmetric": false, "symmetrize": "mean", "normalize": "none",
    "dtype": "float64", "filters": [], "config_hash": "1aa02137"
  },
  "neuron_order": ["DNa03_R_1", "ExR7(ring)_L_1", "..."],
  "pairs": [{"source": "...", "target": "...", "pre_z": 0.21, "post_z": 0.17, "weight": -0.102144}],
  "matrix": [[0.0, -0.102144, "..."], ["...", 0.0, "..."]],
  "matrix_symmetric": [[0.0, -0.09997766037735847, "..."], ["...", 0.0, "..."]],
  "metadata": { "...": "see below" }
}
```

- `matrix` — the **effective** matrix: `matrix_symmetric` when `config.symmetric`, else the
  normalized directed `Z`. Diagonal exactly `0.0`, missing edges `0.0`.
- `matrix_symmetric` — **always present**, symmetric by construction. In symmetric mode it
  is the same array as `matrix`; otherwise it is the `(Z + Zᵀ)/2`-style variant. Phase 03's
  eigen path therefore always has a symmetric matrix to read.
- `neuron_order` — a **list**, not an object: `dumps_json` sorts object keys, which would
  destroy the index ordering if this were a map. The `{id: index}` lookup is derived.
- `pairs[].weight` — the **pre-normalization** unified weight, so the raw directed `Z` is
  fully recoverable from `pairs` alone even under `--normalize`.

`metadata` keys (exact): `n_neurons`, `n_edges`, `n_stored_nonzero`, `n_matrix_zeros`,
`n_weight_positive`, `n_weight_negative`, `n_weight_zero`, `n_self_loops`, `sparsity`,
`density`, `reciprocity`, `matrix_stats`, `weight_stats`, `row_norm_stats`,
`col_norm_stats`, `frobenius_norm`, `largest_singular_value`, `spectral_radius`,
`symmetrized_spectral_radius`, `normalization_scale`, `config_hash`, `numpy_version`,
`generator`, `created_utc`.

**Statistic conventions** (documented because they are easy to get wrong):

* `matrix_stats` (`min/max/mean/std`, `std` with `ddof=0`) is over the **stored (non-zero)
  entries** of the effective matrix; the absent-edge zeros are summarized by `sparsity`.
* `weight_stats` is over the raw unified weights of **all pairs including zeros**
  (so `mean` = −0.095957 while `matrix_stats.mean` = −0.096312 on the reference data).
* `row_norm_stats` / `col_norm_stats` summarize the L1 norms `Σ_j |Z_ij|`.
* `spectral_radius` = `max |eigenvalue|` of the **effective** matrix, and is `null` unless
  symmetric mode was used (per the CLI spec); `symmetrized_spectral_radius` is the same
  quantity for `matrix_symmetric` and is always available (informational only — Phase 03
  computes the authoritative spectra).
* `sparsity` = fraction of the `N²` entries that are exactly zero (reference: 0.894275).

Writing rules: `write_json_atomic` + `dumps_json` (sorted keys, indent 2, trailing
newline, UTF-8, temp file + `os.replace`). Because `dumps_json` cannot disable `NaN`/
`Infinity`, every emitted float is asserted finite before serialization. `created_utc` is
the only non-deterministic field and honours `SOURCE_DATE_EPOCH`.

## 6. `z_matrix.data.json` schema (`--save-data`)

Keys (exact): `provenance`, `config`, `symmetric`, `normalize`, `pairs`, `z_stats`,
`histogram`, `sparsity`, `n_positive`, `n_negative`, `n_zero`, `row_norms`, `col_norms`,
`row_norm_stats`, `col_norm_stats`, `spectral_radius`, `density`, `reciprocity`,
`generator`, `created_utc`.

* `pairs` — the unified weights per pair: `{source, target, pre_z, post_z, weight}`.
* `symmetric` / `normalize` — the mode flags, duplicated here for convenience.
* `z_stats` — `min/max/mean/std` of the effective matrix's stored entries.
* `histogram` — `bins` (edges, `n_bins+1` values), `counts`, `n_bins`, `range`,
  `zero_count`, `degenerate`. Computed over the **stored (non-zero)** entries; absent-edge
  zeros are reported separately via `zero_count`/`sparsity`.
  A numerically degenerate range (all stored weights identical — e.g. `--normalize rows`
  makes them ±1) collapses to a **single bin** with `degenerate: true`, because numpy
  cannot split a zero-width float range.
* `row_norms` / `col_norms` — full L1 vectors (length `N`); `*_norm_stats` their summaries.
* `spectral_radius` — `null` unless `--symmetric`.
* `density` / `reciprocity` — copied from the Phase 01 artifact.

## 7. `z_matrix.npz` sidecar specification

A **derived cache**, never a second source of truth. Its content is a pure function of the
canonical JSON bytes, and it is keyed by that JSON's SHA-256. Nothing in the JSON refers to
the cache, so the §5 schema and JSON determinism are untouched (there is deliberately no
reverse pointer, which would create a cycle).

Fixed array bundle (ordered, little-endian dtypes `"<f8"`/`"<i8"`/`"<U*"`):

| Key | Shape/dtype | Meaning |
|---|---|---|
| `matrix` | `(N, N)` `<f8` | the effective matrix (so `Z` or `Z_sym`) |
| `matrix_symmetric` | `(N, N)` `<f8` | the symmetrized matrix |
| `neuron_order` | `(N,)` `<U*` | row/column index order |
| `edge_rows`, `edge_cols` | `(P,)` `<i8` | `pairs` → matrix coordinates, 1:1 aligned |
| `edge_weights` | `(P,)` `<f8` | unified weight per pair (pre-normalization) |
| `source_json_sha256` | `(1,)` `<U64` | the JSON's digest (cache key / staleness check) |
| `numpy_version` | `(1,)` `<U64` | pinned numpy version (the `.npy` header is part of the byte layout) |

* **Byte-reproducible**: numpy writes every zip entry with the constant
  `date_time = (1980, 1, 1, 0, 0, 0)` and the `.npy` header carries no timestamp, so two
  runs produce identical bytes. Verified empirically (two separate processes → identical
  sha256) and asserted in `test_sidecar_archives_are_written_with_a_fixed_timestamp`.
* **Atomic**: built in memory (`io.BytesIO`) then written to a temp file in the destination
  directory, flushed, `fsync`-ed and `os.replace`-d — a reader never sees a partial cache.
  `write_npz_atomic` lives in the Phase 02 module; `src/utils/io.py` is untouched.
* **Unpickled**: everything is read with `allow_pickle=False`.
* **Staleness**: `load_z_matrix` verifies the digest plus exact array equality; a stale or
  tampered cache is ignored with a warning and the JSON wins. Reading a tampered `.npz`
  directly raises `ZMatrixValidationError`.
* Writes are ordered JSON first, then the cache, so a failed cache write can never lose the
  artifact of record.

## 8. Unification rule — exact specification

```
s = (|pre_z| + |post_z|) / 2                      # strength
g = tanh(alpha * log(post_z / (pre_z + eps)))     # bounded gain, g ∈ (-1, 1)
w = s * g
```

Defaults **`eps = 0.1`, `alpha = 1.0`** reproduce every worked example in
`execution-plans/zscore_context.md` exactly: `0.44`, `0.0` (neutral `0.6/0.7`), `-0.5077`,
`0.2172`, `-0.0362`.

Edge-case contracts:

1. **`zero_policy = "zero-in-zero-out"` (default)** — if `pre_z == 0` or `post_z == 0`
   then `w = 0.0`. A zero z-score means "no measurable relationship"; the raw formula would
   instead drive it to the attenuation limit (`pre=0.5, post=0 → −0.25`). On the reference
   dataset both policies are identical (min z = 0.1). `zero_policy = "formula"` restores
   the raw limit and is exposed via `--zero-policy` for completeness.
2. **`-0.0` is canonicalized to `0.0`** (`json.dumps(-0.0) == "-0.0"` would break byte
   determinism and make "exact zero" ambiguous).
3. All evaluation runs inside `np.errstate(divide="ignore", invalid="ignore")`, so a
   `log(0)` never emits a `RuntimeWarning` (asserted by a subprocess test on the
   `zero_z.gv` fixture).
4. `eps` must be finite and `> 0` (with `eps = 0` and `pre_z = 0` the ratio is `0/0`);
   `alpha` must be finite and `> 0`.
5. Properties asserted in tests: `|g| < 1`, `|w| ≤ s`, `w` non-decreasing in `post_z`,
   non-increasing in `pre_z`, and larger `alpha` ⇒ larger `|w|`.

## 9. Normalization (`--normalize`) and symmetrization (`--symmetric`)

Applied to `Z` in this order: **normalize → symmetrize**, so
`matrix_symmetric == symmetrize(matrix)` always holds.

| `--normalize` | Rule | Effect |
|---|---|---|
| `none` (default) | identity | pristine |
| `rows` | divide each row by `Σ_j \|Z_ij\|` | each non-empty row has L1 norm 1 |
| `cols` | divide each column by its L1 norm | each non-empty column has L1 norm 1 |
| `unit` | divide by `max \|Z\|` | entries bounded to `[-1, 1]` |
| `spectral` | divide by the largest singular value | `σ₁ = 1` |
| `zscore-nonzero` | standardize the **stored** weights in place, zeros stay zero | stored weights have mean 0 / std 1 |

All-zero inputs are returned unchanged (no division by zero). The applied scale is recorded
in `metadata.normalization_scale` (`null` for vector-valued and no-op cases).

**`--symmetric`** computes a symmetric matrix and makes it the effective one, satisfying the
requirement that *both the JSON and the NPZ contain `Z_sym` instead of `Z`*:

- It is the **default and required** prerequisite for a true eigen-decomposition:
  symmetric matrices have real eigenvalues and orthogonal eigenvectors, which is what
  Phase 03's spectral pipeline needs when symmetric modes are desired (the directed `Z`
  path uses SVD instead).
- `--symmetrize` selects the rule: `mean` = `(Z + Zᵀ)/2` (default; exactly symmetric in
  IEEE-754), `sum` = `Z + Zᵀ`, `max-abs`, `min-abs` (sign preserved).
- Both the JSON (`config.symmetric`) and the save-data file (`symmetric`) record whether
  symmetric mode was used; the sidecar's `matrix` array is `Z_sym` as well.
- Reference (symmetric + `unit`): 2136 stored entries, sparsity 0.832720, spectral radius
  4.317409, config hash `f8652585`.

## 10. Config hash and filename policy

`config_hash` = first 8 hex characters of `sha256(dumps_json(hash_fields))`, where
`hash_fields` covers exactly the **matrix-defining** fields:

```
unification, eps, alpha, zero_policy, symmetric, symmetrize, normalize, dtype, filters
```

* `filters` is the Phase 01 `metadata.filters` list of the input artifact, so the hash also
  captures upstream filtering — as required.
* Presentation/IO flags (`--plot`, `--save-data`, `--stats`, `--no-sidecar`,
  `--config-hash`, `--dry-run`) are **deliberately excluded**: they do not change a single
  matrix entry, and hashing them would stop a default-config run from producing the
  canonical `z_matrix.json`.
* `is_default()` compares only the CLI-controlled fields (`eps`, `alpha`, `zero_policy`,
  `symmetric`, `symmetrize`, `normalize`, `dtype`) — **not** `filters` — so a filtered input
  still yields `z_matrix.filtered.json` rather than an extra hash segment.
* Verified hashes: default `1aa02137`; `symmetric=True` `f5688fee`; `normalize=rows`
  `b293d30a`; `eps=0.2` `da730bb6`; `alpha=2.0` `dc54e1fa`; `symmetrize=sum` `3cc081f7`;
  `zero_policy=formula` `aec91950`; with filters `82591f6a` (still `is_default()`).

## 11. Module layout

```
src/matrices/__init__.py               # lazy __getattr__ re-exports (Phase 01 pattern)
src/matrices/build_square_matrix.py     # rule, assembly, normalize/symmetrize, artifacts, CLI
tests/test_build_square_matrix.py      # behaviour, artifacts, CLI, invariants
tests/test_z_matrix_schema.py          # schema + artifact-contract tests
tests/fixtures/zero_z.gv               # a zero z-score (edge-case guard)
```

Unchanged: `src/utils/io.py`, `src/parsing/*`, `pytest.ini`, `parsed_graph.json`, both
Phase 01 test modules (apart from the README documentation update in §23).
`environment.yml` gains exactly one visualisation dependency, `mplcursors`, needed only by
`--interactive` (§16.1).

## 12. Data structures

```python
@dataclass(frozen=True)
class ZMatrixConfig:      # eps, alpha, zero_policy, symmetric, symmetrize, normalize, dtype, filters
    def to_dict() -> dict         # payload written to `config` (incl. config_hash)
    def hash_fields() -> dict     # built directly, never via to_dict() (would recurse)
    def config_hash() -> str      # 8 hex chars
    def is_default() -> bool

@dataclass(frozen=True)
class WeightedPair:       # source, target, pre_z, post_z, weight

@dataclass
class ZMatrix:            # provenance fields, config, neuron_order, pairs,
                          # matrix, matrix_symmetric, metadata, diagnostics, loaded_from
    @property n_neurons, n_edges, index
    def to_dict() -> dict
```

`metadata` holds the JSON-serializable summaries; `diagnostics` holds the full computation
(including the row/column norm vectors and the histogram) used by `--stats` and
`--save-data`; `loaded_from` records where an object was read from and is never serialized,
so `to_dict()` round-trips byte-for-byte. Phase 01's `ValidationReport`/`ValidationIssue`
collection and the `...Error(ValueError)`-with-`.report` convention are reused verbatim.

## 13. Algorithm

1. **Resolve** inputs (`--input`, `--glob`, `--include-filtered`) into a deterministic list
   of Phase 01 JSON artifacts; directory inputs are searched **recursively** (the artifact
   sits one level below the processed root).
2. **Load** the payload (`read_json`) and build `ZMatrixConfig` (including the input's
   `metadata.filters`).
3. **Validate** the input with Phase 01's `validate_payload_schema`, then the config, the
   neuron list and the pairs (§17).
4. **Fuse**: vectorized `unified_weights(pre_z, post_z, eps, alpha, zero_policy)`.
5. **Assemble** the directed `Z` by scatter (after duplicate-coordinate and self-loop
   checks); the diagonal stays exactly `0.0`.
6. **Normalize** (`--normalize`) then **symmetrize** (`--symmetrize`/`--symmetric`); the
   effective matrix is `Z_sym` in symmetric mode.
7. **Diagnose**: counts, statistics, row/column norms, histogram, Frobenius norm, σ₁,
   symmetric spectral radius, density/reciprocity.
8. **Validate the produced payload** (`validate_z_payload_schema` + shape/diagonal/
   symmetry/finiteness checks).
9. **Write** the JSON, then the requested derived artifacts (sidecar, PNG, save-data);
   `--dry-run` writes nothing.
10. **Report**: optional `--stats` block, then the completion box (always, on success).

## 14. CLI reference

```
python -m src.matrices.build_square_matrix \
  -i data/processed/FB4Yaffect_FB45_999prePost_001_all/parsed_graph.json \
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

Exit codes mirror Phase 01: `0` success, `1` any input failed validation/writing (or an
explicitly requested `--interactive` window could not open), `2` no input matched.

| Requirement | Option | Behaviour |
|---|---|---|
| **A** | `--plot` | heatmap PNG at `<stem>/z_matrix.png` **and** a matplotlib GUI popup via `plt.show()`; colormap from `--cmap` (default `viridis`, `plasma` also useful). The popup is skipped automatically on a file-only backend and can be forced off with `--no-popup`; the PNG is written either way |
| **J** | `--interactive` | opens a matplotlib window of the **effective** matrix with an `mplcursors` hover cursor: every cell annotates its source neuron, target neuron, unified weight and `(i, j)` (§16.1). Writes **no** artifact, never changes the matrix/`config_hash`/filenames, opens nothing under `--dry-run`, degrades to a warning on a headless backend, and fails the run (`1`) only when `mplcursors` is missing. `--plot --interactive` writes the PNG and suppresses the plain `--plot` popup (one window) |
| **K** | `--no-popup` | with `--plot`: save the PNG without opening the window. Scoped to `--plot`; it never suppresses an explicitly requested `--interactive` window |
| **B** | `--symmetric` | `Z_sym` replaces `Z` in the JSON, the NPZ and the plot; `config.symmetric` + save-data `symmetric` record it; enables a real eigen-decomposition with real eigenvalues/orthogonal eigenvectors |
| **C** | `--normalize` | `rows`, `cols`, `unit` (÷ `max abs weight`) plus `spectral`, `zscore-nonzero`, `none`; mode recorded in `metadata`, `config` and the save-data file |
| **D** | `--config-hash` | forces `<config_hash8>` into the filenames; non-default configs get it automatically. Hash covers `symmetric`, `normalize`, `filters`, the unified-weight parameters (`eps`, `alpha`, `zero_policy`), `symmetrize` and `dtype` |
| **E** | `--no-sidecar` | suppresses `z_matrix.npz`; the JSON stays canonical and `load_z_matrix()` still works |
| **F** | `--strict` | fails on any validation issue (warnings escalated); applies to both JSON and NPZ writing (a failed cache write is always an error and never leaves a partial file) |
| **H** | `--save-data` | writes `<stem>/z_matrix.data.json` (§6) with per-pair weights, `pre_z`/`post_z`, the symmetric flag, the normalization mode, `min/max/mean/std`, histogram bins, sparsity, positive/negative counts, row/column norms, spectral radius (symmetric mode only) and the Phase 01 density/reciprocity |
| **I** | `--stats` | prints the summary statistics block to the terminal, then the popup-style **ASCII** box. Terminal-only; never opens a window |

Example commands:

```bash
# canonical artifact (z_matrix.json + z_matrix.npz)
python -m src.matrices.build_square_matrix \
  -i data/processed/FB4Yaffect_FB45_999prePost_001_all/parsed_graph.json -o data/processed --stats

# symmetric + row-normalized variant, with plot, save-data and hashed filenames
python -m src.matrices.build_square_matrix \
  -i data/processed/FB4Yaffect_FB45_999prePost_001_all/parsed_graph.json -o data/processed \
  --symmetric --normalize rows --config-hash --plot --save-data --stats

# JSON-only run (no cache) over a whole processed tree, strict
python -m src.matrices.build_square_matrix -i data/processed -o data/processed \
  --no-sidecar --strict --include-filtered

# CI-safe plot (no GUI window) and no writes at all
python -m src.matrices.build_square_matrix -i <artifact> -o data/processed --plot --no-popup \
  --cmap plasma
python -m src.matrices.build_square_matrix -i <artifact> -o data/processed --dry-run --stats

# interactive inspection of the effective matrix (hover tooltips, no artifact written)
python -m src.matrices.build_square_matrix -i <artifact> -o data/processed --interactive

# PNG plus the hover-tooltip window in a single run (one window only)
python -m src.matrices.build_square_matrix -i <artifact> -o data/processed --plot --interactive
```

## 15. Terminal outputs

**`--stats` block** (terminal only):

```
Phase 02 -- Z-matrix statistics
  shape               : 113 x 113
  stored entries      : 1350 of 12769
  min / max           : -0.563605 / 0.507692
  mean / std          : -0.096312 / 0.195449
  sparsity            : 0.894275 (fraction of zero entries)
  positive / negative : 302 / 1048  (5 exact zero weight(s))
  row norms (L1)      : min ... max ... mean ... std ...
  col norms (L1)      : min ... max ... mean ... std ...
  spectral radius     : n/a (run with --symmetric)      # or the value in symmetric mode
  symmetrized radius  : 2.433314 (|eigenvalue| of (Z + Z.T)/2)
  frobenius norm      : 8.005822
  density / reciprocity: 0.107064 / 574
```

**Post-run summary box** (always printed on success; terminal only — *not* a GUI popup):

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

The box reflects the real outcome: `skipped (--no-sidecar)`, `skipped (--dry-run)`,
`not requested (--plot)`, `not requested (--save-data)`, `Phase 02 dry-run (nothing
written)` when nothing was persisted.

## 16. Plot and GUI popup rules

* Two options open a window — `--plot` (after saving the PNG) and `--interactive`
  (§16.1). Nothing else in Phase 02 touches a GUI.
* matplotlib is imported lazily inside `plot_matrix`/`build_interactive_figure`/
  `show_interactive_matrix`, so importing the module never selects a backend. The PNG is
  150 dpi with a colorbar (`unified weight w`) and source/target index axes (the reference
  heatmap is 1093×923 px); the interactive figure is the same picture at 100 dpi sized for
  a screen window.
* Headless detection (`can_popup`) asks matplotlib's backend registry
  (`backend_registry.resolve_backend(name)` → GUI framework, `None` for file-only
  backends) with a fallback to the exact non-interactive name set
  `NON_INTERACTIVE_BACKENDS = {agg, cairo, pdf, pgf, ps, svg, template}` for older
  matplotlib. **Substring matching was a bug**: `"agg" in "qtagg"` is `True`, so the
  desktop backends `qtagg`/`tkagg`/`gtk3agg`/`wxagg` were misclassified as headless and
  no window could ever appear (fixed in this revision; §21).
* Suppression is per-flag: `--plot`'s popup is skipped on a file-only backend or with
  `--no-popup` (PNG still written, reason logged); `--interactive`'s window is skipped only
  on a file-only backend (reason logged, `mplcursors` still required). CI and the test
  suite pin `Agg`, so neither ever blocks.
* The `--stats` output and the summary box are printed to **stdout** and are entirely
  separate from the GUI windows; a run with `--interactive` adds one
  `Interactive: window shown … / skipped (…)` line to the box (absent otherwise, so the
  default box is unchanged).

### 16.1 Interactive mode (`--interactive`)

**Window.** `show_interactive_matrix()` builds `build_interactive_figure(z_matrix)` —
`imshow(z_matrix.matrix, cmap=…, interpolation="nearest", aspect="equal")` plus a colorbar —
then `attach_cell_cursor(image, z_matrix)` and `plt.show()` (blocking). The array rendered is
the **effective** matrix, i.e. bit-identical to the PNG and to `npz["matrix"]`, so
`--normalize`/`--symmetric`/`--strict` need no special handling: the window simply shows
whatever was validated and written.

**Attachment (mplcursors).**

```python
cursor = mplcursors.cursor(image, hover=True, highlight=False,
                           annotation_kwargs=INTERACTIVE_ANNOTATION_KWARGS)

@cursor.connect("add")
def _on_add(selection):
    row, col = (int(part) for part in selection.index)   # AxesImage: index == (row, col)
    selection.annotation.set_text(format_cell_tooltip(z_matrix, row, col, raw_weights=...))
```

* `hover=True` — no click needed; `highlight=False` — mplcursors has no `make_highlight`
  handler for `AxesImage`, so the default would emit a `UserWarning` and copy the image on
  every hover.
* `selection.index` is mplcursors' `(row, col)` tuple for an `AxesImage` (verified against
  mplcursors 0.7.1), which maps directly onto `(i, j)`; `selection.target.index` is gone
  since mplcursors 0.6.
* mplcursors is imported lazily inside `attach_cell_cursor`; if it is missing the run fails
  with `ZMatrixValidationError("mplcursors is required for --interactive … conda install -c
  conda-forge mplcursors")` and exit code `1`.

**Tooltip text** (`format_cell_tooltip`, a pure function — the single source of truth, unit
tested without a GUI):

```
source: <neuron_order[row]>
target: <neuron_order[col]>
unified weight: <matrix[row, col]>
(i, j) = (row, col)
```

plus `stored edge: none (absent)` (and an `(absent edge)` marker on the weight line) when
neither `(row, col)` nor — in symmetric mode — `(col, row)` is a stored pair, and plus
`raw unified weight: …` / `raw unified weight (reverse direction): …` when `--normalize` or
`--symmetric` makes the displayed value differ from the originating pair's weight. Neuron
names come from `ZMatrix.neuron_order` (never from disk), so the tooltip is correct with
`--no-sidecar` and for neurons without any edge; out-of-range indices degrade to
`<row n>` / `<col n>` instead of raising inside the GUI callback. A stored weight of exactly
`0.0` (zero-in-zero-out) is *not* reported as absent — presence is decided by the pair
lookup, not by the value.

**Execution flow.** `main()` calls `show_interactive_matrix` as the last step per input,
after `write_artifact_set` succeeded (a `--strict`/write failure `continue`s earlier, so no
window opens on a failed run) and before the summary box is printed. `--dry-run` warns and
opens nothing. With several inputs one window opens per artifact, sequentially, after a
one-off warning. `--plot --interactive` passes `popup=False` to `write_artifact_set`, so the
PNG is written and exactly one window opens.

## 17. Validation rules

| Check | Severity |
|---|---|
| Input artifact conforms to the Phase 01 schema | Error |
| `neurons` non-empty; `neuron_id` unique, non-empty, contains no `--` | Error |
| Pair endpoints ∈ `neuron_order` | Error |
| No duplicate coordinate, no self-loop | Error |
| `pre_z`/`post_z` finite and `≥ 0` | Error |
| `eps > 0`, `alpha > 0`, valid `--zero-policy`/`--symmetrize`/`--normalize`/dtype | Error |
| `matrix.shape == (N, N)`, diagonal exactly `0.0`, `matrix_symmetric` exactly symmetric | Error |
| All emitted entries finite (guards `NaN`/`Infinity` in JSON) | Error |
| Payload conforms to the Phase 02 schema | Error |
| Sidecar/plot/save-data write failures | Error (never leaves a partial file) |
| Empty `pairs` (all-zero matrix) | Warning (`--strict` ⇒ error) |
| `N < 2` (1×1 matrix, density undefined) | Warning |
| Filtered input | Informational log line (output name carries `.filtered`) |
| Popup skipped (headless/`--no-popup`) | Informational log line |

The filtered-input and popup notices are deliberately **not** validation warnings so
`--strict` stays usable on filtered artifacts and in headless CI.

## 18. Invariants (asserted in tests)

1. `matrix.shape == (N, N) == (len(neuron_order), len(neuron_order))`.
2. `matrix[i, i] == 0.0` for every `i`.
3. `matrix_symmetric == symmetrize(matrix, method)` **exactly**; in symmetric mode
   `matrix == matrix_symmetric`.
4. With `normalize = none`, `matrix[i, j] == pairs[].weight` for every stored edge.
5. `pairs[].weight` is pre-normalization, so the raw directed `Z` is recoverable from
   `pairs` alone.
6. Every emitted float is finite and `-0.0` never appears.
7. Both artifacts written by the CLI are byte-identical across runs under a fixed
   `SOURCE_DATE_EPOCH`.
8. `load_z_matrix(path).to_dict()` reproduces the artifact byte-for-byte.
9. `np.load(sidecar, allow_pickle=False)["matrix"]` is exactly `np.array(payload["matrix"])`,
   and the sidecar's digest matches the JSON's bytes.
10. `--no-sidecar` leaves exactly one file and `load_z_matrix()` still works.

## 19. Determinism

* `created_utc` is the only non-deterministic field, and it derives from
  `SOURCE_DATE_EPOCH` (verified: repeated runs produce identical JSON and NPZ sha256).
* Ordering: `neuron_order` verbatim from the input; `pairs` re-sorted by
  `(source, target)`; arrays in the sidecar always in the same order and endianness.
* `numpy_version` is recorded in both artifacts because the `.npy` header is part of the
  sidecar's byte layout. Cross-platform bit-identity of `log`/`tanh` is **not** claimed.
* `--interactive` cannot affect determinism: it writes nothing, and `--plot --interactive`
  produces byte-identical JSON/NPZ/PNG to `--plot` alone (the suppressed popup is the only
  difference, and a popup is not part of any artifact).

## 20. Testing strategy

**Framework**: `pytest` 9.1.1 with the existing `pytest.ini` (`pythonpath = .`,
`--strict-markers`, the registered `slow` marker — no new markers, no plugin assumptions).
The two Phase 02 modules pin `matplotlib.use("Agg")` so no test can block on a window.

```
fast   /home/mr-miracle/miniconda3/envs/eigen-decomposition/bin/python -m pytest -m "not slow" -q
full   /home/mr-miracle/miniconda3/envs/eigen-decomposition/bin/python -m pytest -q
```

Baseline: **Phase 01 has 106 tests (102 fast + 4 slow), unchanged**; Phase 02 has 162
(`tests/test_build_square_matrix.py` 138, `tests/test_z_matrix_schema.py` 24) = 159 fast + 3
slow. This revision adds 35 fast tests — 19 for the `--interactive` window and 16 for the
`can_popup` backend classification — and modifies no existing test: the summary-box and
`can_popup` cases keep passing unchanged because the new behaviour is purely additive.

The `--interactive` tests stay headless: the tooltip text is exercised through the pure
`format_cell_tooltip`, the window path through `build_interactive_figure` +
`attach_cell_cursor` (with mplcursors' own `select_at` dispatching a *real* image selection),
and `plt.show` is replaced by a spy — a real GUI session is verified manually
(`qtagg` + `$DISPLAY`, hover tooltips checked against known weights). Tests that need the
package use `pytest.importorskip("mplcursors")`, and the "dependency missing" path is
covered by monkeypatching `builtins.__import__`.

**Coverage highlights**: the five documented worked examples and an independent stdlib
implementation of the rule; both `zero_policy` branches; no `RuntimeWarning`
(`warnings.simplefilter("error")` and a subprocess test on `zero_z.gv`); boundedness and
monotonicity; exact tiny-matrix values and the scatter identity; `matrix_symmetric` rules;
all six normalizations (including the degenerate all-identical case); all four
symmetrizations; config-hash sensitivity/stability and the naming policy (including
`--config-hash` and `.filtered`); the sidecar contract, fixed zip timestamp, byte
reproducibility of both artifacts, round-trip loading from both paths, stale-cache
fallback with a warning and a direct tampered-cache error; `--save-data` payload contract;
geometry of the `--stats` block and the completion box; `--plot` PNG validity (magic bytes
+ IHDR size) and the headless path; dry-run behaviour; exit codes `0/1/2`; input-validation
error codes; schema conformance and all negative schema cases; **no `*.csv` anywhere**;
Phase 01's `pairs` still carry no `weight` key; the `--interactive` tooltip contract (names,
weight, `(i, j)`, absent edges, stored zero weights, raw/reverse weight lines, out-of-range
indices), the mplcursors attachment and the `can_popup` backend classification (16
parametrised names).

**Reference numbers** (default `eps=0.1`, `alpha=1.0`, reference dataset — asserted by the
slow tests):

| Quantity | Expected |
|---|---|
| `n_neurons` / `n_edges` | 113 / 1355 |
| `n_stored_nonzero` / `n_matrix_zeros` | 1350 / 11419 |
| `sparsity` | 0.894275197745 |
| positive / negative / zero weights | 302 / 1048 / 5 |
| `matrix_stats` min / max | −0.563605108055 / 0.507692307692 |
| `matrix_stats` mean / std | −0.096311952675 / 0.195449331550 |
| `weight_stats` mean / mean_abs / sum | −0.095956558015220 / 0.172007923289 / −130.021136110623 |
| `row_norm_stats.max` / `col_norm_stats.max` | 7.021678574677 / 7.592547154392 |
| `frobenius_norm` | 8.005821952321897 |
| `largest_singular_value` | 4.441772842851828 |
| `symmetrized_spectral_radius` | 2.433313838239482 |
| `spectral_radius` (non-symmetric) | `null` |
| `density` / `reciprocity` | 0.1070638432364096 / 574 |
| `Z[ExR7(ring)_L_1 → ExR7(ring)_L_2]` | −0.102144 |
| `Z_sym` for the same edge | −0.09997766037735847 |
| symmetric + `unit` variant | 2136 stored, spectral radius 4.317409, hash `f8652585` |

## 21. Risks & mitigations

| Risk | Mitigation |
|---|---|
| **`eps = 0.1` shifts the neutral point** to `post_z = pre_z + eps`: equal pre/post is *attenuating* (`0.6/0.6 → −0.09`), and on this dataset 1048/1355 (77%) weights are negative with `mean w = −0.096` (vs 53% / −0.037 at `eps = 0`) | the rule is exactly as specified in `zscore_context.md`; `--eps`/`--alpha` expose it and the bias is reported in `metadata` so Phases 03–04 can see it. **Flagged for sign-off (§26)** |
| `post_z = 0` would map to maximal *attenuation* under the raw formula | explicit `--zero-policy`, default `zero-in-zero-out`; no effect on the reference dataset |
| `log(0)` → `RuntimeWarning` / `-inf` | `np.errstate` guard, finiteness assertion before writing, dedicated fixture + subprocess test |
| `-0.0` breaks byte determinism | canonicalized to `0.0` |
| `json.dumps` emits `NaN`/`Infinity` (invalid JSON) | finiteness asserted pre-write; `src/utils/io.py` unchanged |
| Two artifacts (JSON + NPZ) drifting into two sources of truth | the cache is strictly derived, keyed by the JSON's sha256, verified on load and ignored with a warning when stale; config/metadata/pairs always come from the JSON |
| Cache byte-instability / `allow_pickle` risk | verified fixed zip `date_time`, fixed-endian dtypes, `allow_pickle=False` everywhere |
| A failed sidecar/plot/save-data write losing the artifact | JSON written first; the cache is built in memory and `os.replace`-d; failures are errors, never partial files |
| A GUI popup blocking CI | file-only-backend auto-detection plus `--no-popup`; the PNG is still produced, the tooltip window only warns |
| `qtagg`/`tkagg`/`gtk3agg`/`wxagg` misread as headless (substring `"agg"`) | `can_popup` now uses matplotlib's backend registry (`resolve_backend()[1] is not None`) with an exact-name fallback; regression-tested for 8 GUI and 8 file-only names (fixed in this revision — previously **no** window could ever open) |
| `mplcursors` missing on a machine that asks for `--interactive` | pinned in `environment.yml`, imported lazily, and the failure is an actionable `ZMatrixValidationError` + exit code `1` (the artifacts are still written) |
| `highlight=True` default warns for `AxesImage` and copies the image on every hover | `highlight=False` explicitly, plus an opaque annotation bbox for readability |
| A blocking window in a recursive batch run | one warning up front; one window per artifact, sequentially, and nothing is opened under `--dry-run` |
| Two windows from `--plot --interactive` | the plain `--plot` popup is suppressed whenever `--interactive` is set |
| Degenerate histogram range (all stored weights identical) | collapses to a single bin with `degenerate: true` |
| Non-default config silently overwriting the canonical artifact | `config_hash8` filename segment (automatic for non-default configs) |
| Dense `N×N` JSON not scaling | fine at `N = 113`; a sparse/alternative cache format is a documented future option (§24) |

## 22. Acceptance criteria

1. Default run on the reference artifact reproduces every number in §20 with zero
   validation errors.
2. Artifacts: `z_matrix.json` always; `z_matrix.npz` unless `--no-sidecar`;
   `z_matrix.png` with `--plot`; `z_matrix.data.json` with `--save-data` — **no CSV**.
3. `--symmetric` puts `Z_sym = (Z + Zᵀ)/2` in the JSON, the NPZ and the plot, and both the
   JSON `config.symmetric` and the save-data `symmetric` flag record it.
4. `--normalize rows|cols|unit` satisfy their defining properties and are recorded in
   `metadata`/`config`/save-data.
5. Non-default configs (and `--config-hash`) produce `<config_hash8>` filenames; the hash
   incorporates `symmetric`, `normalize`, `filters` and the unified-weight parameters.
6. `--plot` writes a valid PNG **and** opens the popup on an interactive backend; `--stats`
   prints the statistics block plus the ASCII box; the box is terminal-only.
7. `--strict` fails on any validation issue; the ten §18 invariants hold.
8. Both artifacts are byte-stable under a fixed `SOURCE_DATE_EPOCH`;
   `load_z_matrix()`/`load_sidecar_arrays()` round-trip.
9. The Phase 01 suite still passes unchanged.
10. `--interactive` opens an `mplcursors` window whose tooltip reports source neuron, target
    neuron, unified weight and `(i, j)` for every cell (verified on `qtagg` + `$DISPLAY`);
    it writes no artifact and leaves JSON/NPZ/PNG byte-identical to a run without it.
11. `--plot --interactive` writes the PNG and opens exactly one window. Under `--dry-run`
    with `--interactive` no window opens and the box still prints, with
    `Interactive: skipped (--dry-run)`. A run without `--interactive` prints the box
    unchanged (no `Interactive:` line).

## 23. Deliverables

- `execution-plans/02_build_square_matrix.md` (this document)
- `src/matrices/__init__.py`, `src/matrices/build_square_matrix.py`
- `tests/test_build_square_matrix.py`, `tests/test_z_matrix_schema.py`,
  `tests/fixtures/zero_z.gv`
- `README.md` Phase 02 section (inputs/outputs, the unified weight, the JSON+NPZ design,
  the full CLI block, example commands, the heatmap, the `--interactive` tooltip window and
  its `--plot` comparison table, Phase 03 loading, symmetric mode vs eigen-decomposition,
  GUI windows vs terminal summary)
- Artifacts: `data/processed/<stem>/z_matrix.json`, `z_matrix.npz` (+ `z_matrix.png`,
  `z_matrix.data.json` for the demo variant)
- **No changes** to `src/utils/io.py`, `src/parsing/*`, `pytest.ini`, the existing tests or
  the existing `parsed_graph.json`; `environment.yml` only gains the `mplcursors` pin
  required by `--interactive` (§16.1)

## 24. Out of scope (explicit non-goals)

Eigendecomposition/SVD/scree plots (Phase 03); clustering, UMAP/PCA, loadings (Phase 04);
reduced dynamical models and Jacobians (Phases 05–06); pair thresholding/filtering
(Phase 01); **multi-file union** (deferred; would be a separate, explicitly flagged
addition); `cent`-weighted matrix entries (carried as Phase 01 metadata for Phase 04);
sparse or alternative cache formats (HDF5, Parquet, per-array `.npy`).

Also explicitly out of scope for `--interactive`: embedding the window in a notebook or web
backend, exporting the tooltip data, click-to-select or neighbour highlighting, and any
interaction that would *write* an artifact — the window is a read-only view of the matrix
that was just built (§16.1).

## 25. How Phase 03 consumes these artifacts

```python
from src.matrices.build_square_matrix import load_sidecar_arrays, load_z_matrix

# fast path: arrays only (no JSON parsing, no pickle)
arrays = load_sidecar_arrays("data/processed/<stem>/z_matrix.npz")   # or pass the .json path
z_directed = arrays["matrix"]              # the effective matrix (Z_sym if --symmetric)
z_symmetric = arrays["matrix_symmetric"]   # always symmetric
order = arrays["neuron_order"]

# full object: config, metadata, per-pair weights (and it verifies the cache)
z = load_z_matrix("data/processed/<stem>/z_matrix.json")
config = z.config            # symmetric / normalize / eps / alpha / config_hash
index = z.index              # neuron_id -> row/column
```

* **Symmetric modes** — with `config.symmetric == True` (or by using `matrix_symmetric`),
  Phase 03 can run a true eigen-decomposition on a symmetric matrix, yielding **real
  eigenvalues and orthogonal eigenvectors** (the requirement behind `--symmetric`).
* **Directed modes** — the directed `Z` (or `matrix` with `symmetric == False`) goes through
  SVD, which is what the directed path requires.
* `metadata.spectral_radius`/`symmetrized_spectral_radius`, `largest_singular_value` and
  `frobenius_norm` are informational sanity checks for Phase 03 to cross-validate against —
  Phase 03 remains the authority on the spectra.
* `metadata.config_hash` ties a matrix to its exact construction recipe, so a Phase 03
  artifact can record which variant it decomposed.

## 26. Flagged decisions and defaults

1. **Unification defaults `eps = 0.1`, `alpha = 1.0`** exactly as documented, exposed as
   `--eps`/`--alpha`, accepting the measured negative-weight bias (77% negative,
   `mean w = −0.096`).
2. **`zero_policy = "zero-in-zero-out"`** — a zero `pre_z`/`post_z` yields `w = 0.0`, not the
   formula limit. No effect on the reference dataset.
3. **`matrix` = effective matrix, `matrix_symmetric` always present** — so `--symmetric`
   replaces `Z` with `Z_sym` as required while Phase 03 always has a symmetric matrix.
4. **Order is normalize → symmetrize**, keeping `matrix_symmetric == symmetrize(matrix)`.
5. **`--config-hash` semantics**: automatic for non-default configs, forced by the flag,
   computed only over matrix-defining fields (IO flags excluded).
6. **Two GUI windows exist** (`--plot`'s plain popup and `--interactive`'s tooltip window),
   each auto-suppressed on a file-only backend; `--no-popup` governs `--plot` only, the
   `--stats`/completion-box output stays terminal-only. When both flags are given the plain
   popup is suppressed so exactly one window opens.
7. **`can_popup()` was fixed to use matplotlib's backend registry** (exact-name fallback)
   instead of substring matching: `"agg" in "qtagg"` is `True`, so the desktop backend used
   to be classified as headless and *no* window could ever open. This is the one behavioural
   change to pre-existing code; it restores the `--plot` popup that requirement A always
   specified, and is required for `--interactive` to work at all.
8. **`--interactive` is presentation-only**: no artifact, no `config_hash`/filename/schema
   impact, no `ZMatrixConfig` field, `--dry-run` opens nothing, a missing `mplcursors` is an
   error (`1`) while a headless backend is only a warning (`0`).
9. **No dependency, config or existing-code changes** beyond the `mplcursors` pin and the
   `can_popup()` fix — `pytest.ini`, `src/utils/io.py`, `src/parsing/*` and every Phase 02
   schema/artifact contract are untouched.
